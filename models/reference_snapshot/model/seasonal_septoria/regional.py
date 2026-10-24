"""Regional seasons with the copied causal T-P-V process and fixed endpoints."""

import numpy as np
import pandas as pd

from .core import simulate_season
from .host import cohort_inputs


def tpv_accumulation(temperature, maximum_temperature, latitude, day_of_year,
                     included, calibration, vernalization_required=True):
    """Vectorized equivalent of the archived winter T-P-V transformation.

    Disabling vernalization constitutes a sensitivity to vernalization demand;
    it does not estimate or validate new spring-wheat stage thresholds.
    """
    t, tx = np.asarray(temperature,float), np.asarray(maximum_temperature,float)
    lat,doy = np.asarray(latitude,float),np.asarray(day_of_year,float)
    mask_raw = np.asarray(included)
    mask = mask_raw.astype(bool)
    if (t.ndim != 2 or t.size == 0 or tx.shape != t.shape or mask.shape != t.shape
            or lat.shape != (len(t),) or doy.shape != (t.shape[1],)
            or not all(np.isfinite(x).all() for x in (t,tx,lat,doy))
            or np.any(abs(lat)>90) or np.any((doy<1)|(doy>366))
            or np.any((mask_raw!=0)&(mask_raw!=1)) or np.any(tx<t-.001)):
        raise ValueError('Invalid regional phenology inputs.')
    if np.any(t[mask]>40):
        raise ValueError('The archived thermal response is unsupported above 40 Celsius daily mean.')
    gdd = np.where(t>30,20-2*(t-30),np.clip(t,0,20))*mask
    # Values outside the crop mask contribute no developmental time.
    gdd[~mask] = 0.
    cumulative = np.cumsum(gdd,axis=1)
    prior = np.column_stack([np.zeros(len(t)),cumulative[:,:-1]])
    declination = .4093*np.sin(2*np.pi*(doy-81)/365)
    daylength = 24/np.pi*np.arccos(np.clip(-np.tan(np.radians(lat[:,None]))*np.tan(declination[None,:]),-1,1))
    pp_factor = np.clip(1-.09*(16-daylength),0,1)
    pp_active = mask & (prior>=calibration['photoperiod_onset_gdd']) & (prior<calibration['photoperiod_stop_gdd'])
    tpp = gdd*np.where(pp_active,pp_factor,1.)
    cumulative_tpp = np.cumsum(tpp,axis=1)
    if not vernalization_required:
        return cumulative_tpp
    prior_tpp = np.column_stack([np.zeros(len(t)),cumulative_tpp[:,:-1]])
    response = np.interp(t,[-4,0,10,16],[0.,1.,1.,0.])
    cumver = np.zeros(len(t))
    increments = np.zeros_like(t)
    for day in range(t.shape[1]):
        active = mask[:,day] & (prior_tpp[:,day]>=calibration['vernalization_onset_tpp'])
        sensitive = prior_tpp[:,day]<calibration['vernalization_stop_tpp']
        updated = cumver+np.where(sensitive,response[:,day],0.)
        hot_early = (updated<10)&(tx[:,day]>30)
        updated = np.where(hot_early,np.maximum(0,cumver-.5*(tx[:,day]-30)),updated)
        cumver = np.where(active,updated,cumver)
        factor = np.where(active&sensitive,.3+.7*np.minimum(cumver/40,1),1.)
        increments[:,day] = tpp[:,day]*factor
    return np.cumsum(increments,axis=1)


def _first_dates(condition, dates):
    """First date along the second axis; absent events retain NaT."""
    found = np.any(condition,axis=1)
    indices = np.argmax(condition,axis=1)
    output = np.asarray(dates,dtype='datetime64[D]')[indices]
    return np.where(found,output,np.datetime64('NaT','D'))


def phenology_window(temperature, maximum_temperature, latitude, day_of_year,
                     included, calibration, terminal_threshold, vernalization_required=True):
    """Locate a terminal event and flag unsupported forcing before it.

    A bounded discovery pass supplies only the location of an endpoint. No
    temperature above40°C enters an accepted crop trajectory: a pre-endpoint
    exceedance is flagged, and post-endpoint weather is excluded. For every
    accepted row the discovery pass equals the original forcing through the
    endpoint, so its event date does not depend on the bounded later values.
    """
    t = np.asarray(temperature,float)
    included = np.asarray(included,bool)
    accumulated = tpv_accumulation(np.minimum(t,40.),maximum_temperature,latitude,
        day_of_year,included,calibration,vernalization_required)
    reached = accumulated>=terminal_threshold
    complete = reached.any(axis=1)
    endpoint = np.argmax(reached,axis=1)
    in_season = included & (np.arange(t.shape[1])[None,:]<=np.where(complete,endpoint,t.shape[1]-1)[:,None])
    unsupported = np.any((t>40)&in_season,axis=1)
    return in_season,complete,endpoint,unsupported


def simulate_grid_seasons(dates, temperature, maximum_temperature, humidity,
                          precipitation, latitude, sowing_dates, parameters,
                          calibration, end_stage=85, vernalization_required=True,
                          cutoffs=(.1,1.,5.), time_step=.25):
    """Produce per-cell crop-season outputs without applying wheat-area weights.

    The main terminal event is predicted BBCH85 (soft dough), rather than an
    inferred observed harvest. Missing terminal development produces a missing
    severity output, not a zero-disease season. A BBCH75 sensitivity uses the
    midpoint between the copied BBCH51 and85 developmental thresholds; that
    interpolation has no independent stage75 validation.
    """
    dates = pd.DatetimeIndex(dates)
    sowing = np.asarray(sowing_dates,dtype='datetime64[D]')
    t,tx,h,r = [np.asarray(x,float) for x in
               (temperature,maximum_temperature,humidity,precipitation)]
    if (dates.hasnans or dates.has_duplicates or len(dates)!=t.shape[1]
            or not dates.equals(pd.date_range(dates[0],dates[-1]))
            or sowing.shape!=(len(t),) or np.isnat(sowing).any()
            or np.any(sowing<dates[0].to_datetime64().astype('datetime64[D]'))
            or np.any(sowing>dates[-1].to_datetime64().astype('datetime64[D]'))):
        raise ValueError('Complete daily dates and in-window sowing dates are required.')
    thresholds = {row['BBCH']:row['Cumulative_t_pp_v_GDD'] for row in calibration['thresholds']}
    if end_stage==85:
        terminal_threshold = thresholds[85]
    elif end_stage==75:
        terminal_threshold = (thresholds[51]+thresholds[85])/2
    else:
        raise ValueError('Supported regional endpoints are BBCH85 or the BBCH75 interpolation sensitivity.')
    included = dates.to_numpy(dtype='datetime64[D]')[None,:]>=sowing[:,None]
    in_season,complete,endpoint,unsupported = phenology_window(t,tx,latitude,dates.dayofyear,
        included,calibration,terminal_threshold,vernalization_required)
    if unsupported.any():
        raise ValueError('The archived thermal response is unsupported above40 Celsius before the crop endpoint.')
    accumulated = tpv_accumulation(t,tx,latitude,dates.dayofyear,in_season,
                                   calibration,vernalization_required)
    active,renewal = cohort_inputs(accumulated,t,thresholds)
    active &= in_season[:,:,None]
    renewal[~active] = 0.
    trajectory = simulate_season(t,h,r,active,renewal,parameters,time_step=time_step)
    row = np.arange(len(t))
    damage = trajectory.damage[row,endpoint+1,:7]*100
    pycnidia = trajectory.pycnidia[row,endpoint+1,:7]*100
    damage[~complete],pycnidia[~complete] = np.nan,np.nan
    visible_dates = np.stack([_first_dates(trajectory.damage[:,1:,:7]*100>=cutoff,dates)
                             for cutoff in cutoffs],axis=-1)
    primary_dates = _first_dates(np.cumsum(trajectory.primary_flow,axis=1).max(axis=2)>=.0001,dates)
    stage_dates = {stage:_first_dates(accumulated>=threshold,dates) for stage,threshold in thresholds.items()}
    return dict(final_damage_percent=damage,final_pycnidia_percent=pycnidia,
                first_visible_dates=visible_dates,primary_exposure_dates=primary_dates,
                endpoint_dates=np.where(complete,dates.to_numpy(dtype='datetime64[D]')[endpoint],np.datetime64('NaT','D')),
                complete_season=complete,stage_dates=stage_dates,
                mass_error=float(np.max(abs(trajectory.state.sum(axis=-1)-1))),
                minimum_state=float(trajectory.state.min()),
                cutoff_percent=np.asarray(cutoffs,float),
                endpoint_stage=end_stage,vernalization_required=vernalization_required)
