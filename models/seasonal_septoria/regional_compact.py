"""Regional operator with compact storage and identical daily diagnostics."""

import numpy as np
import pandas as pd

from .regional import tpv_accumulation,phenology_window,_first_dates
from .host import cohort_inputs
from .rolling_diagnostics import simulate_rolling_diagnostics


def simulate_grid_seasons_compact(dates, temperature, maximum_temperature, humidity,
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
    integration_endpoint = np.where(complete,endpoint,len(dates)-1)
    rolling = simulate_rolling_diagnostics(t,h,r,active,renewal,integration_endpoint,
        [parameters],time_step=time_step,cutoffs=cutoffs)
    damage = rolling['damage'][0,:,:7]*100
    pycnidia = rolling['pycnidia'][0,:,:7]*100
    damage[~complete],pycnidia[~complete] = np.nan,np.nan
    event_days = rolling['first_visible_days'][0,:,:7]
    visible_dates = dates.to_numpy(dtype='datetime64[D]')[np.maximum(event_days,0)]
    visible_dates = np.where(event_days>=0,visible_dates,np.datetime64('NaT','D'))
    primary_days = rolling['primary_exposure_days'][0]
    primary_dates = np.where(primary_days>=0,dates.to_numpy(dtype='datetime64[D]')[np.maximum(primary_days,0)],np.datetime64('NaT','D'))
    stage_dates = {stage:_first_dates(accumulated>=threshold,dates) for stage,threshold in thresholds.items()}
    return dict(final_damage_percent=damage,final_pycnidia_percent=pycnidia,
                first_visible_dates=visible_dates,primary_exposure_dates=primary_dates,
                endpoint_dates=np.where(complete,dates.to_numpy(dtype='datetime64[D]')[endpoint],np.datetime64('NaT','D')),
                complete_season=complete,stage_dates=stage_dates,
                mass_error=rolling['mass_error'],
                minimum_state=rolling['minimum_state'],
                cutoff_percent=np.asarray(cutoffs,float),
                endpoint_stage=end_stage,vernalization_required=vernalization_required)
