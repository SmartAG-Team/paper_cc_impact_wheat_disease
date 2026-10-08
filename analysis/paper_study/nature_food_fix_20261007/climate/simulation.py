"""Local sensitivity adapter, derived from immutable 20261006 climate runner.

Only optional source amount, detection convention and precomputed trajectory
are added. Original season-mask and outcome arithmetic are retained.
"""
import numpy as np
import pandas as pd
from model.seasonal_septoria.overwinter import OverwinterParameters, simulate_overwinter
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.regional import tpv_accumulation, phenology_window
from model.seasonal_septoria.wetness import duration_exposure
from analysis.paper_study.overwinter_leaf_model_20261006.climate import cache_weather as cache
SLOPES=[.0141,.018,.0207]

def simulate_draw_seasons(dates,weather,sowing,latitude,phenology,fit,thresholds,*,return_daily=False,initial_local_source=1.,detection_fraction=.001,trajectory_override=None):
    """Respect actual dates and crop mask; no targets or observed stages enter."""
    dates=pd.DatetimeIndex(dates);raw={key:np.asarray(weather[key],float) for key in cache.FIELDS}
    sowing=np.asarray(sowing,dtype='datetime64[D]');fields,days=raw['tmean_c'].shape
    if not dates.equals(pd.date_range(dates[0],dates[-1])) or len(dates)!=days or sowing.shape!=(fields,):
        raise ValueError('Consecutive dated forcing and one calendar sowing date per draw are required.')
    included=dates.to_numpy(dtype='datetime64[D]')[None,:]>=sowing[:,None]
    finite=np.logical_and.reduce([np.isfinite(raw[key]) for key in cache.FIELDS])
    finite&=(raw['tmax_c']>=raw['tmean_c']-.001)&(raw['rh_mean_pct']>=0)&(raw['rh_mean_pct']<=100)&(raw['precipitation_mm']>=0)
    missing=included&~finite
    first_gap=np.where(missing.any(axis=1),missing.argmax(axis=1),days)
    prefix=included&(np.arange(days)[None,:]<first_gap[:,None])
    safe={key:np.where(finite,raw[key],0.) for key in cache.FIELDS}
    mask,complete,endpoint,unsupported=phenology_window(safe['tmean_c'],safe['tmax_c'],np.asarray(latitude,float),dates.dayofyear,
        prefix,phenology,float(thresholds[85]),True)
    valid=complete&~unsupported
    status=np.where(unsupported,'unsupported_mean_temperature_above40',np.where(complete,'complete',
        np.where(first_gap<days,'missing_weather_before_endpoint','soft_dough_not_reached')))
    mask&=valid[:,None]
    t=np.where(mask,safe['tmean_c'],0.);tx=np.where(mask,safe['tmax_c'],0.)
    a=tpv_accumulation(t,tx,np.asarray(latitude,float),dates.dayofyear,mask,phenology,True)
    host=leaf_host(a,t,thresholds,rank_spacing_units=fit['rank_spacing_units'],forcing_mask=mask,
        juvenile_policy=fit.get('juvenile_policy','handover_31_39'))
    exposure=duration_exposure(t,tx,np.where(mask,safe['rh_mean_pct'],0.),np.where(mask,safe['precipitation_mm'],0.),
        fit['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    trajectory=trajectory_override
    if trajectory is None:trajectory=simulate_overwinter(t,exposure,np.where(mask,safe['precipitation_mm'],0.),host.active,host.renewal,host.area,mask,
        OverwinterParameters(**fit['parameters']),initial_local_source=initial_local_source,
        imported_pressure=np.full_like(t,fit['constant_imported_pressure']),detection_fraction=detection_fraction)
    mass_error=float(np.max(abs(trajectory.state.sum(axis=-1)-1.)))
    if mass_error>1e-10:raise AssertionError('Tissue mass accounting failed.')
    functional=trajectory.damage[:,1:,:3].copy()
    day=np.arange(1,days+1)[None,:,None]
    symptom=trajectory.symptom_day[:,:3]
    functional*=((symptom[:,None,:]>=0)&(day>=symptom[:,None,:]))
    functional*=mask[:,:,None]
    reference=host.area[:,:,:3]/3.
    rows=[]
    for field in range(fields):
        row=dict(status=str(status[field]),valid_complete_season=bool(valid[field]),calendar_sowing_date=pd.Timestamp(sowing[field]),
            BBCH31_date=pd.NaT,BBCH65_date=pd.NaT,BBCH85_date=pd.NaT,grain_fill_days=np.nan)
        stage_indices={stage:int(index[field]) for stage,index in host.stage_day_index.items()}
        for stage in [31,65,85]:
            index=stage_indices[stage];row[f'BBCH{stage}_date']=pd.NaT if index<0 else dates[index]
        anthesis=stage_indices[65];end=stage_indices[85];stem=stage_indices[31]
        gf=np.zeros(days,bool)
        if valid[field]:gf[anthesis:end+1]=True;row['grain_fill_days']=float(gf.sum())
        infected=[];expressed=[];events_in_grain=[]
        for leaf in range(3):
            key=f'F{leaf+1}';infection=int(trajectory.infection_day[field,leaf]);visible=int(trajectory.symptom_day[field,leaf])
            inf_date=pd.NaT if infection<0 or not valid[field] else dates[infection-1]
            sym_date=pd.NaT if visible<0 or not valid[field] else dates[visible-1]
            inf_overlap=(np.arange(1,days+1)>=infection)&(infection>=0)&gf
            sym_overlap=(np.arange(1,days+1)>=visible)&(visible>=0)&gf
            event_in_grain=bool(infection>=anthesis+1 and infection<=end+1 and infection>=0)
            row.update({f'{key}_infection_date':inf_date,f'{key}_symptom_date':sym_date,
                f'{key}_infection_day_after_sowing':np.nan if pd.isna(inf_date) else float((inf_date-pd.Timestamp(sowing[field])).days),
                f'{key}_symptom_day_after_sowing':np.nan if pd.isna(sym_date) else float((sym_date-pd.Timestamp(sowing[field])).days),
                f'{key}_infection_relative_anthesis_days':np.nan if pd.isna(inf_date) or anthesis<0 else float((inf_date-dates[anthesis]).days),
                f'{key}_symptom_relative_anthesis_days':np.nan if pd.isna(sym_date) or anthesis<0 else float((sym_date-dates[anthesis]).days),
                f'{key}_infection_before85':np.nan if not valid[field] else float(infection>=0 and infection<=end+1),
                f'{key}_symptom_before85':np.nan if not valid[field] else float(visible>=0 and visible<=end+1),
                f'{key}_infection_during_grain_fill':np.nan if not valid[field] else float(event_in_grain),
                f'{key}_infected_grain_fill_days':np.nan if not valid[field] else float(inf_overlap.sum()),
                f'{key}_symptomatic_grain_fill_days':np.nan if not valid[field] else float(sym_overlap.sum()),
                f'{key}_infected_grain_fill_fraction':np.nan if not valid[field] else float(inf_overlap.sum()/gf.sum()),
                f'{key}_symptomatic_grain_fill_fraction':np.nan if not valid[field] else float(sym_overlap.sum()/gf.sum())})
            infected.append(inf_overlap);expressed.append(sym_overlap);events_in_grain.append(event_in_grain)
        before=[row[f'F{leaf}_infection_before85'] for leaf in (1,2,3)]
        sym_before=[row[f'F{leaf}_symptom_before85'] for leaf in (1,2,3)]
        row.update(any_top3_infection_before85=np.nan if not valid[field] else float(any(before)),
            all_top3_infection_before85=np.nan if not valid[field] else float(all(before)),
            any_top3_symptom_before85=np.nan if not valid[field] else float(any(sym_before)),
            all_top3_symptom_before85=np.nan if not valid[field] else float(all(sym_before)),
            any_top3_infection_during_grain_fill=np.nan if not valid[field] else float(any(events_in_grain)),
            days_any_top3_infected_grain_fill=np.nan if not valid[field] else float(np.any(infected,axis=0).sum()),
            days_all_top3_infected_grain_fill=np.nan if not valid[field] else float(np.all(infected,axis=0).sum()),
            days_any_top3_symptomatic_grain_fill=np.nan if not valid[field] else float(np.any(expressed,axis=0).sum()),
            days_all_top3_symptomatic_grain_fill=np.nan if not valid[field] else float(np.all(expressed,axis=0).sum()))
        for window,start in [('GS31_85',stem),('GS65_85',anthesis)]:
            ref,loss=np.nan,np.nan
            if valid[field] and start>=0:
                ref=float(reference[field,start:end+1].sum());loss=float((reference[field,start:end+1]*functional[field,start:end+1]).sum())
                assert 0<=loss<=ref+1e-10
            row[f'{window}_reference_had3']=ref;row[f'{window}_lost_had3']=loss
            row[f'{window}_functional_lost_fraction']=np.nan if not np.isfinite(ref) or ref<=0 else loss/ref
            for label,slope in zip(['b0141','b0180','b0207'],SLOPES):row[f'conditional_yield_loss_{window}_{label}_t_ha_per_unit_lai']=loss*slope
        row['mean_temperature_grain_fill_c']=np.nan if not valid[field] else float(safe['tmean_c'][field,gf].mean())
        row['rainfall_grain_fill_mm']=np.nan if not valid[field] else float(safe['precipitation_mm'][field,gf].sum())
        row['exposure_grain_fill_effective_days']=np.nan if not valid[field] else float(exposure[field,gf].sum())
        rows.append(row)
    details=dict(mass_error=mass_error,complete_count=int(valid.sum()))
    if return_daily:details.update(trajectory=trajectory,host=host,accumulation=a,crop_mask=mask,functional_loss=functional,reference_area=reference)
    return pd.DataFrame(rows),details
