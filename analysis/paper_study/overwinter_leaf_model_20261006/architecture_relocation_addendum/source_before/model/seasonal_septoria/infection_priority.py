"""Infection-event timing and top-three grain-fill/yield relevance runtime."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

from .infection_events import EventParameters,simulate_events
from .structural import canopy_host
from .wetness import duration_exposure
from .yield_relevance import top_three_infection_overlap,upper_leaf_had_loss,transfer_had_yield_loss

DEFAULT_MODEL=Path(__file__).with_name('infection_priority_model.json')


def predict_infection_priority(data,accumulation,thresholds,model=None,*,
        reference_leaf_area_index=None,functional_loss_fraction=None,
        reference_yield_t_ha=None,return_daily=False):
    """Forecast effective infection and symptom dates without disease labels.

    Reference leaf area and functional area loss must both be supplied for a
    numerical conditional yield transfer. Infection indicators are never used
    as area-loss fractions. The default grain-fill proxy is anthesis to BBCH85.
    """
    if model is None:model=DEFAULT_MODEL
    if isinstance(model,(str,Path)):model=json.loads(Path(model).read_text())
    if not isinstance(model,dict) or model.get('schema_version')!=1:
        raise ValueError('A schema-1 infection-priority model is required.')
    t=np.asarray(data.temperature,float);a=np.asarray(accumulation,float)
    if a.shape!=t.shape or not hasattr(data,'maximum_temperature'):
        raise ValueError('Development and maximum-temperature forcing are required.')
    if not thresholds[51]<model['anthesis_threshold']<thresholds[85]:
        raise ValueError('Anthesis threshold must lie between heading and soft dough.')
    metadata=data.metadata.sort_values('field_index')
    if (len(metadata)!=len(t) or metadata.field_index.duplicated().any()
            or not np.array_equal(metadata.field_index.to_numpy(),np.arange(len(t)))):
        raise ValueError('One metadata row per ordered forcing field is required.')
    lengths=metadata.forcing_days.to_numpy(float)
    if (not np.isfinite(lengths).all() or np.any(lengths!=np.floor(lengths))
            or np.any((lengths<1)|(lengths>t.shape[1]))):
        raise ValueError('Each forcing window requires a valid positive integer length.')
    valid=np.arange(t.shape[1])[None,:]<lengths[:,None]
    fitted=model['event_model']
    active,renewal,area,births=canopy_host(a,t,thresholds,**fitted['host_parameters'])
    exposure=duration_exposure(t,data.maximum_temperature,data.humidity,data.rain,
        fitted['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    active&=valid[:,:,None]
    event=simulate_events(np.where(valid,t,0.),np.where(valid,exposure,0.),active,
        EventParameters(**fitted['parameters']))
    infection=np.full((len(t),3),np.datetime64('NaT','D'),dtype='datetime64[D]')
    symptom=infection.copy();rows=[];hads=[]
    supplied=(reference_leaf_area_index is not None,functional_loss_fraction is not None)
    if supplied[0]!=supplied[1]:
        raise ValueError('Reference leaf area and functional loss must be supplied together.')
    if reference_yield_t_ha is not None and not all(supplied):
        raise ValueError('Numerical yield requires reference leaf area and functional loss.')
    if all(supplied):
        supplied_area=np.asarray(reference_leaf_area_index,float)
        supplied_loss=np.asarray(functional_loss_fraction,float)
        if supplied_area.shape[:2]!=t.shape or supplied_loss.shape!=supplied_area.shape:
            raise ValueError('Functional canopy arrays must match field/day forcing.')
    for meta in metadata.itertuples():
        field=int(meta.field_index);raw_length=float(meta.forcing_days)
        if not np.isfinite(raw_length) or raw_length!=int(raw_length) or not 1<=raw_length<=t.shape[1]:
            raise ValueError('Each forcing window requires a valid positive integer length.')
        length=int(raw_length);dates=pd.date_range(meta.sowing_date,periods=length)
        for target,indices in [(infection,event.infection_day),(symptom,event.symptom_day)]:
            for leaf,day in enumerate(indices[field,:3]):
                if 1<=day<=length:target[field,leaf]=dates[int(day)-1].to_datetime64().astype('datetime64[D]')
        def stage_date(threshold):
            found=np.flatnonzero(a[field,:length]>=threshold)
            return pd.NaT if not len(found) else dates[int(found[0])]
        anthesis=stage_date(model['anthesis_threshold']);end=stage_date(thresholds[85])
        overlap=top_three_infection_overlap(dates,infection[field:field+1],[anthesis],[end])
        expressed=top_three_infection_overlap(dates,symptom[field:field+1],[anthesis],[end])
        record=dict(field_id=getattr(meta,'field_id',str(field)),field_index=field,
            anthesis_proxy_date=anthesis,soft_dough_date=end,
            complete_grain_fill_window=bool(overlap['complete_window'][0]),
            grain_fill_days=float(overlap['grain_fill_days'][0]),
            days_with_any_top3_infected=float(overlap['days_with_any_top3_infected'][0]),
            days_with_all_top3_infected=float(overlap['days_with_all_top3_infected'][0]),
            days_with_any_top3_symptomatic=float(expressed['days_with_any_top3_infected'][0]))
        for leaf in range(3):
            record[f'F{leaf+1}_infection_date']=infection[field,leaf]
            record[f'F{leaf+1}_symptom_date']=symptom[field,leaf]
            record[f'F{leaf+1}_infected_grain_fill_days']=overlap['infected_grain_fill_days'][0,leaf]
            record[f'F{leaf+1}_infected_grain_fill_fraction']=overlap['infected_grain_fill_fraction'][0,leaf]
        rows.append(record)
        if all(supplied):
            had=upper_leaf_had_loss(dates,supplied_area[field:field+1,:length],
                supplied_loss[field:field+1,:length],[anthesis],[end])
            hads.append(had['lost_had3'][0])
    yield_result=None if not all(supplied) else transfer_had_yield_loss(hads,model['yield_slopes'],
        reference_yield_t_ha=reference_yield_t_ha)
    return dict(model_id=model.get('model_id'),field_outputs=pd.DataFrame(rows),
        infection_dates=infection,symptom_dates=symptom,yield_transfer=yield_result,
        daily_events=event if return_daily else None,conditional_on_inoculum_presence=True,
        daily_valid_mask=(np.arange(t.shape[1]+1)[None,:]<=lengths[:,None]) if return_daily else None,
        calibrated_occurrence_probability=False,direct_infection_dates_validated=False,
        primary_observation='zero/positive symptom detection; top3 assessment brackets',
        yield_window='transferred anthesis proxy to soft dough; truncated-window adaptation')
