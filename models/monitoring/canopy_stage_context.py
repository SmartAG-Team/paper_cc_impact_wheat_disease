"""Causal canopy/weather inputs and a station-balanced calendar-stage control."""
import numpy as np
import pandas as pd

from .canopy_inputs import canopy_prefix

STAGES=(31,51,87)


def calendar_day(dates):
    """A common 365-day calendar; February 29 shares February 28's position."""
    return np.array([pd.Timestamp(2001,d.month,min(d.day,28) if d.month==2 else d.day).dayofyear
        for d in pd.to_datetime(dates)],float)


def stage_context(weather,dates,query_date,latitude,canopy_series):
    """Use exactly the 84 preceding weather days; actual sowing is not an input."""
    weather=np.asarray(weather,float);dates=pd.DatetimeIndex(dates).normalize()
    query=pd.Timestamp(query_date).normalize()
    if weather.shape!=(len(dates),4) or dates.has_duplicates:
        raise ValueError('Unique weather dates with four weather channels required')
    wanted=pd.date_range(query-pd.Timedelta(days=84),query-pd.Timedelta(days=1))
    index=dates.get_indexer(wanted)
    if np.any(index<0):
        raise ValueError('84 complete preceding weather days required')
    previous=weather[index]
    if (not np.isfinite(previous).all() or np.any(previous[:,3]<0) or
            np.any(previous[:,2]>previous[:,0]) or np.any(previous[:,0]>previous[:,1])):
        raise ValueError('84 finite physical weather days required')
    values,names,available=canopy_prefix(canopy_series,query)
    values,names=values.tolist(),list(names)
    for window in [14,42,84]:
        part=previous[-window:]
        values.extend([*part[:,:3].mean(axis=0),float(part[:,3].sum()),
            *part.std(axis=0),float(np.maximum(part[:,0],0).sum()),float((part[:,3]>=1).mean())])
        names.extend([f'stage_weather_{v}_mean_{window}d_c' for v in ['tmean','tmax','tmin']])
        names.append(f'stage_weather_precip_sum_{window}d_mm')
        names.extend([f'stage_weather_{v}_sd_{window}d_{unit}'
            for v,unit in [('tmean','c'),('tmax','c'),('tmin','c'),('precip','mm')]])
        names.extend([f'stage_weather_gdd_base0_sum_{window}d_c_days',
            f'stage_weather_wet_day_fraction_ge1mm_{window}d'])
    angle=2*np.pi*(calendar_day([query])[0]-1)/365
    values.extend([float(np.sin(angle)),float(np.cos(angle)),float(latitude)/60])
    names.extend(['stage_calendar_day_sin','stage_calendar_day_cos','stage_latitude_degrees_per60'])
    assert len(values)==len(names)==85
    return np.asarray(values,float),names,available


def fit_calendar_cdf(records,training_stations):
    """Equal station mass per stage, split among observed cycles at that station."""
    selected=[r for r in records if r['PEP_ID'] in training_stations]
    prior={}
    for index,stage in enumerate(STAGES,1):
        events=pd.DataFrame([dict(station=r['PEP_ID'],date=r['true_dates'][index])
            for r in selected if r['true_dates'][index] is not None])
        if events.empty:
            raise ValueError('At least one training-station event required for every stage')
        weights=1/events.groupby('station').station.transform('size').to_numpy(float)
        values=calendar_day(events.date);order=np.argsort(values,kind='stable')
        prior[stage]=dict(values=values[order],cumulative_weight=np.cumsum(weights[order])/weights.sum())
    return prior


def predict_calendar_cdf(prior,query_dates):
    query=calendar_day(query_dates);predictions=[]
    for stage in STAGES:
        entry=prior[stage];index=np.searchsorted(entry['values'],query,side='right')
        predictions.append(np.r_[0.,entry['cumulative_weight']][index])
    return np.column_stack(predictions)
