"""Six fixed cumulative weather proxies through the day before assessment."""
import numpy as np
import pandas as pd

from model.seasonal_septoria.wetness import duration_exposure

NAMES=['season_rain_total_mm','season_wet_days_ge0p1mm','season_rain_hours_gt0p1mm',
       'season_rh_hours_ge90pct','season_wet_proxy_dose','pre42d_wet_proxy_dose']


def build_season_exposure(weather,fields,queries):
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    if not fields.field_id.is_unique or not queries.global_target_index.is_unique:
        raise ValueError('Unique field and query identities required')
    w=weather.copy()
    w['date']=pd.to_datetime(w.date).dt.normalize()
    if w.duplicated(['location_id','date']).any():
        raise ValueError('Duplicate daily weather')
    locations=dict(tuple(w.groupby('location_id',sort=False)))
    metadata=fields.set_index('field_id')
    cache={}
    for field_id,part in queries.groupby('field_id',sort=False):
        field=metadata.loc[field_id]
        start=pd.Timestamp(field.sowing_date).normalize()
        end=pd.to_datetime(part.date).max().normalize()-pd.Timedelta(days=1)
        if field.site_id not in locations or end<start:
            raise ValueError('Incomplete weather support')
        dates=pd.date_range(start,end)
        daily=locations[field.site_id].set_index('date').reindex(dates)
        columns=['tmean_c','tmax_c','rh_mean_pct','rain_mm','rain_hours_gt_0_1mm','rh_hours_ge_90pct']
        a=daily[columns].to_numpy(float)
        if (not np.isfinite(a).all() or np.any(a[:,3:]<0) or np.any(a[:,4:]>24)):
            raise ValueError('Incomplete or invalid daily weather')
        t,tx,rh,rain,rain_hours,humid_hours=a.T
        proxy=duration_exposure(t,tx,rh,rain)['exposure']
        values=np.column_stack([rain,rain>=.1,rain_hours,humid_hours,proxy])
        cumulative=np.vstack([np.zeros(5),np.cumsum(values,axis=0)])
        cache[field_id]=(start,cumulative,getattr(field,'sowing_basis','unreported'))
    rows,provenance=[],[]
    for query in queries.itertuples(index=False):
        start,cumulative,basis=cache[query.field_id]
        query_date=pd.Timestamp(query.date).normalize()
        length=(query_date-start).days
        if length<1 or length>=len(cumulative):
            raise ValueError('Assessment outside complete weather support')
        rows.append(np.r_[cumulative[length],cumulative[max(0,length-42),4]])
        provenance.append(dict(global_target_index=int(query.global_target_index),field_id=query.field_id,
            weather_start=str(start.date()),weather_end=str((query_date-pd.Timedelta(days=1)).date()),
            weather_days=length,early_days=max(0,length-42),sowing_basis=basis,
            wetness_interpretation='Modeled daily exposure proxy, not measured canopy leaf wetness',weather_cutoff_days=1))
    matrix=np.asarray(rows,float)
    if matrix.shape!=(len(queries),6) or not np.isfinite(matrix).all():
        raise ValueError('Complete six-column exposure matrix required')
    return matrix,NAMES.copy(),pd.DataFrame(provenance)
