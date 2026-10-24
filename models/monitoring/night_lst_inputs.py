"""Past-only clear-sky skin-temperature summaries and daily-air contrasts.

Skin minus daily air/dewpoint temperature is a regional contrast, without
instantaneous collocation or a measured leaf-wetness interpretation.
"""
import numpy as np
import pandas as pd


def qualified_night_pixels(raw_lst,qc,view_time,view_angle):
    raw,q,time,angle=np.broadcast_arrays(raw_lst,qc,view_time,view_angle)
    finite=np.isfinite(raw)&np.isfinite(q)&np.isfinite(time)&np.isfinite(angle)
    q=np.where(np.isfinite(q),q,255).astype(np.int64)
    return (finite&(raw>=7500)&(raw<=65535)&(time>=0)&(time<=240)&(angle>=0)&(angle<=130)&
            ((q&3)<=1)&(((q>>2)&3)==0)&(((q>>4)&3)<=1)&(((q>>6)&3)<=1))


def _dates(values):
    return pd.to_datetime(values,unit='ms' if pd.api.types.is_numeric_dtype(values) else None,
                          format=None if pd.api.types.is_numeric_dtype(values) else 'mixed').dt.normalize()


def temperature_content_columns(names):
    """Satellite-derived thermal channels; matched daily air values are retained."""
    return [i for i,n in enumerate(names) if any(k in n for k in [
        'lst_c','spatial_sd_c','skin_minus_','positive_dewpoint_minus_skin_c'])]


def build_night_features(summaries,weather,queries,lag_days=2):
    if lag_days<1:raise ValueError('Positive lag after the completed daily period required')
    if 'observed_percent' in queries:raise ValueError('Query labels must be absent')
    if queries.global_target_index.duplicated().any():raise ValueError('Unique query IDs required')
    source=summaries.copy();daily=weather.copy()
    for key in ['period_start','period_end']:source[key]=_dates(source[key])
    if not source.sensor.isin(['terra','aqua']).all() or source.duplicated(['site_id','sensor','period_start']).any():
        raise ValueError('Unique daily sensor/site identities required')
    if not source.period_end.eq(source.period_start+pd.Timedelta(days=1)).all():
        raise ValueError('Complete one-day source intervals required')
    source['available_at']=source.period_end+pd.Timedelta(days=lag_days)
    provider={(s,k):g.available_at.sort_values() for (s,k),g in source.groupby(['site_id','sensor'])}
    good=(source.night_lst_c_count.ge(20)&source.valid_mean.ge(.2)&source.valid_mean.le(1)&
          np.isfinite(source[['night_lst_c_mean','night_lst_c_stdDev','view_hour_mean','view_angle_abs_mean']]).all(axis=1))
    source=source.loc[good].copy()
    daily['date']=pd.to_datetime(daily.date,format='mixed').dt.normalize()
    if daily.duplicated(['location_id','date']).any():raise ValueError('Unique daily air records required')
    source=source.merge(daily[['location_id','date','tmin_c','dewpoint_mean_c']],
        left_on=['site_id','period_start'],right_on=['location_id','date'],how='left',validate='many_to_one')
    source['lst_c']=source.night_lst_c_mean/20
    source['spatial_sd_c']=source.night_lst_c_stdDev/10
    source['view_hour']=source.view_hour_mean/24
    source['view_angle']=source.view_angle_abs_mean/65
    source['air_tmin_c']=source.tmin_c/20
    source['air_mean_dewpoint_c']=source.dewpoint_mean_c/20
    source['skin_minus_tmin_c']=(source.night_lst_c_mean-source.tmin_c)/10
    source['skin_minus_mean_dewpoint_c']=(source.night_lst_c_mean-source.dewpoint_mean_c)/10
    source['positive_dewpoint_minus_skin_c']=np.maximum(0,-source.skin_minus_mean_dewpoint_c)
    source['valid_coverage_fraction']=source.valid_mean
    source['air_tmin_available']=source.tmin_c.notna().astype(float)
    source['air_dewpoint_available']=source.dewpoint_mean_c.notna().astype(float)
    variables=['lst_c','spatial_sd_c','view_hour','view_angle','skin_minus_tmin_c',
        'skin_minus_mean_dewpoint_c','positive_dewpoint_minus_skin_c','valid_coverage_fraction',
        'air_tmin_available','air_dewpoint_available','air_tmin_c','air_mean_dewpoint_c']
    physical=[v for v in variables if v in ['lst_c','spatial_sd_c','skin_minus_tmin_c',
                                           'skin_minus_mean_dewpoint_c','positive_dewpoint_minus_skin_c',
                                           'air_tmin_c','air_mean_dewpoint_c']]
    cache={(s,k):g.sort_values('available_at') for (s,k),g in source.groupby(['site_id','sensor'])}
    rows=[];provenance=[];available=[]
    for query in queries.itertuples(index=False):
        now=pd.Timestamp(query.date).normalize();result={};active=False;latest=None
        for sensor in ['terra','aqua']:
            group=cache.get((query.site_id,sensor),source.iloc[:0])
            past=group.loc[group.available_at.le(now)]
            last=past.iloc[-1] if len(past) else None
            age=84. if last is None else float((now-last.available_at).days)
            qualified=last is not None and age<=14
            active=active or qualified
            for variable in variables:
                value=last[variable] if qualified else 0.
                result[f'{sensor}_latest_{variable}']=float(value) if pd.notna(value) else 0.
            result[f'{sensor}_latest_age_84d']=min(age,84)/84
            result[f'{sensor}_latest_available']=float(qualified)
            if last is not None:latest=last.period_end if latest is None else max(latest,last.period_end)
            for width in [14,42,84]:
                context=past.loc[past.available_at.gt(now-pd.Timedelta(days=width))]
                for variable in physical:
                    value=context[variable].mean()
                    result[f'{sensor}_{variable}_mean_{width}d']=float(value) if pd.notna(value) else 0.
                result[f'{sensor}_night_coverage_{width}d']=len(context)/width
                dates=provider.get((query.site_id,sensor),pd.Series([],dtype='datetime64[ns]'))
                result[f'{sensor}_provider_coverage_{width}d']=float(
                    (dates.gt(now-pd.Timedelta(days=width))&dates.le(now)).sum())/width
                for variable in ['air_tmin_available','air_dewpoint_available']:
                    value=context[variable].mean()
                    result[f'{sensor}_{variable}_fraction_{width}d']=float(value) if pd.notna(value) else 0.
        rows.append(result);available.append(active)
        provenance.append(dict(global_target_index=int(query.global_target_index),site_id=query.site_id,
            query_date=str(now.date()),latest_period_end=str(latest.date()) if latest is not None else None,
            night_available=bool(active),availability_lag_days=lag_days,
            air_comparison='Daily retained air/dewpoint, not simultaneous satellite overpass measurements'))
    frame=pd.DataFrame(rows)
    if not np.isfinite(frame.to_numpy()).all():raise ValueError('Finite night-temperature summaries required')
    return frame.to_numpy(),frame.columns.tolist(),np.asarray(available,bool),pd.DataFrame(provenance)
