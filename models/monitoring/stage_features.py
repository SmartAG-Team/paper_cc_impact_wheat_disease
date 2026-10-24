"""Causal crop-stage history and uncalibrated weather-clock summaries.

BBCH context uses only records before the query date and within the declared
season window. Same-day observed stages are exposed by a separate conditional
function, never silently included in the primary information condition.
"""
import numpy as np
import pandas as pd

PRIOR_NAMES=['prior_stage_available','prior_stage_midpoint_scaled','prior_stage_from_scaled',
    'prior_stage_to_scaled','prior_stage_age_30d','prior_stage_count_10',
    'prior_stage_change_per_30d_scaled','prior_stage_degree_days_1000',
    'prior_stage_recorded_code_interaction']
CLOCK_NAMES=['clock_latitude_scaled','clock_longitude_scaled','clock_daylength_fraction',
    'clock_positive_degree_days_1000','clock_frost_days_330','clock_hot_days_330',
    'clock_mean_temperature_scaled']
CURRENT_NAMES=['current_stage_available','current_stage_midpoint_scaled','current_stage_from_scaled',
    'current_stage_to_scaled','current_stage_recorded_code_interaction']


def usable_calendar(calendar):
    frame=calendar.loc[calendar.usable.eq(True)].copy()
    frame['date']=pd.to_datetime(frame.date,errors='raise').dt.normalize()
    if frame.duplicated(['field_id','date']).any():
        raise ValueError('Unique usable field/date stages required')
    values=frame[['stage_from','stage_to','stage_midpoint']].to_numpy(float)
    if (not np.isfinite(values).all() or np.any(values<0) or np.any(values>99)
        or np.any(values[:,0]>values[:,1]) or np.any(values[:,2]<values[:,0])
        or np.any(values[:,2]>values[:,1])):
        raise ValueError('Valid bounded stage interval required')
    return frame


def build_stage_features(weather,fields,queries,calendar):
    names=PRIOR_NAMES+CLOCK_NAMES
    stage=usable_calendar(calendar)
    field=fields.set_index('field_id')
    if field.index.duplicated().any() or queries.global_target_index.duplicated().any():
        raise ValueError('Unique fields and queries required')
    w=weather.copy();w['date']=pd.to_datetime(w.date).dt.normalize()
    if w.duplicated(['location_id','date']).any():
        raise ValueError('Unique daily forcing required')
    wg={key:g.set_index('date').sort_index() for key,g in w.groupby('location_id')}
    sg={key:g.sort_values('date') for key,g in stage.groupby('field_id')}
    rows=[];provenance=[]
    for row in queries.itertuples(index=False):
        meta=field.loc[row.field_id];now=pd.Timestamp(row.date).normalize()
        start=pd.Timestamp(meta.sowing_date);end=now-pd.Timedelta(days=1)
        dates=pd.date_range(start,end)
        if not len(dates):
            raise ValueError('At least one previous weather day required')
        daily=wg[meta.site_id].reindex(dates)
        t=daily.tmean_c.to_numpy(float);tn=daily.tmin_c.to_numpy(float)
        if not np.isfinite(t).all() or not np.isfinite(tn).all():
            raise ValueError('Complete earlier weather required')
        context=sg.get(row.field_id,stage.iloc[:0])
        context=context.loc[context.date.ge(start)&context.date.lt(now)]
        p=np.zeros(len(PRIOR_NAMES))
        latest=None;source_rows=None
        if len(context):
            last=context.iloc[-1];latest=last.date
            later=np.asarray(dates>latest)
            change=0.
            if len(context)>1:
                previous=context.iloc[-2]
                change=30*(last.stage_midpoint-previous.stage_midpoint)/(latest-previous.date).days/100
            p[:]=[1,last.stage_midpoint/100,last.stage_from/100,last.stage_to/100,
                (now-latest).days/30,len(context)/10,change,
                np.maximum(t[later],0).sum()/1000,
                row.leaf_index/6*last.stage_midpoint/100]
            source_rows=str(last.get('source_rows',''))
        declination=.4093*np.sin(2*np.pi*(now.dayofyear-81)/365)
        daylength=24/np.pi*np.arccos(np.clip(-np.tan(np.deg2rad(meta.latitude))*np.tan(declination),-1,1))
        clock=[meta.latitude/90,meta.longitude/180,daylength/24,np.maximum(t,0).sum()/1000,
               (tn<0).sum()/330,(t>=30).sum()/330,t.mean()/30]
        rows.append(np.r_[p,clock])
        provenance.append(dict(global_target_index=int(row.global_target_index),
            latest_stage_date=str(latest.date()) if latest is not None else None,
            latest_stage_source_rows=source_rows,prior_stage_count=len(context),
            weather_start=str(start.date()),weather_end=str(end.date()),
            sowing_basis='declared calendar scenario, not observed sowing',
            contemporaneous_or_future_stage_used=False))
    x=np.asarray(rows,float)
    if not np.isfinite(x).all():
        raise ValueError('Finite stage and clock predictors required')
    return x,names,pd.DataFrame(provenance)


def contemporaneous_features(queries,calendar):
    stage=usable_calendar(calendar).set_index(['field_id','date'])
    records=[]
    for row in queries.itertuples(index=False):
        key=(row.field_id,pd.Timestamp(row.date).normalize())
        values=np.zeros(len(CURRENT_NAMES))
        if key in stage.index:
            current=stage.loc[key]
            values[:]=[1,current.stage_midpoint/100,current.stage_from/100,current.stage_to/100,
                row.leaf_index/6*current.stage_midpoint/100]
        records.append(values)
    return np.asarray(records,float),CURRENT_NAMES.copy()
