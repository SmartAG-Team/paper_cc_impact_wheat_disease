"""Observed subdaily station spell proxies with explicit missingness and day cutoffs."""
import numpy as np
import pandas as pd

EVENT_NAMES=['precip_hours','precip_max_1h','precip_max_3h','precip_max_6h','precip_max_12h',
    'precip_max_24h','precip_run_hours','humid90_hours','humid90_run_hours',
    'humid_thermal_dose','humid_or_precip_run_hours','warm_precip_hours']
CHANNELS=['precip_timing_complete','humidity_timing_complete','joint_timing_complete']


def _run(flag):
    result=np.zeros(len(flag));value=0
    for k,wet in enumerate(flag):
        value=min(24,value+1) if wet else 0;result[k]=value
    return result


def daily_station_events(hourly):
    if hourly.duplicated(['location_id','hour_start']).any():raise ValueError('One station record per field/hour required')
    rows=[]
    for site,raw in hourly.groupby('location_id',sort=False):
        part=raw.copy();part['hour_start']=pd.to_datetime(part.hour_start)
        part=part.set_index('hour_start').sort_index()
        part=part.reindex(pd.date_range(part.index.min(),part.index.max(),freq='h'))
        rain=part.precipitation_mm.where(part.precipitation_mm.between(0,500))
        rh=part.humidity_pct.where(part.humidity_pct.between(0,100))
        temperature=part.temperature_c.where(part.temperature_c.between(-80,60))
        valid_rain=rain.notna();valid_humidity=rh.notna();valid_joint=valid_rain&valid_humidity
        work=pd.DataFrame(index=part.index)
        work['date']=work.index.normalize()
        work['rain_known']=valid_rain;work['rh_known']=valid_humidity
        work['thermal_known']=valid_humidity&temperature.notna()
        work[CHANNELS[0]]=valid_rain.rolling(24,min_periods=24).sum().eq(24)
        work[CHANNELS[1]]=valid_humidity.rolling(24,min_periods=24).sum().eq(24)
        work[CHANNELS[2]]=valid_joint.rolling(24,min_periods=24).sum().eq(24)
        work['precip_hours']=(rain>.1).where(valid_rain)
        work['precip_max_1h']=rain
        for hours in [3,6,12,24]:work[f'precip_max_{hours}h']=rain.rolling(hours,min_periods=hours).sum()
        work['precip_run_hours']=_run((rain>.1)&valid_rain)
        work['humid90_hours']=(rh>=90).where(valid_humidity)
        work['humid90_run_hours']=_run((rh>=90)&valid_humidity)
        work['humid_thermal_dose']=((rh>=90)*np.exp(-.5*((temperature-18)/8)**2)).where(work.thermal_known)
        work['humid_or_precip_run_hours']=_run(((rh>=90)|(rain>.1))&valid_joint)
        work['warm_precip_hours']=((rain>.1)&(temperature>2)).where(valid_rain&temperature.notna())
        work['warm_known']=valid_rain&temperature.notna()
        for date,day in work.groupby('date',sort=True):
            complete={name:bool(len(day)==24 and day[name].all()) for name in CHANNELS}
            record=dict(location_id=site,date=date,**complete)
            for name in EVENT_NAMES:
                if name.startswith('precip_'):
                    available=complete[CHANNELS[0]]
                elif name.startswith('humid90_'):
                    available=complete[CHANNELS[1]]
                elif name=='humid_thermal_dose':
                    available=bool(len(day)==24 and day.thermal_known.all())
                elif name=='warm_precip_hours':
                    available=bool(len(day)==24 and day.warm_known.all())
                else:available=complete[CHANNELS[2]]
                total=name in ['precip_hours','humid90_hours','humid_thermal_dose','warm_precip_hours']
                record[name]=float(day[name].sum() if total else day[name].max()) if available else np.nan
            rows.append(record)
    return pd.DataFrame(rows)


def build_hourly_features(daily,fields,queries):
    if 'observed_percent' in queries:raise ValueError('Hourly weather queries must be label free')
    if daily.duplicated(['location_id','date']).any():raise ValueError('One daily event summary per location/date required')
    names=[f'hourly_{name}_{summary}_{window}d' for window in [7,21,42]
        for name in EVENT_NAMES for summary in ['mean','max']]
    names += [f'hourly_{channel}_{window}d_fraction' for window in [7,21,42] for channel in CHANNELS]
    sow=pd.to_datetime(fields.set_index('field_id').sowing_date)
    groups={key:part.set_index('date').sort_index() for key,part in daily.groupby('location_id')}
    values=[];ledger=[]
    for query in queries.itertuples(index=False):
        now=pd.Timestamp(query.date).normalize();start=sow.loc[query.field_id]
        part=groups.get(query.site_id,pd.DataFrame(columns=[*EVENT_NAMES,*CHANNELS],index=pd.DatetimeIndex([])))
        summaries=[];coverages=[]
        for window in [7,21,42]:
            begin=max(start,now-pd.Timedelta(days=window));expected=max(0,(now-begin).days)
            known=part.loc[(part.index>=begin)&(part.index<now)]
            for name in EVENT_NAMES:
                valid=pd.to_numeric(known[name],errors='raise').dropna()
                summaries.extend([float(valid.mean()),float(valid.max())] if len(valid) else [0.,0.])
            coverages.extend([float(known[channel].fillna(False).sum()/expected) if expected else 0. for channel in CHANNELS])
        values.append([*summaries,*coverages])
        usable=bool(max(coverages[3:5])>=.8 and (now-start).days>=7)
        ledger.append(dict(global_target_index=int(query.global_target_index),location_id=query.site_id,
            latest_weather_day=str((now-pd.Timedelta(days=1)).date()),qualified_hourly_context=usable,
            precipitation_21d_fraction=coverages[3],humidity_21d_fraction=coverages[4],
            interpretation='Observed total-precipitation and air-humidity spell proxies, not measured canopy leafwetness or verified liquidrain'))
    matrix=np.asarray(values,float)
    if not np.isfinite(matrix).all():raise ValueError('Finite hourly event predictors required')
    return matrix,names,pd.DataFrame(ledger)
