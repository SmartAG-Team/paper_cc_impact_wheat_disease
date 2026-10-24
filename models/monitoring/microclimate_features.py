"""Causal radiation/wind/condensation proxies; no measured leaf wetness claim."""
import numpy as np
import pandas as pd


def build_microclimate(weather,fields,queries,history_provenance):
    if 'observed_percent' in queries:raise ValueError('Query labels must be absent')
    if queries.global_target_index.duplicated().any() or fields.field_id.duplicated().any():
        raise ValueError('Unique query and field identities required')
    w=weather.copy();w['date']=pd.to_datetime(w.date,format='mixed').dt.normalize()
    if w.duplicated(['location_id','date']).any():raise ValueError('Unique daily weather required')
    wg={key:g.set_index('date') for key,g in w.groupby('location_id')};f=fields.set_index('field_id')
    provenance=history_provenance.set_index('global_target_index').loc[queries.global_target_index]
    same=pd.to_datetime(provenance.latest_same_date,format='mixed')
    anchor=same.fillna(pd.to_datetime(provenance.latest_any_date,format='mixed'))
    variables=['tmean_c','tmax_c','tmin_c','dewpoint_mean_c','wind_mean_m_s','shortwave_mj_m2',
        'rain_mm','rain_hours_gt_0_1mm','rh_hours_ge_90pct']
    cache={}
    for field,part in queries.groupby('field_id'):
        meta=f.loc[field]
        if not part.site_id.eq(meta.site_id).all() or ('source' in part and not part.source.eq(meta.source).all()):
            raise ValueError('Consistent query/field metadata required')
        start=pd.Timestamp(meta.sowing_date);end=pd.to_datetime(part.date).max()-pd.Timedelta(days=1)
        raw=wg[meta.site_id].reindex(pd.date_range(start,end))[variables].to_numpy(float)
        if not np.isfinite(raw).all() or np.any(raw[:,4:]<0) or np.any(raw[:,7:]>24):
            raise ValueError('Complete physically bounded daily forcing required')
        t,tx,tn,td,wind,solar,rain,rainh,humidh=raw.T
        saturation=lambda v:.6108*np.exp(17.27*v/(v+237.3))
        vpd=np.maximum(0,saturation(t)-saturation(td));dew=np.maximum(0,t-td)
        lower=np.maximum(rainh,humidh);upper=np.minimum(24,rainh+humidh)
        intensity=np.divide(rain,rainh,out=np.zeros_like(rain),where=rainh>0)
        daily=pd.DataFrame(dict(wind_m_s=wind,shortwave_mj_m2=solar,vpd_kpa=vpd,
            dewpoint_depression_c=dew,diurnal_range_c=np.maximum(0,tx-tn),
            wet_lower_hours=lower,wet_upper_hours=upper,
            drying_proxy=solar*vpd*(1+wind/5),rain_intensity_mm_h=intensity,
            rain_duration_missing=((rain>0)&(rainh==0)).astype(float)))
        cache[field]=(start,daily)
    rows=[];ledger=[]
    for i,row in enumerate(queries.itertuples(index=False)):
        start,daily=cache[row.field_id];now=pd.Timestamp(row.date);last=(now-start).days
        if last<1:raise ValueError('At least one earlier weather day required')
        result={}
        for width in [7,21,42]:
            part=daily.iloc[max(0,last-width):last]
            for name in daily:result[f'mc_{name}_mean_{width}d']=float(part[name].mean())
            for name in ['wind_m_s','drying_proxy','rain_intensity_mm_h']:
                result[f'mc_{name}_max_{width}d']=float(part[name].max())
        previous=anchor.iloc[i]
        if pd.notna(previous) and previous>=now:raise ValueError('Strictly earlier disease anchor required')
        first=(previous-start).days+1 if pd.notna(previous) else max(0,last-21)
        if first<0 or first>last:raise ValueError('Supported causal anchor interval required')
        gap=daily.iloc[first:last]
        for name in ['wet_lower_hours','wet_upper_hours','drying_proxy']:
            result['mc_gap_'+name]=float(gap[name].sum())
        result['mc_gap_days']=len(gap)
        rows.append(result)
        ledger.append(dict(global_target_index=int(row.global_target_index),weather_start=str(start.date()),
            weather_end=str((now-pd.Timedelta(days=1)).date()),
            interval_start=str((start+pd.Timedelta(days=first)).date()),
            physical_status='Relative drying and condensation proxies; neither measured canopy wetness nor calibrated evapotranspiration',
            wind_height='Retained source wind speed, no unsupported height conversion'))
    matrix=pd.DataFrame(rows)
    if not np.isfinite(matrix.to_numpy()).all():raise ValueError('Finite microclimate features required')
    return matrix.to_numpy(),matrix.columns.tolist(),pd.DataFrame(ledger)
