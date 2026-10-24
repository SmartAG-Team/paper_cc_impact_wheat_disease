"""Station agreement within prior-day exposure windows of each assessment."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .dwd_station import pair_metrics


def run(root):
    root=Path(root);out=root/'outputs/weather_validation_20261022'
    dwd=pd.read_parquet(out/'dwd/paired_daily.parquet')
    noaa=pd.read_parquet(out/'noaa/elevation_300m/paired_daily.parquet')
    obs=pd.read_csv(root/'outputs/large_error_audit_20261022/row_diagnostics.csv')
    dwd['station_source']='DWD';noaa['station_source']='NOAA_GSOD'
    pairs=pd.concat([dwd,noaa],ignore_index=True)
    groups=dict(tuple(pairs.groupby('site_id')))
    records=[]
    for row in obs.itertuples(index=False):
        for width in [7,21,42]:
            end=pd.Timestamp(row.date)-pd.Timedelta(days=1);start=end-pd.Timedelta(days=width-1)
            block=groups.get(row.site_id)
            entry=dict(global_target_index=row.global_target_index,site_id=row.site_id,date=row.date,
                country=row.country,source=row.source,leaf_index=row.leaf_index,observed_percent=row.observed_percent,
                prediction_percent=row.selected_candidate,abs_error_pp=abs(row.selected_candidate-row.observed_percent),
                weighted_sse=row.weighted_sse,window_days=width,weather_start=str(start.date()),weather_end=str(end.date()))
            if block is not None:
                part=block.loc[block.date.between(start,end)]
                entry['station_source']=block.station_source.iloc[0]
                for variable in ['tmean_c','rh_mean_pct','rh_hours_ge_90pct','precipitation_mm']:
                    a,b=variable+'_forcing',variable+'_station'
                    if a not in part or b not in part:continue
                    valid=part[[a,b]].dropna();n=len(valid)
                    entry[variable+'_paired_days']=n;entry[variable+'_coverage']=n/width
                    if n:
                        delta=valid[a]-valid[b]
                        entry[variable+'_bias']=float(delta.mean());entry[variable+'_rmse']=float(np.sqrt((delta**2).mean()))
                        if variable=='precipitation_mm':
                            entry['paired_station_precip_total_mm']=valid[b].sum();entry['paired_forcing_precip_total_mm']=valid[a].sum()
                entry['rain_timing_status']='aligned UTC day' if entry['station_source']=='DWD' else 'GSOD accumulation may include prior day; diagnostic only'
            records.append(entry)
    windows=pd.DataFrame(records)
    windows.to_csv(out/'assessment_prior_weather_windows.csv',index=False)
    top=obs.nlargest(50,'weighted_sse').global_target_index
    windows.loc[windows.global_target_index.isin(top)].to_csv(out/'largest_errors_prior_weather_windows.csv',index=False)
    pooled=dwd.copy();pooled['site_id']='all_Germany'
    summary=pair_metrics(pooled)
    summary.to_csv(out/'dwd/pooled_metrics.csv',index=False)
    (out/'assessment_window_policy.json').write_text(json.dumps(dict(
        observations=len(obs),weather_cutoff_days=1,windows_days=[7,21,42],
        positive_or_negative_weather_deviations_do_not_prove_disease_error_causation=True,
        noaa_rainfall='GSOD only broad-window diagnostic; daily timing caveat retained; excluded from causal station forcing'),indent=2)+'\n')
    return windows


if __name__=='__main__':
    run(Path(__file__).resolve().parents[2])
