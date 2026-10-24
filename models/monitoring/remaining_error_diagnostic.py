"""Decompose retained rolling-assessment errors and audit first-visit forcing."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.final_study.metrics import hierarchical_weights
from .inputs import sha

ROOT=Path(__file__).resolve().parents[2]


def run(root=ROOT):
    root=Path(root).resolve()
    out=root/'outputs/disease_improvement/remaining_error_diagnostic_20261022'
    if out.exists():
        raise FileExistsError('A new output directory is required')
    prediction=root/'outputs/disease_improvement/history_interval_20261022/assessment_predictions.csv'
    forcing=root/'data/disease_improvement/station_preferred_nl_20261022/daily_weather.parquet'
    fields=root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv'
    f=pd.read_csv(prediction)
    weather=pd.read_parquet(forcing)
    metadata=pd.read_csv(fields)
    assert len(f)==2017 and f.global_target_index.is_unique
    assert not weather.duplicated(['location_id','date']).any()
    assert metadata.field_id.is_unique
    w=hierarchical_weights(f)
    error=f.history_interval-f.observed_percent
    squared=w*error**2
    total=squared.sum()
    first=~f.uses_event_head
    severe=f.observed_percent.gt(75)
    rows=[]
    for key,mask in [('first_field_severe',first&severe),('first_field_other',first&~severe),
                     ('earlier_field_history_severe',~first&severe),('earlier_field_history_other',~first&~severe)]:
        rows.append(dict(population=key,n=int(mask.sum()),
            full_cohort_weight_fraction=float(w[mask].sum()/w.sum()),
            squared_error_share=float(squared[mask].sum()/total),
            rmse_pp=float(np.sqrt(np.average(error[mask]**2,weights=w[mask]))),
            bias_pp=float(np.average(error[mask],weights=w[mask]))))
    decomposition=pd.DataFrame(rows)
    assert decomposition.n.sum()==2017
    np.testing.assert_allclose(decomposition[['full_cohort_weight_fraction','squared_error_share']].sum(),[1.,1.])
    calibration=[]
    for low,high in [(0,10),(10,25),(25,50),(50,75),(75,101)]:
        m=f.history_interval.ge(low)&f.history_interval.lt(high)
        calibration.append(dict(prediction_lower_pp=low,prediction_upper_exclusive_pp=high,n=int(m.sum()),
            full_cohort_weight_fraction=float(w[m].sum()/w.sum()),
            observed_mean_pp=float(np.average(f.loc[m,'observed_percent'],weights=w[m])),
            prediction_mean_pp=float(np.average(f.loc[m,'history_interval'],weights=w[m])),
            bias_pp=float(np.average(error[m],weights=w[m]))))
    cases=f.loc[first&severe,['global_target_index','field_id','site_id','source','source_trial_id',
            'leaf_index','date','latitude','longitude','observed_percent','history_interval']].copy()
    cases=cases.merge(metadata[['field_id','country','cultivar','resistance_class','observed_sowing_date',
                               'plot_id','replicate_id']],on='field_id',validate='m:1')
    assert len(cases)==13
    daily=[]
    for r in cases.itertuples(index=False):
        end=pd.Timestamp(r.date)
        days=weather.loc[weather.location_id.eq(r.site_id)&weather.date.ge(end-pd.Timedelta(days=21))&weather.date.lt(end)].copy()
        assert len(days)==21 and days.precipitation_mm.notna().all() and days.rain_mm.notna().all()
        days['global_target_index']=r.global_target_index
        daily.append(days)
        m=cases.global_target_index.eq(r.global_target_index)
        cases.loc[m,'weather_days_21d']=len(days)
        cases.loc[m,'measured_station_precip_days_21d']=int(days.origin_precipitation_mm.isin(['DWD','METEO_FRANCE','KNMI']).sum())
        cases.loc[m,'precipitation_total_21d_mm']=float(days.precipitation_mm.sum())
        cases.loc[m,'phase_assisted_or_reanalysis_rain_21d_mm']=float(days.rain_mm.sum())
        cases.loc[m,'precipitation_providers']=';'.join(sorted(days.origin_precipitation_mm.unique()))
    out.mkdir(parents=True)
    decomposition.to_csv(out/'error_decomposition.csv',index=False)
    pd.DataFrame(calibration).to_csv(out/'prediction_conditioned_calibration.csv',index=False)
    cases.to_csv(out/'first_field_severe_cases.csv',index=False)
    pd.concat(daily,ignore_index=True).to_parquet(out/'first_field_severe_weather_windows.parquet',index=False)
    summary=dict(evaluation_rows=2017,weighted_rmse_pp=float(np.sqrt(total/w.sum())),
        first_field_severe_cases=13,first_field_severe_weather_days=273,
        first_field_severe_without_station_precipitation=int(cases.measured_station_precip_days_21d.eq(0).sum()),
        first_field_severe_with_complete_station_precipitation=int(cases.measured_station_precip_days_21d.eq(21).sum()),
        first_field_severe_country_counts=cases.country.value_counts().to_dict(),
        current_high_outcome_bias_is_outcome_conditioned=True,
        precipitation_error_established_by_missing_station_coverage=False,
        additional_observations_removed=False,independent_external_validation=False)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    inputs=[prediction,forcing,fields,Path(__file__),root/'analysis/final_study/metrics.py']
    manifest=dict(source_sha256={str(p.relative_to(root)):sha(p) for p in inputs},
                  files_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    run()
