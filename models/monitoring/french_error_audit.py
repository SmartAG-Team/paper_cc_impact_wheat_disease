"""Preserve large-error and regional diagnostics on the complete benchmark."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.final_study.metrics import hierarchical_weights
from .evaluation import score_subset
from .inputs import sha
from .meteo_france_station import ROOT,OUTPUT


def run(root=ROOT,model_output=None):
    root = Path(root)
    model = Path(model_output or root/'outputs/disease_improvement/french_station_20261022').resolve()
    protocol = json.loads((model/'registration.json').read_text())
    frame = pd.read_csv(model/'assessment_predictions.csv')
    fields_path = root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv'
    fields = pd.read_csv(fields_path).set_index('field_id')
    frame['country'] = frame.field_id.map(fields.country)
    weights = hierarchical_weights(frame)
    frame['full_cohort_weight'] = weights
    columns = ['reference_replay','previous_blend','mf_rain_mean','mf_all_mean']
    countries = []
    for country in sorted(frame.country.unique()):
        mask = frame.country.eq(country)
        for name in columns:
            countries.append(dict(country=country,model=name,**score_subset(frame,mask,name)))
    pd.DataFrame(countries).to_csv(model/'country_metrics.csv',index=False)
    for name in columns:
        frame[name+'_error_pp'] = frame[name]-frame.observed_percent
        frame[name+'_weighted_squared_error'] = weights*frame[name+'_error_pp']**2
    frame['squared_error_change_from_previous'] = frame.mf_all_mean_weighted_squared_error-frame.previous_blend_weighted_squared_error
    paired_path = root/protocol.get('station_validation_directory',OUTPUT)/'qualified_paired_daily.parquet'
    pairs = pd.read_parquet(paired_path);pairs['date'] = pd.to_datetime(pairs.date)
    windows = []
    for row in frame.loc[frame.country.eq('FRANCE')].itertuples(index=False):
        end = pd.Timestamp(row.date)-pd.Timedelta(days=1)
        start = end-pd.Timedelta(days=41)
        part = pairs.loc[pairs.site_id.eq(row.site_id)&pairs.date.between(start,end)]
        valid = part.precipitation_mm_station.notna()&part.precipitation_mm_forcing.notna()
        paired = part.loc[valid]
        windows.append(dict(global_target_index=row.global_target_index,site_id=row.site_id,
            start=str(start.date()),end=str(end.date()),paired_days=int(valid.sum()),
            complete42days=bool(valid.sum()==42),
            station_total_on_paired_days_mm=float(paired.precipitation_mm_station.sum()),
            forcing_total_on_paired_days_mm=float(paired.precipitation_mm_forcing.sum()),
            station_minus_forcing_on_paired_days_mm=float((paired.precipitation_mm_station-paired.precipitation_mm_forcing).sum())))
    pd.DataFrame(windows).to_csv(model/'french_assessment_prior_rainfall.csv',index=False)
    frame = frame.merge(pd.DataFrame(windows),on=['global_target_index','site_id'],how='left',validate='one_to_one')
    keep = ['global_target_index','field_id','site_id','source','country','date','leaf_index',
        'observed_percent','prior_assessment_available','full_cohort_weight']+columns+[
        name+suffix for name in columns for suffix in ['_error_pp','_weighted_squared_error']]+[
        'squared_error_change_from_previous','start','end','paired_days','complete42days',
        'station_total_on_paired_days_mm','forcing_total_on_paired_days_mm','station_minus_forcing_on_paired_days_mm']
    frame[keep].sort_values('mf_all_mean_weighted_squared_error',ascending=False).to_csv(model/'assessment_error_ledger.csv',index=False)
    frame[keep].nlargest(50,'mf_all_mean_weighted_squared_error').to_csv(model/'largest50_errors.csv',index=False)
    sums = frame.groupby('country')[['previous_blend_weighted_squared_error','mf_all_mean_weighted_squared_error']].sum()
    result = dict(source_sha256={str(p.relative_to(root)):sha(p) for p in [model/'assessment_predictions.csv',fields_path,paired_path]},
        code_sha256=sha(Path(__file__)),assessments=len(frame),french_assessments=int(frame.country.eq('FRANCE').sum()),
        complete42day_french_rain_windows=int(sum(row['complete42days'] for row in windows)),
        full_cohort_weighting_preserved=True,new_observation_exclusions=0,
        country_weighted_sse=sums.to_dict(),interpretation='Regional changes can arise through modified training calibration, including unchanged non-French forcing; no causal weather attribution.')
    (model/'error_audit_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-output',type=Path)
    args = parser.parse_args()
    print(json.dumps(run(model_output=args.model_output),indent=2))
