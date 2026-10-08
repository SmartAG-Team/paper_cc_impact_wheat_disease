"""Evaluate frozen BASF calibration on the newly acquired Corteva source."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from model.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.calibrate import predict
from model.seasonal_septoria.core import Parameters
from model.seasonal_septoria.endpoints import onset_distance,occurrence_scores
from analysis.paper_study.calibrate_seasonal import metrics,write_json,sha

FROZEN = ROOT/'analysis/paper_study/seasonal_calibration_v1'
DEST = ROOT/'analysis/paper_study/corteva_external_seasonal_v1'


def main():
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError('External evaluation archive already exists.')
    frozen = json.loads((FROZEN/'frozen_parameters_before_validation.json').read_text())
    for name,digest in frozen.items():
        if sha(FROZEN/name)!=digest:
            raise ValueError('Frozen calibration parameters changed.')
    DEST.mkdir(parents=True, exist_ok=True)
    source_path = ROOT/'data/paper_study/observations/corteva_external_assessments.csv'
    source = pd.read_csv(source_path)
    source = source.loc[source.assessment_eligible.eq(True)].copy()
    source['endpoint_series'] = source.physical_unit.astype(str)+'|'+source.organ.astype(str)
    calendar_path = ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv'
    weather_path = ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet'
    data = prepare_fields(source,pd.read_csv(calendar_path),pd.read_parquet(weather_path))
    registered = pd.read_csv(ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv')
    strict_units = set(registered.source_unit.astype(str))
    data.metadata['strict_location_disjoint'] = data.metadata.source_unit.astype(str).isin(strict_units)
    data.targets['strict_location_disjoint'] = data.targets.physical_unit.astype(str).isin(strict_units)
    data.metadata.to_csv(DEST/'external_input_membership.csv',index=False)
    data.excluded.to_csv(DEST/'excluded_assessments.csv',index=False)
    fits = {name:json.loads((FROZEN/file).read_text())['parameters'] for name,file in
        [('seasonal_seir','frozen_main_fit.json'),('primary_only_seasonal','frozen_primary_only_fit.json')]}
    rank_means = json.loads((FROZEN/'frozen_leaf_rank_baseline.json').read_text())
    training = pd.read_csv(FROZEN/'target_membership.csv')
    training = training.loc[training.partition.eq('calibration')]
    global_mean = float(np.average(training.value,weights=training.weight))
    all_predictions,score_rows,onsets,occurrences,numerical = [],[],[],[],[]
    for name in [*fits,'calibration_only_leaf_rank_mean']:
        trajectory = None
        if name in fits:
            forecast,trajectory = predict(data,Parameters(**fits[name]))
            numerical.append(dict(model=name,mass_error=float(np.max(abs(trajectory.state.sum(axis=-1)-1))),
                                   minimum_state=float(trajectory.state.min())))
        else:
            forecast = np.asarray([rank_means.get(str(int(rank)),global_mean) for rank in data.targets.leaf_index])
        scored = data.targets.copy()
        scored['model'],scored['predicted_percent'] = name,forecast
        all_predictions.append(scored)
        for geography,subset in [('full_source_transfer',scored),
                                 ('strict_location_disjoint',scored[scored.strict_location_disjoint])]:
            for leaf_scope,selected in [('all_ordinal_leaves',subset),('upper_three',subset[subset.leaf_index.lt(3)])]:
                if selected.empty:
                    continue
                score_rows.append(dict(model=name,geography=geography,leaf_scope=leaf_scope,
                    endpoint='all_assessments',**metrics(selected)))
                final = selected.sort_values('date').groupby(['field_id','endpoint_series']).tail(1).copy()
                final['weight'] = 1/final.groupby('coordinate_year').field_id.transform('nunique')/final.groupby('field_id').endpoint_series.transform('nunique')
                score_rows.append(dict(model=name,geography=geography,leaf_scope=leaf_scope,
                    endpoint='final_numeric_assessment',**metrics(final)))
            for cutoff in [.1,1.,5.]:
                trial = subset.groupby('field_id').agg(observed=('value','max'),forecast=('predicted_percent','max'))
                occurrences.append(dict(model=name,geography=geography,cutoff_percent=cutoff,
                    endpoint='any_ordinal_leaf_detected_at_assessment_dates',
                    **occurrence_scores(trial.observed.ge(cutoff),trial.forecast.ge(cutoff).astype(float))))
        if trajectory is None:
            continue
        for (field,series),group in scored.groupby(['field_index','endpoint_series']):
            row = group.iloc[0]
            meta = data.metadata.loc[data.metadata.field_index.eq(field)].iloc[0]
            n_days = int(meta.forcing_days)
            organ = int(row.leaf_index)
            primary_days = np.flatnonzero(np.cumsum(trajectory.primary_flow[field,:n_days,organ])>=.0001)
            primary_date = None if not len(primary_days) else str((pd.Timestamp(meta.sowing_date)+pd.Timedelta(days=int(primary_days[0]))).date())
            for cutoff in [.1,1.,5.]:
                ordered = group.sort_values('date')
                positive = ordered[ordered.value.ge(cutoff)]
                upper = None if positive.empty else positive.date.min()
                negative = ordered[ordered.value.lt(cutoff)]
                lower = negative.date.max() if upper is None else negative.loc[negative.date.lt(upper),'date'].max()
                lower = None if pd.isna(lower) else lower
                found = np.flatnonzero(trajectory.damage[field,1:n_days+1,organ]*100>=cutoff)
                predicted = None if len(found)==0 else pd.Timestamp(meta.sowing_date)+pd.Timedelta(days=int(found[0]))
                onsets.append(dict(model=name,field_id=row.field_id,series=series,
                    strict_location_disjoint=bool(row.strict_location_disjoint),cutoff_percent=cutoff,
                    inferred_primary_exposure_date=primary_date,infection_date_observed=False,
                    predicted_visible_date=None if predicted is None else str(predicted.date()),
                    last_negative=None if lower is None else str(lower.date()),
                    first_positive=None if upper is None else str(upper.date()),
                    censoring='right' if upper is None else 'left' if lower is None else 'interval',
                    **onset_distance(predicted,lower,upper)))
    pd.concat(all_predictions,ignore_index=True).to_parquet(DEST/'frozen_external_predictions.parquet',index=False)
    pd.DataFrame(score_rows).to_csv(DEST/'severity_metrics.csv',index=False)
    pd.DataFrame(onsets).to_csv(DEST/'onset_interval_predictions.csv',index=False)
    pd.DataFrame(occurrences).to_csv(DEST/'observed_window_detection_metrics.csv',index=False)
    write_json(DEST/'numerical_checks.json',numerical)
    receipt = dict(status='complete',publication_ready=False,parameters_refitted=False,
        prior_external_model_selection=False,frozen_parameters=frozen,
        evaluated_field_seasons=len(data.metadata),strict_location_disjoint_field_seasons=int(data.metadata.strict_location_disjoint.sum()),
        targets=len(data.targets),source_rows_excluded_by_organ_or_calendar=len(data.excluded),
        source_assessment_sha256=sha(source_path),calendar_sha256=sha(calendar_path),weather_sha256=sha(weather_path),
        late_source_date_retained='2015-58:2015-10-30',
        model_source_sha256={path.name:sha(path) for path in sorted((ROOT/'model/seasonal_septoria').glob('*.py'))},
        sowing_dates_observed=False,true_infection_dates_observed=False,
        full_season_negative_field_classes_available=False,
        shared_rounded_coordinates_are_confirmed_same_fields=False)
    write_json(DEST/'receipt.json',receipt)
    print(pd.DataFrame(score_rows).to_string(index=False),flush=True)
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    main()
