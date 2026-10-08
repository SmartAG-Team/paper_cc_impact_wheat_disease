"""Calibration-only selection of regularized process/weather severity blends."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import sklearn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.paper_study.calibrate_seasonal import sha, write_json
from analysis.paper_study.refine_seasonal_accuracy import onset_records, score
from analysis.paper_study.run_empirical_benchmarks import attach_maxima, endpoint, paired_intervals
from model.seasonal_septoria.empirical import assessment_features, WeightedRidge
from model.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.refinement import (
    blend_severity, calibration_weights, predict_refinement,
)

V1 = ROOT/'analysis/paper_study/seasonal_calibration_v1'
V2 = ROOT/'analysis/paper_study/severity_refinement_v2_20261006'
DEFAULT_DEST = ROOT/'analysis/paper_study/severity_hybrid_v3_20261006'
KEYS = ['field_id', 'endpoint_series', 'date']


def host_at_assessments(data):
    target = data.targets
    return data.host_active[target.field_index.to_numpy(int),
                            target.day_index.to_numpy(int)-1,
                            target.leaf_index.to_numpy(int)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_DEST)
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError('Use a new versioned hybrid archive.')
    dest.mkdir(parents=True)
    protocol = json.loads((V1/'configuration_before_fitting.json').read_text())
    paths = dict(source=V1/'basf_source_assessments_snapshot.csv',
        weather=ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet',
        calendars=ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv',
        phenology=ROOT/'process_model/parameters/calibration.json')
    settings = dict(final_weight=[0., .5, 1.], penalty=[.1, 1., 10., 100., 1000.],
                    empirical_weight=[0., .25, .5, .75, 1.], process_base=['reference_v1', 'refinement_v2'])
    config = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        objective='coordinate-year weighted final recorded leaf-severity RMSE',
        selection='minimum pooled three-location-fold calibration-only final severity RMSE',
        candidates=settings, calibration_years=[2017, 2018], site_fold=protocol['site_fold'],
        prior_2019_and_external_results_known=True, prior_refinement_v2_results_known=True,
        evaluation_role='reused development evidence; no untouched test',
        disease_initialization=False, onset_model='separate unchanged reference_v1 process diagnostic',
        fitting_or_selection_on_2019_or_corteva=False,
        prohibited_features=['disease', 'observed_stage', 'future_weather', 'site_id', 'country'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*paths.values(), Path(__file__),
            ROOT/'model/seasonal_septoria/refinement.py', ROOT/'model/seasonal_septoria/empirical.py',
            V1/'frozen_main_fit.json', V2/'frozen_selected_fit.json']}, sklearn_version=sklearn.__version__)
    write_json(dest/'configuration_before_fitting.json', config)
    weather, calendars = pd.read_parquet(paths['weather']), pd.read_csv(paths['calendars'])
    phenology = json.loads(paths['phenology'].read_text())
    data = prepare_fields(pd.read_csv(paths['source']), calendars, weather)
    data.targets['partition'] = np.where(data.targets.season_year.lt(2019),
                                         'calibration', 'reused_development_2019')
    data.targets.to_csv(dest/'target_membership.csv', index=False)
    data.metadata.to_csv(dest/'field_season_membership.csv', index=False)
    attach_maxima(data, weather)
    features = assessment_features(data, phenology)
    features.to_parquet(dest/'basf_features.parquet', index=False)
    training = data.targets.partition.eq('calibration').to_numpy()
    if set(data.targets.loc[training, 'field_id']) & set(data.targets.loc[~training, 'field_id']):
        raise ValueError('Incomplete field-season partition.')
    folds = data.targets.site_id.map(protocol['site_fold']).to_numpy()
    if set(data.targets.loc[training, 'site_id']) != set(protocol['site_fold']):
        raise ValueError('Calibration site membership changed.')
    active = host_at_assessments(data)
    base_oof = {}
    for name, paths_oof in [
        ('reference_v1', [V1/f'inner_latent30_fold{k}_predictions.parquet' for k in range(3)]),
        ('refinement_v2', [V2/f'inner_fixed_latency_direct_final1_fold{k}_predictions.parquet' for k in range(3)])]:
        archived = pd.concat([pd.read_parquet(p) for p in paths_oof], ignore_index=True)
        mapped = data.targets.loc[training, KEYS+['value']].merge(
            archived[KEYS+['predicted_percent', 'value']], on=KEYS,
            how='left', validate='one_to_one', suffixes=('', '_archived'))
        if mapped.predicted_percent.isna().any() or not np.allclose(mapped.value, mapped.value_archived):
            raise ValueError('Out-of-fold process predictions do not match training observations.')
        base_oof[name] = mapped.predicted_percent.to_numpy()
        config['source_sha256'].update({str(p.relative_to(ROOT)):sha(p) for p in paths_oof})
    write_json(dest/'configuration_before_fitting.json', config)
    selection = []
    final_training = data.targets.loc[training].copy()
    for final_weight in settings['final_weight']:
        for penalty in settings['penalty']:
            ridge_oof = np.full(len(data.targets), np.nan)
            for fold in range(3):
                fit_rows, test_rows = training & (folds != fold), training & (folds == fold)
                if set(data.targets.loc[fit_rows, 'site_id']) & set(data.targets.loc[test_rows, 'site_id']):
                    raise ValueError('Inner location leakage.')
                ridge = WeightedRidge(penalty).fit(features.loc[fit_rows],
                    data.targets.loc[fit_rows, 'value'],
                    calibration_weights(data.targets.loc[fit_rows], final_weight))
                ridge_oof[test_rows] = ridge.predict(features.loc[test_rows])
            if not np.isfinite(ridge_oof[training]).all():
                raise ValueError('Incomplete calibration cross-validation.')
            for base in settings['process_base']:
                for blend in settings['empirical_weight']:
                    forecast = blend_severity(base_oof[base], ridge_oof[training],
                        empirical_weight=blend, host_active=active[training])
                    trial = final_training.copy()
                    trial['predicted_percent'] = forecast
                    result = dict(final_weight=final_weight, penalty=penalty,
                        empirical_weight=blend, process_base=base,
                        **score(endpoint(trial, True, False)))
                    selection.append(result)
            print(f'CV ridge final_weight={final_weight:g}, penalty={penalty:g}: complete', flush=True)
    pd.DataFrame(selection).to_csv(dest/'calibration_only_selection.csv', index=False)
    chosen = min(selection, key=lambda row: (row['rmse'], row['empirical_weight']))
    ridge = WeightedRidge(chosen['penalty']).fit(features.loc[training],
        data.targets.loc[training, 'value'],
        calibration_weights(data.targets.loc[training], chosen['final_weight']))
    reference = json.loads((V1/'frozen_main_fit.json').read_text())
    reference['observation_power'] = 1.
    process = reference if chosen['process_base'] == 'reference_v1' else json.loads((V2/'frozen_selected_fit.json').read_text())
    bundle = dict(estimator=ridge, process_fit=process, feature_names=features.columns.tolist(),
        empirical_weight=chosen['empirical_weight'], selection=chosen,
        calibration_field_ids=sorted(data.targets.loc[training, 'field_id'].unique()),
        observed_disease_assimilated=False, evaluation_role='reused development evidence',
        onset_process_fit=reference, onset_prediction_is_separate=True)
    joblib.dump(bundle, dest/'frozen_severity_hybrid.joblib')
    write_json(dest/'frozen_selected_setting.json', chosen)
    freeze = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        bundle_sha256=sha(dest/'frozen_severity_hybrid.joblib'),
        selection_sha256=sha(dest/'calibration_only_selection.csv'),
        validation_or_external_observations_used=False)
    write_json(dest/'frozen_before_evaluation.json', freeze)
    print(f'FROZEN hybrid setting: {chosen}', flush=True)
    # All reused evaluation scoring follows the selected-bundle freeze.
    external = pd.read_csv(ROOT/'data/paper_study/observations/corteva_external_assessments.csv')
    external = external.loc[external.assessment_eligible.eq(True)].copy()
    external['endpoint_series'] = external.physical_unit.astype(str)+'|'+external.organ.astype(str)
    transfer = prepare_fields(external, calendars, weather)
    strict = set(pd.read_csv(ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv').source_unit.astype(str))
    transfer.targets['partition'] = np.where(transfer.targets.physical_unit.astype(str).isin(strict),
        'reused_external_strict', 'reused_external_overlap')
    attach_maxima(transfer, weather)
    transfer_features = assessment_features(transfer, phenology)
    transfer_features.to_parquet(dest/'corteva_features.parquet', index=False)
    predictions, onsets = [], []
    for source, inputs, x in [('BASF', data, features), ('Corteva', transfer, transfer_features)]:
        reference_forecast, trajectory = predict_refinement(inputs, reference)
        process_forecast, _ = predict_refinement(inputs, process)
        hybrid = blend_severity(process_forecast, ridge.predict(x),
            empirical_weight=chosen['empirical_weight'], host_active=host_at_assessments(inputs))
        for name, forecast in [('reference_v1', reference_forecast), ('selected_hybrid_v3', hybrid)]:
            frame = inputs.targets.copy()
            frame['model'], frame['predicted_percent'], frame['source'] = name, forecast, source
            predictions.append(frame)
        onsets.extend(onset_records(inputs, reference, trajectory,
            'unchanged_reference_v1_onset', inputs.targets.partition))
    predictions = pd.concat(predictions, ignore_index=True)
    predictions['physical_unit'] = predictions.physical_unit.astype(str)
    for column in predictions.select_dtypes(include=['object', 'str']).columns:
        observed = predictions[column].dropna()
        if observed.map(type).nunique() > 1:
            predictions[column] = predictions[column].map(lambda x: None if pd.isna(x) else str(x)).astype('string')
    predictions.to_parquet(dest/'frozen_evaluation_predictions.parquet', index=False)
    pd.DataFrame(onsets).to_csv(dest/'separate_process_onset_predictions.csv', index=False)
    rows, paired = [], []
    generator = np.random.default_rng(20261006)
    for partition, group in predictions.groupby('partition'):
        for final in [False, True]:
            for upper in [False, True]:
                estimand, scope = ('final_numeric_assessment' if final else 'all_assessments'), ('upper_three' if upper else 'all_ordinal_leaves')
                selected = endpoint(group[group.model.eq('selected_hybrid_v3')], final, upper)
                other = endpoint(group[group.model.eq('reference_v1')], final, upper)
                for model, frame in [('reference_v1', other), ('selected_hybrid_v3', selected)]:
                    rows.append(dict(partition=partition, model=model, endpoint=estimand,
                                     leaf_scope=scope, **score(frame)))
                matched = selected.merge(other[KEYS+['predicted_percent']], on=KEYS,
                    validate='one_to_one', suffixes=('', '_reference'))
                matched['comparator_percent'] = matched.predicted_percent_reference
                clusters = np.array(sorted(matched.coordinate_year.unique()))
                draws = generator.integers(0, len(clusters), (2000, len(clusters)))
                paired.append(dict(partition=partition, endpoint=estimand, leaf_scope=scope,
                    **paired_intervals(matched, clusters, draws)))
    pd.DataFrame(rows).to_csv(dest/'severity_metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(dest/'paired_improvement_intervals.csv', index=False)
    if sha(dest/'frozen_severity_hybrid.joblib') != freeze['bundle_sha256']:
        raise ValueError('Hybrid bundle changed during evaluation.')
    snapshot = dest/'source_snapshot'; snapshot.mkdir()
    for file in [Path(__file__), ROOT/'model/seasonal_septoria/refinement.py',
                 ROOT/'model/seasonal_septoria/empirical.py']:
        (snapshot/file.name).write_bytes(file.read_bytes())
    write_json(dest/'receipt.json', dict(status='complete', selection=chosen,
        bundle_sha256=freeze['bundle_sha256'], untouched_test_evaluated=False,
        evaluation_samples_previously_examined=True, onset_model_changed=False,
        regional_projection_version='unchanged_v1', main_manuscript_version='unchanged_v1'))
    print(pd.DataFrame(rows).query("endpoint == 'final_numeric_assessment' and leaf_scope == 'all_ordinal_leaves'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
