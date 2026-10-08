"""Select severity refinements on calibration sites before transfer scoring."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from analysis.paper_study.calibrate_seasonal import metrics, sha, write_json
from analysis.paper_study.run_empirical_benchmarks import endpoint, paired_intervals
from model.seasonal_septoria.endpoints import chronological_partition, onset_distance
from model.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.refinement import (
    fit_refinement, map_observations, predict_refinement,
)

FROZEN = ROOT/'analysis/paper_study/seasonal_calibration_v1'
DEFAULT_DEST = ROOT/'analysis/paper_study/severity_refinement_v2_20261006'


def score(frame):
    result = metrics(frame)
    mean = np.average(frame.value, weights=frame.weight)
    variance = np.average((frame.value-mean)**2, weights=frame.weight)
    result['weighted_r2'] = None if variance == 0 else float(1-result['rmse']**2/variance)
    return result


def onset_records(data, fitted, trajectory, model, partition):
    records = []
    for (field, series), group in data.targets.groupby(['field_index', 'endpoint_series']):
        row = group.iloc[0]
        meta = data.metadata.loc[data.metadata.field_index.eq(field)].iloc[0]
        leaf, length = int(row.leaf_index), int(meta.forcing_days)
        raw = trajectory.pycnidia if row.observation_operator == 'pycnidia' else trajectory.damage
        daily = 100*map_observations(raw[field, 1:length+1, leaf], row.metric,
                                     fitted['observation_power'])
        for cutoff in [.1, 1., 5.]:
            ordered = group.sort_values('date')
            positive = ordered[ordered.value.ge(cutoff)]
            upper = None if positive.empty else positive.date.min()
            negative = ordered[ordered.value.lt(cutoff)]
            lower = negative.date.max() if upper is None else negative.loc[
                negative.date.lt(upper), 'date'].max()
            lower = None if pd.isna(lower) else lower
            found = np.flatnonzero(daily >= cutoff)
            predicted = None if not len(found) else pd.Timestamp(meta.sowing_date)+pd.Timedelta(days=int(found[0]))
            records.append(dict(model=model, partition=partition.loc[group.index[0]],
                field_id=row.field_id, endpoint_series=series,
                coordinate_year=row.coordinate_year, leaf_index=leaf,
                cutoff_percent=cutoff, censoring='right' if upper is None else 'left' if lower is None else 'interval',
                last_negative=None if lower is None else str(lower.date()),
                first_positive=None if upper is None else str(upper.date()),
                predicted_visible_date=None if predicted is None else str(predicted.date()),
                **onset_distance(predicted, lower, upper)))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_DEST)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    dest = args.output.resolve()
    if dest.exists() and (not args.resume or (dest/'receipt.json').exists()):
        raise FileExistsError('Use a new archive; completed results are immutable.')
    dest.mkdir(parents=True, exist_ok=True)
    calendar_path = ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv'
    weather_path = ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet'
    source_path = FROZEN/'basf_source_assessments_snapshot.csv'
    config_path = FROZEN/'configuration_before_fitting.json'
    original_config = json.loads(config_path.read_text())
    candidates = []
    for latent_free, power_free in [(False, False), (True, False), (False, True), (True, True)]:
        family = ('fitted_latency' if latent_free else 'fixed_latency')+('_power_link' if power_free else '_direct')
        for final_weight in [0., .5, 1.]:
            candidates.append(dict(name=f'{family}_final{final_weight:g}',
                latent_free=latent_free, power_free=power_free, final_weight=final_weight))
    identity = {str(path.relative_to(ROOT)): sha(path) for path in [
        source_path, calendar_path, weather_path, config_path,
        Path(__file__), ROOT/'model/seasonal_septoria/refinement.py',
        ROOT/'model/seasonal_septoria/core.py', ROOT/'model/seasonal_septoria/field_data.py']}
    config = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        objective='equal-coordinate-year final recorded ordinal-leaf severity RMSE',
        selection='minimum pooled three-location-fold calibration-only final severity RMSE',
        calibration_years=[2017, 2018], development_evaluation_years=[2019],
        candidates=candidates, site_fold=original_config['site_fold'],
        parameter_bounds=dict(alpha=[1e-8, .1], latent_days=[10., 60.], observation_power=[.25, 4.]),
        beta_fixed=0., prior_2019_and_external_results_known=True,
        evaluation_is_reused_development_evidence=True,
        previously_external_source='Corteva', untouched_test_source_available=False,
        onset_is_secondary_diagnostic=True, sowing_and_phenology_unchanged=True,
        observation_mapping_scope='infection_percent_unspecified_basis',
        random_seed=20261006, input_and_source_sha256=identity)
    contract = dest/'configuration_before_fitting.json'
    if args.resume:
        previous = json.loads(contract.read_text())
        if previous['input_and_source_sha256'] != identity or previous['candidates'] != candidates:
            raise ValueError('Input or source changed during refinement; use a new archive.')
    else:
        write_json(contract, config)
    weather, calendars = pd.read_parquet(weather_path), pd.read_csv(calendar_path)
    data = prepare_fields(pd.read_csv(source_path), calendars, weather)
    data.targets['partition'] = chronological_partition(data.targets, [2019])
    data.metadata['partition'] = chronological_partition(data.metadata, [2019])
    data.targets.to_csv(dest/'target_membership.csv', index=False)
    data.metadata.to_csv(dest/'field_season_membership.csv', index=False)
    training = np.flatnonzero(data.targets.partition.eq('calibration'))
    train_sites = set(data.targets.iloc[training].site_id)
    site_fold = original_config['site_fold']
    if train_sites != set(site_fold):
        raise ValueError('Calibration sites differ from the registered partition.')
    selection = []
    baseline_cv = pd.concat([pd.read_parquet(FROZEN/f'inner_latent30_fold{fold}_predictions.parquet')
                             for fold in range(3)], ignore_index=True)
    baseline_score = score(endpoint(baseline_cv, True, False))
    selection.append(dict(name='reference_v1', **baseline_score))
    fold_labels = data.targets.site_id.map(site_fold).to_numpy()
    for candidate in candidates:
        fold_predictions = []
        for fold in range(3):
            fit_indices = training[fold_labels[training] != fold]
            heldout_indices = training[fold_labels[training] == fold]
            fit_sites = set(data.targets.iloc[fit_indices].site_id)
            heldout_sites = set(data.targets.iloc[heldout_indices].site_id)
            if not len(fit_indices) or not len(heldout_indices) or fit_sites & heldout_sites:
                raise ValueError('Invalid location-grouped fold.')
            fit_path = dest/f"inner_{candidate['name']}_fold{fold}.json"
            if fit_path.exists() and args.resume:
                fitted = json.loads(fit_path.read_text())
            else:
                fitted = fit_refinement(data, fit_indices,
                    **{key: candidate[key] for key in ['latent_free', 'power_free', 'final_weight']})
                write_json(fit_path, fitted)
            forecast, _ = predict_refinement(data, fitted)
            scored = data.targets.iloc[heldout_indices].copy()
            scored['predicted_percent'], scored['inner_fold'] = forecast[heldout_indices], fold
            scored['model'] = candidate['name']
            scored.to_parquet(dest/f"inner_{candidate['name']}_fold{fold}_predictions.parquet", index=False)
            fold_predictions.append(scored)
        pooled = pd.concat(fold_predictions, ignore_index=True)
        result = dict(**candidate, **score(endpoint(pooled, True, False)))
        result['all_assessment_rmse'] = score(endpoint(pooled, False, False))['rmse']
        selection.append(result)
        print(f"calibration CV {candidate['name']}: final RMSE={result['rmse']:.4f}; all RMSE={result['all_assessment_rmse']:.4f}", flush=True)
        pd.DataFrame(selection).to_csv(dest/'calibration_only_selection.csv', index=False)
    chosen = min(selection, key=lambda row: row['rmse'])
    if chosen['name'] == 'reference_v1':
        fitted = json.loads((FROZEN/'frozen_main_fit.json').read_text())
        fitted['observation_power'] = 1.
    else:
        fitted = fit_refinement(data, training,
            **{key: chosen[key] for key in ['latent_free', 'power_free', 'final_weight']})
    fitted['selected_candidate'] = chosen['name']
    fitted['selection_final_rmse'] = chosen['rmse']
    write_json(dest/'frozen_selected_fit.json', fitted)
    snapshot = dest/'source_snapshot'
    snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__), ROOT/'model/seasonal_septoria/refinement.py',
                 ROOT/'model/seasonal_septoria/core.py']:
        (snapshot/path.name).write_bytes(path.read_bytes())
    freeze = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        selected_candidate=chosen['name'], reference_cv_final_rmse=baseline_score['rmse'],
        selected_cv_final_rmse=chosen['rmse'],
        selected_fit_sha256=sha(dest/'frozen_selected_fit.json'),
        selection_table_sha256=sha(dest/'calibration_only_selection.csv'),
        external_observations_read_for_fitting_or_selection=False,
        validation_observations_read_for_fitting_or_selection=False,
        previously_published_external_metrics_known=True)
    write_json(dest/'frozen_before_evaluation.json', freeze)
    print(f"FROZEN: {chosen['name']}; final CV RMSE {baseline_score['rmse']:.4f} -> {chosen['rmse']:.4f}", flush=True)
    # The reused evaluation sources are read and scored only after freezing.
    external = pd.read_csv(ROOT/'data/paper_study/observations/corteva_external_assessments.csv')
    external = external.loc[external.assessment_eligible.eq(True)].copy()
    external['endpoint_series'] = external.physical_unit.astype(str)+'|'+external.organ.astype(str)
    transfer = prepare_fields(external, calendars, weather)
    strict_units = set(pd.read_csv(ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv').source_unit.astype(str))
    transfer.targets['partition'] = np.where(transfer.targets.physical_unit.astype(str).isin(strict_units),
        'reused_external_strict', 'reused_external_overlap')
    reference = json.loads((FROZEN/'frozen_main_fit.json').read_text())
    reference['observation_power'] = 1.
    all_predictions, numerical, onsets = [], [], []
    for dataset, inputs in [('BASF', data), ('Corteva', transfer)]:
        for name, model in [('reference_v1', reference), ('selected_refinement_v2', fitted)]:
            forecast, trajectory = predict_refinement(inputs, model)
            scored = inputs.targets.copy()
            scored['model'], scored['predicted_percent'], scored['source'] = name, forecast, dataset
            all_predictions.append(scored)
            numerical.append(dict(source=dataset, model=name,
                mass_error=float(np.max(abs(trajectory.state.sum(axis=-1)-1))),
                minimum_state=float(trajectory.state.min()),
                prediction_min=float(forecast.min()), prediction_max=float(forecast.max())))
            onsets.extend(onset_records(inputs, model, trajectory, name, inputs.targets.partition))
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions['physical_unit'] = predictions.physical_unit.astype(str)
    for column in predictions.select_dtypes(include=['object', 'str']).columns:
        nonmissing = predictions[column].dropna()
        if nonmissing.map(type).nunique() > 1:
            predictions[column] = predictions[column].map(lambda x: None if pd.isna(x) else str(x)).astype('string')
    predictions.to_parquet(dest/'frozen_evaluation_predictions.parquet', index=False)
    write_json(dest/'numerical_checks.json', numerical)
    pd.DataFrame(onsets).to_csv(dest/'onset_interval_predictions.csv', index=False)
    metric_rows, comparisons = [], []
    generator = np.random.default_rng(20261006)
    for partition, group in predictions.groupby('partition'):
        for final in [False, True]:
            for upper in [False, True]:
                scope = 'upper_three' if upper else 'all_ordinal_leaves'
                estimand = 'final_numeric_assessment' if final else 'all_assessments'
                selected = endpoint(group[group.model.eq('selected_refinement_v2')], final, upper)
                other = endpoint(group[group.model.eq('reference_v1')], final, upper)
                for name, frame in [('reference_v1', other), ('selected_refinement_v2', selected)]:
                    metric_rows.append(dict(partition=partition, model=name, endpoint=estimand,
                        leaf_scope=scope, **score(frame)))
                keys = ['field_id', 'endpoint_series', 'date']
                matched = selected.merge(other[keys+['predicted_percent']], on=keys,
                    how='left', validate='one_to_one', suffixes=('', '_reference'))
                if matched.predicted_percent_reference.isna().any():
                    raise ValueError('Paired evaluation membership differs.')
                matched['comparator_percent'] = matched.predicted_percent_reference
                clusters = np.array(sorted(matched.coordinate_year.unique()))
                draws = generator.integers(0, len(clusters), (2000, len(clusters)))
                comparisons.append(dict(partition=partition, endpoint=estimand,
                    leaf_scope=scope, bootstrap_replicates=2000,
                    **paired_intervals(matched, clusters, draws)))
    pd.DataFrame(metric_rows).to_csv(dest/'severity_metrics.csv', index=False)
    pd.DataFrame(comparisons).to_csv(dest/'paired_improvement_intervals.csv', index=False)
    onset_table = pd.DataFrame(onsets)
    onset_summary = []
    for labels, group in onset_table.groupby(['partition', 'model', 'cutoff_percent', 'censoring']):
        cluster_mean = group.groupby('coordinate_year').compatible.mean()
        onset_summary.append(dict(partition=labels[0], model=labels[1], cutoff_percent=labels[2],
            censoring=labels[3], leaf_series=len(group), coordinate_years=len(cluster_mean),
            coordinate_year_compatibility=float(cluster_mean.mean()),
            raw_compatibility=float(group.compatible.mean()),
            median_signed_distance_days=None if group.delta_days.dropna().empty else float(group.delta_days.median())))
    pd.DataFrame(onset_summary).to_csv(dest/'onset_metrics.csv', index=False)
    if sha(dest/'frozen_selected_fit.json') != freeze['selected_fit_sha256']:
        raise ValueError('Selected model changed during evaluation.')
    write_json(dest/'receipt.json', dict(status='complete',
        selected_candidate=chosen['name'], calibration_site_count=len(train_sites),
        calibration_field_seasons=int(data.metadata.partition.eq('calibration').sum()),
        calibration_targets=len(training), fit_sha256=freeze['selected_fit_sha256'],
        reference_cv_final_rmse=baseline_score['rmse'], selected_cv_final_rmse=chosen['rmse'],
        untouched_test_evaluated=False, evaluation_sets_previously_examined=True,
        regional_projection_version='unchanged_v1', main_manuscript_version='unchanged_v1'))
    print(pd.DataFrame(metric_rows).query("endpoint == 'final_numeric_assessment' and leaf_scope == 'all_ordinal_leaves'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
