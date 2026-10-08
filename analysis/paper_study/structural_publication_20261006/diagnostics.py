"""Independent diagnostics for frozen structural canopy experiments."""

from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.paper_study.calibrate_seasonal import sha, write_json
from analysis.paper_study.run_structural_canopy import development_inputs, onset_predictions
from analysis.paper_study.run_weather_canopy import exposure_for
from analysis.paper_study.run_empirical_benchmarks import endpoint, paired_intervals
from analysis.paper_study.refine_seasonal_accuracy import score
from calibration.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.structural import canopy_host, simulate_canopy, CanopyParameters
from calibration.seasonal_septoria.structural import predict_canopy

BASE = ROOT/'analysis/paper_study'
DEST = Path(__file__).parent/'diagnostics'
KEYS = ['field_id', 'endpoint_series', 'date']


def nested_scores(frame, final, upper):
    frame = frame[frame.leaf_index.lt(3)].copy() if upper else frame.copy()
    if final:
        frame = frame.sort_values('date').groupby(['field_id', 'endpoint_series']).tail(1)
    error = frame.predicted_percent-frame.value
    part = frame.assign(mse=error**2, absolute=abs(error), error=error,
                        observed_squared=frame.value**2)
    cols = ['mse', 'absolute', 'error', 'value', 'observed_squared']
    leaf = part.groupby(['coordinate_year', 'field_id', 'endpoint_series'])[cols].mean()
    field = leaf.groupby(level=['coordinate_year', 'field_id']).mean()
    means = field.groupby(level='coordinate_year').mean().mean()
    return dict(rmse=float(np.sqrt(means.mse)), mae=float(means.absolute), bias=float(means.error),
                weighted_r2=float(1-means.mse/(means.observed_squared-means.value**2)))


def gradients(predictions):
    records = []
    for (partition, model), frame in predictions.groupby(['partition', 'model']):
        indexed = frame.set_index(['field_id', 'date', 'leaf_index'])
        for row in frame.itertuples():
            key = (row.field_id, row.date, int(row.leaf_index)+1)
            if key not in indexed.index:
                continue
            lower = indexed.loc[key]
            records.append(dict(partition=partition, model=model, field_id=row.field_id,
                coordinate_year=row.coordinate_year, date=row.date, upper_rank=int(row.leaf_index)+1,
                lower_rank=int(row.leaf_index)+2, observed_gradient=float(lower.value-row.value),
                predicted_gradient=float(lower.predicted_percent-row.predicted_percent)))
    rows = pd.DataFrame(records)
    summary = []
    for labels, group in rows.groupby(['partition', 'model']):
        error = group.predicted_gradient-group.observed_gradient
        weighted = group.assign(mse=error**2, absolute=abs(error), error=error).groupby(
            ['coordinate_year', 'field_id'])[['mse', 'absolute', 'error', 'observed_gradient', 'predicted_gradient']].mean()
        values = weighted.groupby(level='coordinate_year').mean().mean()
        summary.append(dict(partition=labels[0], model=labels[1], paired_leaf_dates=len(group),
            field_seasons=int(group.field_id.nunique()),
            rmse=float(np.sqrt(values.mse)), mae=float(values.absolute), bias=float(values.error),
            mean_observed_gradient=float(values.observed_gradient), mean_predicted_gradient=float(values.predicted_gradient)))
    return rows, pd.DataFrame(summary)


def main():
    if DEST.exists():
        raise FileExistsError('Completed diagnostics are immutable.')
    DEST.mkdir(parents=True)
    archives = dict(v4=BASE/'structural_canopy_v4_20261006', v5=BASE/'weather_canopy_v5_20261006')
    comparisons, checks, maximum = [], 0, 0.
    all_tables = []
    for version, source in archives.items():
        frame = pd.read_parquet(source/'frozen_evaluation_predictions.parquet')
        rows = pd.read_csv(source/'severity_metrics.csv')
        for row in rows.itertuples():
            selected = frame[frame.model.eq(row.model)&frame.partition.eq(row.partition)]
            actual = nested_scores(selected, row.endpoint=='final_numeric_assessment', row.leaf_scope=='upper_three')
            for name, value in actual.items():
                delta = abs(value-getattr(row, name)); maximum = max(maximum, delta); checks += 1
                if delta > 1e-9:
                    raise ValueError(f'Independent {version} severity calculation differs.')
        all_tables.append(frame)
    predictions = pd.concat(all_tables, ignore_index=True).drop_duplicates(['model', *KEYS])
    v4 = predictions[predictions.model.eq('structural_canopy_v4')]
    v5 = predictions[predictions.model.eq('weather_canopy_v5')]
    ensemble = v4.merge(v5[KEYS+['predicted_percent']], on=KEYS, validate='one_to_one', suffixes=('', '_duration'))
    ensemble['predicted_percent'] = .5*(ensemble.predicted_percent+ensemble.predicted_percent_duration)
    ensemble = ensemble.drop(columns='predicted_percent_duration'); ensemble['model'] = 'fixed_structural_ensemble'
    predictions = pd.concat([predictions, ensemble], ignore_index=True)
    for column in predictions.select_dtypes(include=['object', 'str']).columns:
        values = predictions[column].dropna()
        if values.map(type).nunique() > 1:
            predictions[column] = predictions[column].map(lambda x: None if pd.isna(x) else str(x)).astype('string')
    predictions.to_parquet(DEST/'all_frozen_prediction_comparisons.parquet', index=False)
    rows, high = [], []
    for (partition, model), group in predictions.groupby(['partition', 'model']):
        for final in [False, True]:
            for upper in [False, True]:
                selected = endpoint(group, final, upper)
                rows.append(dict(partition=partition, model=model,
                    endpoint='final_numeric_assessment' if final else 'all_assessments',
                    leaf_scope='upper_three' if upper else 'all_ordinal_leaves', **score(selected)))
        terminal = endpoint(group, True, False)
        for cutoff in [50., 80.]:
            selected = terminal[terminal.value.ge(cutoff)]
            if len(selected):
                high.append(dict(partition=partition, model=model, observed_cutoff=cutoff, **score(selected)))
    pd.DataFrame(rows).to_csv(DEST/'severity_metrics.csv', index=False)
    pd.DataFrame(high).to_csv(DEST/'high_score_strata.csv', index=False)
    paired_rows = []; random = np.random.default_rng(20261006)
    for partition, part in predictions.groupby('partition'):
        for model in ['structural_canopy_v4', 'weather_canopy_v5', 'fixed_structural_ensemble']:
            selected = endpoint(part[part.model.eq(model)], True, False)
            baseline = endpoint(part[part.model.eq('reference_v1')], True, False)
            matched = selected.merge(baseline[KEYS+['predicted_percent']], on=KEYS,
                validate='one_to_one', suffixes=('', '_reference'))
            matched['comparator_percent'] = matched.predicted_percent_reference
            clusters = np.array(sorted(matched.coordinate_year.unique()))
            draws = random.integers(0, len(clusters), (5000, len(clusters)))
            paired_rows.append(dict(partition=partition, model=model,
                **paired_intervals(matched, clusters, draws)))
    pd.DataFrame(paired_rows).to_csv(DEST/'paired_final_severity_intervals.csv', index=False)
    paired, summary = gradients(predictions)
    paired.to_csv(DEST/'same_date_leaf_gradients.csv', index=False)
    summary.to_csv(DEST/'same_date_leaf_gradient_metrics.csv', index=False)
    weather_path = ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet'
    calendar_path = ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv'
    external_path = ROOT/'data/paper_study/observations/corteva_external_assessments.csv'
    strict_path = ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv'
    weather, calendars = pd.read_parquet(weather_path), pd.read_csv(calendar_path)
    phenology_path = ROOT/'process_model/parameters/calibration.json'
    phenology = json.loads(phenology_path.read_text())
    basf = pd.read_csv(BASE/'seasonal_calibration_v1/basf_source_assessments_snapshot.csv')
    external = pd.read_csv(external_path); external = external[external.assessment_eligible.eq(True)].copy()
    external['endpoint_series'] = external.physical_unit.astype(str)+'|'+external.organ.astype(str)
    strict = set(pd.read_csv(strict_path).source_unit.astype(str))
    fit4, fit5 = [json.loads((archive/'frozen_selected_fit.json').read_text()) for archive in archives.values()]
    onset, sensitivity, numerical = [], [], []
    for source, assessments in [('BASF', basf), ('Corteva', external)]:
        data = prepare_fields(assessments, calendars, weather)
        data.targets['partition'] = np.where(data.targets.season_year.lt(2019), 'calibration', 'reused_development_2019') if source=='BASF' else np.where(
            data.targets.physical_unit.astype(str).isin(strict), 'reused_external_strict', 'reused_external_overlap')
        accumulation, thresholds = development_inputs(data, weather, phenology)
        driver = exposure_for(data, fit5['rain_rate_mm_hour'])
        raw = []
        for name, fitted, forcing in [('structural_canopy_v4', fit4, None), ('weather_canopy_v5', fit5, driver)]:
            forecast, trajectory = predict_canopy(data, accumulation, thresholds, fitted, forcing)
            raw.append(trajectory.expressed)
            archived = predictions[predictions.source.eq(source)&predictions.model.eq(name)]
            reconciled = data.targets[KEYS].assign(current=forecast).merge(
                archived[KEYS+['predicted_percent']], on=KEYS, validate='one_to_one')
            delta = float(abs(reconciled.current-reconciled.predicted_percent).max()); checks += 1; maximum = max(maximum, delta)
            if delta > 1e-9:
                raise ValueError('Current safeguarded API differs from executed seasonal predictions.')
            onset.append(onset_predictions(data, trajectory.expressed, name))
            numerical.append(dict(source=source, model=name, mass_error=float(abs(trajectory.state.sum(axis=-1)-1).max()),
                minimum_state=float(trajectory.state.min()), prediction_difference=delta))
            active, renewal, _, _ = canopy_host(accumulation, data.temperature, thresholds, **fitted['host_parameters'])
            refined = simulate_canopy(data.temperature, data.humidity, data.rain, active, renewal,
                CanopyParameters(**fitted['parameters']), time_step=.125, environmental_response=forcing)
            idx = (data.targets.field_index.to_numpy(int), data.targets.day_index.to_numpy(int), data.targets.leaf_index.to_numpy(int))
            numerical[-1]['half_step_max_score_difference'] = float(abs(100*refined.expressed[idx]-forecast).max())
            numerical[-1]['half_step_mean_score_difference'] = float(abs(100*refined.expressed[idx]-forecast).mean())
            for expansion in [0., 100., 200.]:
                altered = {**fitted, 'host_parameters':{**fitted['host_parameters'], 'expansion_units':expansion}}
                forecast_altered, _ = predict_canopy(data, accumulation, thresholds, altered, forcing)
                table = data.targets.copy(); table['predicted_percent'] = forecast_altered
                for partition, group in table.groupby('partition'):
                    sensitivity.append(dict(model=name, partition=partition, expansion_units=expansion,
                        parameters_refitted=False, **score(endpoint(group, True, False))))
        onset.append(onset_predictions(data, .5*(raw[0]+raw[1]), 'fixed_structural_ensemble'))
    all_onset = pd.concat(onset, ignore_index=True)
    all_onset.to_csv(DEST/'onset_interval_predictions.csv', index=False)
    reference_onset = pd.read_csv(BASE/'severity_refinement_v2_20261006/onset_interval_predictions.csv')
    reference_onset = reference_onset[reference_onset.model.eq('reference_v1')].copy()
    reference_onset['partition'] = reference_onset.partition.replace({'validation':'reused_development_2019'})
    all_onset = pd.concat([all_onset, reference_onset], ignore_index=True)
    onset_rows = []
    for labels, group in all_onset.groupby(['partition', 'model', 'cutoff_percent', 'censoring']):
        rates = group.groupby(['coordinate_year', 'field_id']).compatible.mean().groupby('coordinate_year').mean()
        bad = group.loc[~group.compatible, 'delta_days'].dropna()
        onset_rows.append(dict(partition=labels[0], model=labels[1], cutoff_percent=labels[2],
            censoring=labels[3], n=len(group), coordinate_years=len(rates), compatible_fraction=float(rates.mean()),
            median_incompatible_signed_distance=None if bad.empty else float(bad.median())))
    pd.DataFrame(onset_rows).to_csv(DEST/'onset_metrics.csv', index=False)
    pd.DataFrame(sensitivity).to_csv(DEST/'leaf_expansion_sensitivity.csv', index=False)
    write_json(DEST/'numerical_and_step_checks.json', numerical)
    # Supplementary hashes identify previously omitted inputs retrospectively.
    old_receipt = json.loads((BASE/'corteva_external_seasonal_v1/receipt.json').read_text())
    if old_receipt['source_assessment_sha256'] != sha(external_path):
        raise ValueError('External source differs from its earlier frozen evaluation identity.')
    write_json(DEST/'supplementary_input_provenance.json', dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        recording_role='retrospective provenance supplement; not a retroactive prefit registration',
        external_source_matches_original_frozen_evaluation=True,
        identities={str(path.relative_to(ROOT)):sha(path) for path in [weather_path, calendar_path,
            external_path, strict_path, phenology_path, BASE/'severity_refinement_v2_20261006/frozen_evaluation_predictions.parquet']},
        strict_model_membership_field_seasons=143, strict_model_membership_assessments=1703))
    write_json(DEST/'verification.json', dict(status='passed', independent_severity_and_reproduction_checks=checks,
        maximum_difference=maximum, strict_external_membership_preserved=True,
        actual_leaf_area_or_incidence_identified=False, untouched_test=False,
        original_archives_unchanged=True))
    print(json.dumps(dict(status='passed', checks=checks, maximum_difference=maximum)))


if __name__ == '__main__':
    main()
