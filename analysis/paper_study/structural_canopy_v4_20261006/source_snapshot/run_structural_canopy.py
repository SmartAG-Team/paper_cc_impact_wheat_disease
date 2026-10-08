"""Freeze a structural canopy score model before reused transfer evaluation."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.paper_study.calibrate_seasonal import sha, write_json
from analysis.paper_study.refine_seasonal_accuracy import score
from analysis.paper_study.run_empirical_benchmarks import attach_maxima, endpoint, paired_intervals
from model.seasonal_septoria.endpoints import onset_distance
from model.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.regional import tpv_accumulation
from model.seasonal_septoria.structural import fit_canopy, predict_canopy

FROZEN = ROOT/'analysis/paper_study/seasonal_calibration_v1'
DEFAULT = ROOT/'analysis/paper_study/structural_canopy_v4_20261006'
KEYS = ['field_id', 'endpoint_series', 'date']


def development_inputs(data, weather, phenology):
    attach_maxima(data, weather)
    accumulated = np.zeros_like(data.temperature)
    for meta in data.metadata.itertuples():
        field, days = int(meta.field_index), int(meta.forcing_days)
        dates = pd.date_range(meta.sowing_date, periods=days)
        accumulated[field, :days] = tpv_accumulation(data.temperature[field:field+1, :days],
            data.maximum_temperature[field:field+1, :days], [meta.latitude], dates.dayofyear,
            np.ones((1, days), bool), phenology)[0]
        accumulated[field, days:] = accumulated[field, days-1]
    thresholds = {row['BBCH']:row['Cumulative_t_pp_v_GDD'] for row in phenology['thresholds']}
    return accumulated, thresholds


def onset_predictions(data, expressed, model):
    rows = []
    for (field, series), group in data.targets.groupby(['field_index', 'endpoint_series']):
        row = group.iloc[0]
        meta = data.metadata.loc[data.metadata.field_index.eq(field)].iloc[0]
        daily = 100*expressed[field, 1:int(meta.forcing_days)+1, int(row.leaf_index)]
        for cutoff in [.1, 1., 5.]:
            ordered = group.sort_values('date')
            positive = ordered[ordered.value.ge(cutoff)]
            upper = None if positive.empty else positive.date.min()
            negative = ordered[ordered.value.lt(cutoff)]
            lower = negative.date.max() if upper is None else negative.loc[negative.date.lt(upper), 'date'].max()
            lower = None if pd.isna(lower) else lower
            found = np.flatnonzero(daily >= cutoff)
            predicted = None if not len(found) else pd.Timestamp(meta.sowing_date)+pd.Timedelta(days=int(found[0]))
            rows.append(dict(model=model, partition=row.partition, field_id=row.field_id,
                coordinate_year=row.coordinate_year, endpoint_series=series, leaf_index=row.leaf_index,
                cutoff_percent=cutoff, last_negative=lower, first_positive=upper,
                predicted_visible_date=predicted,
                censoring='right' if upper is None else 'left' if lower is None else 'interval',
                **onset_distance(predicted, lower, upper)))
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(); dest = args.output.resolve()
    if dest.exists() and (not args.resume or (dest/'receipt.json').exists()):
        raise FileExistsError('Use a new structural experiment archive.')
    dest.mkdir(parents=True, exist_ok=True)
    paths = dict(source=FROZEN/'basf_source_assessments_snapshot.csv',
        weather=ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet',
        calendars=ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv',
        phenology=ROOT/'process_model/parameters/calibration.json')
    protocol = json.loads((FROZEN/'configuration_before_fitting.json').read_text())
    phenology = json.loads(paths['phenology'].read_text())
    thresholds = {row['BBCH']:row['Cumulative_t_pp_v_GDD'] for row in phenology['thresholds']}
    candidates = []
    for flag in [.3, .6, .9]:
        coupled = flag*(thresholds[51]-thresholds[31])/2
        for label, interval in [('coupled', coupled), ('80', 80.), ('120', 120.), ('160', 160.)]:
            for delay in [20., 30.]:
                for amplified in [False, True]:
                    candidates.append(dict(name=f'flag{flag:g}_spacing{label}_delay{delay:g}_amp{int(amplified)}',
                        host_parameters=dict(flag_fraction=flag, leaf_interval=interval, expansion_units=100.),
                        amplification_free=amplified, latent_days=delay))
    dependencies = [Path(__file__), *paths.values(),
        ROOT/'analysis/paper_study/STRUCTURAL_REVISION_PROTOCOL_20261006.txt',
        ROOT/'analysis/paper_study/calibrate_seasonal.py',
        ROOT/'analysis/paper_study/refine_seasonal_accuracy.py',
        ROOT/'analysis/paper_study/run_empirical_benchmarks.py',
        FROZEN/'configuration_before_fitting.json', FROZEN/'frozen_main_fit.json',
        *sorted((ROOT/'model/seasonal_septoria').glob('*.py')),
        *sorted((ROOT/'process_model').glob('*.py')),
        *sorted((ROOT/'process_model/_support').glob('*.py'))]
    identity = {str(path.relative_to(ROOT)):sha(path) for path in dependencies}
    config = dict(registered_utc=datetime.now(timezone.utc).isoformat(), candidates=candidates,
        selected_endpoint='final_numeric_assessment_all_ordinal_leaves', final_loss_weight=.5,
        selection='minimum calibration-only pooled location-fold final RMSE',
        calibration_years=[2017, 2018], site_fold=protocol['site_fold'],
        effective_initiation_bounds=[1e-6, .1], effective_amplification_bounds=[1e-5, .3],
        prior_evaluation_and_refinement_exposure=True, untouched_test=False,
        observation_semantics='INFECT percentage proxy; unknown denominator',
        input_and_source_sha256=identity)
    contract = dest/'configuration_before_fitting.json'
    if args.resume:
        prior = json.loads(contract.read_text())
        if prior['input_and_source_sha256'] != identity:
            raise ValueError('Dependencies changed; start a new version.')
    else:
        write_json(contract, config)
    weather, calendars = pd.read_parquet(paths['weather']), pd.read_csv(paths['calendars'])
    data = prepare_fields(pd.read_csv(paths['source']), calendars, weather)
    data.targets['partition'] = np.where(data.targets.season_year.lt(2019),
        'calibration', 'reused_development_2019')
    data.targets.to_csv(dest/'target_membership.csv', index=False)
    data.metadata.to_csv(dest/'field_season_membership.csv', index=False)
    accumulation, thresholds = development_inputs(data, weather, phenology)
    np.savez_compressed(dest/'development_inputs.npz', accumulation=accumulation)
    training = np.flatnonzero(data.targets.partition.eq('calibration'))
    folds = data.targets.site_id.map(protocol['site_fold']).to_numpy()
    if set(data.targets.iloc[training].site_id) != set(protocol['site_fold']):
        raise ValueError('Original calibration site membership changed.')
    selection = []
    for index, candidate in enumerate(candidates):
        cross_predictions = []
        for fold in range(3):
            fitted_rows = training[folds[training] != fold]
            heldout_rows = training[folds[training] == fold]
            if set(data.targets.iloc[fitted_rows].site_id) & set(data.targets.iloc[heldout_rows].site_id):
                raise ValueError('Location fold leakage.')
            fit_path = dest/f"inner_{candidate['name']}_fold{fold}.json"
            if fit_path.exists() and args.resume:
                fitted = json.loads(fit_path.read_text())
            else:
                fitted = fit_canopy(data, accumulation, thresholds, fitted_rows,
                    candidate['host_parameters'], candidate['amplification_free'], candidate['latent_days'])
                write_json(fit_path, fitted)
            forecast, _ = predict_canopy(data, accumulation, thresholds, fitted)
            scored = data.targets.iloc[heldout_rows].copy()
            scored['model'], scored['predicted_percent'], scored['inner_fold'] = candidate['name'], forecast[heldout_rows], fold
            scored.to_parquet(dest/f"inner_{candidate['name']}_fold{fold}_predictions.parquet", index=False)
            cross_predictions.append(scored)
        pooled = pd.concat(cross_predictions, ignore_index=True)
        result = dict(**candidate, **score(endpoint(pooled, True, False)),
                      all_assessment_rmse=score(endpoint(pooled, False, False))['rmse'])
        selection.append(result)
        write_json(dest/'calibration_only_selection.json', selection)
        print(f"CV {index+1}/{len(candidates)} {candidate['name']}: final {result['rmse']:.4f}, all {result['all_assessment_rmse']:.4f}", flush=True)
    chosen = min(selection, key=lambda row: row['rmse'])
    fitted = fit_canopy(data, accumulation, thresholds, training, chosen['host_parameters'],
                        chosen['amplification_free'], chosen['latent_days'])
    fitted['selected_candidate'], fitted['selection_cv_rmse'] = chosen['name'], chosen['rmse']
    write_json(dest/'frozen_selected_fit.json', fitted)
    freeze = dict(frozen_utc=datetime.now(timezone.utc).isoformat(),
        fit_sha256=sha(dest/'frozen_selected_fit.json'), selection=chosen,
        evaluation_observations_used=False, prior_evaluation_exposure=True)
    write_json(dest/'frozen_before_evaluation.json', freeze)
    snapshot = dest/'source_snapshot'; snapshot.mkdir(exist_ok=True)
    for path in [Path(__file__), ROOT/'model/seasonal_septoria/structural.py',
                 ROOT/'analysis/paper_study/STRUCTURAL_REVISION_PROTOCOL_20261006.txt']:
        (snapshot/path.name).write_bytes(path.read_bytes())
    print(f"FROZEN structural model: {chosen['name']} with CV RMSE {chosen['rmse']:.4f}", flush=True)
    external = pd.read_csv(ROOT/'data/paper_study/observations/corteva_external_assessments.csv')
    external = external.loc[external.assessment_eligible.eq(True)].copy()
    external['endpoint_series'] = external.physical_unit.astype(str)+'|'+external.organ.astype(str)
    transfer = prepare_fields(external, calendars, weather)
    strict = set(pd.read_csv(ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv').source_unit.astype(str))
    transfer.targets['partition'] = np.where(transfer.targets.physical_unit.astype(str).isin(strict),
        'reused_external_strict', 'reused_external_overlap')
    transfer_accumulation, _ = development_inputs(transfer, weather, phenology)
    np.savez_compressed(dest/'external_development_inputs.npz', accumulation=transfer_accumulation)
    predictions, onsets, numerical = [], [], []
    reference_archive = pd.read_parquet(ROOT/'analysis/paper_study/severity_refinement_v2_20261006/frozen_evaluation_predictions.parquet')
    for source, inputs, accumulated in [('BASF', data, accumulation), ('Corteva', transfer, transfer_accumulation)]:
        forecast, trajectory = predict_canopy(inputs, accumulated, thresholds, fitted)
        reference = reference_archive[reference_archive.source.eq(source)&reference_archive.model.eq('reference_v1')]
        scored = inputs.targets.copy()
        scored['model'], scored['source'], scored['predicted_percent'] = 'structural_canopy_v4', source, forecast
        predictions.append(scored)
        baseline = inputs.targets.merge(reference[KEYS+['predicted_percent']], on=KEYS,
            how='left', validate='one_to_one')
        if baseline.predicted_percent.isna().any():
            raise ValueError('Reference prediction membership differs.')
        baseline['model'], baseline['source'] = 'reference_v1', source
        predictions.append(baseline)
        onsets.append(onset_predictions(inputs, trajectory.expressed, 'structural_canopy_v4'))
        numerical.append(dict(source=source,
            mass_error=float(abs(trajectory.state.sum(axis=-1)-1).max()),
            minimum_state=float(trajectory.state.min()),
            maximum_score=float(forecast.max()), minimum_score=float(forecast.min())))
    predictions = pd.concat(predictions, ignore_index=True)
    predictions['physical_unit'] = predictions.physical_unit.astype(str)
    for column in predictions.select_dtypes(include=['object', 'str']).columns:
        nonmissing = predictions[column].dropna()
        if nonmissing.map(type).nunique() > 1:
            predictions[column] = predictions[column].map(lambda value: None if pd.isna(value) else str(value)).astype('string')
    predictions.to_parquet(dest/'frozen_evaluation_predictions.parquet', index=False)
    pd.concat(onsets).to_csv(dest/'onset_interval_predictions.csv', index=False)
    write_json(dest/'numerical_checks.json', numerical)
    rows, paired = [], []; generator = np.random.default_rng(20261006)
    for partition, group in predictions.groupby('partition'):
        for final in [False, True]:
            for upper in [False, True]:
                name = 'final_numeric_assessment' if final else 'all_assessments'
                scope = 'upper_three' if upper else 'all_ordinal_leaves'
                selected = endpoint(group[group.model.eq('structural_canopy_v4')], final, upper)
                baseline = endpoint(group[group.model.eq('reference_v1')], final, upper)
                for model, frame in [('reference_v1', baseline), ('structural_canopy_v4', selected)]:
                    rows.append(dict(model=model, partition=partition, endpoint=name, leaf_scope=scope, **score(frame)))
                matched = selected.merge(baseline[KEYS+['predicted_percent']], on=KEYS,
                    validate='one_to_one', suffixes=('', '_reference'))
                matched['comparator_percent'] = matched.predicted_percent_reference
                clusters = np.array(sorted(matched.coordinate_year.unique()))
                draws = generator.integers(0, len(clusters), (2000, len(clusters)))
                paired.append(dict(partition=partition, endpoint=name, leaf_scope=scope,
                    **paired_intervals(matched, clusters, draws)))
    pd.DataFrame(rows).to_csv(dest/'severity_metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(dest/'paired_improvement_intervals.csv', index=False)
    if sha(dest/'frozen_selected_fit.json') != freeze['fit_sha256']:
        raise ValueError('Frozen fit changed during evaluation.')
    write_json(dest/'receipt.json', dict(status='complete', selected_candidate=chosen['name'],
        selection_cv_rmse=chosen['rmse'], fit_sha256=freeze['fit_sha256'],
        evaluation_is_reused_development=True, untouched_test_evaluated=False,
        manuscript_and_regional_model_unchanged=True))
    print(pd.DataFrame(rows).query("endpoint == 'final_numeric_assessment' and leaf_scope == 'all_ordinal_leaves'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
