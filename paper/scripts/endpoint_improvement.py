"""Two training-only extensions of the retained same-field monitoring filter.

The four covariance parameters and bounded prior mean are fitted jointly on
historical inner-coordinate-held-out baseline forecasts. Current observations
enter the fitting loss as targets; prediction uses only strictly earlier dates
within the same field. This is adaptive internal development, not independent
validation or evidence of autonomous climate-response skill.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits

from models.monitoring.discrepancy_filter import _prepare, _predict
from models.monitoring.discrepancy_filter import predict_discrepancy
from models.monitoring.discrepancy_experiment import forecast_baseline
from models.monitoring.evaluation import score_predictions, score_subset, paired_interval
from models.monitoring.innovation_common import ROOT, load_cohort, register
from models.monitoring.inputs import sha
from paper.analyses.source_support.metrics import hierarchical_weights

BASELINE = ROOT / 'data/final_monitoring/folds'
RESTORED = ROOT / 'data/final_monitoring/evaluation'
CANDIDATES = ['endpoint_joint_mean', 'endpoint_native_growth']
REFERENCE = 'baseline_prediction'


def native_growth(frame):
    """Extrapolated logit change from two earlier dates of one native code."""
    frame = frame.reset_index(drop=True)
    dates = pd.to_datetime(frame.date).to_numpy(dtype='datetime64[D]').astype(int)
    if frame[['field_id', 'leaf_index', 'date']].isna().any().any():
        raise ValueError('Known field, native code and assessment date required')
    y = frame.observed_percent.to_numpy(float) / 100.
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Finite bounded severity required')
    result = np.zeros(len(frame))
    for _, group in frame.groupby(['field_id', 'leaf_index'], sort=False):
        ix = group.index.to_numpy()
        for now in np.unique(dates[ix]):
            earlier = np.unique(dates[ix][dates[ix] < now])
            if len(earlier) < 2:
                continue
            a, b = earlier[-2:]
            ya, yb = [y[ix[dates[ix] == d]].mean() for d in [a, b]]
            change = logit(np.clip(yb, .005, .995)) - logit(np.clip(ya, .005, .995))
            result[ix[dates[ix] == now]] = change * (now - b) / (b - a)
    return result


def _mean_prepared(theta, prepared, trend):
    linked = prepared['base']
    shift = theta[4] + (np.exp(theta[5]) - 1.) * linked
    if len(theta) == 7:
        shift = shift + theta[6] * trend
    mean = linked + shift
    raw = expit(mean)
    endpoints = (prepared['raw_base'] == 0.) | (prepared['raw_base'] == 1.)
    raw[endpoints] = prepared['raw_base'][endpoints]
    return dict(prepared, base=mean, raw_base=raw, residual=prepared['residual'] - shift)


def predict_endpoint(parameters, frame, baseline, *, growth):
    theta = np.asarray(parameters, float)
    if theta.shape != (7 if growth else 6,) or not np.isfinite(theta).all():
        raise ValueError('Finite aligned registered endpoint parameters required')
    prepared = _prepare(frame, baseline, 'logit')
    trend = native_growth(frame) if growth else None
    return _predict(theta[:4], _mean_prepared(theta, prepared, trend))


def fit_endpoint(frame, baseline, weights, initial_kernel, *, growth, initial_mean=None):
    prepared = _prepare(frame, baseline, 'logit')
    target = frame.observed_percent.to_numpy(float) / 100.
    weight = np.asarray(weights, float)
    if weight.shape != target.shape or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError('Aligned finite positive scientific weights required')
    weight = weight / weight.sum()
    trend = native_growth(frame) if growth else None
    initial_mean = np.zeros(2) if initial_mean is None else np.asarray(initial_mean, float)
    initial = np.r_[initial_kernel, initial_mean, [0.] if growth else []]
    reference = np.r_[np.log([1., 1., 30., 30.]), np.zeros(len(initial) - 4)]
    trace = []

    def objective(theta):
        prediction = _predict(theta[:4], _mean_prepared(theta, prepared, trend)) / 100.
        penalty = 1e-5 * np.sum((theta[:4] - reference[:4]) ** 2)
        penalty += 1e-4 * np.sum(theta[4:] ** 2)
        value = float(weight @ (prediction - target) ** 2 + penalty)
        trace.append(value)
        return value

    initial_objective = objective(initial)
    bounds = [tuple(x) for x in np.log([[.01, 100.], [.01, 100.], [2., 180.], [2., 180.]])]
    bounds += [(-5., 5.), (np.log(.1), np.log(5.))]
    if growth:
        bounds.append((0., 1.))
    optimized = minimize(objective, initial, method='L-BFGS-B', bounds=bounds,
                         options=dict(maxiter=60, ftol=1e-10))
    if not np.isfinite(optimized.fun) or not np.isfinite(optimized.x).all():
        raise RuntimeError('Nonfinite endpoint fit')
    return dict(parameters=optimized.x, growth=growth, converged=bool(optimized.success),
        iterations=int(optimized.nit), message=str(optimized.message),
        objective=float(optimized.fun), initial_objective=initial_objective, trace=trace,
        fitting='Original hierarchy-weighted prequential severity MSE with weak parameter shrinkage',
        training_baseline='Cached five-inner-coordinate-held-out process/history forecasts; joint conditional fit')


def visit_context(frame):
    """Whole-day visit labels and strictly earlier native-code metadata."""
    frame = frame.reset_index(drop=True)
    dates = pd.to_datetime(frame.date)
    result = pd.DataFrame(index=frame.index)
    first = dates.groupby(frame.field_id).transform('min')
    last = dates.groupby(frame.field_id).transform('max')
    result['first_visit'] = dates.eq(first)
    result['last_visit'] = dates.eq(last)
    result['only_visit'] = first.eq(last)
    result['prior_same_nativecode_count'] = 0
    result['prior_same_nativecode_visit_count'] = 0
    result['prior_same_nativecode_lastdate'] = ''
    for _, group in frame.groupby(['field_id', 'leaf_index'], sort=False):
        ix = group.index.to_numpy()
        for now in dates.iloc[ix].unique():
            query = ix[dates.iloc[ix].eq(now)]
            past = ix[dates.iloc[ix].lt(now)]
            result.loc[query, 'prior_same_nativecode_count'] = len(past)
            result.loc[query, 'prior_same_nativecode_visit_count'] = dates.iloc[past].nunique()
            if len(past):
                result.loc[query, 'prior_same_nativecode_lastdate'] = dates.iloc[past].max().strftime('%Y-%m-%d')
    return result


def _write_scores(frames, out):
    metrics, subgroups, intervals, forecast_bins = [], [], [], []
    for design, frame in frames.items():
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        masks = dict(low_le10=frame.observed_percent.le(10), severe_above75=frame.observed_percent.gt(75),
            first_visit=frame.first_visit, last_visit=frame.last_visit, only_visit=frame.only_visit,
            earlier_field=~frame.first_visit, BASF=frame.source.eq('BASF'), Corteva=frame.source.eq('Corteva'),
            first_visit_severe=frame.first_visit & frame.observed_percent.gt(75),
            last_visit_severe=frame.last_visit & frame.observed_percent.gt(75),
            two_earlier_nativecode_dates=frame.prior_same_nativecode_visit_count.ge(2))
        weight = hierarchical_weights(frame)
        for name in [REFERENCE, *CANDIDATES]:
            metrics.append(dict(evaluation=design, model=name, **score_predictions(frame, name)))
            for label, mask in masks.items():
                if mask.any():
                    subgroups.append(dict(evaluation=design, model=name, population=label,
                                          **score_subset(frame, mask, name)))
            for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100.0000001)]:
                mask = frame[name].ge(lo) & frame[name].lt(hi)
                if mask.any():
                    forecast_bins.append(dict(evaluation=design, model=name, lower=lo, upper=min(hi, 100),
                        n=int(mask.sum()), original_weight_mass=float(weight[mask].sum()),
                        forecast_mean=float(np.average(frame.loc[mask, name], weights=weight[mask])),
                        observed_mean=float(np.average(frame.loc[mask, 'observed_percent'], weights=weight[mask]))))
        for name in CANDIDATES:
            lower, upper = paired_interval(frame, name, REFERENCE)
            intervals.append(dict(evaluation=design, model=name, reference=REFERENCE, lower=lower, upper=upper,
                interpretation='Paired site bootstrap within repeatedly reused development cohort; not external confirmation'))
    metrics, subgroups = pd.DataFrame(metrics), pd.DataFrame(subgroups)
    metrics.to_csv(out / 'metrics.csv', index=False)
    subgroups.to_csv(out / 'subgroup_metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'paired_intervals.csv', index=False)
    pd.DataFrame(forecast_bins).to_csv(out / 'forecast_bin_calibration.csv', index=False)
    decisions = {}
    for name in CANDIDATES:
        gates = {}
        for design in frames:
            full = metrics.loc[metrics.evaluation.eq(design)].set_index('model')
            group = subgroups.loc[subgroups.evaluation.eq(design)].set_index(['model', 'population'])
            gates[design + '_overall_rmse_improves'] = bool(full.loc[name, 'rmse'] < full.loc[REFERENCE, 'rmse'])
            for population in ['severe_above75', 'low_le10', 'first_visit', 'last_visit']:
                gates[design + '_' + population + '_rmse_improves'] = bool(
                    group.loc[(name, population), 'rmse'] < group.loc[(REFERENCE, population), 'rmse'])
            gates[design + '_severe_absolute_bias_improves'] = bool(
                abs(group.loc[(name, 'severe_above75'), 'bias']) < abs(group.loc[(REFERENCE, 'severe_above75'), 'bias']))
        required = [gates[d + suffix] for d in frames for suffix in [
            '_overall_rmse_improves', '_severe_above75_rmse_improves', '_severe_absolute_bias_improves']]
        decisions[name] = dict(core_endpoint_criteria_pass=all(required), gates=gates,
                              independent_external_validation=False, publication_promoted=False)
    (out / 'candidate_decisions.json').write_text(json.dumps(decisions, indent=2) + '\n')
    return metrics, subgroups, decisions


def run(output=None):
    out = Path(output or ROOT / 'outputs/gcb_endpoint_improvement_20261023').resolve()
    codes = [Path(__file__), ROOT / 'models/monitoring/discrepancy_filter.py',
        ROOT / 'models/monitoring/discrepancy_experiment.py',
        ROOT / 'models/monitoring/field_history_models.py',
        ROOT / 'models/monitoring/robust_models.py',
        ROOT / 'models/monitoring/history_interval_hybrid.py']
    registration = register(out, CANDIDATES[0], dict(
        candidate_limit=2, candidate_names=CANDIDATES,
        hypothesis='Historical mean calibration and a measured native-code growth increment may reduce low/high and early/late monitoring errors',
        evidence='Observed-outcome extreme bias does not establish forecast miscalibration; prior forecasts are already close to calibrated at their extremes',
        prior_experiments_checked=['severity_calibration_support_repair_20261022', 'error_dynamics_20261022',
            'random_coefficients_20261022', 'field_meta_kernel_20261022', 'offset_growth_20261022'],
        candidates=dict(endpoint_joint_mean='Retained logit GP with jointly fitted intercept and positive slope of prior logit',
            endpoint_native_growth='Joint mean plus one [0,1] coefficient for the extrapolated logit difference of two strictly earlier same-native-code visit means'),
        fitting='Six/seven parameters jointly fitted to original hierarchy-weighted prequential percentage MSE; L-BFGS-B60; covariance prior 1e-5 and added coefficient ridge 1e-4',
        baseline_training='Reuse exact saved five-inner-coordinate-held-out historical baseline forecasts; all years at coordinates excluded by saved baseline membership',
        fitting_interpretation='Joint conditional-model estimation on historical targets; added parameters are not held-out during their own fitting and no independently calibrated uncertainty is claimed',
        outer_design='All original20 chronological/chronological+50km folds run; all2017 assessment rows in both designs, without screen-based support changes',
        information='Strictly earlier own-field observations; whole query day withheld across native codes; same-code nominal equality only; original weather ends q-1',
        no_new_inputs=True, satellite=False, other_current_season_fields=False,
        no_cultivar_invention=True, original_weights=True, original_rows=True,
        adaptive_internal_comparison=True, climate_model_changed=False,
        promotion='No active/publication pointer changes; endpoint criteria require lower overall and severe RMSE and lower absolute severe bias in both designs'), codes)
    original_registration = json.loads((BASELINE / 'registration.json').read_text())
    for key in ['source_sha256', 'code_sha256']:
        registration['source_sha256'].update(original_registration[key])
    inputs = [*sorted(BASELINE.glob('*_model.joblib')), BASELINE / 'registration.json',
        BASELINE / 'assessment_predictions.csv', BASELINE / 'spatial_predictions.csv',
        RESTORED / 'assessment_predictions.csv', RESTORED / 'spatial_predictions.csv',
        RESTORED / 'selection.json', ROOT / 'data/final_climate/scientific_manifest.json',
        ROOT / 'data/final_models/manifest.json']
    for path in inputs:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    assert all(sha(ROOT / name) == digest for name, digest in registration['source_sha256'].items())
    (out / 'registration.json').write_text(json.dumps(registration, indent=2) + '\n')
    obs, arrays, history, _, splits, position = load_cohort()
    context_metadata = visit_context(obs)
    context_metadata.index = obs.global_target_index
    parents = {label: pd.read_csv(RESTORED / name, float_precision='round_trip').set_index('global_target_index')
        for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    parts, progress, provenance = {}, [], []
    started = time.monotonic()
    for split in splits:
        print('start', split['fold'], flush=True)
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        ti, vi = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        train = obs.iloc[ti].reset_index(drop=True)
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        prior_path = BASELINE / (split['fold'] + '_model.joblib')
        prior = joblib.load(prior_path)
        assert prior['train_ids'] == split['train'] and prior['test_ids'] == split['test']
        inner_fold = np.full(len(train), -1)
        held = []
        memberships = []
        for member in prior['inner_memberships']:
            a, b = [position.loc[member[key]].to_numpy() for key in ['train', 'test']]
            assert set(zip(obs.iloc[a].latitude, obs.iloc[a].longitude)).isdisjoint(
                set(zip(obs.iloc[b].latitude, obs.iloc[b].longitude)))
            assert set(member['train']).isdisjoint(member['test'])
            assert set(member['train']) | set(member['test']) == set(split['train'])
            held.extend(member['test'])
            member_position = train.global_target_index.isin(member['test']).to_numpy()
            inner_fold[member_position] = member['inner_fold']
            memberships.append(member)
        assert sorted(held) == sorted(split['train']) and (inner_fold >= 0).all()
        oof, weights = prior['inner_predictions'], hierarchical_weights(train)
        assert oof.shape == (len(train),) and np.isfinite(oof).all()
        fits = {}
        fits[CANDIDATES[0]] = fit_endpoint(train, oof, weights,
            prior['corrections']['logit']['parameters'], growth=False)
        fits[CANDIDATES[1]] = fit_endpoint(train, oof, weights,
            fits[CANDIDATES[0]]['parameters'][:4], growth=True,
            initial_mean=fits[CANDIDATES[0]]['parameters'][4:6])
        fitted = dict(filters=fits, baseline_path=str(prior_path), baseline_sha256=sha(prior_path),
            train_ids=split['train'], test_ids=split['test'], historical_baseline=oof,
            original_training_weight=weights, inner_memberships=memberships,
            training_baseline_coordinate_excluded=True, adaptive_internal_comparison=True)
        model_path = out / (split['fold'] + '_model.joblib')
        joblib.dump(fitted, model_path)
        ci = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        assert set(obs.iloc[ci].global_target_index).isdisjoint(split['train'])
        qi = pd.Series(np.arange(len(ci)), index=ci).loc[vi].to_numpy()
        baseline = forecast_baseline(prior['base'], arrays, history, obs, ci)
        retained = predict_discrepancy(prior['corrections']['logit']['parameters'], obs.iloc[ci], baseline, 'logit')

        def forecast(bundle):
            assert sha(Path(bundle['baseline_path'])) == bundle['baseline_sha256']
            return {name: predict_endpoint(fit['parameters'], obs.iloc[ci], baseline, growth=fit['growth'])
                    for name, fit in bundle['filters'].items()}

        values, replay = forecast(fitted), forecast(joblib.load(model_path))
        for name in CANDIDATES:
            np.testing.assert_array_equal(values[name], replay[name])
        part = parents[label].loc[split['test'], [
            'field_id', 'site_id', 'source', 'coordinate_year', 'endpoint_series', 'leaf_index', 'date',
            'day_index', 'partition', 'observed_percent', 'season_year', 'latitude', 'longitude']].reset_index()
        np.testing.assert_array_equal(retained[qi], parents[label].loc[split['test'], 'prediction_percent'])
        part[REFERENCE] = retained[qi]
        for name in CANDIDATES:
            part[name] = values[name][qi]
        part['native_growth_logit'] = native_growth(obs.iloc[ci])[qi]
        part = part.join(context_metadata.loc[split['test']].reset_index(drop=True))
        part['evaluation'], part['outer_fold'] = label, split['fold']
        # Mutate query-day and future outcomes, including all codes; verify the
        # new model's complete prediction prefix remains exactly unchanged.
        context = obs.iloc[ci].copy()
        cutoff = sorted(context.date.unique())[len(context.date.unique()) // 2]
        changed = context.copy()
        changed.loc[changed.date.ge(cutoff), 'observed_percent'] = 100. - changed.loc[
            changed.date.ge(cutoff), 'observed_percent']
        prefix = context.date.le(cutoff).to_numpy()
        for name, fit in fits.items():
            checked = predict_endpoint(fit['parameters'], changed, baseline, growth=fit['growth'])
            np.testing.assert_array_equal(values[name][prefix], checked[prefix])
        parts.setdefault(label, []).append(part)
        train_table = train[['global_target_index', 'field_id', 'latitude', 'longitude', 'date',
                             'leaf_index', 'observed_percent']].copy()
        train_table['inner_fold'] = inner_fold
        train_table['coordinate_excluded_baseline'] = oof
        train_table['original_training_weight'] = weights
        train_table['native_growth_logit'] = native_growth(train)
        train_table.to_csv(out / (split['fold'] + '_training.csv'), index=False)
        provenance.append(dict(fold=split['fold'], train_ids=split['train'], test_ids=split['test'],
            inner_memberships=memberships, last_training_year=int(train.season_year.max()),
            first_test_year=int(obs.iloc[vi].season_year.min()),
            training_prediction='Inner coordinate-held-out backbone; six/seven conditional parameters jointly estimated on historical targets',
            source_model=str(prior_path.relative_to(ROOT)), source_sha256=sha(prior_path)))
        progress.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            train_rows=len(train), test_rows=len(vi),
            fits={name: {key: (value.tolist() if isinstance(value, np.ndarray) else value)
                        for key, value in fit.items() if key != 'trace'} for name, fit in fits.items()},
            split_rmse={name: score_predictions(part, name)['rmse'] for name in [REFERENCE, *CANDIDATES]}))
        (out / 'fit_progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        (out / 'training_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print('finished', split['fold'], 'seconds', round(time.monotonic()-started, 1),
              'rmse', {name: round(progress[-1]['split_rmse'][name], 4) for name in [REFERENCE, *CANDIDATES]}, flush=True)
    results = {label: pd.concat(pieces, ignore_index=True) for label, pieces in parts.items()}
    all_predictions = []
    for label, frame in results.items():
        frame['original_hierarchy_weight'] = hierarchical_weights(frame)
        frame.to_csv(out / ('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
        all_predictions.append(frame)
    pd.concat(all_predictions, ignore_index=True).to_csv(out / 'diagnostic_predictions.csv', index=False)
    metrics, groups, decisions = _write_scores(results, out)
    verification = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(), fitted_splits=len(progress),
        candidates_fitted=len(CANDIDATES), complete_assessment_support={label: len(frame) for label, frame in results.items()},
        saved_model_replay=True, baseline_prediction_reproduced_exactly=True,
        whole_query_day_and_future_mutation_invariant=True, historical_baseline_coordinate_exclusions_verified=True,
        no_satellite_or_new_cultivar=True, active_pointer_changed=False, climate_model_changed=False,
        source_unchanged=all(sha(ROOT / name) == digest for name, digest in registration['source_sha256'].items()),
        code_unchanged=all(sha(ROOT / name) == digest for name, digest in registration['code_sha256'].items()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert verification['source_unchanged'] and verification['code_unchanged']
    (out / 'replay_receipt.json').write_text(json.dumps(verification, indent=2) + '\n')
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    lines = ['Endpoint development comparison, 2026-10-23', '',
        'All 20 original folds and 2017 rows per design retained. Adaptive internal development; no independent external validation.',
        'Retained baseline and frozen climate sources unchanged; active/publication pointers unchanged.', '',
        metrics.to_string(index=False), '',
        groups.loc[groups.population.isin(['low_le10', 'severe_above75', 'first_visit', 'last_visit'])].to_string(index=False), '',
        json.dumps(decisions, indent=2), '',
        'Training provenance, fold models, exact replay receipt, original weights and whole-day visit contexts are in ' + str(out),
        'Forecast-bin calibration diagnoses forecast calibration; outcome-defined bins describe conditional errors and cannot establish miscalibration.']
    Path('/tmp/gcb_endpoint_improvement_summary_20261023.md').write_text('\n'.join(lines) + '\n')
    print(metrics.to_string(index=False), flush=True)
    return metrics, decisions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
