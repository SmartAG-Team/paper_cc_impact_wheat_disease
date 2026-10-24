"""One fixed severe-priority training-risk extension; original evaluation mass.

Training outcomes above 75% receive exactly twice their original hierarchy
mass. Prediction has no outcome-bin correction, and every assessment is scored
with the original scientific weights. The fixed multiplier is not tuned.
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
from threadpoolctl import threadpool_limits

from . import endpoint_improvement as original

NAME = 'endpoint_severe_priority2'
PREVIOUS = original.ROOT / 'outputs/gcb_endpoint_improvement_20261023'


def training_risk_weights(training, scientific_weights):
    y = training.observed_percent.to_numpy(float)
    weight = np.asarray(scientific_weights, float)
    if weight.shape != y.shape or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError('Aligned finite positive scientific weights required')
    if not np.isfinite(y).all() or np.any((y < 0) | (y > 100)):
        raise ValueError('Finite bounded historical training targets required')
    return weight * np.where(y > 75., 2., 1.)


def fit_severe_priority(training, baseline, scientific_weights, initial_kernel):
    risk = training_risk_weights(training, scientific_weights)
    fitted = original.fit_endpoint(training, baseline, risk, initial_kernel, growth=False)
    fitted.update(fitting='Hierarchy-weighted prequential severity MSE with fixed 2x historical severe training risk',
        severe_threshold_exclusive=75., severe_training_multiplier=2., evaluation_weight_multiplier=1.)
    return fitted


def _score(frames, out):
    metrics, groups, intervals, bins = [], [], [], []
    for design, frame in frames.items():
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        weights = original.hierarchical_weights(frame)
        masks = dict(low_le10=frame.observed_percent.le(10), severe_above75=frame.observed_percent.gt(75),
            first_visit=frame.first_visit, last_visit=frame.last_visit, only_visit=frame.only_visit,
            earlier_field=~frame.first_visit, BASF=frame.source.eq('BASF'), Corteva=frame.source.eq('Corteva'),
            first_visit_severe=frame.first_visit & frame.observed_percent.gt(75),
            last_visit_severe=frame.last_visit & frame.observed_percent.gt(75))
        for name in [original.REFERENCE, NAME]:
            metrics.append(dict(evaluation=design, model=name, **original.score_predictions(frame, name)))
            for population, mask in masks.items():
                if mask.any():
                    groups.append(dict(evaluation=design, model=name, population=population,
                                       **original.score_subset(frame, mask, name)))
            for lo, hi in [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100.0000001)]:
                mask = frame[name].ge(lo) & frame[name].lt(hi)
                if mask.any():
                    bins.append(dict(evaluation=design, model=name, lower=lo, upper=min(hi, 100),
                        n=int(mask.sum()), original_weight_mass=float(weights[mask].sum()),
                        forecast_mean=float(np.average(frame.loc[mask, name], weights=weights[mask])),
                        observed_mean=float(np.average(frame.loc[mask, 'observed_percent'], weights=weights[mask]))))
        lower, upper = original.paired_interval(frame, NAME, original.REFERENCE)
        intervals.append(dict(evaluation=design, model=NAME, reference=original.REFERENCE, lower=lower, upper=upper,
                              interpretation='Paired site bootstrap within adaptively reused development cohort'))
    metrics, groups = pd.DataFrame(metrics), pd.DataFrame(groups)
    metrics.to_csv(out / 'metrics.csv', index=False)
    groups.to_csv(out / 'subgroup_metrics.csv', index=False)
    pd.DataFrame(bins).to_csv(out / 'forecast_bin_calibration.csv', index=False)
    pd.DataFrame(intervals).to_csv(out / 'paired_intervals.csv', index=False)
    gates = {}
    for design in frames:
        full = metrics.loc[metrics.evaluation.eq(design)].set_index('model')
        subgroup = groups.loc[groups.evaluation.eq(design)].set_index(['model', 'population'])
        gates[design + '_overall_rmse_improves'] = bool(full.loc[NAME, 'rmse'] < full.loc[original.REFERENCE, 'rmse'])
        gates[design + '_integral_rmse_improves'] = bool(full.loc[NAME, 'integral_rmse'] < full.loc[original.REFERENCE, 'integral_rmse'])
        for population in ['low_le10', 'severe_above75', 'first_visit', 'last_visit']:
            gates[design + '_' + population + '_rmse_improves'] = bool(
                subgroup.loc[(NAME, population), 'rmse'] < subgroup.loc[(original.REFERENCE, population), 'rmse'])
        gates[design + '_severe_absolute_bias_improves'] = bool(
            abs(subgroup.loc[(NAME, 'severe_above75'), 'bias']) < abs(subgroup.loc[(original.REFERENCE, 'severe_above75'), 'bias']))
    core = [gates[design + suffix] for design in frames for suffix in [
        '_overall_rmse_improves', '_severe_above75_rmse_improves', '_severe_absolute_bias_improves']]
    decision = dict(model=NAME, core_endpoint_criteria_pass=all(core), all_endpoint_gates_pass=all(gates.values()),
        gates=gates, independent_external_validation=False, publication_promoted=False,
        training_risk=dict(severe_threshold_exclusive=75, multiplier=2, fixed_without_sweep=True),
        evaluation='Original hierarchy weights and complete original support')
    (out / 'candidate_decision.json').write_text(json.dumps(decision, indent=2) + '\n')
    # Empirical comparison describes only tested candidates, not a statistical
    # lower bound on errors attainable with these data or another architecture.
    comparison = pd.concat([pd.read_csv(PREVIOUS / 'metrics.csv'), metrics.loc[metrics.model.eq(NAME)]], ignore_index=True)
    comparison.to_csv(out / 'development_comparison.csv', index=False)
    subgroup_comparison = pd.concat([pd.read_csv(PREVIOUS / 'subgroup_metrics.csv'),
        groups.loc[groups.model.eq(NAME)]], ignore_index=True)
    subgroup_comparison.to_csv(out / 'development_subgroup_comparison.csv', index=False)
    return metrics, groups, decision


def run(output=None):
    out = Path(output or original.ROOT / 'outputs/gcb_endpoint_severe_priority_20261023').resolve()
    registration = original.register(out, NAME, dict(candidate_limit=1, candidate=NAME,
        hypothesis='Fixed 2x weighting of historical severe training targets may counter endpoint shrinkage while preserving acceptable complete-cohort skill',
        inherited_architecture='Joint mean logit GP, six parameters; identical bounds, ridge, L-BFGS-B60 and historical coordinate-held-out baseline forecasts',
        training_risk='Original hierarchy mass multiplied by exactly2 iff historical training severity>75; global normalization inside fitting only',
        evaluation='All2017 assessment rows per design, original unmodified hierarchy weights; no outcome-based record removal or evaluation reweighting',
        no_parameter_sweep=True, added_prediction_inputs=False,
        query='Strictly earlier own-field observations, whole query day across all codes withheld, weather ends q-1, no other current-season field observations',
        inner='Saved all-year coordinate exclusions apply to historical backbone; conditional parameters fitted jointly on historical outer-training targets',
        outer='All original20 chronological/chronological+50km splits, no temporal screen',
        adaptive_internal_comparison=True, independent_external_validation=False,
        baseline_and_climate_unchanged=True, publication_pointer_unchanged=True),
        [Path(__file__), Path(original.__file__)])
    previous_registration = json.loads((PREVIOUS / 'registration.json').read_text())
    for key in ['source_sha256', 'code_sha256']:
        registration['source_sha256'].update(previous_registration[key])
    for path in [PREVIOUS / 'registration.json', PREVIOUS / 'verification.json',
                 PREVIOUS / 'metrics.csv', PREVIOUS / 'subgroup_metrics.csv', PREVIOUS / 'diagnostic_predictions.csv']:
        registration['source_sha256'][str(path.relative_to(original.ROOT))] = original.sha(path)
    assert all(original.sha(original.ROOT / p) == h for p, h in registration['source_sha256'].items())
    (out / 'registration.json').write_text(json.dumps(registration, indent=2) + '\n')
    obs, arrays, history, _, splits, position = original.load_cohort()
    parents = {design: pd.read_csv(PREVIOUS / name, float_precision='round_trip').set_index('global_target_index')
        for design, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    parts, progress, provenance = {}, [], []
    started = time.monotonic()
    for split in splits:
        print('start', split['fold'], flush=True)
        design = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        ti, vi = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        training = obs.iloc[ti].reset_index(drop=True)
        assert training.season_year.max() < obs.iloc[vi].season_year.min()
        prior_path = original.BASELINE / (split['fold'] + '_model.joblib')
        prior = joblib.load(prior_path)
        assert prior['train_ids'] == split['train'] and prior['test_ids'] == split['test']
        held = []
        for inner in prior['inner_memberships']:
            a, b = [position.loc[inner[key]].to_numpy() for key in ['train', 'test']]
            assert set(zip(obs.iloc[a].latitude, obs.iloc[a].longitude)).isdisjoint(
                set(zip(obs.iloc[b].latitude, obs.iloc[b].longitude)))
            assert set(inner['train']) | set(inner['test']) == set(split['train'])
            held.extend(inner['test'])
        assert sorted(held) == sorted(split['train'])
        weight = original.hierarchical_weights(training)
        risk = training_risk_weights(training, weight)
        fit = fit_severe_priority(training, prior['inner_predictions'], weight, prior['corrections']['logit']['parameters'])
        fitted = dict(filter=fit, baseline_path=str(prior_path), baseline_sha256=original.sha(prior_path),
            train_ids=split['train'], test_ids=split['test'], historical_baseline=prior['inner_predictions'],
            original_training_weight=weight, training_risk_weight=risk,
            inner_memberships=prior['inner_memberships'], evaluation_weight_multiplier=1.)
        model_path = out / (split['fold'] + '_model.joblib')
        joblib.dump(fitted, model_path)
        ci = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        assert set(obs.iloc[ci].global_target_index).isdisjoint(split['train'])
        qi = pd.Series(np.arange(len(ci)), index=ci).loc[vi].to_numpy()
        unfiltered = original.forecast_baseline(prior['base'], arrays, history, obs, ci)
        baseline = original.predict_discrepancy(prior['corrections']['logit']['parameters'], obs.iloc[ci], unfiltered, 'logit')
        values = original.predict_endpoint(fit['parameters'], obs.iloc[ci], unfiltered, growth=False)
        replay = joblib.load(model_path)
        assert original.sha(Path(replay['baseline_path'])) == replay['baseline_sha256']
        np.testing.assert_array_equal(values, original.predict_endpoint(replay['filter']['parameters'], obs.iloc[ci], unfiltered, growth=False))
        np.testing.assert_array_equal(baseline[qi], parents[design].loc[split['test'], original.REFERENCE])
        part = parents[design].loc[split['test']].reset_index().copy()
        part[NAME] = values[qi]
        # Whole query day and all later outcomes can be altered without any
        # change in predictions on that day or the earlier complete prefix.
        context = obs.iloc[ci].copy()
        cutoff = sorted(context.date.unique())[len(context.date.unique()) // 2]
        changed = context.copy()
        mask = changed.date.ge(cutoff)
        changed.loc[mask, 'observed_percent'] = 100 - changed.loc[mask, 'observed_percent']
        prefix = context.date.le(cutoff).to_numpy()
        np.testing.assert_array_equal(values[prefix], original.predict_endpoint(
            fit['parameters'], changed, unfiltered, growth=False)[prefix])
        parts.setdefault(design, []).append(part)
        ledger = training[['global_target_index', 'field_id', 'date', 'leaf_index', 'observed_percent']].copy()
        ledger['coordinate_excluded_baseline'] = prior['inner_predictions']
        ledger['original_hierarchy_weight'], ledger['training_risk_weight'] = weight, risk
        ledger.to_csv(out / (split['fold'] + '_training.csv'), index=False)
        severe = training.observed_percent.gt(75).to_numpy()
        provenance.append(dict(fold=split['fold'], train_ids=split['train'], test_ids=split['test'],
            inner_memberships=prior['inner_memberships'], original_severe_weight_fraction=float(weight[severe].sum()/weight.sum()),
            risk_severe_weight_fraction=float(risk[severe].sum()/risk.sum()), last_training_year=int(training.season_year.max()),
            first_test_year=int(obs.iloc[vi].season_year.min()), fixed_severe_multiplier=2.))
        progress.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            fit={k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in fit.items() if k != 'trace'},
            baseline_rmse=original.score_predictions(part, original.REFERENCE)['rmse'],
            candidate_rmse=original.score_predictions(part, NAME)['rmse']))
        (out / 'fit_progress.json').write_text(json.dumps(progress, indent=2) + '\n')
        (out / 'training_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
        print('finished', split['fold'], round(time.monotonic()-started, 1),
              'rmse', round(progress[-1]['candidate_rmse'], 5), flush=True)
    frames = {design: pd.concat(pieces, ignore_index=True) for design, pieces in parts.items()}
    for design, frame in frames.items():
        np.testing.assert_array_equal(frame.original_hierarchy_weight, original.hierarchical_weights(frame))
        frame.to_csv(out / ('assessment_predictions.csv' if design == 'temporal' else 'spatial_predictions.csv'), index=False)
    pd.concat(frames.values(), ignore_index=True).to_csv(out / 'diagnostic_predictions.csv', index=False)
    metrics, groups, decision = _score(frames, out)
    verification = dict(passed=True, completed_utc=datetime.now(timezone.utc).isoformat(), fitted_splits=len(progress),
        candidates_fitted=1, all_fits_converged=all(x['fit']['converged'] for x in progress), saved_model_replay=True,
        baseline_reproduced_exactly=True, complete_support={d: len(f) for d, f in frames.items()},
        original_evaluation_weights_exact=True, whole_query_day_future_mutation_invariant=True,
        unchanged_prior_experiment=True, climate_sources_unchanged=True, active_pointer_changed=False,
        source_unchanged=all(original.sha(original.ROOT / p) == h for p, h in registration['source_sha256'].items()),
        code_unchanged=all(original.sha(original.ROOT / p) == h for p, h in registration['code_sha256'].items()),
        artifacts_sha256={p.name: original.sha(p) for p in out.iterdir() if p.is_file()})
    assert verification['source_unchanged'] and verification['code_unchanged']
    (out / 'replay_receipt.json').write_text(json.dumps(verification, indent=2) + '\n')
    (out / 'verification.json').write_text(json.dumps(verification, indent=2) + '\n')
    lines = ['Fixed severe-priority training-risk comparison', '', metrics.to_string(index=False), '',
        groups.loc[groups.population.isin(['low_le10', 'severe_above75', 'first_visit', 'last_visit'])].to_string(index=False),
        '', json.dumps(decision, indent=2), '',
        'The tested development frontier does not identify a theoretical accuracy ceiling. All comparisons reuse the same development cohort.',
        'Retained baseline, prior experiment receipts, original evaluation weights and frozen climate sources remain unchanged.']
    Path('/tmp/gcb_endpoint_severe_priority_summary_20261023.md').write_text('\n'.join(lines)+'\n')
    print(metrics.to_string(index=False), flush=True)
    return metrics, decision


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
