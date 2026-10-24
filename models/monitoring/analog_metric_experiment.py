"""Sequential test of supervised state-space analog retrieval."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .analog_metric_model import fit_analogs, predict_analogs
from .discrepancy_experiment import forecast_baseline, CACHE
from .weather_sequence_model import design
from .innovation_common import ROOT, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY = 'analog_metric_direct'
NAMES = [PRIMARY, 'analog_metric_residual']
CONTROL = 'analog_metric_unfiltered'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/analog_metric_20261022').resolve()
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in [
        'analog_metric_model', 'discrepancy_experiment', 'discrepancy_filter', 'crossfit_router',
        'field_history_models', 'models', 'robust_models', 'history_interval_hybrid',
        'weather_sequence_model', 'dynamic_model']]
    registration = register(out, PRIMARY, dict(
        hypothesis='A learned similarity measure can retrieve relevant historical disease states despite mixed meteorological and history dimensions',
        predictors='Original145 causal predictors plus baseline fraction; weighted training standardization, scale floor .01, clip8',
        metric='Shared normalized positive diagonal metric and Gaussian bandwidth; mass-weighted same-source/native-code kernel',
        training='300 Adam steps lr.04; coordinate-excluded neighbor prediction MSE with original scientific weights; .0005 mean squared metric logits and .0001 squared log bandwidth regularization',
        exclusions='Every observation at the query training coordinate, across fields and years, excluded from its neighbor library; hard same-source and native-code match',
        prior='Effective-neighbor coefficient ESS/(ESS+2) combines retrieved estimate with reference; unsupported endpoints retain reference',
        formulations='Direct percentage retrieval or retrieval of honest inner-coordinate-held-out baseline errors',
        baseline='Original cached process/history reference, original five inner partitions provide training-only baseline errors and baseline covariate',
        interpretation='Learned retrieval training objective is not independent validation; no claim of verified native-code anatomy',
        inference='Global metric frozen before outer query; strictly earlier observations already encoded in retained causal predictors; query labels never accessed',
        selection='Spatial iff either fixed formulation beats original temporal reference; unfiltered control cannot advance',
        adaptive_extension=True), codes)
    for path in [CACHE/'registration.json', *sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    x, _, _ = design(arrays, history, obs.drop(columns='observed_percent'))
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            assess({'temporal': frame}, PRIMARY, out, extra_models=(*NAMES[1:], CONTROL))
            if min(score_predictions(frame, n)['rmse'] for n in NAMES) >= score_predictions(frame, REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both learned-analog formulations fail temporal reference improvement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start', split['fold'], flush=True)
        prior = joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids'] == split['train'] and prior['test_ids'] == split['test']
        oof = .5*(prior['inner_predictions'][:, 0]+prior['inner_predictions'][:, 1])
        base = {k: prior['experts'][k] for k in ['process', 'first', 'event', 'training_ids']}
        analogs = {link: fit_analogs(x[ti], train, oof, hierarchical_weights(train), link)
            for link in ['direct', 'residual']}
        fitted = dict(base=base, analogs=analogs, train_ids=split['train'], test_ids=split['test'],
            inner_memberships=prior['inner_memberships'], training_baseline=oof)
        path = out/f'{split["fold"]}_model.joblib'
        joblib.dump(fitted, path)

        def forecast(bundle):
            baseline = forecast_baseline(bundle['base'], arrays, history, obs, vi)
            values = {link: predict_analogs(f, x[vi], obs.iloc[vi].drop(columns='observed_percent'), baseline)
                for link, f in bundle['analogs'].items()}
            values['unfiltered'] = baseline
            return values

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key], replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name, key in zip([*NAMES, CONTROL], ['direct', 'residual', 'unfiltered']):
            part[name] = values[key]
        np.testing.assert_allclose(part[CONTROL], part[REFERENCE], rtol=0, atol=1e-11)
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            trace={key: value['trace'] for key, value in analogs.items()}))
        (out/'fit_progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print('finished', split['fold'], 'seconds', round(time.monotonic()-started, 1),
            'rmse', {n: round(score_predictions(part, n)['rmse'], 3) for n in NAMES}, flush=True)
    results = {label: pd.concat(p, ignore_index=True) for label, p in parts.items()}
    for label, frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
    metrics, _ = assess(results, PRIMARY, out, extra_models=(*NAMES[1:], CONTROL))
    groups = pd.read_csv(out/'subgroup_metrics.csv')
    decisions = {n: goal_decision(metrics, groups.loc[groups.population.eq('severe_above75')], n) for n in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions, indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True, saved_model_replay=True,
        fitted_splits=len(records), source_unchanged=all(sha(ROOT/p) == h for p, h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}), indent=2)+'\n')
    return metrics, decisions


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
