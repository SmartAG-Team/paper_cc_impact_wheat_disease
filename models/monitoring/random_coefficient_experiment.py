"""Assess nonstationary source-native random-coefficient forecast-error filters."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .random_coefficient_filter import fit_coefficients, predict_coefficients
from .discrepancy_experiment import forecast_baseline, CACHE
from .innovation_common import ROOT, DATA, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY = 'random_coefficient_linear'
NAMES = [PRIMARY, 'random_coefficient_logit']
CONTROL = 'random_coefficient_unfiltered'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/random_coefficients_20261022').resolve()
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in [
        'random_coefficient_filter', 'discrepancy_experiment', 'discrepancy_filter', 'crossfit_router',
        'field_history_models', 'models', 'robust_models', 'history_interval_hybrid']]
    registration = register(out, PRIMARY, dict(
        hypothesis='Native-code specific field offsets and thermal-time slopes co-vary, enabling nonstationary correction between assessment dates',
        basis='Seven nominal code intercepts and seven code-specific slopes against original GDD/1000minus2; no anatomical ordering',
        fitting='Separate source-specific dense14x14 covariance;30 fixed regularized EM-like iterations on honest baseline errors',
        initial='Intercept/slope sd1/2 forlogit,.2/.5 forlinear; noise.7/.1; covariance.8moment+.2diagmoment+.01initial; noise floor.5/.04',
        weights='Original scientific fieldmass; within each field original normalized observationweights with maxfour assessmentday evidenceunits',
        inference='Strict earlier fielddays only, equalmass perassessmentday split amongcodes,maxfourunits; linear posterior mean clipped, logit posterior predictive mean via9pointGaussHermite',
        baseline='Original cached process/history reference; all first visits and unsupported source/nativecodes unchanged',
        controls='Unfiltered exact baseline; linear and logit fixed beforefit',
        limitations='Regularized empirical-Bayes covariance proxy, not measured biological coupling or unpenalized maximum likelihood; monitoring only',
        selection='Spatial iff either fixed formulation beats original temporal reference', adaptive_extension=True), codes)
    for path in [DATA/'feature_schema.json', CACHE/'registration.json', *sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    schema = json.loads((DATA/'feature_schema.json').read_text())
    clock = arrays['station'][:, schema['station_feature_names'].index('clock_positive_degree_days_1000')]
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            assess({'temporal': frame}, PRIMARY, out, extra_models=(*NAMES[1:], CONTROL))
            if min(score_predictions(frame, n)['rmse'] for n in NAMES) >= score_predictions(frame, REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both random-coefficient filters fail temporal reference improvement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start', split['fold'], flush=True)
        prior = joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids'] == split['train'] and prior['test_ids'] == split['test']
        oof = .5*(prior['inner_predictions'][:, 0]+prior['inner_predictions'][:, 1])
        base = {k: prior['experts'][k] for k in ['process', 'first', 'event', 'training_ids']}
        filters = {link: fit_coefficients(train, oof, clock[ti], hierarchical_weights(train), link)
            for link in ['linear', 'logit']}
        fitted = dict(base=base, filters=filters, train_ids=split['train'], test_ids=split['test'],
            inner_memberships=prior['inner_memberships'], training_baseline=oof)
        path = out/f'{split["fold"]}_model.joblib'
        joblib.dump(fitted, path)
        ci = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qi = pd.Series(np.arange(len(ci)), index=ci).loc[vi].to_numpy()

        def forecast(bundle):
            baseline = forecast_baseline(bundle['base'], arrays, history, obs, ci)
            values = {link: predict_coefficients(f, obs.iloc[ci], baseline, clock[ci])[qi]
                for link, f in bundle['filters'].items()}
            values['unfiltered'] = baseline[qi]
            return values

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key], replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name, key in zip([*NAMES, CONTROL], ['linear', 'logit', 'unfiltered']):
            part[name] = values[key]
        np.testing.assert_allclose(part[CONTROL], part[REFERENCE], rtol=0, atol=1e-11)
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            trace={link: {key: value['trace'] for key, value in f['sources'].items()} for link, f in filters.items()}))
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
