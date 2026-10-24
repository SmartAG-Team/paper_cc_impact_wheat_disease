"""Sequential fixed-recipe ordered categorical boosting assessment."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .ordered_category_model import fit_ordered, predict_ordered
from .weather_sequence_model import design
from .innovation_common import ROOT, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY = 'ordered_category_direct'
NAMES = [PRIMARY, 'ordered_category_growth']


def _anchor(priors, frame, measured, available):
    categories = frame.source.astype(str)+'|'+frame.leaf_index.astype(str)
    fallback = np.array([priors['means'].get(k, priors['pooled']) for k in categories])
    return 100*np.where(available, measured, fallback)


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/ordered_category_20261022').resolve()
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in [
        'ordered_category_model', 'weather_sequence_model', 'dynamic_model', 'field_history_models']]
    registration = register(out, PRIMARY, dict(
        hypothesis='Ordered categorical learners can represent source/native-code and country differences beyond a common numerical tree design',
        predictors='Original145 causal features plus categorical source,nativecode,source-code,country; no field/site/trial identifiers',
        architecture='CatBoost1.2.10 CPU Ordered,has_timeTrue; trainingrows stablysortedbydate;500trees depth4 lr.04 L2=10 border32',
        ensemble='Three fixed seeds20261022to24, Bayesian rowbootstrap temperature1/random_strength1; complete original scientific sampleweights per member',
        targets='Directpercentage and observed-history anchored percentagegrowth; weighted source-native prior with .04pooledshrink for no earlier measured anchor',
        anchor='Retained field_history_anchor rule; nearest numeric native-code fallback remains an unvalidated inherited proxy, not verified anatomy',
        estimation='Original hierarchical weights passed directly tolibrary and normalizedonlybyglobalmean; noresamplingweightapproximation',
        chronology='Alloutertrainseasons precedetest; chronologicalorderedfittingisnotanadditionaltemporalvalidation',
        selection='Spatial iff either fixedformulation beatsoriginaltemporalreference; no outer earlystopping or hyperparameterselection',
        official_software_source='https://catboost.ai/docs/en/concepts/python-reference_catboostregressor_fit',
        adaptive_extension=True), codes)
    obs, arrays, history, frames, splits, position = load_cohort()
    x, measured, available = design(arrays, history, obs.drop(columns='observed_percent'))
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            assess({'temporal': frame}, PRIMARY, out, extra_models=NAMES[1:])
            if min(score_predictions(frame, n)['rmse'] for n in NAMES) >= score_predictions(frame, REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both ordered categorical formulations fail temporal reference improvement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start', split['fold'], flush=True)
        w = hierarchical_weights(train); mass = w/w.sum(); y = train.observed_percent.to_numpy(float)/100
        priors = dict(pooled=float(np.average(y, weights=mass)), means={})
        category = train.source.astype(str)+'|'+train.leaf_index.astype(str)
        for code in category.unique():
            mask = category.eq(code).to_numpy()
            priors['means'][code] = float((np.sum(mass[mask]*y[mask])+.04*priors['pooled'])/(mass[mask].sum()+.04))
        anchor = _anchor(priors, train, measured[ti], available[ti])
        models = {link: [fit_ordered(x[ti], train, anchor, w, link, seed=20261022+seed)
            for seed in range(3)] for link in ['direct', 'growth']}
        fitted = dict(models=models, priors=priors, train_ids=split['train'], test_ids=split['test'])
        path = out/f'{split["fold"]}_model.joblib'
        joblib.dump(fitted, path)

        def forecast(bundle):
            query = obs.iloc[vi].drop(columns='observed_percent')
            anchor = _anchor(bundle['priors'], query, measured[vi], available[vi])
            return {link: np.mean([predict_ordered(model, x[vi], query, anchor) for model in members], axis=0)
                for link, members in bundle['models'].items()}

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key], replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name, key in zip(NAMES, ['direct', 'growth']):
            part[name] = values[key]
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            trees_per_member={link: [m['model'].tree_count_ if m['model'] is not None else 0 for m in members]
                for link, members in models.items()}))
        (out/'fit_progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print('finished', split['fold'], 'seconds', round(time.monotonic()-started, 1),
            'rmse', {n: round(score_predictions(part, n)['rmse'], 3) for n in NAMES}, flush=True)
    results = {label: pd.concat(p, ignore_index=True) for label, p in parts.items()}
    for label, frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
    metrics, _ = assess(results, PRIMARY, out, extra_models=NAMES[1:])
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
