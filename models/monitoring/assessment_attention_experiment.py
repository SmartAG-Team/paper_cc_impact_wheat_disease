"""Sequential evaluation of attention over earlier assessment records."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .assessment_attention import fit_attention, predict_attention
from .discrepancy_experiment import forecast_baseline, CACHE
from .weather_sequence_model import design
from .innovation_common import ROOT, DATA, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY = 'assessment_attention'
NAMES = [PRIMARY, 'assessment_attention_half']
CONTROL = 'assessment_attention_unfiltered'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/assessment_attention_20261022').resolve()
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in [
        'assessment_attention', 'discrepancy_experiment', 'discrepancy_filter', 'crossfit_router',
        'field_history_models', 'models', 'robust_models', 'history_interval_hybrid',
        'weather_sequence_model', 'dynamic_model']]
    registration = register(out, PRIMARY, dict(
        hypothesis='Query-dependent attention can select relevant individual earlier assessments beyond fixed history summaries',
        tokens='Up to32 most recent strictly earlier field records: source2/nativecode7 one-hot, severity, forecast, innovation, age/30, earlierclockminus2, clockgap, samecodeflag; learned null token',
        predictors='Original145 causal query features plus reference fraction; weighted scale floor.01 and clip8',
        architecture='32dim query encoder with layernorm/tanh; 32dim token encoder; fourhead crossattention; 32dim tanh residual head; dropout.1',
        fitting='Original hierarchy-weighted unclipped residual MSE; inference clips0to100. AdamW lr.003 wd.01 gradientclip1,300epochs, lr times.3 at60% and85%; three seeds20261022to24',
        initialization='Final output layer zero initialized; firstvisit residual multiplied by zero and exact baseline restored after ensemble',
        training_context='Honest inner-coordinate-heldout baseline and earlier forecast innovations from original cached five inner folds',
        query_context='Baseline fitted only on outer historical training; query field earlier actual labels may update tokens, same-day labels withheld together',
        formulations='Fixed full correction and half correction; no outer epoch/hyperparameter selection',
        interpretation='Monitoring model; nominal source-native code interactions, no verified anatomical mapping; learned memory weights are not causal effects',
        selection='Spatial iff either fixed formulation beats original temporal reference; unfiltered control cannot advance',
        adaptive_extension=True), codes)
    for path in [DATA/'feature_schema.json', CACHE/'registration.json', *sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    x, _, _ = design(arrays, history, obs.drop(columns='observed_percent'))
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
                    reason='Both learned attention formulations fail temporal reference improvement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start', split['fold'], flush=True)
        prior = joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids'] == split['train'] and prior['test_ids'] == split['test']
        oof = .5*(prior['inner_predictions'][:, 0]+prior['inner_predictions'][:, 1])
        base = {k: prior['experts'][k] for k in ['process', 'first', 'event', 'training_ids']}
        attention = fit_attention(x[ti], train, oof, clock[ti], hierarchical_weights(train))
        fitted = dict(base=base, attention=attention, train_ids=split['train'], test_ids=split['test'],
            inner_memberships=prior['inner_memberships'], training_baseline=oof)
        path = out/f'{split["fold"]}_model.joblib'
        joblib.dump(fitted, path)
        ci = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qi = pd.Series(np.arange(len(ci)), index=ci).loc[vi].to_numpy()

        def forecast(bundle):
            baseline = forecast_baseline(bundle['base'], arrays, history, obs, ci)
            values = {name: predict_attention(bundle['attention'], x[ci], obs.iloc[ci], baseline, clock[ci], factor)[qi]
                for name, factor in zip(NAMES, [1., .5])}
            values[CONTROL] = baseline[qi]
            return values

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key], replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name in [*NAMES, CONTROL]:
            part[name] = values[name]
        np.testing.assert_allclose(part[CONTROL], part[REFERENCE], rtol=0, atol=1e-11)
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            trace=[value['trace'] for value in attention['members']]))
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
