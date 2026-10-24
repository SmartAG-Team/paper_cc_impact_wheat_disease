"""Sequential assessment of historical curve particles on the frozen cohort."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .curve_particle_model import fit_curve_bank, predict_curve_bank
from .discrepancy_experiment import forecast_baseline, CACHE
from .innovation_common import ROOT, DATA, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY = 'curve_particle_same'
NAMES = [PRIMARY, 'curve_particle_field']
FREE, CONTROL = 'curve_particle_free', 'curve_particle_unfiltered'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/curve_particles_20261022').resolve()
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in [
        'curve_particle_model', 'discrepancy_experiment', 'discrepancy_filter', 'crossfit_router',
        'field_history_models', 'models', 'robust_models', 'history_interval_hybrid']]
    registration = register(out, PRIMARY, dict(
        hypothesis='Historical field-level curve shapes constrain nonlinear growth between assessments',
        clock='Original day-minus-one accumulated positive thermal time, GDD/1000; station feature clock_positive_degree_days_1000',
        curve='100*sigmoid(a+b*(clock-2)); a in [-25,25], b in [0,50]; weighted fractional least squares, max100 evaluations',
        regularization='Gaussian residual penalties .05*(a-prior_a)/6 and .05*(b-prior_b)/10; source/native-code pooled then individual-field curves',
        missing_code='Latent proposals borrow source/code pooled curve plus observed-code mean intercept shift and median slope ratio clipped .25 to4; not new observations',
        particles='Five thermal phase shifts [-.2,-.1,0,.1,.2] times three slope multipliers [.5,1,2], equal within historical field',
        prior='Original hierarchical field mass times 300km Gaussian geographical kernel; same source only; nominal native-code support',
        conditioning='12pp Gaussian proxy, equal mass per assessment day split among codes, max4 day evidence units, 1% prior refresh; strictly earlier field days only',
        variants='same: observed same-code anchor plus expected curve increment; field: additionally absolute posterior when only other codes observed; first visit unchanged',
        diagnostic='Free curve prior with no held-field labels; unknown source/code fallback to training-only weighted mean',
        limitations='Monotone curves cannot explain observed decreases; prior clock inherits uncertain sowing; native codes are not verified anatomy; regularized likelihood is heuristic',
        selection='Spatial iff either fixed monitored variant beats original temporal reference; free/control cannot advance',
        adaptive_extension=True), codes)
    for path in [DATA/'feature_schema.json', CACHE/'registration.json', *sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    schema = json.loads((DATA/'feature_schema.json').read_text())
    clock_index = schema['station_feature_names'].index('clock_positive_degree_days_1000')
    clock = arrays['station'][:, clock_index]
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            assess({'temporal': frame}, PRIMARY, out, extra_models=(*NAMES[1:], FREE, CONTROL))
            if min(score_predictions(frame, n)['rmse'] for n in NAMES) >= score_predictions(frame, REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both monitored curve-particle variants fail temporal reference improvement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start', split['fold'], flush=True)
        prior = joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids'] == split['train'] and prior['test_ids'] == split['test']
        base = {k: prior['experts'][k] for k in ['process', 'first', 'event', 'training_ids']}
        bank = fit_curve_bank(clock[ti], train, hierarchical_weights(train))
        fitted = dict(base=base, bank=bank, train_ids=split['train'], test_ids=split['test'], clock_index=clock_index)
        path = out/f'{split["fold"]}_model.joblib'
        joblib.dump(fitted, path)
        ci = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qi = pd.Series(np.arange(len(ci)), index=ci).loc[vi].to_numpy()

        def forecast(bundle):
            baseline = forecast_baseline(bundle['base'], arrays, history, obs, ci)
            values = predict_curve_bank(bundle['bank'], clock[ci], obs.iloc[ci], baseline)
            values['unfiltered'] = baseline
            return {k: v[qi] for k, v in values.items()}

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key], replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name, key in zip([*NAMES, FREE, CONTROL], ['same', 'field', 'free', 'unfiltered']):
            part[name] = values[key]
        np.testing.assert_allclose(part[CONTROL], part[REFERENCE], rtol=0, atol=1e-11)
        part['curve_particle_ess'] = values['ess']
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            bank_fields={key: len(value['weight']) for key, value in bank['banks'].items()},
            training_rows=len(train), training_ids=split['train']))
        (out/'fit_progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print('finished', split['fold'], 'seconds', round(time.monotonic()-started, 1),
            'rmse', {n: round(score_predictions(part, n)['rmse'], 3) for n in NAMES}, flush=True)
    results = {label: pd.concat(p, ignore_index=True) for label, p in parts.items()}
    for label, frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
    metrics, _ = assess(results, PRIMARY, out, extra_models=(*NAMES[1:], FREE, CONTROL))
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
