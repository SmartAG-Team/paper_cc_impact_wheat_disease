"""Joint latent epidemic-phase curves on the original disease assessment support."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .joint_phase_model import fit_phase, predict_phase
from .discrepancy_experiment import forecast_baseline, CACHE
from .innovation_common import ROOT, DATA, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY = 'joint_phase_posterior'
NAMES = [PRIMARY, 'joint_phase_anchor']
FREE, CONTROL = 'joint_phase_free', 'joint_phase_unfiltered'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/joint_phase_20261022').resolve()
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in [
        'joint_phase_model', 'discrepancy_experiment', 'discrepancy_filter', 'crossfit_router',
        'field_history_models', 'models', 'robust_models', 'history_interval_hybrid']]
    registration = register(out, PRIMARY, dict(
        hypothesis='Jointly pooled category curves with uncertain field phase and growth can generalize with fewer shared parameters than high-dimensional learners',
        predictors='Only18 retained pure weather/geographical/calendar features: station indices0to11,43,44,45,47,48,49; clock46 separate; observation-derived features excluded',
        mean='100sigmoid(categoryintercept+positivecategoryslope*(clock-2)*fieldgrowth+sourceweatherlinear+fieldpressure)',
        latent_prior='Seven Gaussian pressure nodes times growthfactors .6,1,1/.6 with probabilities.25,.5,.25; learnedsourcepressuresdexpclipped[-2.3,1.4]',
        fitting='Joint field-marginal endpoint-compatible censoredGaussian likelihoodproxy sd15pp;400Adamsteps lr.02 gradientclip5; weightsoriginalfieldmass and maxfourdayunits of normalized originalwithinfield observationweights',
        regularization='Weathermean-square.02; categoryinterceptshrink.0002; logslopeshrinkto8 .002; logpressuresdshrinktozero .002',
        observation_interpretation='Endpoint likelihood is a statistical proxy, not verified measurement censoring protocol',
        variants='Strictpastfieldposteriorcurve mean or same-codeobservedanchor+posteriorcurveincrement; nohistorymonitoringbaselineexact; unsupportedsource/code baselineexact',
        query_updates='Maxfour equallyweighted earlierday likelihoodunits,1%priorrefresh; allsame-daycodeswithheldtogether',
        diagnostic='Freepriorcurve withnoheld-field labels; unsupportedcategorytrainingweightedmean; diagnosticcannotadvancespatial',
        limitations='Latentpressure/growth are unverified proxies, not measuredsowing/cultivar/anatomy; weatherlinearity is a modelingrestriction',
        selection='Spatial iff eitherfixedmonitoringvariant beatsoriginaltemporalreference', adaptive_extension=True), codes)
    for path in [DATA/'feature_schema.json', CACHE/'registration.json', *sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    schema = json.loads((DATA/'feature_schema.json').read_text())
    names = schema['station_feature_names']
    indices = list(range(12))+[43, 44, 45, 47, 48, 49]
    assert all(names[k].startswith(('weather_', 'clock_')) for k in indices)
    assert names[46] == 'clock_positive_degree_days_1000'
    x, clock = arrays['station'][:, indices], arrays['station'][:, 46]
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            assess({'temporal': frame}, PRIMARY, out, extra_models=(*NAMES[1:], FREE, CONTROL))
            if min(score_predictions(frame, n)['rmse'] for n in NAMES) >= score_predictions(frame, REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both joint latent-phase monitoring variants fail temporal reference improvement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start', split['fold'], flush=True)
        prior = joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids'] == split['train'] and prior['test_ids'] == split['test']
        base = {k: prior['experts'][k] for k in ['process', 'first', 'event', 'training_ids']}
        phase = fit_phase(x[ti], clock[ti], train, hierarchical_weights(train))
        fitted = dict(base=base, phase=phase, train_ids=split['train'], test_ids=split['test'],
            feature_names=[names[k] for k in indices], feature_indices=indices)
        path = out/f'{split["fold"]}_model.joblib'
        joblib.dump(fitted, path)
        ci = np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qi = pd.Series(np.arange(len(ci)), index=ci).loc[vi].to_numpy()

        def forecast(bundle):
            baseline = forecast_baseline(bundle['base'], arrays, history, obs, ci)
            values = predict_phase(bundle['phase'], x[ci], clock[ci], obs.iloc[ci], baseline)
            values['unfiltered'] = baseline
            return {key: value[qi] for key, value in values.items()}

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key], replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name, key in zip([*NAMES, FREE, CONTROL], ['posterior', 'anchor', 'free', 'unfiltered']):
            part[name] = values[key]
        np.testing.assert_allclose(part[CONTROL], part[REFERENCE], rtol=0, atol=1e-11)
        part['joint_phase_ess'] = values['ess']
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started, trace=phase['trace']))
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
