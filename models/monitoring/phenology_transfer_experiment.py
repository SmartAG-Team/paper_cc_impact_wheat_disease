"""Assess trait-supervised development features and phase-specific disease growth."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .phenology_phase_growth import fit_phase_growth, predict_phase_growth, phase_weights
from .discrepancy_experiment import forecast_baseline, CACHE
from .weather_sequence_model import design
from .innovation_common import ROOT, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

TRAIT_DATA = ROOT/'data/disease_improvement/phenology_transfer_20261022'
PRIMARY = 'phenology_phase_growth'
NAMES = [PRIMARY, 'phenology_shared_growth']
CONTROL = 'phenology_unfiltered'


def _category(frame):
    return frame.source.astype(str)+'|'+frame.leaf_index.astype(str)


def _anchor(priors, frame, measured, available):
    fallback = np.array([priors['means'].get(k, priors['pooled']) for k in _category(frame)])
    return 100*np.where(available, measured, fallback)


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/phenology_transfer_20261022').resolve()
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in [
        'phenology_phase_growth', 'phenology_transfer', 'phenology_transfer_data',
        'discrepancy_experiment', 'discrepancy_filter', 'crossfit_router', 'field_history_models',
        'models', 'robust_models', 'history_interval_hybrid', 'weather_sequence_model', 'dynamic_model']]
    trait_verification = json.loads((TRAIT_DATA/'verification.json').read_text())
    assert trait_verification['passed'] and trait_verification['source_unchanged']
    trait_registration = json.loads((TRAIT_DATA/'registration.json').read_text())
    for path, digest in trait_registration['source_sha256'].items():
        assert sha(Path(path)) == digest
    registration = register(out, PRIMARY, dict(
        hypothesis='Historically supervised crop development supplies more useful disease-growth structure than uncalibrated thermal age',
        trait_source='EJA-PGN-Wheat raw DWD cycles endingbefore2013;10181cycles/632stations, alldocumentedsourcecoordinates >50km fromEVERYcurrentdiseaselocation; completehistoricalrelocationsunknown',
        trait_labels='Observed emergence10,stemelongation31,heading51,yellowripeness87; sourceinternal85 maps87; noanatomicalleafmapping',
        trait_fit='Recompute thermal/photoperiod/vernalization clocks andrefitallgates/empiricalstageCDFs onlyonadmittedhistoricaltraits, equalstationmass perstage',
        predictors='145retained causalfeatures+59newtraitdevelopment/coverage features; processstatecolumns unchanged',
        architecture='SharedgrowthHGB orfourphaseweightedgrowth experts; gates1-P31,P31-P51,P51-P87,P87 fromTPV; .01gatefloor/renormalization',
        fitting='20members80%coordinateyear,200HGBiterations depth3/leaves15/minleaf10/L2=3/lr.05/noearlystop; originalscientificmass normalizedgloballybeforephaseweighting',
        anchor='Retained strictlyearlier measuredhistory orsource-native .04pooledshrunkprior; restorepercentagegrowthonce; inheritednumericnearest-codeanchor unverified',
        fallback='Noearlierfieldhistory OR source-nativeendpointabsentfromoutertraining preservesexactreference; noneofthequerytargetsaredropped',
        information='Alltrait/queryweatherenddayminus1, currentretainedestimatedsowingscenario; noqueryobservedBBCH or cultivar',
        selection='Spatial iff eitherfixedformulation beatsoriginaltemporalreference; unchanged10%andseveregates',
        interpretation='Trait-trainedbiologicalrepresentation, not independentdiseasevalidation or verifiedactualphenology/EPPOleafprotocol',
        adaptive_extension=True), codes)
    registration['source_sha256'].update(trait_registration['source_sha256'])
    for path in [*sorted(TRAIT_DATA.glob('*')), CACHE/'registration.json', *sorted(CACHE.glob('*_model.joblib'))]:
        if path.is_file():
            registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, arrays, history, frames, splits, position = load_cohort()
    with np.load(TRAIT_DATA/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'], obs.global_target_index)
        phenology, stage_probability = source['phenology'], source['stage_probability']
    x, measured, available = design(arrays, history, obs.drop(columns='observed_percent'))
    x = np.column_stack([x, phenology])
    phase = phase_weights(stage_probability[:, 2])
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km' and label not in parts:
            frame = pd.concat(parts['temporal'], ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv', index=False)
            assess({'temporal': frame}, PRIMARY, out, extra_models=(*NAMES[1:], CONTROL))
            if min(score_predictions(frame, n)['rmse'] for n in NAMES) >= score_predictions(frame, REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both trait-informed growth formulations fail temporal reference improvement'), indent=2)+'\n')
                break
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]
        assert train.season_year.max() < obs.iloc[vi].season_year.min()
        print('start', split['fold'], flush=True)
        cached = joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert cached['train_ids'] == split['train'] and cached['test_ids'] == split['test']
        base = {k: cached['experts'][k] for k in ['process', 'first', 'event', 'training_ids']}
        w = hierarchical_weights(train); mass = w/w.sum(); y = train.observed_percent.to_numpy(float)/100
        priors = dict(pooled=float(np.average(y, weights=mass)), means={})
        category = _category(train)
        for key in category.unique():
            mask = category.eq(key).to_numpy()
            priors['means'][key] = float((np.sum(mass[mask]*y[mask])+.04*priors['pooled'])/(mass[mask].sum()+.04))
        anchor = _anchor(priors, train, measured[ti], available[ti])
        growth = fit_phase_growth(x[ti], train, anchor, w, phase[ti])
        fitted = dict(base=base, growth=growth, priors=priors, supported_categories=category.unique().tolist(),
            train_ids=split['train'], test_ids=split['test'])
        path = out/f'{split["fold"]}_model.joblib'
        joblib.dump(fitted, path)

        def forecast(bundle):
            query = obs.iloc[vi].drop(columns='observed_percent')
            baseline = forecast_baseline(bundle['base'], arrays, history, obs, vi)
            anchor = _anchor(bundle['priors'], query, measured[vi], available[vi])
            supported = _category(query).isin(bundle['supported_categories']).to_numpy()
            use = available[vi] & supported
            values = predict_phase_growth(bundle['growth'], x[vi], anchor, phase[vi], baseline, use)
            values['unfiltered'], values['uses_growth'] = baseline, use
            return values

        values, replay = forecast(fitted), forecast(joblib.load(path))
        for key in values:
            np.testing.assert_array_equal(values[key], replay[key])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name, key in zip([*NAMES, CONTROL], ['stage', 'shared', 'unfiltered']):
            part[name] = values[key]
        part['phenology_uses_growth'] = values['uses_growth']
        np.testing.assert_allclose(part[CONTROL], part[REFERENCE], rtol=0, atol=1e-11)
        parts.setdefault(label, []).append(part)
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            members=len(growth['members']), models_per_member=5, corrected_rows=int(values['uses_growth'].sum())))
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
