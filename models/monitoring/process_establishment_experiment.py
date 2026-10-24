"""Outer-fold disease-magnitude fitting of the label-free shared process bank."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .autonomous_weather import OUTPUTS
from .evaluation import score_subset
from .innovation_common import ROOT, load_cohort, register, assess, goal_decision
from .inputs import sha
from .process_establishment import fit_observation, predict_observation
from .process_establishment_data import PROCESS_DATA

PRIOR = ROOT/'outputs/disease_improvement/autonomous_weather_20261022'
NAMES = ['process_generic', 'process_selected', 'process_establishment']


def fit_bank(bank, physics, train, weights):
    bank, physics = np.asarray(bank), np.asarray(physics)
    if bank.shape != (len(physics), len(train), 3, 8) or physics.shape[1:] != (3,) or (physics <= 0).any():
        raise ValueError('Aligned process bank and positive physical parameters required')
    penalty = 1e-4*np.sum(np.log(physics/np.array([.01, 1., 20.]))**2, axis=1)
    generic = int(np.argmin(penalty)); losses, fits = [], {}
    for mixture in [False, True]:
        for index, states in enumerate(bank):
            fitted = fit_observation(states, train, weights, mixture)
            fits[(mixture, index)] = fitted
            losses.append(dict(mixture=mixture, physics_index=index, primary=float(physics[index, 0]),
                secondary=float(physics[index, 1]), latency=float(physics[index, 2]),
                training_mse=fitted['training_mse'], regularizer=fitted['regularizer'],
                physics_penalty=float(penalty[index]), objective=fitted['objective']+float(penalty[index]),
                optimizer_success=fitted['optimizer_success']))
    models = {'process_generic': dict(physics_index=generic, fit=fits[(False, generic)])}
    for name, mixture in [('process_selected', False), ('process_establishment', True)]:
        winner = min((row for row in losses if row['mixture'] == mixture), key=lambda row: row['objective'])
        index = winner['physics_index']
        models[name] = dict(physics_index=index, fit=fits[(mixture, index)])
    return dict(models=models, losses=losses, all_fits=fits, physics=physics)


def predict_bank(fitted, bank, queries):
    return {name: predict_observation(model['fit'], bank[model['physics_index']], queries)
        for name, model in fitted['models'].items()}


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/process_establishment_20261022').resolve()
    receipt = json.loads((PROCESS_DATA/'verification.json').read_text()); assert receipt['passed']
    for name, digest in receipt['artifacts_sha256'].items():
        assert sha(PROCESS_DATA/name) == digest
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in [
        'process_establishment', 'process_establishment_data', 'variety_resistance']]
    registration = register(out, NAMES[-1], dict(
        process='27genericprimary/secondary/latencygrid; fixedprocessbaseline; nominalstatisticalobservationoperators, BASF0fixedF1',
        selection='OutertrainingweightedfractionMSE+declaredoperator/arrivalshrinkage+1e-4squaredlogdistancetogeneric(.01,1,20)',
        establishment='Shared0/2/6wetdayintroductions+zeroescapedcomponent; fivealternatingupdates',
        cultivar='Neutral susceptibility1; noverifiedvarietyidentitiesorresistanceratingsinqualifiedcohort; explicitscenarioAPIavailablewithoutimputation',
        information='Fourweather/qminus1/RFestimatedsowing/disjointpre2013cropdevelopment; nohistory/observedBBCH/satellite',
        evaluation='All20originalsplits2017recordsperdesign; alltargetsretained; separateBASFflagdiagnostic',
        deployment='NoFigure1orclimatepromotion; assumedhost37/39,terminaldonor87; noyieldqualification'), codes)
    paths = [*PROCESS_DATA.glob('*.*'), PRIOR/'assessment_predictions.csv', PRIOR/'spatial_predictions.csv',
        PRIOR/'verification.json']
    for path in paths:
        registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, _, _, frames, splits, position = load_cohort()
    arrays = dict(np.load(PROCESS_DATA/'query_states.npz'))
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    prior = {label: pd.read_csv(PRIOR/name, float_precision='round_trip').set_index('global_target_index')
        for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    parts, progress, started = {}, [], time.monotonic()
    for split in splits:
        print('start', split['fold'], flush=True)
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        ti = position.loc[split['train']].to_numpy(); qi = position.loc[split['test']].to_numpy()
        train = obs.iloc[ti]; weight = hierarchical_weights(train)
        fitted = fit_bank(arrays['damage'][:, ti], arrays['physics'], train, weight)
        fitted.update(train_ids=split['train'], test_ids=split['test'], weights=weight,
            information_setting='autonomous', variety_resistance_used=False)
        path = out/f'{split["fold"]}_model.joblib'; joblib.dump(fitted, path)
        part = frames[label].loc[split['test']].reset_index().copy()
        queries = obs.iloc[qi][['source', 'leaf_index']]
        predicted = predict_bank(fitted, arrays['damage'][:, qi], queries)
        replay = predict_bank(joblib.load(path), arrays['damage'][:, qi], queries)
        for name in NAMES:
            np.testing.assert_array_equal(predicted[name], replay[name]); part[name] = predicted[name]
        for name in OUTPUTS:
            part[name] = prior[label].loc[split['test'], name].to_numpy()
        parts.setdefault(label, []).append(part)
        progress.append(dict(fold=split['fold'], seconds=time.monotonic()-started,
            selected={n: int(m['physics_index']) for n, m in fitted['models'].items()},
            failed_optimizers=sum(not row['optimizer_success'] for row in fitted['losses'])))
        (out/'fit_progress.json').write_text(json.dumps(progress, indent=2)+'\n')
        print('finished', split['fold'], round(time.monotonic()-started, 1), flush=True)
    results = {label: pd.concat(part, ignore_index=True) for label, part in parts.items()}
    for label, frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
    metrics, _ = assess(results, NAMES[-1], out, extra_models=[*NAMES[:-1], *OUTPUTS])
    groups = pd.read_csv(out/'subgroup_metrics.csv'); severe = groups.loc[groups.population.eq('severe_above75')]
    original = {n: goal_decision(metrics, severe, n) for n in NAMES}
    autonomous = {n: goal_decision(metrics, severe, n, reference='climate_reference') for n in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(original, indent=2)+'\n')
    (out/'autonomous_goal_decisions.json').write_text(json.dumps(autonomous, indent=2)+'\n')
    flags = []
    for label, frame in results.items():
        mask = frame.source.eq('BASF')&frame.leaf_index.eq(0)
        for name in [*NAMES, 'climate_reference']:
            flags.append(dict(evaluation=label, model=name, endpoint='Verified BASF flag F1 only; diagnostic', **score_subset(frame, mask, name)))
    pd.DataFrame(flags).to_csv(out/'qualified_flag_metrics.csv', index=False)
    verification = dict(passed=True, fitted_splits=len(progress), process_fits=20*54,
        saved_model_replay=True, information_setting='autonomous', publication_model_promoted=False,
        source_unchanged=all(sha(ROOT/p) == v for p, v in registration['source_sha256'].items()),
        code_unchanged=all(sha(ROOT/p) == v for p, v in registration['code_sha256'].items()),
        goal_achieved=any(v['achieved'] for v in original.values()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert verification['source_unchanged'] and verification['code_unchanged']
    (out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    return metrics, original, autonomous


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
