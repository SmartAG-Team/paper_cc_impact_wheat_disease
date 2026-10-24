"""Matched autonomous climate-compatible control and renewal representation."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .autonomous_weather import fit_autonomous, predict_autonomous, OUTPUTS
from .autonomous_weather_data import AUTONOMOUS
from .innovation_common import ROOT, load_cohort, register, assess, goal_decision
from .evaluation import distances_km, paired_interval
from .inputs import sha


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/autonomous_weather_20261022').resolve()
    receipt = json.loads((AUTONOMOUS/'verification.json').read_text())
    assert receipt['passed'] and receipt['source_unchanged']
    for path, digest in receipt['artifacts_sha256'].items():
        assert sha(AUTONOMOUS/path) == digest
    codes = [Path(__file__)] + [Path(__file__).with_name(name+'.py') for name in [
        'autonomous_weather', 'autonomous_weather_data', 'phenology_phase_growth', 'phenology_transfer',
        'phenology_transfer_data', 'sowing_distribution_data']]
    codes += [ROOT/'model/seasonal_septoria/core.py', ROOT/'model/seasonal_septoria/wetness.py']
    registration = register(out, 'climate_renewal_phase', dict(
        comparison='Autonomous weather-only predictions; monitoring reference retained as a separately labeled information setting',
        runtime='Four weather variables only; no observed disease, BBCH, Tmin, subdaily weather or satellite ancestry',
        predictors='34causalweather/nominalendpoint/fixedgenericprocess; candidateadds59historicaldevelopment+36thermal-delaydosefeatures',
        model='20paired80%coordinateyearmembers; sharedHGBcontrol/sharedrenewal/fourphaseexperts;200iterations,depth3,leaves15,min10,lr.05,L2=3,noearlystop;phasefloor.01',
        target='Fullseveritypercentage,zeroobservationanchor; allscientificweights preserved beforephaseweighting',
        unsupported='Nominalendpointindicators withlearnedpooledweatherbehavior; nohistory-conditionedfallback',
        evaluation='All20originalsplits; nooutcome-basedearlyscreen,datadeletion,orrecipechange',
        monitoring_goal='Originalsixgatesunchanged; autonomouscomparisonreportedseparately',
        autonomous_comparison='DescriptivepairedRMSEintervals and10%+severegatesagainstfreshautonomouscontrol; notindependentvalidation',
        biological_interpretation='Assumeddelaybank10/20/30thermal-daysat18C, threegenerichostgates; noempiricalparameterdistribution or nativeleafanatomy',
        deployment='Nochange to hash-bound climate model; extra previousannualmeanT forcing andnative-to-anatomical endpointqualification required beforeintegration'), codes)
    data_registration = json.loads((AUTONOMOUS/'registration.json').read_text())
    registration['source_sha256'].update(data_registration['source_sha256'])
    for path in AUTONOMOUS.iterdir():
        if path.is_file():
            registration['source_sha256'][str(path.relative_to(ROOT))] = sha(path)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    obs, _, _, frames, splits, position = load_cohort()
    with np.load(AUTONOMOUS/'predictors.npz') as arrays:
        np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
        baseline, augmented, phase = [arrays[key] for key in ['baseline', 'augmented', 'phase']]
    parts, records, started = {}, [], time.monotonic()
    for split in splits:
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        ti, vi = position.loc[split['train']].to_numpy(), position.loc[split['test']].to_numpy()
        train, query = obs.iloc[ti], obs.iloc[vi]
        assert train.season_year.max() < query.season_year.min()
        distance = None
        if label == 'spatial_50km':
            distance = float(distances_km(train[['latitude', 'longitude']], query[['latitude', 'longitude']]).min())
            assert distance >= 50.
        weights = hierarchical_weights(train)
        print('start', split['fold'], 'train', len(ti), 'test', len(vi), flush=True)
        fitted = fit_autonomous(baseline[ti], augmented[ti], train, weights, phase[ti])
        fitted.update(train_ids=split['train'], test_ids=split['test'], information_setting='autonomous')
        path = out/f'{split["fold"]}_model.joblib'; joblib.dump(fitted, path)
        values = predict_autonomous(fitted, baseline[vi], augmented[vi], phase[vi])
        replay = predict_autonomous(joblib.load(path), baseline[vi], augmented[vi], phase[vi])
        for name in OUTPUTS:
            np.testing.assert_array_equal(values[name], replay[name])
        part = frames[label].loc[split['test']].reset_index().copy()
        for name in OUTPUTS:
            part[name] = values[name]
        parts.setdefault(label, []).append(part)
        # Verify fitted initialization from actual preserved member weights.
        initial_error = 0.
        for member, reference in zip(fitted['growth']['members'], fitted['baseline']):
            ix = position.loc[member['training_ids']].to_numpy()
            local = train.global_target_index.isin(member['training_ids']).to_numpy()
            mass = weights[local]/weights[local].mean()
            target = train.observed_percent.to_numpy(float)[local]
            gate = np.maximum(phase[ix], .01); gate /= gate.sum(axis=1, keepdims=True)
            for model, weight in [(reference, mass), (member['shared'], mass),
                    *[(head, mass*gate[:, k]) for k, head in enumerate(member['stage'])]]:
                initial_error = max(initial_error, abs(float(model._baseline_prediction.ravel()[0])-float(np.average(target, weights=weight))))
        assert initial_error < 1e-10
        records.append(dict(fold=split['fold'], elapsed_seconds=time.monotonic()-started,
            train=len(ti), test=len(vi), members=20, heads_per_member=6,
            spatial_minimum_km=distance, initialization_max_error=initial_error, saved_replay_exact=True))
        (out/'fit_progress.json').write_text(json.dumps(records, indent=2)+'\n')
        print('finished', split['fold'], 'seconds', round(time.monotonic()-started, 1), flush=True)
    results = {label: pd.concat(values, ignore_index=True) for label, values in parts.items()}
    for label, frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label == 'temporal' else 'spatial_predictions.csv'), index=False)
    metrics, _ = assess(results, OUTPUTS[-1], out, extra_models=OUTPUTS[:-1])
    groups = pd.read_csv(out/'subgroup_metrics.csv')
    severe = groups.loc[groups.population.eq('severe_above75')]
    monitoring = {name: goal_decision(metrics, severe, name) for name in OUTPUTS}
    (out/'all_goal_decisions.json').write_text(json.dumps(monitoring, indent=2)+'\n')
    comparisons = {}
    for name in OUTPUTS[1:]:
        decision = goal_decision(metrics, severe, name, reference=OUTPUTS[0])
        decision['interpretation'] = 'Autonomous adaptive internal comparison only; not the original monitoring goal'
        decision['paired_rmse_difference_intervals'] = {label: paired_interval(frame, name, OUTPUTS[0])
            for label, frame in results.items()}
        comparisons[name] = decision
    (out/'autonomous_comparisons.json').write_text(json.dumps(comparisons, indent=2)+'\n')
    verification = dict(passed=True, fitted_splits=len(records), new_heads=len(records)*20*6,
        history_conditioned=False, satellite_used=False, original_monitoring_goal_achieved=any(d['achieved'] for d in monitoring.values()),
        source_unchanged=all(sha(ROOT/p) == digest for p, digest in registration['source_sha256'].items()),
        code_unchanged=all(sha(ROOT/p) == digest for p, digest in registration['code_sha256'].items()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()})
    assert verification['source_unchanged'] and verification['code_unchanged']
    (out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    return metrics, monitoring, comparisons


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    with threadpool_limits(limits=1):
        run(parser.parse_args().output)
