"""Replay calibration and shared-process experiments and qualify scenarios."""
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from model.seasonal_septoria.overwinter import OverwinterParameters
from .autonomous_monitoring import coordinate_oob
from .autonomous_weather import OUTPUTS, WEATHER
from .autonomous_weather_data import AUTONOMOUS, PHENO, EXTENSION
from .audit_climate_calibration import deployment_decision
from .evaluation import score_predictions, score_subset
from .innovation_common import ROOT, DATA, load_cohort, goal_decision
from .inputs import sha
from .process_establishment import predict_observation, simulate_establishment
from .process_establishment_data import PROCESS_DATA
from .process_establishment_experiment import NAMES, predict_bank
from .process_scenario import simulate_queries
from .severity_calibration import predict_calibration

CALIBRATION = ROOT/'outputs/disease_improvement/severity_calibration_support_repair_20261022'
PROCESS = ROOT/'outputs/disease_improvement/process_establishment_20261022'
PRIOR = ROOT/'outputs/disease_improvement/autonomous_weather_20261022'


def _authenticate(out):
    receipt = json.loads((out/'verification.json').read_text()); assert receipt['passed']
    for name, digest in receipt['artifacts_sha256'].items():
        assert sha(out/name) == digest, (out, name)
    registration = json.loads((out/'registration.json').read_text())
    for key in ['source_sha256', 'code_sha256']:
        for path, digest in registration[key].items():
            assert sha(ROOT/path) == digest, path
            if key == 'code_sha256':
                snapshot = out/('code_snapshots' if (out/'code_snapshots').exists() else 'code_snapshot')/path
                assert sha(snapshot) == digest


def _metrics(out, frames, names, reference='climate_reference'):
    recorded = pd.read_csv(out/'metrics.csv', float_precision='round_trip')
    groups = pd.read_csv(out/'subgroup_metrics.csv', float_precision='round_trip')
    for design, frame in frames.items():
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        for name in names:
            actual = score_predictions(frame, name)
            row = recorded.loc[recorded.evaluation.eq(design)&recorded.model.eq(name)].iloc[0]
            for metric, value in actual.items():
                value = np.nan if value is None else value
                np.testing.assert_allclose(row[metric], value, rtol=1e-10, atol=1e-9, equal_nan=True)
            severe = score_subset(frame, frame.observed_percent.gt(75), name)
            row = groups.loc[groups.evaluation.eq(design)&groups.model.eq(name)&groups.population.eq('severe_above75')].iloc[0]
            for metric, value in severe.items():
                value = np.nan if value is None else value
                np.testing.assert_allclose(row[metric], value, rtol=1e-10, atol=1e-9, equal_nan=True)
    severe = groups.loc[groups.population.eq('severe_above75')]
    for filename, ref in [('all_goal_decisions.json', 'irish_station_candidate'), ('autonomous_goal_decisions.json', reference)]:
        stored = json.loads((out/filename).read_text())
        for name in names:
            assert stored[name] == goal_decision(recorded, severe, name, reference=ref)


def verify():
    out = ROOT/'outputs/disease_improvement/calibration_process_verification_20261022'
    if out.exists():
        raise FileExistsError('Fresh verification destination required')
    for parent in [CALIBRATION, PROCESS, PROCESS_DATA]: _authenticate(parent)
    obs, _, _, original, splits, position = load_cohort()
    a = dict(np.load(AUTONOMOUS/'predictors.npz')); bank = dict(np.load(PROCESS_DATA/'query_states.npz'))
    forcing = dict(np.load(PROCESS_DATA/'forcing.npz'))
    states = bank['damage']; np.testing.assert_array_equal(bank['target_ids'], obs.global_target_index)
    args = [forcing[k] for k in ['temperature', 'exposure', 'rain', 'active', 'renewal', 'area', 'valid']]
    for index, values in enumerate(bank['physics']):
        params = replace(OverwinterParameters(), primary_scale=values[0], secondary_scale=values[1], latent_reference_days=values[2])
        for k, threshold in enumerate(bank['introduction_thresholds']):
            trajectory = simulate_establishment(*args, params, threshold)
            damage = trajectory[forcing['field_index'], forcing['day_index'], :, -3:].sum(-1)
            np.testing.assert_array_equal(damage, states[index, :, k])
    frames = {parent: {design: pd.read_csv(parent/name, float_precision='round_trip')
        for design, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
        for parent in [CALIBRATION, PROCESS, PRIOR]}
    no_support, optimizer_failures = 0, 0
    for split in splits:
        design = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        ti, qi = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        training = obs.iloc[ti]; weight = hierarchical_weights(training)
        assert training.season_year.max() < obs.iloc[qi].season_year.min()
        assert not set(training.field_id)&set(obs.iloc[qi].field_id)
        prior = joblib.load(PRIOR/f'{split["fold"]}_model.joblib')
        fitted = joblib.load(CALIBRATION/f'{split["fold"]}_model.joblib')
        oob = coordinate_oob(prior, a['baseline'][ti], a['augmented'][ti], a['phase'][ti], training)
        np.testing.assert_array_equal(oob['count'], fitted['oob']['count'])
        for name in OUTPUTS:
            np.testing.assert_array_equal(oob['predictions'][name], fitted['oob']['predictions'][name])
        for parent in [CALIBRATION, PROCESS]:
            bundle = joblib.load(parent/f'{split["fold"]}_model.joblib')
            assert bundle['train_ids'] == split['train'] and bundle['test_ids'] == split['test']
            np.testing.assert_array_equal(bundle['weights'], weight)
        part = frames[CALIBRATION][design].set_index('global_target_index').loc[split['test']]
        parent = frames[PRIOR][design].set_index('global_target_index').loc[split['test']]
        for name, fit in fitted['calibrators'].items():
            covered = np.flatnonzero(oob['count'] > 0)
            if not len(covered):
                assert fit['fit_status'] == 'no_honest_calibration_support'; no_support += 1
                assert len(fit['calibration_rows']) == 0
            else:
                np.testing.assert_array_equal(fit['calibration_rows'], covered)
                np.testing.assert_allclose(fit['normalized_weight'], weight[covered]/weight[covered].sum(), atol=1e-15)
            prefix = name.split('_')[1]
            source = dict(baseline=OUTPUTS[0], shared=OUTPUTS[1], phase=OUTPUTS[2])[prefix]
            np.testing.assert_allclose(predict_calibration(fit, parent[source].to_numpy()), part[name], atol=1e-12, rtol=1e-14)
        bundle = joblib.load(PROCESS/f'{split["fold"]}_model.joblib')
        for (mixture, index), fit in bundle['all_fits'].items():
            pred = predict_observation(fit, states[index, ti], training[['source', 'leaf_index']])/100.
            error = float(np.sum((weight/weight.sum())*(pred-training.observed_percent.to_numpy()/100.)**2))
            np.testing.assert_allclose(error, fit['training_mse'], atol=1e-15)
            assert np.isfinite(fit['objective']); optimizer_failures += int(not fit['optimizer_success'])
        for name, values in predict_bank(bundle, states[:, qi], obs.iloc[qi][['source', 'leaf_index']]).items():
            actual = frames[PROCESS][design].set_index('global_target_index').loc[split['test'], name]
            np.testing.assert_allclose(values, actual, atol=1e-12, rtol=1e-14)
        print('replayed', split['fold'], flush=True)
    calibration_names = list(json.loads((CALIBRATION/'all_goal_decisions.json').read_text()))
    _metrics(CALIBRATION, frames[CALIBRATION], calibration_names)
    _metrics(PROCESS, frames[PROCESS], NAMES)
    # Recalculate both crop development and disease under each perturbation.
    weather = pd.read_parquet(DATA/'daily_weather.parquet', columns=['location_id', 'date', *WEATHER])
    extra = pd.read_parquet(EXTENSION, columns=['location_id', 'date', *WEATHER])
    weather.date, extra.date = pd.to_datetime(weather.date), pd.to_datetime(extra.date)
    existing = pd.MultiIndex.from_frame(weather[['location_id', 'date']])
    extra = extra.loc[~pd.MultiIndex.from_frame(extra[['location_id', 'date']]).isin(existing)]
    weather = pd.concat([weather, extra], ignore_index=True)
    fields = pd.read_csv(AUTONOMOUS/'field_sowing_scenarios.csv')
    prior = joblib.load(PHENO/'development_prior.joblib')
    params = OverwinterParameters(primary_scale=.1, secondary_scale=10., latent_reference_days=20.)
    scenarios = []
    for name, warming, rainfall, susceptibility in [('baseline', 0, 1, 1), ('warming2', 2, 1, 1),
            ('rainfall80', 0, .8, 1), ('rainfall120', 0, 1.2, 1), ('susceptibility25', 0, 1, .25), ('susceptibility200', 0, 1, 2)]:
        w = weather.copy(); w['tmean_c'] += warming; w['tmax_c'] += warming; w['precipitation_mm'] *= rainfall
        result = simulate_queries(w, fields, obs, prior, params, susceptibility)
        if name == 'baseline': np.testing.assert_array_equal(result['damage'], states[25])
        scenarios.append(dict(scenario=name, warming_c=warming, rainfall_multiplier=rainfall,
            susceptibility_multiplier=susceptibility, largest_mass_error=result['largest_mass_error'],
            smallest_state=result['smallest_state'], mean_latent_damage=float(result['damage'].mean()),
            actual_variety_rating=False, numerical_qualification_only=True))
    original_goals = json.loads((PROCESS/'all_goal_decisions.json').read_text())
    auto_goals = json.loads((PROCESS/'autonomous_goal_decisions.json').read_text())
    deployment = deployment_decision(dict(autonomous_input_contract=True, no_satellite_ancestry=True,
        fold_fit_excludes_query_labels=True, comparable_endpoint_mapping=False, magnitude_validated=False,
        original_goal_passed=any(d['achieved'] for d in original_goals.values()),
        autonomous_goal_passed=any(d['achieved'] for d in auto_goals.values()),
        scenario_sowing_inputs_ready=False, climate_perturbations_pass=True))
    out.mkdir(parents=True)
    pd.DataFrame(scenarios).to_csv(out/'scenario_numerics.csv', index=False)
    result = dict(passed=True, outer_folds=20, calibrator_states=120, unsupported_identity_maps=no_support,
        process_fits=1080, process_optimizer_failures=optimizer_failures, physical_bank_replay='bitexact81trajectories',
        complete_metric_goal_replay=True, saved_prediction_replay=True, original_memberships_weights=True,
        OOB_group_exclusion_recomputed=True, scenario_numerics_only=True, actual_variety_effect_estimated=False,
        deployment=deployment, source_sha256={str(p.relative_to(ROOT)): sha(p) for p in [
            CALIBRATION/'verification.json', PROCESS/'verification.json', PROCESS_DATA/'verification.json',
            Path(__file__), Path(__file__).with_name('process_scenario.py'), Path(__file__).with_name('variety_resistance.py')]},
        artifact_sha256={'scenario_numerics.csv': sha(out/'scenario_numerics.csv')})
    (out/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)
    return out


if __name__ == '__main__':
    with threadpool_limits(limits=1): verify()
