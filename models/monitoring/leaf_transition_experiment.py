"""Registered fixed-recipe leaf-state transition development experiment."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from model.seasonal_septoria.core import Parameters
from .evaluation import paired_interval, promotion_decision, score_predictions, score_subset
from .inputs import sha
from .leaf_state import STATE_NAMES, build_state_features
from .leaf_transition_model import fit_ensemble, predict_ensemble, route_transition
from .rf_sowing_finalize import assessment_populations

ROOT = Path(__file__).resolve().parents[2]
NAME = 'leaf_state_transition'
PARENT = ROOT/'data/final_monitoring/cohort'
DATA = ROOT/'data/disease_improvement/station_preferred_ie_20261022'


def assess(frames, history, position, output):
    metrics, groups, intervals = [], [], []
    for evaluation, frame in frames.items():
        h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        masks = assessment_populations(frame, h)
        names = [NAME, 'irish_station_candidate', 'reference_replay']
        if evaluation == 'temporal':
            names += ['current']
        for name in names:
            metrics.append(dict(evaluation=evaluation, model=name, **score_predictions(frame, name)))
            if name != NAME:
                lo, hi = paired_interval(frame, NAME, name)
                intervals.append(dict(evaluation=evaluation, model=NAME, reference=name, lower=lo, upper=hi))
            for label, mask in masks.items():
                groups.append(dict(evaluation=evaluation, population=label, model=name,
                    **score_subset(frame, mask, name)))
    pd.DataFrame(metrics).to_csv(output/'metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(output/'subgroup_metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(output/'paired_intervals.csv', index=False)
    temporal = frames['temporal']
    h = history.iloc[position.loc[temporal.global_target_index]].reset_index(drop=True)
    masks = assessment_populations(temporal, h); decisions = {}
    for reference in ['irish_station_candidate', 'current', 'reference_replay']:
        delta = {key: score_subset(temporal, masks[key], NAME)['rmse']
            -score_subset(temporal, masks[key], reference)['rmse']
            for key in ['cold', 'warm', 'BASF', 'Corteva']}
        severe = abs(score_subset(temporal, masks['severe_above75'], NAME)['bias'])
        severe -= abs(score_subset(temporal, masks['severe_above75'], reference)['bias'])
        ratio = score_subset(temporal, masks['new_site'], NAME)['rmse']
        ratio /= score_subset(temporal, masks['new_site'], reference)['rmse']
        decision = promotion_decision(score_predictions(temporal, reference),
            score_predictions(temporal, NAME), paired_interval(temporal, NAME, reference),
            independent=False, subgroup_deltas=delta, severe_bias_delta=severe, new_site_ratio=ratio)
        guards = {k: v for k, v in decision['gates'].items() if k != 'independent_validation'}
        decisions[reference] = dict(internal_guards=guards, passes_current_cohort_guards=all(guards.values()),
            subgroup_rmse_deltas=delta, severe_absolute_bias_change=severe, new_site_rmse_ratio=ratio)
    incremental = {}
    for evaluation, frame in frames.items():
        new, old = [score_predictions(frame, name) for name in [NAME, 'irish_station_candidate']]
        incremental[evaluation+'_rmse_noninferiority'] = new['rmse'] <= old['rmse']
        incremental[evaluation+'_integral_noninferiority'] = new['integral_rmse'] <= old['integral_rmse']
    (output/'assessment_decision.json').write_text(json.dumps(dict(references=decisions,
        incremental_parent_guards=incremental, balanced_incremental_improvement=all(incremental.values()),
        evaluation_scope='Current dated untreated cohort; retrospective rolling field history',
        external_validation_required_for_internal_experiment=False,
        independent_external_validation=False, publication_model_promoted=False), indent=2)+'\n')
    return pd.DataFrame(metrics)


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/leaf_transition_20261022').resolve()
    if out.exists():
        raise FileExistsError('Fresh leaf-transition output directory required')
    obs = pd.read_csv(PARENT/'training_observation_ledger.csv')
    pd.testing.assert_frame_equal(obs, pd.read_csv(DATA/'training_observation_ledger.csv'))
    arrays = np.load(PARENT/'predictors.npz')
    for key in arrays.files:
        np.testing.assert_array_equal(arrays[key], np.load(DATA/'predictors.npz')[key])
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    parent = joblib.load(PARENT/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=parent['base']['history_names'])
    q = obs.drop(columns='observed_percent')
    fields = pd.read_csv(DATA/'fields.csv')
    weather = pd.read_parquet(DATA/'daily_weather.parquet')
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    sources = [PARENT/n for n in ['training_observation_ledger.csv', 'predictors.npz',
        'split_membership.json', 'assessment_predictions.csv', 'spatial_predictions.csv',
        'fitted_candidate.joblib', 'fitted_candidate_lock.json', 'verification.json']]
    sources += [DATA/n for n in ['manifest.json', 'READY.json', 'fields.csv', 'daily_weather.parquet', 'feature_schema.json']]
    sources += [ROOT/'data/disease_improvement/rf_sowing_20261022/leaf_scoring_convention.csv']
    codes = [Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['leaf_state',
        'leaf_transition_model', 'models', 'robust_models', 'field_history_models',
        'history_interval_hybrid', 'history', 'evaluation', 'rf_sowing_finalize', 'infection_event_experiment']]
    codes += [ROOT/'analysis/final_study/metrics.py', ROOT/'model/seasonal_septoria/core.py']
    out.mkdir(parents=True)
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='A native-leaf daily infection-state proxy updated from earlier visible severity improves history-informed progression',
        recipe='Retained parent first-field and process predictions; new state-baseline residual history ensemble; final 50:50 on history rows',
        parameters=asdict(Parameters()), time_step_days=.25, leaf_codes=list(range(7)),
        assimilation='Proportionally rescale visible V/I/R to observed percentage and nonvisible S/E1/E2/E3 to its complement; zero-mass blocks fall back to V or S',
        state_interpretation='Unvalidated latent-state proxy; all native leaf cohorts continuously active; no observed emergence or cultivar resistance',
        predictors='Retained event68 + unchanged history35 + state13; squared-error HGB fixed original settings',
        estimator=dict(max_iter=200, max_leaf_nodes=15, max_depth=3, min_samples_leaf=10,
            learning_rate=.05, l2_regularization=1., early_stopping=False, random_state=20261021),
        ensemble=dict(members=20, fraction=.8, seed=20261022),
        weather='Qualified station-preferred daily temperature/RH/liquid-rain estimate; fallback and phase provenance retained; previous-day cutoff',
        sowing='Retained calendar scenario; RF estimates deliberately not combined with this independent hypothesis',
        disease_history='Strictly earlier same-field observations, including earlier other-leaf observations; all same-day observations withheld',
        disease_training_observations=2304, expected_evaluation_rows_per_design=2017,
        outer_splits_preserved=True, first_field_predictions_preserved=True,
        hyperparameter_search=False, posthoc_development=True,
        user_authorized_current_dataset_evaluation=True, external_validation_required_for_internal_experiment=False,
        independent_external_validation=False, publication_model_promoted=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        environment={n:version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'joblib']})
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in codes:
        dest = out/'code_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); dest.write_bytes(path.read_bytes())
    for name in ['training_observation_ledger.csv', 'predictors.npz', 'split_membership.json']:
        (out/name).write_bytes((PARENT/name).read_bytes())
    states, names, ledger = build_state_features(weather, fields, obs, q)
    np.savez_compressed(out/'state_predictors.npz', states=states, target_ids=obs.global_target_index)
    (out/'state_schema.json').write_text(json.dumps(dict(names=names, columns=len(names),
        visible_baseline_columns=[4, 5, 6], interpretation=registration['state_interpretation']), indent=2)+'\n')
    ledger.to_json(out/'state_provenance.json', orient='records', indent=2)
    np.testing.assert_array_equal(states[:, names.index('state_same_assimilated')], history.prior_assessment_available)
    any_history = history[['prior_assessment_available', 'other_leaf_available']].to_numpy().any(axis=1)
    np.testing.assert_array_equal(states[:, names.index('state_any_assimilated')], any_history.astype(float))
    print('prepared states', states.shape, 'history records', int(any_history.sum()), flush=True)
    splits = json.loads((PARENT/'split_membership.json').read_text())
    frames = {label: pd.read_csv(PARENT/name).set_index('global_target_index') for label, name in
        [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    parts = dict(temporal=[], spatial_50km=[])
    for split in splits:
        train, test = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        assert obs.iloc[train].season_year.max() < obs.iloc[test].season_year.min()
        bundle = fit_ensemble(arrays['history_events'][train], history.iloc[train], states[train], obs.iloc[train])
        joblib.dump(bundle, out/(split['fold']+'_transition.joblib'))
        new = predict_ensemble(bundle, arrays['history_events'][test], history.iloc[test], states[test], q.iloc[test])
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        part = frames[label].loc[split['test']].reset_index().copy()
        part['transition_history_component'] = new
        part[NAME], part['uses_leaf_transition'] = route_transition(part.irish_station_candidate,
            part.candidate_process, new, history.iloc[test])
        parts[label].append(part)
        print('fitted', split['fold'], flush=True)
    results = {label: pd.concat(records, ignore_index=True) for label, records in parts.items()}
    for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame = results[label]
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        assert (~frame.uses_leaf_transition).sum() == 494
        frame.to_csv(out/name, index=False)
    metrics = assess(results, history, position, out)
    transition = fit_ensemble(arrays['history_events'], history, states, obs)
    fitted = dict(parent=parent, transition=transition,
        training_ids=obs.global_target_index.astype(int).tolist(), minimum_assessment_year=2020)
    joblib.dump(fitted, out/'fitted_candidate.joblib')
    contract = dict(created_utc=datetime.now(timezone.utc).isoformat(), recipe=NAME,
        inference='analysis.disease_improvement.leaf_transition_model.predict', minimum_assessment_year=2020,
        widths=dict(process=37, station=53, history_events=68, history=35, states=13),
        state_builder='analysis.disease_improvement.leaf_state.build_state_features',
        strict_current_query_label_and_previous_day_weather_cutoff=True,
        state_interpretation=registration['state_interpretation'],
        user_authorized_current_dataset_evaluation=True, independent_external_validation=False,
        publication_model_promoted=False, sha256={name:sha(out/name) for name in ['registration.json',
            'predictors.npz', 'state_predictors.npz', 'state_schema.json', 'state_provenance.json',
            'training_observation_ledger.csv', 'split_membership.json', 'fitted_candidate.joblib']})
    (out/'fitted_candidate_lock.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(metrics.to_string(index=False), flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
