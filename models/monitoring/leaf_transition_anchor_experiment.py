"""Adaptive controlled ablation: updated leaf states with the measured anchor."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval, score_predictions, score_subset
from .inputs import sha
from .leaf_state import STATE_NAMES
from .leaf_transition_anchor import fit_ensemble, predict_ensemble
from .leaf_transition_experiment import DATA, PARENT, ROOT
from .leaf_transition_model import route_transition
from .rf_sowing_finalize import assessment_populations

NAME = 'leaf_state_anchor'
STATE_RUN = ROOT/'outputs/disease_improvement/leaf_transition_20261022'


def assess(frames, history, position, out):
    metrics, groups, intervals = [], [], []
    for label, frame in frames.items():
        h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        masks = assessment_populations(frame, h)
        for name in [NAME, 'irish_station_candidate', 'leaf_state_transition']:
            metrics.append(dict(evaluation=label, model=name, **score_predictions(frame, name)))
            if name != NAME:
                lo, hi = paired_interval(frame, NAME, name)
                intervals.append(dict(evaluation=label, model=NAME, reference=name, lower=lo, upper=hi))
            for pop, mask in masks.items():
                groups.append(dict(evaluation=label, population=pop, model=name, **score_subset(frame, mask, name)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(groups).to_csv(out/'subgroup_metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    guards = {}
    for label, frame in frames.items():
        new, old = [score_predictions(frame, n) for n in [NAME, 'irish_station_candidate']]
        guards[label+'_rmse_noninferiority'] = new['rmse'] <= old['rmse']
        guards[label+'_integral_noninferiority'] = new['integral_rmse'] <= old['integral_rmse']
    decision = dict(incremental_parent_guards=guards, balanced_incremental_improvement=all(guards.values()),
        adaptive_development=True, independent_external_validation=False,
        external_validation_required_for_internal_experiment=False, publication_model_promoted=False)
    (out/'assessment_decision.json').write_text(json.dumps(decision, indent=2)+'\n')
    return pd.DataFrame(metrics)


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/leaf_transition_anchor_20261022').resolve()
    if out.exists():
        raise FileExistsError('Fresh anchor-ablation destination required')
    obs = pd.read_csv(PARENT/'training_observation_ledger.csv')
    arrays = np.load(PARENT/'predictors.npz')
    state = np.load(STATE_RUN/'state_predictors.npz')
    np.testing.assert_array_equal(state['target_ids'], obs.global_target_index)
    z = state['states']; assert z.shape == (2304, 13)
    parent = joblib.load(PARENT/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=parent['base']['history_names'])
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    splits = json.loads((PARENT/'split_membership.json').read_text())
    sources = [PARENT/n for n in ['training_observation_ledger.csv', 'predictors.npz',
        'split_membership.json', 'assessment_predictions.csv', 'spatial_predictions.csv', 'fitted_candidate.joblib']]
    sources += [STATE_RUN/n for n in ['registration.json', 'state_predictors.npz', 'state_provenance.json',
        'state_schema.json', 'fitted_candidate_lock.json', 'verification.json', 'assessment_predictions.csv', 'spatial_predictions.csv']]
    codes = [Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['leaf_transition_anchor',
        'leaf_transition_model', 'leaf_state', 'leaf_transition_experiment', 'models', 'field_history_models',
        'history_interval_hybrid', 'robust_models', 'evaluation', 'rf_sowing_finalize', 'infection_event_experiment']]
    codes += [ROOT/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    reg = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='State information is useful while direct propagation is a poor residual baseline; retain the earlier measured-severity anchor',
        adaptive_followup_after_primary_results=True, primary_results_inspected=True,
        recipe='Parent first-field/process retained; event68+state13 input to original field_split head; existing measured-severity anchor and final 50:50',
        ensemble=dict(members=20, fraction=.8, seed=20261022), fixed_original_estimator_settings=True,
        state_names=STATE_NAMES, no_hyperparameter_search=True,
        parent_tables_parser='round_trip; preserve original saved parent prediction decimals',
        current_dated_untreated_dataset=True, independent_external_validation=False,
        external_validation_required_for_internal_experiment=False, publication_model_promoted=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes})
    (out/'registration.json').write_text(json.dumps(reg, indent=2)+'\n')
    for path in codes:
        dst = out/'code_snapshot'/path.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True); dst.write_bytes(path.read_bytes())
    for name in ['training_observation_ledger.csv', 'predictors.npz', 'split_membership.json']:
        (out/name).write_bytes((PARENT/name).read_bytes())
    for name in ['state_predictors.npz', 'state_schema.json', 'state_provenance.json']:
        (out/name).write_bytes((STATE_RUN/name).read_bytes())
    frames = {}; primary = {}
    for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frames[label] = pd.read_csv(PARENT/name, float_precision='round_trip').set_index('global_target_index')
        primary[label] = pd.read_csv(STATE_RUN/name, float_precision='round_trip').set_index('global_target_index')
    parts = dict(temporal=[], spatial_50km=[])
    for split in splits:
        train, test = [position.loc[split[k]].to_numpy() for k in ['train', 'test']]
        assert obs.iloc[train].season_year.max() < obs.iloc[test].season_year.min()
        fitted = fit_ensemble(arrays['history_events'][train], history.iloc[train], z[train], obs.iloc[train])
        joblib.dump(fitted, out/(split['fold']+'_transition.joblib'))
        new = predict_ensemble(fitted, arrays['history_events'][test], history.iloc[test], z[test], obs.iloc[test].drop(columns='observed_percent'))
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        part = frames[label].loc[split['test']].reset_index().copy()
        part['leaf_state_transition'] = primary[label].loc[split['test'], 'leaf_state_transition'].to_numpy()
        part['anchor_history_component'] = new
        part[NAME], part['uses_leaf_transition'] = route_transition(part.irish_station_candidate,
            part.candidate_process, new, history.iloc[test])
        parts[label].append(part)
        print('fitted anchor', split['fold'], flush=True)
    results = {label:pd.concat(records, ignore_index=True) for label, records in parts.items()}
    for label, name in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame = results[label]
        assert len(frame) == 2017 and frame.global_target_index.is_unique and (~frame.uses_leaf_transition).sum() == 494
        frame.to_csv(out/name, index=False)
    metrics = assess(results, history, position, out)
    transition = fit_ensemble(arrays['history_events'], history, z, obs)
    full = dict(parent=parent, transition=transition, minimum_assessment_year=2020,
        training_ids=obs.global_target_index.astype(int).tolist())
    joblib.dump(full, out/'fitted_candidate.joblib')
    contract = dict(created_utc=datetime.now(timezone.utc).isoformat(), recipe=NAME,
        inference='analysis.disease_improvement.leaf_transition_anchor.predict', minimum_assessment_year=2020,
        widths=dict(process=37, station=53, history_events=68, history=35, states=13),
        adaptive_development=True, independent_external_validation=False, publication_model_promoted=False,
        sha256={name:sha(out/name) for name in ['registration.json', 'fitted_candidate.joblib',
            'training_observation_ledger.csv', 'split_membership.json', 'state_predictors.npz',
            'predictors.npz', 'state_schema.json', 'state_provenance.json']})
    (out/'fitted_candidate_lock.json').write_text(json.dumps(contract, indent=2)+'\n')
    print(metrics.to_string(index=False), flush=True)
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.output)
