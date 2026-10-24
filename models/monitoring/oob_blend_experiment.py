"""Evaluate one training-only, history-availability-specific blend calibration."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval, promotion_decision, score_predictions, score_subset
from .infection_event_experiment import populations
from .inputs import sha
from .oob_history_blend import apply_weights, fit_weights, oob_predictions

ROOT = Path(__file__).resolve().parents[2]
NAME = 'oob_history_blend'


def score_frames(frames, history, position, out):
    metrics, intervals, groups = [], [], []
    for label, frame in frames.items():
        for name in [NAME, 'history_interval', 'reference_replay']+(['current'] if label == 'temporal' else []):
            metrics.append(dict(evaluation=label, model=name, **score_predictions(frame, name)))
            if name != NAME:
                low, high = paired_interval(frame, NAME, name)
                intervals.append(dict(evaluation=label, model=NAME, reference=name, lower=low, upper=high))
        if label == 'temporal':
            h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
            masks = populations(frame, h)
            for population, mask in masks.items():
                for name in [NAME, 'history_interval', 'reference_replay']:
                    groups.append(dict(population=population, model=name, **score_subset(frame, mask, name)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    pd.DataFrame(groups).to_csv(out/'subgroup_metrics.csv', index=False)
    frame = frames['temporal']
    decisions = {}
    for reference in ['history_interval', 'current', 'reference_replay']:
        deltas = {key:score_subset(frame, masks[key], NAME)['rmse']-score_subset(frame, masks[key], reference)['rmse']
                  for key in ['cold', 'warm', 'BASF', 'Corteva']}
        severe = abs(score_subset(frame, masks['severe_above75'], NAME)['bias'])-abs(score_subset(frame, masks['severe_above75'], reference)['bias'])
        ratio = score_subset(frame, masks['new_site'], NAME)['rmse']/score_subset(frame, masks['new_site'], reference)['rmse']
        decisions[reference] = promotion_decision(score_predictions(frame, reference), score_predictions(frame, NAME),
            paired_interval(frame, NAME, reference), independent=False, subgroup_deltas=deltas,
            severe_bias_delta=severe, new_site_ratio=ratio)
        decisions[reference].update(subgroup_rmse_deltas=deltas, severe_absolute_bias_change=severe, new_site_rmse_ratio=ratio)
    (out/'adoption_decision.json').write_text(json.dumps(dict(references=decisions,
        independent_external_validation=False, publication_model_promoted=False), indent=2)+'\n')
    return pd.DataFrame(metrics)


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    base = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    events = root/'outputs/disease_improvement/infection_events_20261022'
    parent = root/'outputs/disease_improvement/history_interval_20261022'
    out = Path(output or root/'outputs/disease_improvement/oob_blend_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    obs = pd.read_csv(base/'training_observation_ledger.csv')
    arrays, event_arrays = np.load(base/'predictors.npz'), np.load(events/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    np.testing.assert_array_equal(event_arrays['target_ids'], obs.global_target_index)
    components = joblib.load(parent/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=components['base']['history_names'])
    designs = dict(process=arrays['process'], station=arrays['station'], event_station=event_arrays['history_events'])
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    splits = json.loads((base/'split_membership.json').read_text())
    sources = [base/n for n in ['training_observation_ledger.csv', 'predictors.npz', 'split_membership.json']]
    sources += [events/'predictors.npz']
    sources += [parent/n for n in ['assessment_predictions.csv', 'spatial_predictions.csv',
                                  'fitted_candidate.joblib', 'fitted_candidate_lock.json', 'verification.json']]
    for split in splits:
        sources += [base/f"{split['fold']}_{component}.joblib" for component in ['process', 'history']]
        sources.append(events/f"{split['fold']}_history.joblib")
    codes = [Path(__file__), Path(__file__).with_name('oob_history_blend.py')]
    codes += [Path(__file__).with_name(name+'.py') for name in ['models', 'robust_models', 'field_history_models',
              'history_interval_hybrid', 'infection_event_experiment', 'evaluation']]
    codes += [root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Fixed equal weights dilute complementary component skill across history-availability groups',
        groups=['first_field', 'same_leaf', 'other_leaf_only'],
        fitting='One convex history-component weight per group, using only whole-coordinate-year out-of-bag training predictions; no component refit',
        regularization='Ten equivalent coordinate-year groups centered on neutral50:50; penalty10*weighted_mean_squared_component_disagreement',
        minimum_oob_members=2, insufficient_oob='Omit only from weight calibration; retain every evaluation and training observation in the original components',
        changes='Final mixture only; station rainfall, predictors, history policy, component estimators, all original splits and all outcomes unchanged',
        posthoc_development=True, hyperparameter_search=False,
        query_outcomes_used_in_calibration=False, independent_external_validation=False,
        publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)):sha(p) for p in codes})
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in codes:
        dest = out/'code_snapshot'/path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    frames = {label:pd.read_csv(parent/filename).set_index('global_target_index')
              for label,filename in [('temporal','assessment_predictions.csv'), ('spatial_50km','spatial_predictions.csv')]}
    predictions = dict(temporal=[], spatial_50km=[])
    for split in splits:
        fold = split['fold']
        train, test = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        training = obs.iloc[train].reset_index(drop=True)
        htrain = history.iloc[train].reset_index(drop=True)
        held = obs.iloc[test]
        assert training.season_year.max() < held.season_year.min()
        assert set(training.global_target_index).isdisjoint(held.global_target_index)
        pmodel, hmodel = [joblib.load(base/f'{fold}_{key}.joblib') for key in ['process', 'history']]
        emodel = joblib.load(events/f'{fold}_history.joblib')
        p, h, counts = oob_predictions(pmodel, hmodel, emodel, {k:v[train] for k,v in designs.items()}, htrain, training)
        calibration = fit_weights(p, h, htrain, training, counts)
        (out/f'{fold}_calibration.json').write_text(json.dumps(calibration, indent=2)+'\n')
        training[['global_target_index', 'coordinate_year', 'observed_percent']].assign(
            process=p, history_component=h, oob_members=counts,
            prior_assessment_available=htrain.prior_assessment_available.to_numpy(),
            other_leaf_available=htrain.other_leaf_available.to_numpy()).to_csv(out/f'{fold}_oob.csv', index=False)
        label = 'temporal' if fold.startswith('temporal') else 'spatial_50km'
        part = frames[label].loc[split['test']].reset_index().copy()
        htest = history.iloc[test].reset_index(drop=True)
        query = held.drop(columns='observed_percent')
        part[NAME] = apply_weights(calibration, part.candidate_process, part.interval_history_component, htest, query)
        predictions[label].append(part)
        print(fold, {k:round(v['history_weight'],4) for k,v in calibration['groups'].items()}, flush=True)
    output_frames = {label:pd.concat(parts, ignore_index=True) for label,parts in predictions.items()}
    for label,filename in [('temporal','assessment_predictions.csv'), ('spatial_50km','spatial_predictions.csv')]:
        output_frames[label].to_csv(out/filename, index=False)
    p, h, counts = oob_predictions(components['base']['process'], components['base']['history'],
        components['events_history'], designs, history, obs)
    full_calibration = fit_weights(p, h, history, obs, counts)
    (out/'full_calibration.json').write_text(json.dumps(full_calibration, indent=2)+'\n')
    obs[['global_target_index', 'coordinate_year', 'observed_percent']].assign(
        process=p, history_component=h, oob_members=counts,
        prior_assessment_available=history.prior_assessment_available.to_numpy(),
        other_leaf_available=history.other_leaf_available.to_numpy()).to_csv(out/'full_oob.csv', index=False)
    fitted = dict(components=components, calibration=full_calibration, training_ids=components['training_ids'])
    joblib.dump(fitted, out/'fitted_candidate.joblib')
    contract = json.loads((parent/'fitted_candidate_lock.json').read_text())
    contract.update(created_utc=datetime.now(timezone.utc).isoformat(), recipe=NAME,
        role='Development candidate with training-only regularized convex mixture calibration',
        inference='analysis.disease_improvement.oob_history_blend.predict',
        parent_contract_sha256=sha(parent/'fitted_candidate_lock.json'),
        sha256={n:sha(out/n) for n in ['registration.json', 'fitted_candidate.joblib', 'full_calibration.json']})
    (out/'fitted_candidate_lock.json').write_text(json.dumps(contract, indent=2)+'\n')
    return score_frames(output_frames, history, position, out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output).to_string(index=False))
