"""Retain first-field prediction and add event inputs after earlier field visits."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval, promotion_decision, score_predictions, score_subset
from .field_history_models import predict_ensemble
from .infection_event_experiment import populations
from .inputs import sha
from .robust_models import predict_cluster_ensemble
from .verify_french_weather import write_verification_receipt

ROOT = Path(__file__).resolve().parents[2]
NAME = 'history_interval'


def history_prediction(original, events, history, queries):
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    old, new = np.asarray(original, float), np.asarray(events, float)
    if (old.shape != (len(queries),) or new.shape != old.shape or len(history) != len(queries)
            or not np.isfinite(np.c_[old, new]).all()
            or np.any((old < 0)|(old > 100)|(new < 0)|(new > 100))):
        raise ValueError('Aligned bounded component predictions required')
    available = history[['prior_assessment_available', 'other_leaf_available']].to_numpy(float)
    if not np.isfinite(available).all() or not np.isin(available, [0, 1]).all():
        raise ValueError('Finite binary earlier-field availability required')
    mask = available.any(axis=1)
    return np.where(mask, new, old), mask


def predict(fitted, process, station, event_station, history, queries):
    """Schema-checked future native-domain inference; metadata qualification is upstream."""
    base = fitted['base']
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    dates = pd.to_datetime(queries.date, errors='coerce', format='mixed')
    if (dates.isna().any() or not dates.dt.year.eq(queries.season_year).all()
            or not queries.season_year.ge(base['minimum_assessment_year']).all()
            or not queries.source.isin(['BASF', 'Corteva']).all()
            or not queries.leaf_index.isin(range(7)).all()):
        raise ValueError('Queries outside the frozen future native-source domain')
    arrays = [np.asarray(value, float) for value in [process, station, event_station]]
    if (any(a.shape != (len(queries), width) or not np.isfinite(a).all()
            for a, width in zip(arrays, [37, 53, 68]))
            or history.columns.tolist() != base['history_names']
            or history.shape != (len(queries), len(base['history_names']))
            or not np.isfinite(history.to_numpy(float)).all()):
        raise ValueError('Complete aligned predictor schema required')
    p = predict_cluster_ensemble(base['process'], arrays[0], history, queries)
    old = predict_ensemble(base['history'], arrays[1], history, queries)
    new = predict_ensemble(fitted['events_history'], arrays[2], history, queries)
    selected, _ = history_prediction(old, new, history, queries)
    return .5*(p+selected)


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    parent = root/'outputs/disease_improvement/infection_events_20261022'
    base = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    out = Path(output or root/'outputs/disease_improvement/history_interval_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    obs = pd.read_csv(base/'training_observation_ledger.csv')
    arrays = np.load(base/'predictors.npz')
    base_bundle = joblib.load(base/'fitted_candidate.joblib')
    events_bundle = joblib.load(parent/'fitted_candidates.joblib')
    assert joblib.hash(events_bundle['base']) == joblib.hash(base_bundle)
    history = pd.DataFrame(arrays['history'], columns=base_bundle['history_names'])
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    sources = [parent/n for n in ['registration.json', 'assessment_predictions.csv', 'spatial_predictions.csv',
        'fitted_candidates.joblib', 'fitted_candidate_lock.json', 'verification.json', 'split_membership.json']]
    sources += [base/n for n in ['predictors.npz', 'training_observation_ledger.csv', 'fitted_candidate.joblib',
                               'fitted_candidate_lock.json', 'verification.json']]
    codes = [Path(__file__), Path(__file__).with_name('infection_event_experiment.py'),
             Path(__file__).with_name('evaluation.py'), root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        rule='Unchanged process component; original first-field history head; event-enhanced history head whenever any strictly earlier field observation exists; fixed50:50 final blend',
        selection_condition='Prior availability only; no current severity, error, country or time threshold',
        fitting='No refit: reuse independently fitted heads with identical training pools and whole-cluster samples',
        motivation='First-field event expansion worsens development accuracy; visit-interval features have a directly observed start only after earlier field context',
        posthoc_refinement=True, independent_external_validation=False, publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)): sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)): sha(p) for p in codes})
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in codes:
        dest = out/'code_snapshot'/path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    metrics, intervals, groups = [], [], []
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame = pd.read_csv(parent/filename)
        h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        component, mask = history_prediction(frame.candidate_history, frame.events_history, h,
                                             frame.drop(columns='observed_percent'))
        frame['interval_history_component'] = component
        frame['uses_event_head'] = mask
        frame[NAME] = .5*(frame.candidate_process+component)
        frame.to_csv(out/filename, index=False)
        for name in [NAME, 'station_candidate', 'events_history_mean', 'reference_replay']+(['current'] if label == 'temporal' else []):
            metrics.append(dict(evaluation=label, model=name, **score_predictions(frame, name)))
            if name != NAME:
                interval = paired_interval(frame, NAME, name)
                intervals.append(dict(evaluation=label, reference=name, lower=interval[0], upper=interval[1]))
        if label == 'temporal':
            for population, subset in populations(frame, h).items():
                for name in [NAME, 'station_candidate', 'reference_replay']:
                    groups.append(dict(population=population, model=name, **score_subset(frame, subset, name)))
            frame.loc[frame.global_target_index.isin([479, 1750, 413, 159, 1231]),
                ['global_target_index', 'observed_percent', 'station_candidate', NAME, 'uses_event_head']].to_csv(out/'hard_cases.csv', index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    pd.DataFrame(groups).to_csv(out/'subgroup_metrics.csv', index=False)
    fitted = dict(base=base_bundle, events_history=events_bundle['history'],
                  training_ids=base_bundle['training_ids'], minimum_assessment_year=2020)
    joblib.dump(fitted, out/'fitted_candidate.joblib')
    contract = json.loads((base/'fitted_candidate_lock.json').read_text())
    contract.update(created_utc=datetime.now(timezone.utc).isoformat(),
        role='Posthoc development candidate using conditional predictor sets in independent heads',
        recipe=NAME, posthoc_refinement=True, current_labels_used_in_switch=False,
        predictor_widths=dict(process=37, first_field_history=53, event_history=68, disease_history=35),
        inference='analysis.disease_improvement.history_interval_hybrid.predict; frozen code snapshot supplied',
        sha256={n: sha(out/n) for n in ['registration.json', 'fitted_candidate.joblib']},
        parent_event_contract_sha256=sha(parent/'fitted_candidate_lock.json'),
        parent_base_contract_sha256=sha(base/'fitted_candidate_lock.json'))
    (out/'fitted_candidate_lock.json').write_text(json.dumps(contract, indent=2)+'\n')
    return pd.DataFrame(metrics)


def verify(root, output):
    root, out = Path(root).resolve(), Path(output).resolve()
    protocol = json.loads((out/'registration.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        assert sha(root/path) == digest, path
    for path, digest in protocol['code_sha256'].items():
        assert sha(out/'code_snapshot'/path) == digest, path
    for path in ['analysis/disease_improvement/infection_event_experiment.py',
                 'analysis/disease_improvement/evaluation.py', 'analysis/final_study/metrics.py']:
        assert sha(root/path) == protocol['code_sha256'][path]
    parent = root/'outputs/disease_improvement/infection_events_20261022'
    base = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    original = joblib.load(base/'fitted_candidate.joblib')
    event_bundle = joblib.load(parent/'fitted_candidates.joblib')
    fitted = joblib.load(out/'fitted_candidate.joblib')
    assert joblib.hash(fitted['base']) == joblib.hash(original)
    assert joblib.hash(fitted['events_history']) == joblib.hash(event_bundle['history'])
    assert fitted['training_ids'] == original['training_ids'] and len(fitted['training_ids']) == 2304
    contract = json.loads((out/'fitted_candidate_lock.json').read_text())
    for path, digest in contract['sha256'].items():
        assert sha(out/path) == digest, path
    assert contract['parent_event_contract_sha256'] == sha(parent/'fitted_candidate_lock.json')
    assert contract['parent_base_contract_sha256'] == sha(base/'fitted_candidate_lock.json')
    obs = pd.read_csv(base/'training_observation_ledger.csv')
    arrays = np.load(base/'predictors.npz')
    history = pd.DataFrame(arrays['history'], columns=original['history_names'])
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    frames, masks = {}, {}
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame, old = pd.read_csv(out/filename), pd.read_csv(parent/filename)
        pd.testing.assert_frame_equal(frame[old.columns], old, check_dtype=False)
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        mask = h.prior_assessment_available.eq(1)|h.other_leaf_available.eq(1)
        selected = np.where(mask, frame.events_history, frame.candidate_history)
        np.testing.assert_array_equal(mask, frame.uses_event_head)
        np.testing.assert_allclose(selected, frame.interval_history_component, atol=1e-10)
        np.testing.assert_allclose(.5*(frame.candidate_process+selected), frame[NAME], atol=1e-10)
        np.testing.assert_allclose(frame.loc[~mask, NAME], frame.loc[~mask, 'station_candidate'], atol=1e-10)
        frames[label] = frame
        if label == 'temporal':
            masks = populations(frame, h)
    metrics, intervals, groups = [pd.read_csv(out/n) for n in ['metrics.csv', 'paired_intervals.csv', 'subgroup_metrics.csv']]
    for row in metrics.itertuples(index=False):
        for key, value in score_predictions(frames[row.evaluation], row.model).items():
            if value is not None:
                np.testing.assert_allclose(value, getattr(row, key), atol=1e-9)
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose(paired_interval(frames[row.evaluation], NAME, row.reference), [row.lower, row.upper], atol=1e-9)
    for row in groups.itertuples(index=False):
        for key, value in score_subset(frames['temporal'], masks[row.population], row.model).items():
            if value is not None:
                np.testing.assert_allclose(value, getattr(row, key), atol=1e-9)
    frame = frames['temporal']
    decisions = {}
    for reference in ['station_candidate', 'current', 'reference_replay']:
        differences = {key: score_subset(frame, masks[key], NAME)['rmse']-score_subset(frame, masks[key], reference)['rmse']
                       for key in ['cold', 'warm', 'BASF', 'Corteva']}
        severe = abs(score_subset(frame, masks['severe_above75'], NAME)['bias'])-abs(score_subset(frame, masks['severe_above75'], reference)['bias'])
        ratio = score_subset(frame, masks['new_site'], NAME)['rmse']/score_subset(frame, masks['new_site'], reference)['rmse']
        decision = promotion_decision(score_predictions(frame, reference), score_predictions(frame, NAME),
            paired_interval(frame, NAME, reference), independent=False, subgroup_deltas=differences,
            severe_bias_delta=severe, new_site_ratio=ratio)
        decision.update(subgroup_rmse_deltas=differences, severe_absolute_bias_change=severe, new_site_rmse_ratio=ratio)
        decisions[reference] = decision
    (out/'adoption_decision.json').write_text(json.dumps(dict(references=decisions,
        independent_external_validation=False, publication_model_promoted=False), indent=2)+'\n')
    result = dict(verified=True, training_records=2304, evaluation_records_per_condition=2017,
        original_components_and_training_identical=True, first_field_predictions_retained=494,
        earlier_field_queries_use_event_head=1523, query_labels_used_in_switch=False,
        metrics_recomputed=len(metrics), intervals_recomputed=len(intervals), subgroup_rows_recomputed=len(groups),
        parent_new_component_replay_receipt_sha256=sha(parent/'verification.json'),
        independent_external_validation=False, publication_model_promoted=False, verifier_sha256=sha(Path(__file__)))
    write_verification_receipt(out, result, Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.verify:
            print(json.dumps(verify(ROOT, args.verify), indent=2))
        else:
            print(run(output=args.output).to_string(index=False))
