"""Replay causal state features, fitted transitions and complete-support scores."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import distances_km, paired_interval, promotion_decision, score_predictions, score_subset
from .history import build_history
from .inputs import sha
from .leaf_state import STATE_NAMES, build_state_features
from .leaf_transition_experiment import DATA, NAME, PARENT, ROOT
from .leaf_transition_model import predict, predict_transition
from .rf_sowing_finalize import assessment_populations
from .robust_models import predict_cluster_ensemble
from .history_interval_hybrid import predict as predict_parent
from .verify_first_visit_screens import manual_integral, manual_weights
from .verify_french_weather import write_verification_receipt


def verify(output):
    out = Path(output).resolve()
    reg = json.loads((out/'registration.json').read_text())
    for path, digest in reg['source_sha256'].items():
        assert sha(ROOT/path) == digest, path
    for path, digest in reg['code_sha256'].items():
        assert sha(ROOT/path) == digest == sha(out/'code_snapshot'/path), path
    assert not reg['hyperparameter_search'] and not reg['external_validation_required_for_internal_experiment']
    obs = pd.read_csv(out/'training_observation_ledger.csv')
    pd.testing.assert_frame_equal(obs, pd.read_csv(PARENT/'training_observation_ledger.csv'))
    assert len(obs) == 2304 and obs.global_target_index.is_unique
    arrays = np.load(out/'predictors.npz'); original = np.load(PARENT/'predictors.npz')
    for key in arrays.files:
        np.testing.assert_array_equal(arrays[key], original[key])
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    states = np.load(out/'state_predictors.npz')
    np.testing.assert_array_equal(states['target_ids'], obs.global_target_index)
    z = states['states']; assert z.shape == (2304, 13)
    parent = joblib.load(PARENT/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=parent['base']['history_names'])
    q = obs.drop(columns='observed_percent')
    actual_history, hp = build_history(obs, q)
    np.testing.assert_array_equal(actual_history.to_numpy(), history.to_numpy())
    weather = pd.read_parquet(DATA/'daily_weather.parquet'); fields = pd.read_csv(DATA/'fields.csv')
    rebuilt, names, ledger = build_state_features(weather, fields, obs, q)
    np.testing.assert_array_equal(z, rebuilt)
    assert names == STATE_NAMES == json.loads((out/'state_schema.json').read_text())['names']
    saved_ledger = json.loads((out/'state_provenance.json').read_text())
    assert json.loads(ledger.to_json(orient='records')) == saved_ledger
    for a, b in zip(ledger.prior_target_ids, hp.prior_target_ids):
        assert set(a) == set(b)
    np.testing.assert_allclose(z[:, :7].sum(axis=1), 1, rtol=0, atol=1e-10)
    assert (z[:, :7] >= 0).all()
    perturbations = 0
    # Both providers, same-leaf history, other-leaf-only history, and initial visits.
    masks = [history.prior_assessment_available.eq(1)&obs.source.eq('BASF'),
        history.prior_assessment_available.eq(1)&obs.source.eq('Corteva'),
        history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(1),
        history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(0)]
    choose = [int(np.flatnonzero(mask)[0]) for mask in masks]
    for index in choose:
        query = q.iloc[[index]]; day = pd.Timestamp(query.date.iloc[0])
        changed_obs = obs.copy()
        changed_obs.loc[pd.to_datetime(changed_obs.date).ge(day), 'observed_percent'] = 100-changed_obs.observed_percent
        changed_weather = weather.copy()
        changed_weather.loc[pd.to_datetime(changed_weather.date).ge(day), ['tmean_c', 'rh_mean_pct', 'rain_mm']] = [35., 40., 999.]
        probe, _, _ = build_state_features(changed_weather, fields, changed_obs, query)
        np.testing.assert_array_equal(z[[index]], probe)
        perturbations += 1
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    assert (out/'split_membership.json').read_bytes() == (PARENT/'split_membership.json').read_bytes()
    splits = json.loads((out/'split_membership.json').read_text())
    frames = {}
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame = pd.read_csv(out/filename); old = pd.read_csv(PARENT/filename)
        pd.testing.assert_frame_equal(frame[old.columns], old)
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        frames[label] = frame
    count = 0

    def replay(bundle, training_ids, test_ids):
        nonlocal count
        assert bundle['training_ids'] == training_ids and len(bundle['members']) == 20
        assert bundle['seed'] == 20261022 and bundle['fraction'] == .8
        pool = obs.set_index('global_target_index').loc[training_ids]
        clusters = np.asarray(sorted(pool.coordinate_year.unique()))
        n = max(1, int(np.ceil(.8*len(clusters))))
        assert bundle['clusters_per_member'] == n
        rng = np.random.default_rng(20261022); predictions = []
        test = position.loc[test_ids].to_numpy()
        for member in bundle['members']:
            selected = rng.choice(clusters, n, replace=False)
            expected = pool.loc[pool.coordinate_year.isin(selected)].index.tolist()
            assert member['training_ids'] == expected
            loc = position.loc[expected].to_numpy()
            available = history.iloc[loc][['prior_assessment_available', 'other_leaf_available']].to_numpy().any(axis=1)
            assert member['transition_training_ids'] == np.asarray(expected)[available].tolist()
            if member['estimator'] is not None:
                params = member['estimator'].get_params()
                for key, value in reg['estimator'].items():
                    assert params[key] == value
                assert params['loss'] == 'squared_error'
            predictions.append(predict_transition(member, arrays['history_events'][test], history.iloc[test], z[test], q.iloc[test]))
            count += 1
        return np.mean(predictions, axis=0)

    seen = dict(temporal=[], spatial_50km=[]); minimum = np.inf
    for split in splits:
        train, test = [position.loc[split[key]].to_numpy() for key in ['train', 'test']]
        assert set(split['train']).isdisjoint(split['test'])
        assert obs.iloc[train].season_year.max() < obs.iloc[test].season_year.min()
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km':
            distance = distances_km(obs.iloc[train][['latitude', 'longitude']], obs.iloc[test][['latitude', 'longitude']]).min()
            minimum = min(minimum, distance); assert distance >= 50
            assert set(obs.iloc[train].site_id).isdisjoint(obs.iloc[test].site_id)
        else:
            assert split['train'] == obs.loc[obs.season_year.lt(obs.iloc[test].season_year.iloc[0]), 'global_target_index'].tolist()
        part = frames[label].set_index('global_target_index').loc[split['test']]
        bundle = joblib.load(out/(split['fold']+'_transition.joblib'))
        value = replay(bundle, split['train'], split['test'])
        np.testing.assert_allclose(value, part.transition_history_component, atol=1e-10, rtol=0)
        available = history.iloc[test][['prior_assessment_available', 'other_leaf_available']].to_numpy().any(axis=1)
        expected = np.where(available, .5*(part.candidate_process+value), part.irish_station_candidate)
        np.testing.assert_array_equal(available, part.uses_leaf_transition)
        np.testing.assert_allclose(expected, part[NAME], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(part.loc[~available, NAME], part.loc[~available, 'irish_station_candidate'])
        seen[label] += split['test']
        print('verified', split['fold'], flush=True)
    for label, ids in seen.items():
        assert len(ids) == len(set(ids)) == 2017 and set(ids) == set(frames[label].global_target_index)
        assert (~frames[label].uses_leaf_transition).sum() == 494
    full = joblib.load(out/'fitted_candidate.joblib')
    assert joblib.hash(full['parent']) == joblib.hash(parent)
    ids = obs.global_target_index.astype(int).tolist()
    assert full['training_ids'] == ids
    fitted_values = replay(full['transition'], ids, ids)
    assert count == 420
    # Real fitted arithmetic with synthetic future query dates, not forecast validation.
    take = choose[:3]+[choose[3]]
    queries = q.iloc[take].copy(); queries['date'] = '2020-06-01'; queries['season_year'] = 2020
    h = history.iloc[take]; x = [arrays[key][take] for key in ['process', 'station', 'history_events']]
    old = predict_parent(parent, *x, h, queries)
    process = predict_cluster_ensemble(parent['base']['process'], x[0], h, queries)
    available = h[['prior_assessment_available', 'other_leaf_available']].to_numpy().any(axis=1)
    expected = np.where(available, .5*(process+fitted_values[take]), old)
    np.testing.assert_allclose(predict(full, *x, h, z[take], queries), expected, atol=1e-10, rtol=0)
    for bad in [queries.assign(observed_percent=10.), queries.assign(season_year=2019)]:
        try:
            predict(full, *x, h, z[take], bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid future query accepted')
    metrics = pd.read_csv(out/'metrics.csv'); groups = pd.read_csv(out/'subgroup_metrics.csv')
    intervals = pd.read_csv(out/'paired_intervals.csv')
    assert len(metrics) == 7 and len(groups) == 63 and len(intervals) == 5
    for row in metrics.itertuples(index=False):
        frame = frames[row.evaluation]
        for key, value in score_predictions(frame, row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, key), value, atol=1e-9, rtol=0)
        direct = np.sqrt(np.average((frame[row.model]-frame.observed_percent)**2, weights=manual_weights(frame)))
        np.testing.assert_allclose(row.rmse, direct, atol=1e-10, rtol=0)
        np.testing.assert_allclose(row.integral_rmse, manual_integral(frame, row.model), atol=1e-9, rtol=0)
    for row in groups.itertuples(index=False):
        frame = frames[row.evaluation]; h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        for key, value in score_subset(frame, assessment_populations(frame, h)[row.population], row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, key), value, atol=1e-9, rtol=0)
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose([row.lower, row.upper], paired_interval(frames[row.evaluation], NAME, row.reference), atol=1e-10, rtol=0)
    decision = json.loads((out/'assessment_decision.json').read_text())
    frame = frames['temporal']; h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
    masks = assessment_populations(frame, h)
    for reference, saved in decision['references'].items():
        delta = {k:score_subset(frame, masks[k], NAME)['rmse']-score_subset(frame, masks[k], reference)['rmse'] for k in ['cold', 'warm', 'BASF', 'Corteva']}
        severe = abs(score_subset(frame, masks['severe_above75'], NAME)['bias'])-abs(score_subset(frame, masks['severe_above75'], reference)['bias'])
        ratio = score_subset(frame, masks['new_site'], NAME)['rmse']/score_subset(frame, masks['new_site'], reference)['rmse']
        actual = promotion_decision(score_predictions(frame, reference), score_predictions(frame, NAME),
            paired_interval(frame, NAME, reference), independent=False, subgroup_deltas=delta, severe_bias_delta=severe, new_site_ratio=ratio)
        guards = {k:v for k,v in actual['gates'].items() if k != 'independent_validation'}
        assert guards == saved['internal_guards'] and saved['passes_current_cohort_guards'] == all(guards.values())
    for evaluation, frame in frames.items():
        new, old = [score_predictions(frame, n) for n in [NAME, 'irish_station_candidate']]
        assert decision['incremental_parent_guards'][evaluation+'_rmse_noninferiority'] == (new['rmse'] <= old['rmse'])
        assert decision['incremental_parent_guards'][evaluation+'_integral_noninferiority'] == (new['integral_rmse'] <= old['integral_rmse'])
    assert decision['balanced_incremental_improvement'] == all(decision['incremental_parent_guards'].values())
    assert decision['independent_external_validation'] is False and decision['publication_model_promoted'] is False
    contract = json.loads((out/'fitted_candidate_lock.json').read_text())
    for name, digest in contract['sha256'].items():
        assert sha(out/name) == digest
    result = dict(verified=True, source_and_code_hashes_match=True, state_predictors_rebuilt_exactly=True,
        history35_rebuilt_exactly=True, all_parent_predictors_and_observations_unchanged=True,
        training_records=2304, fields=218, state_features=13, replayed_members=count,
        evaluation_records_per_design=2017, first_field_predictions_preserved_per_design=494,
        history_informed_predictions_per_design=1523, prior_observation_ids_verified=True,
        minimum_spatial_distance_km=float(minimum), causal_label_and_weather_perturbations=perturbations,
        full_parent_bundle_identical=True, future_inference_arithmetic=True, current_query_labels_rejected=True,
        metric_rows_recomputed=7, subgroup_rows_recomputed=63, intervals_recomputed=5,
        decisions_recomputed=3, incremental_guards_recomputed=4,
        independent_external_validation=False, publication_model_promoted=False, verifier_sha256=sha(__file__),
        verifier_dependencies_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            Path(__file__).with_name('verify_first_visit_screens.py'), Path(__file__).with_name('verify_french_weather.py')]})
    write_verification_receipt(out, result, Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/disease_improvement/leaf_transition_20261022')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.output), indent=2))
