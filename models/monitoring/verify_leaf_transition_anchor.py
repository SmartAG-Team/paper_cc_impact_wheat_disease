"""Replay the state-feature ablation with its retained observation anchor."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import distances_km, paired_interval, score_predictions, score_subset
from .field_history_models import predict_field_history
from .history_interval_hybrid import predict as predict_parent
from .inputs import sha
from .leaf_transition_anchor import predict
from .leaf_transition_anchor_experiment import NAME, PARENT, ROOT, STATE_RUN
from .rf_sowing_finalize import assessment_populations
from .robust_models import predict_cluster_ensemble
from .verify_first_visit_screens import manual_integral, manual_weights
from .verify_french_weather import write_verification_receipt


def verify(output):
    out = Path(output).resolve()
    reg = json.loads((out/'registration.json').read_text())
    for path, digest in reg['source_sha256'].items():
        assert sha(ROOT/path) == digest, path
    for path, digest in reg['code_sha256'].items():
        assert sha(ROOT/path) == digest == sha(out/'code_snapshot'/path), path
    assert reg['adaptive_followup_after_primary_results'] and reg['no_hyperparameter_search']
    obs = pd.read_csv(out/'training_observation_ledger.csv')
    pd.testing.assert_frame_equal(obs, pd.read_csv(PARENT/'training_observation_ledger.csv'))
    assert len(obs) == 2304
    arrays = np.load(out/'predictors.npz'); old = np.load(PARENT/'predictors.npz')
    for key in arrays.files:
        np.testing.assert_array_equal(arrays[key], old[key])
    for name in ['state_predictors.npz', 'state_schema.json', 'state_provenance.json']:
        assert (out/name).read_bytes() == (STATE_RUN/name).read_bytes()
    assert json.loads((STATE_RUN/'verification.json').read_text())['verified']
    z = np.load(out/'state_predictors.npz')['states']
    parent = joblib.load(PARENT/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=parent['base']['history_names'])
    design = np.c_[arrays['history_events'], z]; assert design.shape == (2304, 81)
    query = obs.drop(columns='observed_percent')
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    assert (out/'split_membership.json').read_bytes() == (PARENT/'split_membership.json').read_bytes()
    splits = json.loads((out/'split_membership.json').read_text())
    frames = {}; primary = {}
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame = pd.read_csv(out/filename, float_precision='round_trip')
        original = pd.read_csv(PARENT/filename, float_precision='round_trip')
        pd.testing.assert_frame_equal(frame[original.columns], original, check_exact=True)
        primary[label] = pd.read_csv(STATE_RUN/filename, float_precision='round_trip')
        np.testing.assert_array_equal(frame.leaf_state_transition, primary[label].leaf_state_transition)
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        frames[label] = frame
    members = heads = 0

    def replay(bundle, training_ids, test_ids):
        nonlocal members, heads
        assert bundle['training_ids'] == training_ids and len(bundle['members']) == 20
        assert bundle['seed'] == 20261022 and bundle['fraction'] == .8
        pool = obs.set_index('global_target_index').loc[training_ids]
        clusters = np.asarray(sorted(pool.coordinate_year.unique()))
        n = max(1, int(np.ceil(.8*len(clusters))))
        assert bundle['clusters_per_member'] == n
        rng = np.random.default_rng(20261022); values = []
        test = position.loc[test_ids].to_numpy()
        for member in bundle['members']:
            selected = rng.choice(clusters, n, replace=False)
            expected = pool.loc[pool.coordinate_year.isin(selected)].index.tolist()
            assert member['training_ids'] == expected and member['spec'] == dict(structure='field_split')
            assert member['n_features'] == 81
            for model in member['models'].values():
                if model is None:
                    continue
                settings = model.get_params()
                for key, value in dict(max_iter=200, max_depth=3, max_leaf_nodes=15,
                        min_samples_leaf=10, learning_rate=.05, l2_regularization=1.,
                        early_stopping=False, random_state=20261021, loss='squared_error').items():
                    assert settings[key] == value
                heads += 1
            values.append(predict_field_history(member, design[test], history.iloc[test], query.iloc[test]))
            members += 1
        return np.mean(values, axis=0)

    minimum = np.inf; seen = dict(temporal=[], spatial_50km=[])
    for split in splits:
        train, test = [position.loc[split[k]].to_numpy() for k in ['train', 'test']]
        assert set(split['train']).isdisjoint(split['test'])
        assert obs.iloc[train].season_year.max() < obs.iloc[test].season_year.min()
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'spatial_50km':
            distance = distances_km(obs.iloc[train][['latitude', 'longitude']], obs.iloc[test][['latitude', 'longitude']]).min()
            minimum = min(minimum, distance); assert distance >= 50
            assert set(obs.iloc[train].site_id).isdisjoint(obs.iloc[test].site_id)
        else:
            assert split['train'] == obs.loc[obs.season_year.lt(obs.iloc[test].season_year.iloc[0]), 'global_target_index'].tolist()
        values = replay(joblib.load(out/(split['fold']+'_transition.joblib')), split['train'], split['test'])
        part = frames[label].set_index('global_target_index').loc[split['test']]
        np.testing.assert_allclose(values, part.anchor_history_component, atol=1e-10, rtol=0)
        available = history.iloc[test][['prior_assessment_available', 'other_leaf_available']].to_numpy().any(axis=1)
        expected = np.where(available, .5*(part.candidate_process+values), part.irish_station_candidate)
        np.testing.assert_array_equal(available, part.uses_leaf_transition)
        np.testing.assert_allclose(expected, part[NAME], atol=1e-10, rtol=0)
        np.testing.assert_array_equal(part.loc[~available, NAME], part.loc[~available, 'irish_station_candidate'])
        seen[label] += split['test']
        print('verified anchor', split['fold'], flush=True)
    for label, ids in seen.items():
        assert len(ids) == len(set(ids)) == 2017 and set(ids) == set(frames[label].global_target_index)
        assert (~frames[label].uses_leaf_transition).sum() == 494
    full = joblib.load(out/'fitted_candidate.joblib')
    assert joblib.hash(full['parent']) == joblib.hash(parent)
    ids = obs.global_target_index.astype(int).tolist()
    full_values = replay(full['transition'], ids, ids)
    assert members == 420
    masks = [history.prior_assessment_available.eq(1),
        history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(1),
        history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(0)]
    take = [int(np.flatnonzero(m)[0]) for m in masks]
    q = query.iloc[take].copy(); q['date'] = '2020-06-01'; q['season_year'] = 2020
    h = history.iloc[take]; x = [arrays[k][take] for k in ['process', 'station', 'history_events']]
    old_prediction = predict_parent(parent, *x, h, q)
    process = predict_cluster_ensemble(parent['base']['process'], x[0], h, q)
    expected = np.where([True, True, False], .5*(process+full_values[take]), old_prediction)
    np.testing.assert_allclose(predict(full, *x, h, z[take], q), expected, atol=1e-10, rtol=0)
    for bad in [q.assign(observed_percent=50), q.assign(season_year=2019)]:
        try:
            predict(full, *x, h, z[take], bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid query accepted')
    metrics = pd.read_csv(out/'metrics.csv'); groups = pd.read_csv(out/'subgroup_metrics.csv')
    intervals = pd.read_csv(out/'paired_intervals.csv')
    assert len(metrics) == 6 and len(groups) == 54 and len(intervals) == 4
    for row in metrics.itertuples(index=False):
        f = frames[row.evaluation]
        for key, value in score_predictions(f, row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, key), value, atol=1e-9, rtol=0)
        direct = np.sqrt(np.average((f[row.model]-f.observed_percent)**2, weights=manual_weights(f)))
        np.testing.assert_allclose(row.rmse, direct, atol=1e-10, rtol=0)
        np.testing.assert_allclose(row.integral_rmse, manual_integral(f, row.model), atol=1e-9, rtol=0)
    for row in groups.itertuples(index=False):
        f = frames[row.evaluation]; h = history.iloc[position.loc[f.global_target_index]].reset_index(drop=True)
        for key, value in score_subset(f, assessment_populations(f, h)[row.population], row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row, key), value, atol=1e-9, rtol=0)
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose([row.lower, row.upper], paired_interval(frames[row.evaluation], NAME, row.reference), atol=1e-10, rtol=0)
    decision = json.loads((out/'assessment_decision.json').read_text())
    for label, f in frames.items():
        new, previous = [score_predictions(f, n) for n in [NAME, 'irish_station_candidate']]
        assert decision['incremental_parent_guards'][label+'_rmse_noninferiority'] == (new['rmse'] <= previous['rmse'])
        assert decision['incremental_parent_guards'][label+'_integral_noninferiority'] == (new['integral_rmse'] <= previous['integral_rmse'])
    assert decision['balanced_incremental_improvement'] == all(decision['incremental_parent_guards'].values())
    assert decision['independent_external_validation'] is False and decision['publication_model_promoted'] is False
    for name, digest in json.loads((out/'fitted_candidate_lock.json').read_text())['sha256'].items():
        assert sha(out/name) == digest
    result = dict(verified=True, source_and_code_hashes_match=True, full_parent_bundle_identical=True,
        parent_prediction_decimals_preserved_exactly=True, state13_identical_to_verified_primary=True,
        training_records=2304, evaluation_records_per_design=2017, first_field_predictions_preserved_per_design=494,
        replayed_members=members, checked_estimator_heads=heads, minimum_spatial_distance_km=float(minimum),
        metric_rows_recomputed=6, subgroup_rows_recomputed=54, intervals_recomputed=4, incremental_guards_recomputed=4,
        real_future_inference_arithmetic=True, current_query_labels_rejected=True,
        adaptive_development=True, independent_external_validation=False, publication_model_promoted=False,
        verifier_sha256=sha(__file__), verifier_dependencies_sha256={str(p.relative_to(ROOT)):sha(p) for p in [
            Path(__file__).with_name('verify_first_visit_screens.py'), Path(__file__).with_name('verify_french_weather.py')]})
    write_verification_receipt(out, result, Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/disease_improvement/leaf_transition_anchor_20261022')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.output), indent=2))
