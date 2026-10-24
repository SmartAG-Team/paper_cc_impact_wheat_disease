"""Replay the integrated candidate and audit every unchanged validation boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import distances_km, paired_interval, promotion_decision, score_predictions, score_subset
from .history import build_history
from .inputs import sha
from .verify_field_history import frozen_module
from .verify_french_weather import write_verification_receipt

ROOT = Path(__file__).resolve().parents[2]


def verify_source_slice(actual, source):
    """A concatenated union schema preserves every source value and no extras."""
    pd.testing.assert_frame_equal(actual[source.columns].reset_index(drop=True),
                                  source.reset_index(drop=True), check_dtype=False)
    extra_columns = actual.columns.difference(source.columns)
    assert actual[extra_columns].isna().all().all()


def verify(root, output):
    root, out = Path(root).resolve(), Path(output).resolve()
    protocol = json.loads((out/'registration.json').read_text())
    for path, digest in protocol['source_sha256'].items():
        assert sha(root/path) == digest, path
    for path, digest in protocol['code_sha256'].items():
        assert sha(out/'code_snapshot'/path) == digest, path
    for name in ['evaluation.py', 'history.py']:
        key = 'analysis/disease_improvement/'+name
        assert sha(root/key) == protocol['code_sha256'][key]
    assert sha(root/'analysis/final_study/metrics.py') == protocol['code_sha256']['analysis/final_study/metrics.py']
    station_dir = root/'data/disease_improvement/coordinate_augmentation_station_20261022'
    station_manifest = json.loads((station_dir/'manifest.json').read_text())
    for group in ['source_sha256', 'code_sha256', 'output_sha256']:
        for path, digest in station_manifest[group].items():
            assert sha(root/path) == digest, path
    snapshot = out/'code_snapshot/analysis/disease_improvement'
    models = frozen_module(snapshot/'models.py', 'frozen_candidate_base')
    robust = frozen_module(snapshot/'robust_models.py', 'analysis.disease_improvement.frozen_candidate_robust')
    robust.predict_candidate = models.predict_candidate
    history_models = frozen_module(snapshot/'field_history_models.py', 'analysis.disease_improvement.frozen_candidate_history')
    history_models._category = models._category
    obs = pd.read_csv(out/'training_observation_ledger.csv')
    old_dir = root/'outputs/disease_improvement/field_history_20261022'
    original = pd.read_csv(old_dir/'observation_ledger.csv')
    extra = pd.read_csv(station_dir/'observations.csv')
    verify_source_slice(obs.iloc[:len(original)], original)
    verify_source_slice(obs.iloc[len(original):], extra)
    assert len(obs) == 2304 and obs.global_target_index.is_unique
    arrays = np.load(out/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    history, provenance = build_history(obs, obs.drop(columns='observed_percent'))
    np.testing.assert_allclose(history.to_numpy(), arrays['history'], atol=1e-12)
    old_arrays = np.load(old_dir/'predictors.npz')
    station_arrays = np.load(station_dir/'predictors.npz')
    raw_arrays = np.load(root/'data/disease_improvement/coordinate_augmentation_20261022/predictors.npz')
    raw_original = np.load(root/'outputs/disease_improvement/station_repairs_20261022_v4/predictors.npz')
    np.testing.assert_array_equal(arrays['process'], np.vstack([raw_original['reanalysis'], raw_arrays['matrix']]))
    original_station = np.vstack([old_arrays['matrix'], station_arrays['station_stage']])
    override = protocol.get('station_design_override')
    if override is None:
        np.testing.assert_array_equal(arrays['station'], original_station)
    else:
        override_path = root/override
        override_manifest = json.loads(override_path.with_name('manifest.json').read_text())
        for key in ['source_sha256', 'code_sha256', 'output_sha256']:
            for path, digest in override_manifest[key].items():
                directory = override_path.parent if key == 'output_sha256' else root
                assert sha(directory/path) == digest, path
        override_arrays = np.load(override_path)
        np.testing.assert_array_equal(override_arrays['target_ids'], obs.global_target_index)
        np.testing.assert_array_equal(arrays['station'], override_arrays['station'])
        unchanged = obs.site_id.ne('era5loc_c36c557e4afc').to_numpy()
        np.testing.assert_array_equal(arrays['station'][unchanged], original_station[unchanged])
    index = obs.set_index('global_target_index')
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    for row in provenance.itertuples(index=False):
        if row.prior_target_ids:
            assert pd.to_datetime(index.loc[row.prior_target_ids, 'date']).lt(pd.Timestamp(index.loc[row.global_target_index, 'date'])).all()
    frames = {}
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame = pd.read_csv(out/filename)
        old = pd.read_csv(old_dir/filename).set_index('global_target_index').loc[frame.global_target_index].reset_index()
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        pd.testing.assert_frame_equal(frame[old.columns], old, check_dtype=False)
        np.testing.assert_allclose(frame.station_candidate, .5*(frame.candidate_process+frame.candidate_history), atol=1e-12)
        frames[label] = frame
    old_splits = {s['fold']: s for s in json.loads((old_dir/'spatial_membership.json').read_text())}
    added_ids = set(extra.global_target_index)
    assignments = protocol['site_assignment']
    ordered = sorted(original.site_id.unique(), key=lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    assert all(assignments[s] == i % 3 for i, s in enumerate(ordered))
    assert all(assignments[s] == int(hashlib.sha256(str(s).encode()).hexdigest(), 16) % 3
               for s in set(extra.site_id)-set(original.site_id))
    ensembles = members = 0
    minimum_distance = np.inf

    def replay(fitted, train, test, expected, component):
        nonlocal ensembles, members
        assert fitted['training_ids'] == train
        eligible = index.loc[train]
        clusters = np.asarray(sorted(eligible.coordinate_year.unique()))
        count = max(1, int(np.ceil(len(clusters)*.8)))
        rng = np.random.default_rng(20261022)
        assert len(fitted['members']) == 20 and fitted['fraction'] == .8 and fitted['seed'] == 20261022
        for model in fitted['members']:
            selected = rng.choice(clusters, size=count, replace=False)
            expected_ids = eligible.loc[eligible.coordinate_year.isin(selected)].index.tolist()
            assert model['training_ids'] == expected_ids
            members += 1
        loc = position.loc[test].to_numpy()
        fn = robust.predict_cluster_ensemble if component == 'process' else history_models.predict_ensemble
        prediction = fn(fitted, arrays[component if component == 'process' else 'station'][loc],
                        history.iloc[loc], index.loc[test].reset_index().drop(columns='observed_percent'))
        if expected is not None:
            np.testing.assert_allclose(prediction, expected, atol=1e-10)
        ensembles += 1
        return prediction

    memberships = json.loads((out/'split_membership.json').read_text())
    assert len(memberships) == 20
    seen = {'temporal': [], 'spatial_50km': []}
    for split in memberships:
        train, test = split['train'], split['test']
        held, eligible = index.loc[test], index.loc[train]
        assert eligible.season_year.max() < held.season_year.min()
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label == 'temporal':
            expected = obs.loc[obs.season_year.lt(held.season_year.iloc[0]), 'global_target_index'].tolist()
            assert train == expected
        else:
            old_split = old_splits[split['original_fold']]
            assert test == old_split['test']
            assert [x for x in train if x not in added_ids] == old_split['train']
            fold = int(re.search(r'sitefold(\d+)', split['original_fold']).group(1))
            additions = extra.loc[extra.season_year.lt(held.season_year.iloc[0]) & extra.site_id.map(assignments).ne(fold)]
            if len(additions):
                d = distances_km(additions[['latitude', 'longitude']], held[['latitude', 'longitude']])
                additions = additions.loc[d.min(axis=1) >= 50]
            assert split['added_train_ids'] == additions.global_target_index.tolist()
            assert train == old_split['train']+additions.global_target_index.tolist()
            distance = float(distances_km(eligible[['latitude', 'longitude']], held[['latitude', 'longitude']]).min())
            assert distance >= 50
            minimum_distance = min(minimum_distance, distance)
        assert split['added_train_ids'] == sorted(set(train)&added_ids)
        part = frames[label].set_index('global_target_index').loc[test]
        for component in ['process', 'history']:
            fitted = joblib.load(out/f"{split['fold']}_{component}.joblib")
            replay(fitted, train, test, part['candidate_'+component], component)
        seen[label] += test
    for label in seen:
        assert len(seen[label]) == len(set(seen[label])) == 2017
        assert set(seen[label]) == set(frames[label].global_target_index)
    metrics = pd.read_csv(out/'metrics.csv')
    for row in metrics.itertuples(index=False):
        for key, value in score_predictions(frames[row.evaluation], row.model).items():
            if value is not None:
                np.testing.assert_allclose(value, getattr(row, key), atol=1e-9)
    intervals = pd.read_csv(out/'paired_intervals.csv')
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose(paired_interval(frames[row.evaluation], 'station_candidate', row.reference),
                                   [row.lower, row.upper], atol=1e-9)
    frame = frames['temporal']
    h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
    masks = dict(cold=~frame.prior_assessment_available, warm=frame.prior_assessment_available,
        first_field=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(0),
        first_leaf_other_history=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(1),
        BASF=frame.source.eq('BASF'), Corteva=frame.source.eq('Corteva'),
        severe_above75=frame.observed_percent.gt(75), new_site=frame.new_site)
    groups = pd.read_csv(out/'subgroup_metrics.csv')
    for row in groups.itertuples(index=False):
        for key, value in score_subset(frame, masks[row.population], row.model).items():
            if value is not None:
                np.testing.assert_allclose(value, getattr(row, key), atol=1e-9)
    lock = json.loads((out/'fitted_candidate_lock.json').read_text())
    for path, digest in lock['sha256'].items():
        assert sha(out/path) == digest, path
    bundle = joblib.load(out/'fitted_candidate.joblib')
    ids = obs.global_target_index.astype(int).tolist()
    assert bundle['training_ids'] == ids and bundle['minimum_assessment_year'] == 2020
    assert bundle['history_names'] == history.columns.tolist() and bundle['blend_weights'] == [.5, .5]
    full = [replay(bundle[c], ids, ids, None, c) for c in ['process', 'history']]
    full_prediction = .5*(full[0]+full[1])
    assert np.isfinite(full_prediction).all() and np.all((full_prediction >= 0)&(full_prediction <= 100))
    decisions = {}
    for reference in ['current', 'reference_replay']:
        deltas = {k: score_subset(frame, masks[k], 'station_candidate')['rmse']-score_subset(frame, masks[k], reference)['rmse']
                  for k in ['cold', 'warm', 'BASF', 'Corteva']}
        severe = abs(score_subset(frame, masks['severe_above75'], 'station_candidate')['bias'])-abs(score_subset(frame, masks['severe_above75'], reference)['bias'])
        ratio = score_subset(frame, masks['new_site'], 'station_candidate')['rmse']/score_subset(frame, masks['new_site'], reference)['rmse']
        decision = promotion_decision(score_predictions(frame, reference), score_predictions(frame, 'station_candidate'),
            paired_interval(frame, 'station_candidate', reference), independent=False,
            subgroup_deltas=deltas, severe_bias_delta=severe, new_site_ratio=ratio)
        decision.update(subgroup_rmse_deltas=deltas, severe_absolute_bias_change=severe, new_site_rmse_ratio=ratio)
        decisions[reference] = decision
    (out/'adoption_decision.json').write_text(json.dumps(dict(references=decisions,
        independent_external_validation=False, publication_model_promoted=False), indent=2)+'\n')
    result = dict(verified=True, training_records=len(obs), original_evaluation_records=2017,
        added_records_training_only=len(extra), saved_ensembles_replayed=ensembles,
        component_models_replayed=members, source_hashes=len(protocol['source_sha256']),
        nested_station_manifest_hashes_verified=True, history_reconstructed=True,
        all_dates_strict=True, all_original_evaluation_values_preserved=True,
        old_spatial_membership_preserved=True, minimum_spatial_distance_km=minimum_distance,
        metrics_recomputed=len(metrics), intervals_recomputed=len(intervals), subgroup_rows_recomputed=len(groups),
        full_fit_replay_is_validation=False, full_fit_prediction_sha256=hashlib.sha256(full_prediction.tobytes()).hexdigest(),
        independent_external_validation=False, publication_model_promoted=False, verifier_sha256=sha(Path(__file__)))
    write_verification_receipt(out, result, Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(ROOT, args.output), indent=2))
