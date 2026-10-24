"""Rebuild timing inputs and replay all fixed event-component combinations."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import distances_km, paired_interval, promotion_decision, score_predictions, score_subset
from .history import build_history
from .infection_event_features import build_event_features
from .inputs import WEATHER, sha
from .verify_field_history import frozen_module
from .verify_french_weather import write_verification_receipt

ROOT = Path(__file__).resolve().parents[2]


def verify(root, output):
    root, out = Path(root).resolve(), Path(output).resolve()
    registration = json.loads((out/'registration.json').read_text())
    for path, digest in registration['source_sha256'].items():
        assert sha(root/path) == digest, path
    for path, digest in registration['code_sha256'].items():
        assert sha(out/'code_snapshot'/path) == digest, path
    for path in ['analysis/disease_improvement/infection_event_features.py', 'analysis/disease_improvement/features.py',
                 'analysis/disease_improvement/history.py', 'analysis/disease_improvement/evaluation.py', 'analysis/final_study/metrics.py']:
        assert sha(root/path) == registration['code_sha256'][path], path
    base = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    data = root/'data/disease_improvement/infection_events_20261022'
    manifest = json.loads((data/'manifest.json').read_text())
    for key in ['source_sha256', 'code_sha256', 'output_sha256']:
        for path, digest in manifest[key].items():
            target = root/path if key != 'output_sha256' else data/path
            assert sha(target) == digest, path
    for name in ['training_observation_ledger.csv', 'split_membership.json']:
        assert (out/name).read_bytes() == (base/name).read_bytes()
    assert (out/'predictors.npz').read_bytes() == (data/'predictors.npz').read_bytes()
    obs = pd.read_csv(out/'training_observation_ledger.csv')
    arrays, old = np.load(out/'predictors.npz'), np.load(base/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    np.testing.assert_array_equal(np.c_[arrays['process_events'][:, :34], arrays['process_events'][:, -3:]], old['process'])
    np.testing.assert_array_equal(np.c_[arrays['history_events'][:, :50], arrays['history_events'][:, -3:]], old['station'])
    history, history_provenance = build_history(obs, obs.drop(columns='observed_percent'))
    np.testing.assert_array_equal(history.to_numpy(), arrays['history'])
    np.testing.assert_array_equal(arrays['history'], old['history'])
    fields = pd.concat([pd.read_csv(root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv'),
                       pd.read_csv(root/'data/disease_improvement/coordinate_augmentation_station_20261022/fields.csv')], ignore_index=True)
    assert fields.field_id.is_unique and fields.field_id.nunique() == 218
    original_weather = pd.read_parquet(root/WEATHER)
    additional = pd.read_parquet(root/'data/disease_improvement/coordinate_augmentation_20261022/daily_weather.parquet')
    reanalysis = pd.concat([original_weather, additional.loc[~additional.location_id.isin(original_weather.location_id)]], ignore_index=True)
    station = pd.read_parquet(root/'data/disease_improvement/station_preferred_nl_20261022/daily_weather.parquet')
    provenances = []
    for label, weather, key, first in [('process', reanalysis, 'process_event_features', 34),
                                       ('history', station, 'station_event_features', 50)]:
        events, names, provenance = build_event_features(weather, fields, obs.drop(columns='observed_percent'), history_provenance)
        np.testing.assert_allclose(events, arrays[key], rtol=0, atol=1e-12)
        np.testing.assert_array_equal(arrays[label+'_events'][:, first:first+15], arrays[key])
        assert pd.to_datetime(provenance.weather_end).eq(pd.to_datetime(obs.date)-pd.Timedelta(days=1)).all()
        known = provenance.anchor_date.notna()
        assert pd.to_datetime(provenance.loc[known, 'anchor_date']).lt(pd.to_datetime(obs.loc[known, 'date'])).all()
        provenances.append(provenance)
    snapshot = out/'code_snapshot/analysis/disease_improvement'
    models = frozen_module(snapshot/'models.py', 'frozen_event_models')
    robust = frozen_module(snapshot/'robust_models.py', 'analysis.disease_improvement.frozen_event_robust')
    robust.predict_candidate = models.predict_candidate
    field_models = frozen_module(snapshot/'field_history_models.py', 'analysis.disease_improvement.frozen_event_field')
    field_models._category = models._category
    experiment = frozen_module(snapshot/'infection_event_experiment.py', 'analysis.disease_improvement.frozen_event_experiment')
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    indexed = obs.set_index('global_target_index')
    frames = {}
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame, parent = pd.read_csv(out/filename), pd.read_csv(base/filename)
        pd.testing.assert_frame_equal(frame[parent.columns], parent, check_dtype=False)
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        for name, spec in registration['blends'].items():
            np.testing.assert_allclose(frame[name], frame[spec['components']].mean(axis=1), atol=1e-10)
        frames[label] = frame
    ensembles = members = 0
    minimum_distance = np.inf

    def replay(fitted, train, test, component, expected=None):
        nonlocal ensembles, members
        assert fitted['training_ids'] == train
        eligible = indexed.loc[train]
        clusters = np.asarray(sorted(eligible.coordinate_year.unique()))
        count = max(1, int(np.ceil(len(clusters)*.8)))
        rng = np.random.default_rng(20261022)
        width = 52 if component == 'process' else 68
        assert len(fitted['members']) == 20 and fitted['fraction'] == .8 and fitted['seed'] == 20261022
        for member in fitted['members']:
            selected = rng.choice(clusters, size=count, replace=False)
            ids = eligible.loc[eligible.coordinate_year.isin(selected)].index.tolist()
            assert member['training_ids'] == ids and member['n_features'] == width
            for head in member['models'].values():
                if head is not None:
                    assert head.n_features_in_ == width+35+(5 if component == 'history' else 0)
            members += 1
        positions = position.loc[test].to_numpy()
        fn = robust.predict_cluster_ensemble if component == 'process' else field_models.predict_ensemble
        values = fn(fitted, arrays[component+'_events'][positions], history.iloc[positions],
                    indexed.loc[test].reset_index().drop(columns='observed_percent'))
        if expected is not None:
            np.testing.assert_allclose(values, expected, atol=1e-10)
        assert np.isfinite(values).all() and np.all((values >= 0)&(values <= 100))
        ensembles += 1
        return values

    seen = dict(temporal=[], spatial_50km=[])
    for split in json.loads((out/'split_membership.json').read_text()):
        train, test = split['train'], split['test']
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        assert indexed.loc[train, 'season_year'].max() < indexed.loc[test, 'season_year'].min()
        if label == 'spatial_50km':
            distance = float(distances_km(indexed.loc[train, ['latitude', 'longitude']], indexed.loc[test, ['latitude', 'longitude']]).min())
            assert distance >= 50
            minimum_distance = min(minimum_distance, distance)
        part = frames[label].set_index('global_target_index').loc[test]
        for component in ['process', 'history']:
            replay(joblib.load(out/f"{split['fold']}_{component}.joblib"), train, test, component, part['events_'+component])
        seen[label] += test
    for label, ids in seen.items():
        assert len(ids) == len(set(ids)) == 2017 and set(ids) == set(frames[label].global_target_index)
    metrics, intervals, groups = [pd.read_csv(out/name) for name in ['metrics.csv', 'paired_intervals.csv', 'subgroup_metrics.csv']]
    for row in metrics.itertuples(index=False):
        for key, value in score_predictions(frames[row.evaluation], row.model).items():
            if value is not None:
                np.testing.assert_allclose(value, getattr(row, key), atol=1e-9)
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose(paired_interval(frames[row.evaluation], row.model, row.reference),
                                   [row.lower, row.upper], atol=1e-9)
    temporal = frames['temporal']
    h = history.iloc[position.loc[temporal.global_target_index]].reset_index(drop=True)
    masks = experiment.populations(temporal, h)
    for row in groups.itertuples(index=False):
        for key, value in score_subset(temporal, masks[row.population], row.model).items():
            if value is not None:
                np.testing.assert_allclose(value, getattr(row, key), atol=1e-9)
    contract = json.loads((out/'fitted_candidate_lock.json').read_text())
    for path, digest in contract['sha256'].items():
        assert sha(out/path) == digest, path
    fitted = joblib.load(out/'fitted_candidates.joblib')
    assert joblib.hash(fitted['base']) == joblib.hash(joblib.load(base/'fitted_candidate.joblib'))
    all_ids = obs.global_target_index.astype(int).tolist()
    assert fitted['training_ids'] == all_ids and fitted['minimum_assessment_year'] == 2020
    assert fitted['history_names'] == history.columns.tolist()
    full = {component: replay(fitted[component], all_ids, all_ids, component) for component in ['process', 'history']}
    decisions = {}
    for name in registration['blends']:
        decisions[name] = {}
        for reference in ['station_candidate', 'current', 'reference_replay']:
            deltas = {key: score_subset(temporal, masks[key], name)['rmse']-score_subset(temporal, masks[key], reference)['rmse']
                      for key in ['cold', 'warm', 'BASF', 'Corteva']}
            severe = abs(score_subset(temporal, masks['severe_above75'], name)['bias'])-abs(score_subset(temporal, masks['severe_above75'], reference)['bias'])
            ratio = score_subset(temporal, masks['new_site'], name)['rmse']/score_subset(temporal, masks['new_site'], reference)['rmse']
            decision = promotion_decision(score_predictions(temporal, reference), score_predictions(temporal, name),
                paired_interval(temporal, name, reference), independent=False,
                subgroup_deltas=deltas, severe_bias_delta=severe, new_site_ratio=ratio)
            decision.update(subgroup_rmse_deltas=deltas, severe_absolute_bias_change=severe, new_site_rmse_ratio=ratio)
            decisions[name][reference] = decision
    (out/'adoption_decision.json').write_text(json.dumps(dict(candidates=decisions,
        independent_external_validation=False, publication_model_promoted=False), indent=2)+'\n')
    result = dict(verified=True, training_records=len(obs), evaluation_records_per_condition=2017,
        source_hashes=len(registration['source_sha256']), event_features_rebuilt=15,
        reanalysis_and_station_events_rebuilt=True, all_original_predictor_columns_identical=True,
        all_prior_dates_strict=True, saved_ensembles_replayed=ensembles, component_models_replayed=members,
        minimum_spatial_distance_km=minimum_distance, metrics_recomputed=len(metrics),
        intervals_recomputed=len(intervals), subgroup_rows_recomputed=len(groups),
        parent_full_models_identical=True, full_fit_replay_is_validation=False,
        independent_external_validation=False, publication_model_promoted=False, verifier_sha256=sha(Path(__file__)))
    write_verification_receipt(out, result, Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(ROOT, args.output), indent=2))
