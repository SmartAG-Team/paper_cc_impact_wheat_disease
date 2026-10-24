"""Independently check added-record boundaries and replay augmented models."""
import argparse
import importlib.util
import json
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .evaluation import distances_km, score_predictions
from .inputs import sha

ROOT = Path(__file__).resolve().parents[2]


def verify(root, output):
    root, output = Path(root), Path(output)
    registration = json.loads((output/'registration.json').read_text())
    for path, digest in registration['source_sha256'].items():
        assert sha(root/path) == digest, path
    for path, digest in registration['code_sha256'].items():
        assert sha(output/'code_snapshot'/path) == digest, path
    extra = root/'data/disease_improvement/coordinate_augmentation_20261022'
    manifest = json.loads((extra/'manifest.json').read_text())
    for path, digest in manifest['source_sha256'].items():
        assert sha(root/path) == digest, path
    added = pd.read_csv(extra/'observations.csv')
    assert added.source_file.nunique() == 1
    assert sha(Path(added.source_file.iloc[0])) == manifest['raw_source_sha256']
    obs = pd.read_csv(output/'training_observation_ledger.csv')
    base = root/'outputs/disease_improvement/station_repairs_20261022_v4'
    prior = root/'outputs/disease_improvement/history_anchor_20261022'
    original = pd.read_csv(base/'observation_ledger.csv')
    old_ids, added_ids = set(original.global_target_index), set(added.global_target_index)
    assert len(added) == 18 and added.field_id.nunique() == 2
    assert added_ids.isdisjoint(old_ids) and obs.global_target_index.is_unique
    pd.testing.assert_frame_equal(obs.iloc[:len(original)][original.columns], original,
                                  check_dtype=False)
    pd.testing.assert_frame_equal(obs.iloc[len(original):][added.columns].reset_index(drop=True),
                                  added, check_dtype=False)
    original_splits = {s['fold']: s for s in json.loads((base/'spatial_membership.json').read_text())}
    memberships = json.loads((output/'split_membership.json').read_text())
    indexed = obs.set_index('global_target_index')
    for split in memberships:
        train_ids, test_ids = set(split['train']), set(split['test'])
        assert train_ids.isdisjoint(test_ids) and test_ids <= old_ids
        train, test = indexed.loc[split['train']], indexed.loc[split['test']]
        assert test.season_year.nunique() == 1
        year = int(test.season_year.iloc[0])
        assert train.season_year.lt(year).all()
        assert set(split['added_train_ids']) == train_ids & added_ids
        if split['fold'].startswith('temporal'):
            assert train_ids == set(obs.loc[obs.season_year.lt(year), 'global_target_index'])
        else:
            old_split = original_splits[split['fold']]
            assert train_ids & old_ids == set(old_split['train'])
            assert test_ids == set(old_split['test'])
            new_train = train.loc[train.index.isin(added_ids)]
            if len(new_train):
                distances = distances_km(new_train[['latitude','longitude']], test[['latitude','longitude']])
                assert (distances.min(axis=1) >= 50).all()
                fold = int(re.search(r'sitefold(\d+)', split['fold']).group(1))
                assert new_train.site_id.map(registration['site_assignment']).ne(fold).all()
    for filename, date_column in [('feature_provenance.csv', 'weather_end'),
                                  ('stage_provenance.csv', 'weather_end'),
                                  ('stage_provenance.csv', 'latest_stage_date')]:
        provenance = pd.read_csv(extra/filename)
        dates = pd.to_datetime(provenance[date_column])
        query_dates = pd.to_datetime(indexed.loc[provenance.global_target_index, 'date']).reset_index(drop=True)
        assert (dates.isna() | dates.lt(query_dates)).all()
    arrays = np.load(output/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    names = json.loads((root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json').read_text())['history_names']
    history = pd.DataFrame(arrays['history'], columns=names)
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    spec = importlib.util.spec_from_file_location('frozen_augmented_models', output/'code_snapshot/analysis/disease_improvement/models.py')
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    metrics = pd.read_csv(output/'metrics.csv')
    model_count = member_count = 0
    for label, filename in [('temporal','assessment_predictions.csv'), ('spatial_50km','spatial_predictions.csv')]:
        frame, old = pd.read_csv(output/filename), pd.read_csv(prior/filename)
        assert set(frame.global_target_index) == set(old.global_target_index)
        assert not frame.global_target_index.duplicated().any()
        aligned = old.set_index('global_target_index').loc[frame.global_target_index].reset_index()
        pd.testing.assert_frame_equal(frame[aligned.columns], aligned, check_dtype=False)
        for name, blend in registration['blends'].items():
            np.testing.assert_allclose(frame[name], frame[blend['components']].to_numpy() @ blend['weights'], atol=1e-10)
        for name in list(registration['recipes']) + list(registration['blends']):
            assert frame[name].between(0,100).all()
            recalculated = score_predictions(frame, name)
            saved = metrics.loc[metrics.evaluation.eq(label)&metrics.model.eq(name)].iloc[0]
            for key, value in recalculated.items():
                np.testing.assert_allclose(value, saved[key], atol=1e-9)
        if label != 'temporal':
            continue
        for year, part in frame.groupby('season_year'):
            eligible_ids = set(obs.loc[obs.season_year.lt(year),'global_target_index'])
            selected = position.loc[part.global_target_index].to_numpy()
            for name in registration['recipes']:
                fitted = joblib.load(output/f'{year}_{name}.joblib')
                assert set(fitted['training_ids']) == eligible_ids
                predictions = []
                for member in fitted['members']:
                    ids = set(member['training_ids'])
                    assert ids <= eligible_ids
                    member_train = obs.loc[obs.global_target_index.isin(ids)]
                    whole_clusters = obs.loc[obs.season_year.lt(year)&obs.coordinate_year.isin(member_train.coordinate_year)]
                    assert ids == set(whole_clusters.global_target_index)
                    design = arrays['station' if name.endswith('_split') else 'shared']
                    predictions.append(frozen.predict_candidate(member, design[selected], history.iloc[selected],
                                                                part.drop(columns='observed_percent')))
                    member_count += 1
                np.testing.assert_allclose(np.mean(predictions,axis=0), part[name], atol=1e-10)
                model_count += 1
    result = dict(verified=True, original_evaluation_assessments=2017, added_training_assessments=len(added),
        original_training_records_unchanged=True, original_splits_unchanged=True,
        added_records_obey_year_and_spatial_boundaries=True, saved_models_replayed=model_count,
        component_models_replayed=member_count, metrics_recomputed=len(metrics),
        stage_and_weather_strictly_prior=True, independent_external_validation=False,
        publication_model_promoted=False, verifier_sha256=sha(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(ROOT,args.output),indent=2))
