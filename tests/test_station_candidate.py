import numpy as np
import pandas as pd
import pytest


def test_extra_training_preserves_old_members_and_excludes_future_near_and_held_sites():
    from models.monitoring.station_candidate import extend_spatial_split
    frame = pd.DataFrame(dict(global_target_index=[1, 2, 10, 11, 12, 13],
        season_year=[2017, 2019, 2018, 2019, 2018, 2018],
        site_id=['old', 'test', 'eligible', 'future', 'near', 'held'],
        latitude=[48., 50., 47., 47., 50.1, 47.], longitude=[1.]*6))
    assignment = dict(old=1, test=0, eligible=1, future=1, near=1, held=0)
    train, test, added = extend_spatial_split(frame, [10, 11, 12, 13],
        dict(fold='2019:sitefold0:buffer50km', train=[1], test=[2]), assignment)
    assert train == [1, 10] and test == [2] and added == [10]
    # Disease values do not enter eligibility or ordering.
    frame['observed_percent'] = [100., 0., 100., 100., 0., 100.]
    assert extend_spatial_split(frame, [10, 11, 12, 13],
        dict(fold='2019:sitefold0:buffer50km', train=[1], test=[2]), assignment) == (train, test, added)


def test_bundle_prediction_rejects_labels_and_out_of_domain_queries(monkeypatch):
    from models.monitoring import station_candidate as module
    monkeypatch.setattr(module, 'predict_cluster_ensemble', lambda *a: np.array([20.]))
    monkeypatch.setattr(module, 'predict_history_ensemble', lambda *a: np.array([80.]))
    query = pd.DataFrame(dict(source=['BASF'], leaf_index=[1], season_year=[2020], date=['2020-06-01']))
    bundle = dict(process={}, history={}, minimum_assessment_year=2020,
                  history_names=['a'], blend_weights=[.5, .5])
    history = pd.DataFrame(dict(a=[0.]))
    assert module.predict_bundle(bundle, np.zeros((1, 37)), np.zeros((1, 53)), history, query)[0] == 50.
    with pytest.raises(ValueError, match='labels'):
        module.predict_bundle(bundle, np.zeros((1, 37)), np.zeros((1, 53)), history,
                              query.assign(observed_percent=90.))
    for changed in [query.assign(source='external'), query.assign(leaf_index=9), query.assign(season_year=2019),
                    query.assign(date='2019-06-01'), query.assign(date='invalid')]:
        with pytest.raises(ValueError, match='domain'):
            module.predict_bundle(bundle, np.zeros((1, 37)), np.zeros((1, 53)), history, changed)


def test_training_union_schema_preserves_source_values_and_missing_extra_columns():
    from models.monitoring.verify_station_candidate import verify_source_slice
    original = pd.DataFrame(dict(global_target_index=[1], first_only=['a']))
    addition = pd.DataFrame(dict(global_target_index=[2], added_only=[10.]))
    union = pd.concat([original, addition], ignore_index=True)
    verify_source_slice(union.iloc[:1], original)
    verify_source_slice(union.iloc[1:], addition)
    union.loc[0, 'added_only'] = 99.
    with pytest.raises(AssertionError):
        verify_source_slice(union.iloc[:1], original)


def test_station_override_rejects_reordered_targets_and_keeps_complete_schema(tmp_path):
    from models.monitoring.station_candidate import load_station_design
    path = tmp_path/'predictors.npz'
    values = np.ones((2, 53))
    np.savez(path, target_ids=[2, 1], station=values)
    with pytest.raises(ValueError, match='identity'):
        load_station_design(path, [1, 2])
    np.testing.assert_array_equal(load_station_design(path, [2, 1]), values)
    np.savez(path, target_ids=[2, 1], station=np.ones((2, 52)))
    with pytest.raises(ValueError, match='schema'):
        load_station_design(path, [2, 1])
