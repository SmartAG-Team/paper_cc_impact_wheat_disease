"""Leakage and serialization boundaries for retrospective empirical baselines."""

import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def benchmark_module():
    try:
        return importlib.import_module("analysis.paper_study.structural_publication_20261006.nested_benchmarks.run_benchmarks")
    except ModuleNotFoundError:
        pytest.fail("The isolated nested empirical benchmarks are not implemented")


def training_targets():
    return pd.DataFrame([
        dict(field_id="field-A", coordinate_year="A|2017", endpoint_series="A|LEAF 1",
            date=pd.Timestamp("2017-05-01"), leaf_index=0, value=0.),
        dict(field_id="field-A", coordinate_year="A|2017", endpoint_series="A|LEAF 1",
            date=pd.Timestamp("2017-06-01"), leaf_index=0, value=20.),
        dict(field_id="field-B", coordinate_year="B|2017", endpoint_series="B|LEAF 3",
            date=pd.Timestamp("2017-06-01"), leaf_index=2, value=60.),
    ])


def test_means_use_combined_final_all_training_weights_and_pooled_fallback():
    module = benchmark_module()
    target = training_targets()
    pooled = module.fit_empirical("weighted_training_mean", target, None)
    by_leaf = module.fit_empirical("leaf_rank_training_mean", target, None)
    assert pooled["pooled_mean"] == 37.5
    assert by_leaf["leaf_means"] == {"0": 15., "2": 60.}
    prediction_targets = pd.DataFrame(dict(value=[np.nan] * 3, leaf_index=[0, 2, 1]))
    assert module.predict_empirical(by_leaf, prediction_targets, None).tolist() == [15., 60., 37.5]


def test_serialized_ridge_preserves_train_only_standardization_and_clipping():
    module = benchmark_module()
    from model.seasonal_septoria.empirical import WeightedRidge
    from model.seasonal_septoria.refinement import calibration_weights

    target = training_targets()
    x = pd.DataFrame(dict(tmean_7d=[0., 2., 4.]))
    fitted = module.fit_empirical("weather_development_ridge", target, x, penalty=1.)
    reference = WeightedRidge(1.).fit(x, target.value, calibration_weights(target, .5))
    heldout = pd.DataFrame(dict(tmean_7d=[-1e5, 1., 1e5]))
    actual = module.predict_empirical(json.loads(json.dumps(fitted)),
        pd.DataFrame(dict(value=[np.nan] * 3, leaf_index=[0, 0, 0])), heldout)
    np.testing.assert_allclose(actual, reference.predict(heldout), atol=1e-12, rtol=0)
    assert actual[0] == 0. and actual[-1] == 100.
    assert fitted["mean"] == reference.mean.tolist()
    assert fitted["mean"] == [2.75]


def test_prediction_boundary_rejects_withheld_disease_labels():
    module = benchmark_module()
    with pytest.raises(ValueError, match="redact"):
        module.predict_empirical(dict(family="weighted_training_mean", pooled_mean=10.),
            pd.DataFrame(dict(value=[90.], leaf_index=[0])), None)


def test_feature_boundary_ignores_stage_identity_disease_and_future_forcing():
    module = benchmark_module()
    from model.seasonal_septoria.field_data import FieldData

    target = pd.DataFrame(dict(field_index=[0, 1], field_id=["A", "B"], day_index=[3, 3],
        leaf_index=[0, 0], date=pd.to_datetime(["2017-10-03", "2017-10-03"]),
        value=[5., 90.], stage_from=[10, 99], stage_to=[11, 99], site_id=["A", "B"], country=["X", "Y"]))
    metadata = pd.DataFrame(dict(field_index=[0, 1], field_id=["A", "B"], forcing_days=[8, 8],
        sowing_date=["2017-10-01"] * 2, latitude=[50., 50.]))
    data = FieldData(np.full((2, 8), 12.), np.full((2, 8), 85.), np.full((2, 8), 2.),
        np.ones((2, 8, 8), bool), np.zeros((2, 8, 8)), metadata, target, pd.DataFrame())
    data.maximum_temperature = np.full((2, 8), 16.)
    root = Path(__file__).resolve().parents[4]
    phenology = json.loads((root / "process_model/parameters/calibration.json").read_text())
    baseline = module.safe_features(data, np.array([0]), phenology)
    data.targets["value"], data.targets["stage_from"], data.targets["stage_to"] = 99., 99, 99
    data.targets["site_id"], data.targets["country"] = "changed", "changed"
    for name in ["temperature", "humidity", "rain", "maximum_temperature"]:
        getattr(data, name)[0, 3:] = 0.
        getattr(data, name)[1, :] = 0.
    pd.testing.assert_frame_equal(baseline, module.safe_features(data, np.array([0]), phenology))
    assert not set(baseline) & {"value", "stage_from", "stage_to", "site_id", "country", "latitude", "longitude"}


def test_feature_columns_reject_unknown_identity_or_disease_inputs():
    module = benchmark_module()
    with pytest.raises(ValueError, match="feature"):
        module.validate_feature_columns(["tmean_7d", "site_id"])


def test_frozen_membership_reconstruction_rejects_changed_target_identity():
    module = benchmark_module()
    from analysis.paper_study.structural_evaluation.membership import spatial_folds, inner_folds, membership_records

    target = pd.DataFrame(dict(site_id=list("ABCDEFGHIJ"), field_id=list("abcdefghij"),
        season_year=[2017] * 10, date=["2017-06-01"] * 10,
        endpoint_series=["LEAF 1"] * 10, coordinate_year=[f"{x}|2017" for x in "ABCDEFGHIJ"]))
    outer = spatial_folds(target, n_splits=5)[0]
    membership = membership_records(target, outer, inner_folds(target, outer))
    restored, inners = module.reconstruct_memberships(target, membership, require_nine=False)
    assert len(restored) == 1 and len(inners[outer.name]) == 3
    assert restored[0].test.tolist() == outer.test.tolist()
    changed = target.copy(); changed.loc[0, "field_id"] = "a-different-field"
    with pytest.raises(ValueError, match="identity|membership"):
        module.reconstruct_memberships(changed, membership, require_nine=False)
