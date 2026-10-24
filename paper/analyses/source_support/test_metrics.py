"""Hand-calculated endpoint and denominator contracts."""
import importlib

import numpy as np
import pandas as pd
import pytest


def metrics():
    spec = importlib.util.find_spec("paper.analyses.source_support.metrics")
    assert spec is not None, "Standalone final metrics implementation is missing"
    return importlib.import_module(spec.name)


def test_unequal_fields_leaves_and_repeated_assessments_have_equal_coordinate_mass():
    m = metrics()
    frame = pd.DataFrame(dict(coordinate_year=["A", "A", "A", "A", "B"],
        field_id=["f1", "f1", "f1", "f2", "f3"], leaf_index=[0, 0, 1, 0, 0],
        observed_percent=[0., 0., 0., 0., 0.], prediction=[1., 3., 5., 7., 9.]))
    np.testing.assert_array_equal(m.hierarchical_weights(frame), [1/8, 1/8, 1/4, 1/2, 1])
    score = m.disease_scores(frame, "prediction")
    assert score["rmse"] == pytest.approx(np.sqrt(56.5))
    assert score["bias"] == pytest.approx(7.125)


def test_irregular_observed_window_uses_trapezoids_without_endpoint_extrapolation():
    m = metrics()
    frame = pd.DataFrame(dict(partition=["test"] * 4, field_id=["f1"] * 3 + ["f2"],
        coordinate_year=["A"] * 4, source=["BASF"] * 4, leaf_index=[0] * 4,
        endpoint_series=["s1"] * 3 + ["s2"],
        date=["2020-01-01", "2020-01-03", "2020-01-08", "2020-01-20"],
        observed_percent=[0., 20., 10., 90.], prediction=[10., 30., 10., 0.]))
    result = m.observed_windows(frame, ["prediction"])
    assert len(result) == 1
    row = result.iloc[0]
    assert row["span_days"] == 7
    assert row["observed_audpc"] == 95
    assert row["prediction_audpc"] == 140
    assert row["observed_percent"] == pytest.approx(95/7)


def test_missing_candidate_median_retains_reference_support_and_nan_full_point():
    m = metrics()
    baseline = np.asarray([[[0., .6, .8], [0., .55, .7], [0., .5, .6], [0., .5, .55]]])
    candidate = baseline.copy()
    candidate[:, 3] = [0., .1, .2]
    target = np.asarray([[1, 1, 1, 1]])
    cycles = pd.DataFrame(dict(cycle_id=["a"], PEP_ID=[1], historical_station_group=["test"]))
    rows = m.phenology_metrics(dict(baseline=baseline, candidate=candidate), target, cycles, references=("baseline",))
    row = rows.loc[rows.model.eq("candidate") & rows.cohort.eq("all")].iloc[0]
    assert row["observed_events"] == row["common_resolved_events"] == 4
    assert row["duration_pairs"] == 1
    assert row["lost_reference_events"] == row["unresolved_observed_events"] == 1
    assert np.isnan(row["stage_equal_mae"])
    assert np.isnan(row["terminal_mae"])
    assert np.isnan(row["duration_mae"])
    assert np.isfinite(row["stage_equal_crps"])
    guard = m.promotion_guards(candidate, target, baseline)
    assert not guard["finite_metrics"]
    assert not guard["no_lost_baseline_events"]
    assert not guard["no_lost_baseline_pairs"]


def test_crps_uses_all_observed_integer_days_including_right_censored_events():
    m = metrics()
    cdf = np.asarray([[[.2, .4, .7], [.1, .2, .4], [0., .1, .2], [0., 0., .1]]])
    target = np.asarray([[1, 5, -1, 2]])
    actual = m.restricted_crps(cdf, target)
    np.testing.assert_allclose(actual[:, [0, 1, 3]], [[.49, .21, .81]])
    assert np.isnan(actual[0, 2])


def test_retained_input_integrity_rejects_changed_bytes_of_equal_length(tmp_path):
    import hashlib
    import json
    m = metrics()
    data = tmp_path / "predictions.csv"
    data.write_bytes(b"value\n1\n")
    path_manifest = tmp_path / "manifest.json"
    path_manifest.write_text(json.dumps(dict(files={"original/predictions.csv": dict(path=str(data),
        bytes=data.stat().st_size, sha256=hashlib.sha256(data.read_bytes()).hexdigest())})))
    data.write_bytes(b"value\n2\n")
    with pytest.raises(ValueError, match="changed"):
        m.RetainedFiles(path_manifest).resolve("original/predictions.csv")
