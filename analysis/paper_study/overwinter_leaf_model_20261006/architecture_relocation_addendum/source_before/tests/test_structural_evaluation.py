"""Guards for retrospective nested evaluation, without costly model fitting."""

import importlib

import numpy as np
import pandas as pd
import pytest


def evaluation_module():
    try:
        return importlib.import_module("analysis.paper_study.structural_evaluation.membership")
    except ModuleNotFoundError:
        pytest.fail("Retrospective nested membership guards are not implemented")


@pytest.fixture
def histories():
    # A source unit can recur in a second source; physical location is the
    # cross-source holdout identity and all assessments belong to its history.
    rows = []
    for site, years in {
        "A": [2014, 2016, 2019], "B": [2014, 2017],
        "C": [2015, 2018], "D": [2016, 2017], "E": [2017, 2018],
        "F": [2018, 2019], "G": [2019], "H": [2015, 2016],
        "I": [2014, 2018], "J": [2016, 2019],
    }.items():
        for year in years:
            for source in (["basf", "corteva"] if site == "A" else ["corteva"]):
                field = f"{source}|unit-{site}|{year}"
                for day, value in [("05-01", 0.), ("06-01", 12.)]:
                    rows.append(dict(site_id=site, season_year=year, dataset_id=source,
                        field_id=field, endpoint_series=f"unit-{site}|LEAF 1",
                        coordinate_year=f"{site}|{year}", date=f"{year}-{day}",
                        leaf_index=0, value=value))
    return pd.DataFrame(rows)


def test_spatial_outer_folds_hold_out_complete_location_histories(histories):
    module = evaluation_module()
    folds = module.spatial_folds(histories, n_splits=5, seed=20261006)
    assert len(folds) == 5
    heldout = []
    for fold in folds:
        train, test = histories.iloc[fold.train], histories.iloc[fold.test]
        assert set(train.site_id).isdisjoint(test.site_id)
        assert set(train.field_id).isdisjoint(test.field_id)
        assert set(np.r_[fold.train, fold.test]) == set(range(len(histories)))
        for site in test.site_id.unique():
            assert len(test[test.site_id.eq(site)]) == len(histories[histories.site_id.eq(site)])
        heldout.extend(fold.test)
    assert sorted(heldout) == list(range(len(histories)))


def test_forward_folds_exclude_future_years_and_previously_observed_sites(histories):
    module = evaluation_module()
    folds = module.chronological_folds(histories, years=[2016, 2017, 2018, 2019])
    expected_sites = {2016: {"D", "J"}, 2017: {"E"}, 2018: {"F"}, 2019: {"G"}}
    for fold in folds:
        train, test = histories.iloc[fold.train], histories.iloc[fold.test]
        assert train.season_year.lt(fold.test_year).all()
        assert test.season_year.eq(fold.test_year).all()
        assert set(test.site_id) == expected_sites[fold.test_year]
        assert set(test.site_id).isdisjoint(train.site_id)
        assert set(fold.train).isdisjoint(fold.test)
        assert set(fold.excluded) == set(range(len(histories))) - set(fold.train) - set(fold.test)


def test_inner_folds_cannot_include_outer_holdout_or_split_field_seasons(histories):
    module = evaluation_module()
    outer = module.spatial_folds(histories, n_splits=5)[0]
    inners = module.inner_folds(histories, outer, n_splits=3)
    heldout = []
    for inner in inners:
        module.guard_nested_membership(histories, outer, inner)
        assert set(inner.train) | set(inner.test) == set(outer.train)
        assert set(inner.train).isdisjoint(outer.test)
        assert set(inner.test).isdisjoint(outer.test)
        heldout.extend(inner.test)
    assert sorted(heldout) == sorted(outer.train)
    contaminated = module.EvaluationFold("bad", "inner", np.r_[inners[0].train, outer.test[:1]], inners[0].test)
    with pytest.raises(ValueError, match="outer"):
        module.guard_nested_membership(histories, outer, contaminated)
    split_field = module.EvaluationFold("bad", "inner", inners[0].train[1:], inners[0].test)
    with pytest.raises(ValueError, match="complete|partition|field"):
        module.guard_nested_membership(histories, outer, split_field)


def test_membership_guard_rejects_duplicate_and_overlapping_rows(histories):
    module = evaluation_module()
    outer = module.spatial_folds(histories, n_splits=5)[0]
    valid = module.inner_folds(histories, outer, n_splits=3)[0]
    duplicate = module.EvaluationFold("bad", "inner", np.r_[valid.train, valid.train[0]], valid.test)
    with pytest.raises(ValueError, match="(?i)unique"):
        module.guard_nested_membership(histories, outer, duplicate)
    overlapping = module.EvaluationFold("bad", "inner", valid.train, np.r_[valid.test, valid.train[:2]])
    with pytest.raises(ValueError, match="overlap"):
        module.guard_nested_membership(histories, outer, overlapping)


def test_forward_membership_guard_rejects_a_future_training_year(histories):
    module = evaluation_module()
    valid = module.chronological_folds(histories, years=[2016])[0]
    future = np.flatnonzero(histories.season_year.eq(2019) & histories.site_id.eq("A"))
    invalid = module.EvaluationFold("bad", "forward", np.r_[valid.train, future], valid.test, test_year=2016)
    with pytest.raises(ValueError, match="future|chronolog"):
        module.guard_outer_membership(histories, invalid)


def runner_module():
    try:
        return importlib.import_module("analysis.paper_study.structural_evaluation.run")
    except ModuleNotFoundError:
        pytest.fail("Retrospective nested fit isolation and recovery are not implemented")


def test_duration_rate_uses_unique_training_weather_and_ignores_heldout_values():
    module = runner_module()
    metadata = pd.DataFrame([
        dict(field_index=0, site_id="train", first_forcing_date="2017-01-01", last_forcing_date="2017-01-03"),
        dict(field_index=1, site_id="train", first_forcing_date="2017-01-01", last_forcing_date="2017-01-03"),
    ])
    weather = pd.DataFrame([
        dict(location_id="train", date="2017-01-01", tmean_c=10., precipitation_mm=4., rain_hours_gt_0_1mm=2.),
        dict(location_id="train", date="2017-01-02", tmean_c=10., precipitation_mm=2., rain_hours_gt_0_1mm=2.),
        dict(location_id="train", date="2017-01-03", tmean_c=0., precipitation_mm=100., rain_hours_gt_0_1mm=1.),
        dict(location_id="heldout", date="2017-01-01", tmean_c=10., precipitation_mm=900., rain_hours_gt_0_1mm=1.),
    ])
    rate = module.estimate_training_rain_rate(metadata, weather)
    assert rate["rain_rate_mm_hour"] == pytest.approx(1.5)
    assert rate["unique_training_location_dates"] == 3
    assert rate["eligible_positive_rain_days"] == 2
    altered = weather.copy()
    altered.loc[altered.location_id.eq("heldout"), "precipitation_mm"] = 99999.
    assert module.estimate_training_rain_rate(metadata, altered) == rate


def test_fit_subsets_remove_withheld_targets_and_remap_all_field_indices():
    from model.seasonal_septoria.field_data import FieldData

    module = runner_module()
    targets = pd.DataFrame([
        dict(field_index=0, field_id="old", value=90., day_index=2, leaf_index=0),
        dict(field_index=1, field_id="heldout", value=80., day_index=2, leaf_index=0),
        dict(field_index=2, field_id="selected", value=5., day_index=1, leaf_index=0),
        dict(field_index=2, field_id="selected", value=15., day_index=2, leaf_index=0),
    ])
    metadata = pd.DataFrame(dict(field_index=[0, 1, 2], field_id=["old", "heldout", "selected"]))
    array = np.array([[1., 2.], [3., 4.], [5., 6.]])
    data = FieldData(array, array, array, np.ones((3, 2, 8), bool),
        np.zeros((3, 2, 8)), metadata, targets, pd.DataFrame())
    isolated = module.subset_fields(data, np.array([2, 3]))
    assert isolated.targets.value.tolist() == [5., 15.]
    assert isolated.targets.field_index.tolist() == [0, 0]
    assert isolated.targets.global_target_index.tolist() == [2, 3]
    assert isolated.metadata.field_id.tolist() == ["selected"]
    assert isolated.temperature.tolist() == [[5., 6.]]
    with pytest.raises(ValueError, match="complete"):
        module.subset_fields(data, np.array([2]))
    redacted = module.subset_fields(data, np.array([2, 3]), redact_values=True)
    assert redacted.targets.value.isna().all()
    assert targets.value.tolist() == [90., 80., 5., 15.]


def test_completed_fit_recovery_requires_exact_hash_identity(tmp_path):
    module = runner_module()
    path = tmp_path / "fit.json"
    identity = {"input_sha256": "abc", "membership_sha256": "def"}
    assert module.read_completed_fit(path, identity) is None
    module.write_frozen_json(path, dict(status="fit_complete", identity=identity,
        fitted=dict(parameters=dict(initiation=.001))))
    assert module.read_completed_fit(path, identity)["parameters"]["initiation"] == .001
    with pytest.raises(ValueError, match="identity|hash"):
        module.read_completed_fit(path, dict(identity, input_sha256="changed"))
    with pytest.raises(FileExistsError):
        module.write_frozen_json(path, {})


def test_preparation_plan_is_frozen_without_claiming_experiment_completion(tmp_path, histories):
    module = runner_module()
    outers = evaluation_module().spatial_folds(histories)
    plan = module.freeze_membership_plan(tmp_path, histories, outers)
    assert len(plan) == 5
    membership = pd.read_csv(tmp_path / "nested_membership_before_fitting.csv")
    assert len(membership.query("level == 'outer' and role == 'validation'")) == len(histories)
    assert not (tmp_path / "receipt.json").exists()


def test_frozen_csv_recovery_does_not_duplicate_results(tmp_path):
    module = runner_module()
    path = tmp_path / "metrics.csv"
    frame = pd.DataFrame(dict(model=["canopy", "seir"], rmse=[12., 15.]))
    assert hasattr(module, "save_frozen_csv"), "CSV recovery must verify a frozen result rather than append it"
    module.save_frozen_csv(path, frame)
    module.save_frozen_csv(path, frame)
    assert pd.read_csv(path).rmse.tolist() == [12., 15.]
    with pytest.raises(ValueError, match="frozen|differ"):
        module.save_frozen_csv(path, pd.DataFrame(dict(model=["canopy", "seir"], rmse=[99., 15.])))


def test_summary_keeps_final_leaf_endpoints_and_bootstraps_full_site_histories(histories):
    module = runner_module()
    rows = []
    for family, error in [("structural_canopy", 1.), ("original_seir", 3.)]:
        part = histories.copy()
        part["model"] = family
        part["evaluation_kind"], part["report_scope"] = "spatial", "pooled"
        part["predicted_percent"] = part.value + error
        rows.append(part)
    metrics, paired = module.summarize_predictions(pd.concat(rows), bootstrap_draws=200)
    final = metrics.query("endpoint == 'final_numeric_assessment' and leaf_scope == 'all_ordinal_leaves'")
    assert final.set_index("model").rmse.to_dict() == {"original_seir": 3., "structural_canopy": 1.}
    assert final.n.tolist() == [23, 23]
    assert paired.cluster_count.unique().tolist() == [10]
    assert paired.rmse_difference.tolist() == [-2., -2., -2., -2.]
    assert paired.rmse_difference_lower.tolist() == [-2., -2., -2., -2.]


def test_prediction_boundary_rejects_unredacted_disease_values():
    from types import SimpleNamespace

    module = runner_module()
    data = SimpleNamespace(targets=pd.DataFrame(dict(value=[1., 3.])))
    with pytest.raises(ValueError, match="redacted"):
        module.predict_record("original_seir", data, None, None, {})


def test_interrupted_prediction_write_leaves_no_completed_archive(tmp_path, monkeypatch):
    module = runner_module()
    path = tmp_path / "predictions.parquet"

    # A parquet writer can fail after opening and writing its destination.
    # Replace only that I/O boundary and observe the archive's recovery state.
    def interrupted_writer(frame, destination, **kwargs):
        from pathlib import Path
        Path(destination).write_bytes(b"partial parquet content")
        raise RuntimeError("interrupted parquet write")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", interrupted_writer)
    with pytest.raises(RuntimeError, match="interrupted"):
        module._save_frame(path, pd.DataFrame(dict(value=[1., 2.])))
    assert not path.exists()


def test_source_changes_prevent_completion_receipt(tmp_path):
    module = runner_module()
    path = tmp_path / "source.py"
    path.write_text("initiation_max = .1\n")
    assert hasattr(module, "verify_dependencies"), "Completion must verify source identities again"
    hashes = {str(path): module.sha(path)}
    module.verify_dependencies(hashes)
    path.write_text("initiation_max = 1.\n")
    with pytest.raises(ValueError, match="changed|hash"):
        module.verify_dependencies(hashes)
