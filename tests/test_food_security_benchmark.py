"""Scientific invariants for the fresh autonomous retrospective benchmark."""
import hashlib
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def module():
    name = "paper.analyses.food_security.benchmark"
    assert importlib.util.find_spec(name) is not None, "autonomous benchmark is not implemented"
    return importlib.import_module(name)


NAMES = [f"{name}_{'log1p_' if name == 'rain' else ''}mean_{width}d"
         for width in (7, 21, 42) for name in ("temperature", "humidity", "rain")]
NAMES += ["elapsed_days_scaled", "calendar_sin", "calendar_cos"]
NAMES += [f"crop_cdf_{stage}" for stage in (10, 31, 51, 87)]


def fixture_frame():
    rows, features = [], []
    for year in (2014, 2015, 2016):
        for site in ("a", "b", "c", "d", "e", "f"):
            for day in (1, 4, 11):
                rows.append(dict(global_target_index=len(rows), site_id=site,
                    field_id=f"{site}-{year}", coordinate_year=f"{site}|{year}",
                    season_year=year, source="source", leaf_index=0,
                    endpoint_series="flag", date=f"{year}-05-{day:02}", day_index=day,
                    partition="test", observed_percent=float(day * 2 + ord(site) - 97)))
                features.append([10., 80., .5, 9., 78., .7, 8., 75., .8, day / 330, .2, -.9])
    return pd.DataFrame(rows), np.asarray(features)


def test_first_twelve_features_exclude_crop_values_and_use_one_based_days():
    b = module()
    observations = pd.DataFrame(dict(field_index=[0, 0], day_index=[1, 3]))
    features = np.zeros((1, 3, 16))
    features[0, 0, :12] = np.arange(12)
    features[0, 2, :12] = np.arange(12) + 50
    arrays = dict(features=features, valid=np.ones((1, 3), bool),
                  horizons=np.array([3]), feature_names=np.array(NAMES))
    selected, available = b.assessment_features(observations, arrays)
    np.testing.assert_array_equal(selected, features[0, [0, 2], :12])
    assert available.tolist() == [True, True]
    arrays["features"][..., 12:] = np.nan
    changed, available = b.assessment_features(observations, arrays)
    np.testing.assert_array_equal(changed, selected)
    assert available.all()
    arrays["feature_names"][0] = "prior_observed_severity"
    with pytest.raises(ValueError, match="feature schema"):
        b.assessment_features(observations, arrays)


def test_folds_exclude_same_sites_and_later_years_and_are_outcome_independent():
    b = module()
    frame, features = fixture_frame()
    predictions, folds, membership = b.autonomous_predictions(frame, features)
    assert set(predictions.global_target_index) == set(frame.loc[frame.season_year.gt(2014), "global_target_index"])
    assert predictions[b.MODELS].notna().all().all()
    for _, fold in folds.loc[folds.test_rows.gt(0)].iterrows():
        rows = membership.loc[membership.fold_id.eq(fold.fold_id)]
        train = rows.loc[rows.role.eq("train")]
        test = rows.loc[rows.role.eq("test")]
        assert train.season_year.max() < test.season_year.min()
        assert set(train.site_id).isdisjoint(test.site_id)
    changed = frame.copy()
    changed.loc[changed.season_year.eq(2016), "observed_percent"] = 99
    altered, other_folds, other_members = b.autonomous_predictions(changed, features)
    np.testing.assert_allclose(predictions[b.MODELS], altered[b.MODELS])
    pd.testing.assert_frame_equal(folds, other_folds)
    pd.testing.assert_frame_equal(membership, other_members)


def test_unseen_training_leaves_use_shared_visible_pooled_fallback():
    b = module()
    frame, features = fixture_frame()
    frame.loc[frame.season_year.eq(2016), "leaf_index"] = 6
    predictions, folds, members = b.autonomous_predictions(frame, features)
    new = predictions.loc[predictions.season_year.eq(2016)]
    assert new.unseen_leaf.all()
    assert new.prediction_status.eq("unseen_leaf_pooled_fallback").all()
    assert new[b.MODELS].notna().all().all()
    for name in b.MODELS[1:]:
        np.testing.assert_array_equal(new[name], new[b.MODELS[0]])


def test_unavailable_training_fold_preserves_records_and_undefined_metrics():
    b = module()
    frame, features = fixture_frame()
    take = frame.site_id.eq("a")
    predictions, folds, members = b.autonomous_predictions(frame.loc[take].reset_index(drop=True), features[take])
    assert len(predictions) == 6
    assert predictions[b.MODELS].isna().all().all()
    summary = b.score_table(predictions)
    assert summary.rmse.isna().all()
    assert summary.n_available.eq(0).all()
    assert summary.status.eq("no_paired_support").all()


def test_windows_integrate_actual_dates_without_endpoint_extrapolation():
    b = module()
    frame, _ = fixture_frame()
    frame = frame.loc[frame.site_id.eq("a") & frame.season_year.eq(2015)].copy()
    frame["observed_percent"] = [0., 30., 60.]
    frame[b.MODELS[0]] = [10., 40., 70.]
    windows = b.observed_windows(frame, [b.MODELS[0]])
    assert len(windows) == 1
    assert windows.span_days.iloc[0] == 10
    assert windows.observed_audpc.iloc[0] == 360
    assert windows.observed_percent.iloc[0] == 36
    assert windows[b.MODELS[0] + "_audpc"].iloc[0] == 460
    frame.loc[frame.index[1], b.MODELS[0]] = np.nan
    missing = b.observed_windows(frame, [b.MODELS[0]])
    assert np.isnan(missing[b.MODELS[0] + "_audpc"].iloc[0])


def test_conditional_paired_intervals_preserve_site_clusters_and_low_support():
    b = module()
    frame, _ = fixture_frame()
    for name in b.MODELS:
        frame[name] = frame.observed_percent + 5
    result = b.paired_intervals(frame, bootstrap_draws=20)
    supported = result.loc[result.n_available.gt(0)]
    assert supported.value.eq(0).all()
    assert supported.lower.eq(0).all() and supported.upper.eq(0).all()
    assert result.loc[result.n_available.eq(0), "value"].isna().all()
    one_site = b.paired_intervals(frame.loc[frame.site_id.eq("a")], bootstrap_draws=20)
    assert one_site.lower.isna().all()
    assert one_site.loc[one_site.n_available.gt(0), "status"].eq("insufficient_sites").all()


def test_actual_training_dates_cannot_exceed_the_declared_season():
    b = module()
    frame, features = fixture_frame()
    frame.loc[0, "date"] = "2016-05-01"
    with pytest.raises(ValueError, match="season|date"):
        b.autonomous_predictions(frame, features)


def retained_fixture(root: Path):
    frame, _ = fixture_frame()
    fields = frame[["field_id", "site_id", "season_year", "source"]].drop_duplicates().reset_index(drop=True)
    fields["field_index"] = np.arange(len(fields))
    fields["sowing_date"] = fields.season_year.map(lambda y: f"{y}-05-01")
    fields["first_forcing_date"] = fields.sowing_date
    observations = frame.drop(columns="season_year").merge(fields[["field_id", "field_index"]], on="field_id", validate="many_to_one")
    target = root / "data/final_study/evidence/disease_joint_forecast/inputs"
    target.mkdir(parents=True)
    observations.to_csv(target / "observations.csv", index=False)
    fields.to_csv(target / "fields.csv", index=False)
    features = np.zeros((len(fields), 11, 16))
    features[..., 0] = 10; features[..., 1] = 80; features[..., 2] = .5
    features[..., 9] = np.arange(1, 12) / 330
    np.savez(target / "features.npz", features=features, valid=np.ones((len(fields), 11), bool),
        field_ids=fields.field_id.to_numpy(str), horizons=np.full(len(fields), 11), feature_names=np.asarray(NAMES))
    records = {}
    for path in target.iterdir():
        records["archive/" + path.name] = dict(path=path.relative_to(root).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    (root / "data/final_study/path_manifest.json").write_text(json.dumps(dict(files=records)))
    return target


def test_run_authenticates_inputs_and_writes_truthful_receipt(tmp_path):
    b = module()
    root = tmp_path / "root"
    target = retained_fixture(root)
    output = root / "outputs/result"
    receipt = b.run(output, root=root, bootstrap_draws=10)
    assert receipt["qualification"]["climate_ready"] is False
    assert receipt["qualification"]["prospective_validation"] is False
    assert receipt["test_records"] == 36
    assert len(receipt["authenticated_inputs"]) == 3
    assert (output / "fold_membership.csv").is_file()
    assert (output / "weather_strata.csv").is_file()
    assert (output / "receipt.json").is_file()
    with pytest.raises(FileExistsError):
        b.run(output, root=root, bootstrap_draws=10)
    with (target / "observations.csv").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="changed|hash|size"):
        b.run(tmp_path / "altered", root=root, bootstrap_draws=10)
