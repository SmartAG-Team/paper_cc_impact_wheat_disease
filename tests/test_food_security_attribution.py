"""Actual-day, endpoint and support contracts for diagnostic attribution."""
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from models.climate.kernel import FIELDS, simulate_draw_seasons


ROOT = Path(__file__).resolve().parents[1]


def api():
    return importlib.import_module("paper.analyses.food_security.attribution")


@pytest.fixture(scope="module")
def season():
    dates = pd.date_range("1999-10-01", periods=366)
    weather = dict(tmean_c=np.full((2, len(dates)), 12.),
                   tmax_c=np.full((2, len(dates)), 18.),
                   rh_mean_pct=np.full((2, len(dates)), 90.),
                   precipitation_mm=np.full((2, len(dates)), 1.))
    sowing = np.array(["1999-10-01", "1999-10-03"], dtype="datetime64[D]")
    phenology = json.loads((ROOT / "data/final_climate/phenology_parameters.json").read_text())
    fit = json.loads((ROOT / "data/final_climate/disease_parameters.json").read_text())["fitted"]
    thresholds = {int(k): v for k, v in json.loads(
        (ROOT / "data/final_climate/stage_thresholds.json").read_text())["all_stage_thresholds"].items()}
    original, details = simulate_draw_seasons(dates, weather, sowing,
        np.array([48., 48.]), phenology, fit, thresholds, return_daily=True)
    return dates, weather, sowing, phenology, fit, thresholds, original, details


def test_elapsed_alignment_keeps_leap_day_in_actual_sowing_age():
    source = pd.date_range("2000-02-27", periods=4)
    target = pd.date_range("2001-02-27", periods=4)
    actual = api().align_elapsed_days(source, np.array(["2000-02-27"], dtype="datetime64[D]"),
        target, np.array(["2001-02-27"], dtype="datetime64[D]"), np.array([[10., 11., 12., 13.]]))
    # 2001-03-01 is age two: the donor is 2000-02-29, not 2000-03-01.
    np.testing.assert_array_equal(actual, [[10., 11., 12., 13.]])


def test_alignment_leaves_absent_donor_dates_missing():
    actual = api().align_elapsed_days(pd.date_range("2000-02-27", periods=2),
        np.array(["2000-02-27"], dtype="datetime64[D]"),
        pd.date_range("2001-02-27", periods=3),
        np.array(["2001-02-27"], dtype="datetime64[D]"), np.array([[10., 11.]]))
    np.testing.assert_allclose(actual[:, :2], [[10., 11.]])
    assert np.isnan(actual[0, 2])


def test_baseline_corner_matches_frozen_diagonal_and_preserves_mass(season):
    dates, weather, _, _, fit, _, original, details = season
    actual = api().simulate_corner(dates, weather, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("baseline"))
    assert actual.valid.tolist() == [True, True]
    np.testing.assert_allclose(actual.lost_had3, original.GS65_85_lost_had3, rtol=0, atol=1e-10)
    np.testing.assert_allclose(actual.reference_had3, original.GS65_85_reference_had3, rtol=0, atol=1e-10)
    assert actual.mass_error < 1e-10


@pytest.mark.parametrize("invalid", [np.nan, 41.])
def test_active_invalid_weather_is_unavailable_not_zero(season, invalid):
    dates, weather, _, _, fit, _, original, details = season
    changed = {key: value.copy() for key, value in weather.items()}
    index = np.flatnonzero(details["crop_mask"][0])[20]
    changed["tmean_c"][0, index] = invalid
    if np.isfinite(invalid):
        changed["tmax_c"][0, index] = invalid + 2.
    actual = api().simulate_corner(dates, changed, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("baseline"))
    assert actual.valid.tolist() == [False, True]
    assert np.isnan(actual.lost_had3[0])
    assert actual.status[0] in ("missing_or_invalid_active_weather", "unsupported_active_temperature_above40")


def test_post_endpoint_invalid_weather_does_not_discard_valid_season(season):
    dates, weather, _, _, fit, _, original, details = season
    changed = {key: value.copy() for key, value in weather.items()}
    for key in FIELDS:
        changed[key][~details["crop_mask"]] = np.nan
    actual = api().simulate_corner(dates, changed, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("baseline"))
    assert actual.valid.all()
    np.testing.assert_allclose(actual.lost_had3, original.GS65_85_lost_had3, rtol=0, atol=1e-10)


def test_endpoint_scenarios_shift_only_declared_thresholds(season):
    dates, weather, sowing, phenology, fit, thresholds, original, _ = season
    changed = api().setting_thresholds(thresholds, api().Setting("terminal_late", terminal_scale=1.05))
    assert changed[65] == thresholds[65]
    replay, _ = simulate_draw_seasons(dates, weather, sowing, np.array([48., 48.]),
        phenology, fit, changed)
    assert (replay.BBCH85_date > original.BBCH85_date).all()
    assert replay.BBCH65_date.equals(original.BBCH65_date)
    assert changed[39] == thresholds[39]


def test_reference_senescence_reduces_accounted_canopy_without_feedback(season):
    dates, weather, _, _, fit, _, original, details = season
    baseline = api().simulate_corner(dates, weather, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("baseline"))
    senescent = api().simulate_corner(dates, weather, details,
        original.valid_complete_season.to_numpy(), fit,
        api().Setting("reference_senescence", reference_senescence=True))
    assert np.all(senescent.reference_had3 < baseline.reference_had3)
    assert np.all(senescent.lost_had3 < baseline.lost_had3)
    np.testing.assert_array_equal(senescent.symptom_day, baseline.symptom_day)
    np.testing.assert_array_equal(details["reference_area"], details["host"].area[:, :, :3] / 3.)


def test_ungated_operator_retains_subthreshold_loss(season):
    dates, weather, _, _, fit, _, original, details = season
    baseline = api().simulate_corner(dates, weather, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("baseline"))
    ungated = api().simulate_corner(dates, weather, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("ungated", detection_gated=False))
    assert np.all(ungated.lost_had3 >= baseline.lost_had3)


@pytest.mark.parametrize("key,value", [("tmean_c", -81.), ("tmax_c", 81.)])
def test_weather_outside_wetness_domain_is_unavailable(season, key, value):
    dates, weather, _, _, fit, _, original, details = season
    changed = {name: values.copy() for name, values in weather.items()}
    changed[key][0, np.flatnonzero(details["crop_mask"][0])[20]] = value
    actual = api().simulate_corner(dates, changed, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("baseline"))
    assert actual.valid.tolist() == [False, True]
    assert np.isnan(actual.lost_had3[0])


def test_ungated_damage_is_positive_when_detection_is_never_reached(season):
    dates, weather, _, _, fit, _, original, details = season
    changed = {key: value.copy() for key, value in weather.items()}
    changed["rh_mean_pct"][:] = 0.
    changed["precipitation_mm"][:] = 1e-6
    baseline = api().simulate_corner(dates, changed, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("baseline"))
    ungated = api().simulate_corner(dates, changed, details,
        original.valid_complete_season.to_numpy(), fit, api().Setting("ungated", detection_gated=False))
    np.testing.assert_array_equal(baseline.lost_had3, [0., 0.])
    assert np.all(ungated.lost_had3 > 0)
    assert np.all(ungated.symptom_day == -1)


def test_fixed_overlap_window_uses_same_actual_sowing_ages(season):
    dates, _, sowing, _, _, _, _, details = season
    # A target calendar shifted by one year spans a nonleap winter. Stage
    # indexes differ by three days in this hand-made host fixture.
    future_dates = pd.date_range("2000-10-01", periods=366)
    future_sowing = np.array(["2000-10-01", "2000-10-03"], dtype="datetime64[D]")
    starts = details["host"].stage_day_index[65]
    ends = details["host"].stage_day_index[85]
    reference = (starts, ends)
    future = (starts - 3, ends - 3)
    windows = api().fixed_overlap_windows(dates, sowing, reference,
        future_dates, future_sowing, future)
    np.testing.assert_array_equal(windows[0][0], starts)
    np.testing.assert_array_equal(windows[0][1], ends - 3)
    np.testing.assert_array_equal(windows[1][0], starts)
    np.testing.assert_array_equal(windows[1][1], ends - 3)


def test_common_support_conserves_shapley_sum_and_missing_corners():
    result = api().components([1., 1.], [5., np.nan], [0., 0.], [4., 4.])
    np.testing.assert_allclose(result["weather"][0], 4.)
    np.testing.assert_allclose(result["host"][0], -1.)
    np.testing.assert_allclose(result["net"][0], 3.)
    assert result["common"].tolist() == [True, False]
    assert np.isnan(result["weather"][1])
    np.testing.assert_allclose(result["weather"] + result["host"], result["net"], equal_nan=True)


def test_summaries_preserve_duplicate_draws_and_across_setting_population():
    rows = []
    for setting in ["baseline", "diagnostic"]:
        for draw, cell, weight in [(0, "a", .1), (1, "a", .1), (2, "b", .8)]:
            rows.append(dict(setting=setting, model="m", reference_year=1991, future_year=2071,
                spatial_draw_id=draw, cell_id=cell, area_mean_weight=weight,
                y00=1., y10=5., y01=0., y11=4. if (setting == "baseline" or cell == "a") else np.nan))
    summary, pairs = api().summarize_pairs(pd.DataFrame(rows))
    baseline = summary.loc[(summary.setting == "baseline") & (summary.model == "m")
        & (summary.support == "across_settings")].iloc[0]
    assert baseline.valid_pairs == 2  # duplicate a draws retain both design weights
    assert baseline.common_area_time_fraction == pytest.approx(.2)
    assert baseline.net_had_days == pytest.approx(3.)
    assert baseline.weather_had_days == pytest.approx(4.)
    assert baseline.host_had_days == pytest.approx(-1.)
    assert baseline.offset_percent == pytest.approx(25.)
    assert pairs.common_across_settings.sum() == 4


def unequal_model_pairs():
    rows = []
    for model, first_weight, corner_values in [
        ("a", .9, (1., 4., 0., 3.)), ("b", .1, (2., 12., 0., 10.))]:
        for draw, weight in [(0, first_weight), (1, 1-first_weight)]:
            rows.append(dict(setting="baseline", model=model, reference_year=1991,
                future_year=2071, spatial_draw_id=draw, cell_id=str(draw),
                area_mean_weight=weight, y00=corner_values[0], y10=corner_values[1],
                y01=corner_values[2], y11=corner_values[3] if draw == 0 else np.nan))
    return pd.DataFrame(rows)


def test_ensemble_averages_model_ratios_before_offset_with_unequal_support():
    summary, _ = api().summarize_pairs(unequal_model_pairs())
    result = summary.loc[(summary.model == "ensemble")
        & (summary.support == "within_setting")].iloc[0]
    # Model a: weather3, host-1, net2; model b: weather10, host-2, net8.
    assert result.weather_had_days == pytest.approx(6.5)
    assert result.host_had_days == pytest.approx(-1.5)
    assert result.net_had_days == pytest.approx(5.)
    assert result.offset_percent == pytest.approx(100*1.5/6.5)
    assert result.common_area_time_fraction == pytest.approx(.5)
    assert result.valid_pairs == 2
    assert result.complete_model_ensemble


def test_ensemble_stays_unavailable_when_one_member_has_no_supported_pairs():
    pairs = unequal_model_pairs()
    pairs.loc[pairs.model == "b", "y11"] = np.nan
    summary, _ = api().summarize_pairs(pairs)
    result = summary.loc[(summary.model == "ensemble")
        & (summary.support == "within_setting")].iloc[0]
    assert np.isnan(result.net_had_days)
    assert np.isnan(result.weather_had_days)
    assert np.isnan(result.offset_percent)
    assert result.requested_models == 2
    assert result.available_model_estimates == 1
    assert not result.complete_model_ensemble


def test_ensemble_stays_unavailable_when_a_requested_model_is_absent():
    pairs = unequal_model_pairs()
    pairs = pairs.loc[pairs.model == "a"]
    summary, _ = api().summarize_pairs(pairs, expected_models=["a", "b"])
    result = summary.loc[(summary.model == "ensemble")
        & (summary.support == "within_setting")].iloc[0]
    assert np.isnan(result.net_had_days)
    assert result.requested_models == 2
    assert result.available_model_estimates == 1
    assert not result.complete_model_ensemble


def test_missing_required_parameter_registration_is_rejected_before_replay():
    manifest = json.loads((ROOT / "data/final_climate/scientific_manifest.json").read_text())
    manifest["inputs"] = [row for row in manifest["inputs"]
                          if row["path"] != "data/final_climate/disease_parameters.json"]
    with pytest.raises(ValueError, match="registered.*disease_parameters"):
        api()._authenticated_inputs(ROOT, manifest, [], [])


def minimal_run_root(tmp_path):
    root = tmp_path / "repo"
    directory = root / "data/final_climate"
    directory.mkdir(parents=True)
    (directory / "scientific_manifest.json").write_text(json.dumps(
        dict(models=["m"], scenarios=["ssp585"], inputs=[], forcing=[], core_sources=[])))
    return root


def test_run_refuses_existing_destination_before_input_replay(tmp_path):
    root = minimal_run_root(tmp_path)
    destination = root / "outputs/already_exists"
    destination.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        api().run(destination, root=root, models=["m"], reference_years=[1991])
    assert list(destination.iterdir()) == []


def test_run_refuses_source_destination_without_creating_it(tmp_path):
    root = minimal_run_root(tmp_path)
    destination = root / "analysis/new_results"
    with pytest.raises(ValueError, match="frozen/source"):
        api().run(destination, root=root, models=["m"], reference_years=[1991])
    assert not destination.exists()

