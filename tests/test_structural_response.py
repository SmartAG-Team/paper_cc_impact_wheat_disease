"""Observation-link and withheld-outcome guards for the bounded experiment."""

import importlib

import numpy as np
import pandas as pd
import pytest

from calibration.seasonal_septoria.field_data import FieldData
from model.seasonal_septoria.structural import CanopyParameters, canopy_host, simulate_canopy


def response_module():
    try:
        return importlib.import_module("analysis.paper_study.structural_response.response")
    except ModuleNotFoundError:
        pytest.fail("The empirical dose-to-score observation response is not implemented")


def synthetic_fields():
    days = 110
    temperature = np.full((2, days), 18.)
    humidity = np.full_like(temperature, 90.)
    rain = np.ones_like(temperature)
    accumulation = np.tile(np.arange(1., days + 1) * 18, (2, 1))
    thresholds = {10: 10., 31: 100., 51: 200., 85: 3000.}
    host = dict(flag_fraction=.3, leaf_interval=120., expansion_units=100.)
    active, renewal, _, _ = canopy_host(accumulation, temperature, thresholds, **host)
    truth = simulate_canopy(temperature, humidity, rain, active, renewal,
        CanopyParameters(initiation=.012, amplification=0., latent_days=20.))
    rows = []
    for field in range(2):
        for leaf in [0, 2]:
            for day in [45, 65, 90]:
                fraction = truth.expressed[field, day, leaf]
                score = float(100 * (1 - np.exp(-(-np.log1p(-fraction)) ** 2)))
                rows.append(dict(field_index=field, field_id=f"field{field}",
                    coordinate_year=f"site{field}|2018", endpoint_series=f"field{field}|F{leaf+1}",
                    date=pd.Timestamp("2018-01-01") + pd.Timedelta(days=day-1),
                    day_index=day, leaf_index=leaf, value=score,
                    site_id=f"site{field}", season_year=2018,
                    metric="infection_percent_unspecified_basis", partition="calibration"))
    metadata = pd.DataFrame([dict(field_index=i, field_id=f"field{i}", site_id=f"site{i}",
        forcing_days=days, sowing_date="2018-01-01", first_forcing_date="2018-01-01",
        last_forcing_date="2018-04-20") for i in range(2)])
    data = FieldData(temperature, humidity, rain, active, renewal, metadata,
        pd.DataFrame(rows), pd.DataFrame())
    data.maximum_temperature = temperature + 6
    weather = pd.DataFrame([dict(location_id=f"site{i}", date=date,
        tmean_c=18., tmax_c=24., rh_mean_pct=90., precipitation_mm=1.,
        rain_hours_gt_0_1mm=2.) for i in range(2)
        for date in pd.date_range("2018-01-01", periods=days)])
    candidate = dict(name="daily_or_delay20_nonlinear", weather_operator="daily_or",
        latent_days=20., observation_link="nonlinear", host_parameters=host)
    return data, accumulation, thresholds, weather, candidate


def test_response_preserves_exact_endpoints_and_is_strictly_monotone():
    module = response_module()
    fractions = np.array([0., 1e-12, .001, .1, .5, .9, 1.])
    for power in [.5, 1., 2., 4.]:
        actual = module.observation_response(fractions, power)
        assert actual[0] == 0.
        assert actual[-1] == 100.
        assert np.all(np.diff(actual) > 0)
    np.testing.assert_allclose(module.observation_response(fractions, 1.),
        100 * fractions, rtol=1e-12, atol=1e-14)
    # D = 1 - exp(-.5), b = 2 gives O = 100 * (1 - exp(-.25)).
    assert module.observation_response(.3934693402873666, 2.) == pytest.approx(22.119921692859513)


@pytest.mark.parametrize("fraction,power", [(-.1, 1.), (1.1, 1.),
    (np.nan, 1.), (.5, 0.), (.5, np.nan), (.5, .49), (.5, 4.01)])
def test_invalid_response_inputs_fail(fraction, power):
    with pytest.raises(ValueError):
        response_module().observation_response(fraction, power)


def test_nonlinear_fit_recovers_known_link_without_reading_withheld_scores():
    module = response_module()
    data, accumulation, thresholds, weather, candidate = synthetic_fields()
    train = np.flatnonzero(data.targets.field_index.eq(0))
    first = module.fit_response(data, accumulation, thresholds, train, candidate, weather)
    assert first["observation_power"] == pytest.approx(2., abs=.002)
    assert first["parameters"]["initiation"] == pytest.approx(.012, abs=2e-5)
    assert first["parameters"]["amplification"] == 0.
    assert first["calibration_field_ids"] == ["field0"]
    altered = data.targets.field_index.eq(1)
    data.targets.loc[altered, "value"] = np.nan
    data.temperature[1] = -30.
    weather.loc[weather.location_id.eq("site1"), "precipitation_mm"] = 99999.
    second = module.fit_response(data, accumulation, thresholds, train, candidate, weather)
    assert second["parameters"] == first["parameters"]
    assert second["observation_power"] == first["observation_power"]
    assert second["training_weighted_sse"] == first["training_weighted_sse"]


def test_duration_preprocessing_excludes_withheld_weather_and_outcomes():
    module = response_module()
    data, accumulation, thresholds, weather, candidate = synthetic_fields()
    candidate.update(weather_operator="duration_proxy", observation_link="identity")
    train = np.flatnonzero(data.targets.field_index.eq(0))
    first = module.fit_response(data, accumulation, thresholds, train, candidate, weather)
    assert first["rain_rate_mm_hour"] == .5
    data.targets.loc[data.targets.field_index.eq(1), "value"] = 100.
    weather.loc[weather.location_id.eq("site1"), "rain_hours_gt_0_1mm"] = .001
    second = module.fit_response(data, accumulation, thresholds, train, candidate, weather)
    assert second["rain_rate_mm_hour"] == .5
    assert second["parameters"] == first["parameters"]
    assert second["observation_power"] == 1.


def test_fitting_cannot_split_field_histories_or_accept_fractional_membership():
    module = response_module()
    data, accumulation, thresholds, weather, candidate = synthetic_fields()
    with pytest.raises(ValueError, match="complete"):
        module.fit_response(data, accumulation, thresholds, [0], candidate, weather)
    with pytest.raises(ValueError, match="integers"):
        module.fit_response(data, accumulation, thresholds, [.9], candidate, weather)


def test_mapped_prediction_preserves_raw_process_state_and_rejects_labels():
    module = response_module()
    data, accumulation, thresholds, _, candidate = synthetic_fields()
    fitted = dict(parameters=dict(initiation=.012, amplification=0., latent_days=20.),
        host_parameters=candidate["host_parameters"], observation_power=2., weather_operator="daily_or")
    with pytest.raises(ValueError, match="redacted"):
        module.predict_response(data, accumulation, thresholds, fitted)
    data.targets["value"] = np.nan
    actual, trajectory = module.predict_response(data, accumulation, thresholds, fitted)
    active, renewal, _, _ = canopy_host(accumulation, data.temperature, thresholds,
        **candidate["host_parameters"])
    original = simulate_canopy(data.temperature, data.humidity, data.rain, active, renewal,
        CanopyParameters(**fitted["parameters"]))
    np.testing.assert_array_equal(trajectory.state, original.state)
    assert actual.min() > 0.
    identity, unchanged = module.predict_response(data, accumulation, thresholds,
        dict(fitted, observation_power=1.))
    idx = (data.targets.field_index.to_numpy(), data.targets.day_index.to_numpy(),
        data.targets.leaf_index.to_numpy())
    np.testing.assert_allclose(identity, 100 * original.expressed[idx], atol=1e-12)
    np.testing.assert_array_equal(unchanged.state, original.state)


def test_onset_uses_mapped_daily_scores_and_retains_interval_censoring():
    module = response_module()
    data, _, _, _, _ = synthetic_fields()
    data.targets = data.targets.iloc[:2].copy()
    data.targets["date"] = pd.to_datetime(["2018-01-01", "2018-01-03"])
    data.targets["value"] = [0., 20.]
    expressed = np.zeros((2, 111, 8))
    expressed[0, 2:, 0] = .2
    identity = module.response_onsets(data, expressed, 1., "identity")
    nonlinear = module.response_onsets(data, expressed, 4., "nonlinear")
    at_five = identity.loc[identity.cutoff_percent.eq(5.)].iloc[0]
    transformed = nonlinear.loc[nonlinear.cutoff_percent.eq(5.)].iloc[0]
    assert pd.Timestamp(at_five.predicted_visible_date) == pd.Timestamp("2018-01-02")
    assert at_five.compatible
    assert pd.isna(transformed.predicted_visible_date)
    assert not transformed.compatible
    assert transformed.missing_prediction
    assert transformed.censoring == "interval"


def test_same_date_gradient_does_not_join_leaf_assessments_on_different_dates():
    module = response_module()
    frame = pd.DataFrame(dict(field_id=["a", "a", "a", "b", "b"],
        coordinate_year=["site|2018"] * 5,
        date=pd.to_datetime(["2018-05-01", "2018-05-01", "2018-06-01", "2018-05-01", "2018-06-01"]),
        leaf_index=[0, 1, 2, 0, 1], value=[10., 30., 90., 5., 95.],
        predicted_percent=[12., 25., 30., 5., 50.]))
    actual = module.same_date_gradients(frame)
    assert len(actual) == 1
    assert actual.iloc[0].observed_gradient == 20.
    assert actual.iloc[0].predicted_gradient == 13.
    assert actual.iloc[0].gradient_error == -7.


def test_registration_freezes_identical_bounds_before_any_fit(tmp_path):
    try:
        module = importlib.import_module("analysis.paper_study.structural_response.run")
    except ModuleNotFoundError:
        pytest.fail("The preregistered response experiment runner is not implemented")
    path = tmp_path / "dependency.py"
    path.write_text("frozen = 1\n")
    configuration = module.register_experiment(tmp_path / "experiment", {"siteA": 0}, [path])
    assert len(configuration["candidates"]) == 8
    for candidate in configuration["candidates"]:
        assert candidate["initiation_bounds"] == [1e-6, 1.]
        assert candidate["observation_power_bounds"] == [.5, 4.]
    assert (tmp_path / "experiment/configuration_before_fitting.json").exists()
    assert not (tmp_path / "experiment/frozen_selected_fit.json").exists()
    path.write_text("frozen = 2\n")
    with pytest.raises(ValueError, match="changed|hash"):
        module.verify_registered_dependencies(configuration)


def test_candidate_selection_requires_all_original_calibration_folds():
    module = importlib.import_module("analysis.paper_study.structural_response.run")
    data, _, _, _, _ = synthetic_fields()
    with pytest.raises(ValueError, match="three|3"):
        module.original_calibration_membership(data.targets, {"site0": 0, "site1": 1})
    data.targets.loc[data.targets.field_index.eq(1), "season_year"] = 2019
    with pytest.raises(ValueError, match="2017|2018|calibration"):
        module.original_calibration_membership(data.targets, {"site0": 0, "site1": 1})


def test_severity_summary_retains_endpoint_and_leaf_scope_hierarchies():
    module = importlib.import_module("analysis.paper_study.structural_response.run")
    frame = pd.DataFrame(dict(field_id=["a", "a", "a", "a", "b"],
        coordinate_year=["siteA|2018"] * 4 + ["siteB|2018"],
        endpoint_series=["a|F1", "a|F1", "a|F4", "a|F4", "b|F1"],
        date=pd.to_datetime(["2018-05-01", "2018-06-01", "2018-05-01", "2018-06-01", "2018-06-01"]),
        leaf_index=[0, 0, 3, 3, 0], value=[0., 10., 20., 30., 50.],
        predicted_percent=[0., 12., 20., 26., 52.], model=["test"] * 5,
        partition=["reused_development_2019"] * 5))
    metrics = module.severity_summary(frame)
    final = metrics.query("endpoint == 'final_numeric_assessment' and leaf_scope == 'all_ordinal_leaves'").iloc[0]
    # Each site has equal mass; site A splits its mass equally across two leaves.
    assert final.rmse == pytest.approx(np.sqrt(7.))
    assert final.mae == 2.5
    assert final.bias == .5
    assert final.n == 3
    assert "weighted_r2" in final.index
    assert len(metrics) == 4
