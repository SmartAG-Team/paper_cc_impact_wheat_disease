"""Structural canopy models preserve latency, leaf growth and causal forcing."""

import numpy as np
import pandas as pd
import pytest

from model.seasonal_septoria.field_data import FieldData
from model.seasonal_septoria.wetness import duration_exposure

from model.seasonal_septoria.structural import (
    CanopyParameters, canopy_host, simulate_canopy,
    fit_canopy, predict_canopy,
)


def forcing(days=100):
    t = np.full((1, days), 18.)
    h = np.full_like(t, 90.)
    r = np.ones_like(t)
    active = np.ones((1, days, 2), bool)
    renewal = np.zeros(active.shape)
    return t, h, r, active, renewal


def test_amplification_requires_expressed_symptoms_and_preserves_mass():
    args = forcing()
    empty = simulate_canopy(*args, CanopyParameters(initiation=0., amplification=.1))
    assert np.count_nonzero(empty.expressed) == 0
    baseline = simulate_canopy(*args, CanopyParameters(initiation=.003, amplification=0.))
    expanded = simulate_canopy(*args, CanopyParameters(initiation=.003, amplification=.1))
    assert expanded.expressed[0, -1, 0] > baseline.expressed[0, -1, 0]+.2
    assert not expanded.expressed[0, :31, 0].any()
    assert expanded.state.min() >= 0.
    np.testing.assert_allclose(expanded.state.sum(axis=-1), 1., atol=1e-12)


def test_post_assessment_weather_cannot_change_previous_canopy_predictions():
    args = forcing()
    first = simulate_canopy(*args, CanopyParameters())
    args[0][:, 60:] = 0.
    args[1][:, 60:] = 0.
    args[2][:, 60:] = 0.
    second = simulate_canopy(*args, CanopyParameters())
    np.testing.assert_array_equal(first.expressed[:, :61], second.expressed[:, :61])


def test_leaf_spacing_is_independent_of_flag_timing_and_growth_dilutes():
    accumulation = np.arange(0., 1000., 20.)[None, :]
    thresholds = {10:100., 31:300., 51:500., 85:1200.}
    active, renewal, area, births = canopy_host(accumulation, np.full_like(accumulation, 18.),
        thresholds, flag_fraction=.6, leaf_interval=100., expansion_units=100.)
    np.testing.assert_allclose(births, [420., 320., 220., 120., 100., 100., 100., 100.])
    assert not active[0, 21, 0]  # preceding-day accumulation remains below birth
    assert active[0, 22, 0]
    assert 0 < area[0, 22, 0] < 1
    assert renewal[0, 23, 0] > 0
    assert renewal[0, 35, 0] == 0.
    assert not np.any(renewal[~active])


def test_inactive_leaf_cannot_retain_or_express_disease():
    t, h, r, active, renewal = forcing()
    active[:, 40:, 1] = False
    result = simulate_canopy(t, h, r, active, renewal, CanopyParameters())
    assert not result.expressed[:, 41:, 1].any()
    with pytest.raises(ValueError):
        simulate_canopy(t, h, -r, active, renewal, CanopyParameters())


def test_complete_leaf_replacement_removes_queued_exposure_before_expression():
    t, h, r, active, renewal = forcing()
    renewal[:, 5, 0] = 1.
    result = simulate_canopy(t, h, r, active, renewal,
        CanopyParameters(initiation=.003, amplification=0., latent_days=30.))
    assert not result.expressed[0, :36, 0].any()
    assert result.expressed[0, 31, 1] > 0.


def test_partial_leaf_expansion_dilutes_pending_exposure_at_expression():
    t, h, r, active, renewal = forcing()
    baseline = simulate_canopy(t, h, r, active, renewal,
        CanopyParameters(initiation=.003, amplification=0., latent_days=30.))
    renewal[:, 20, 0] = .5
    diluted = simulate_canopy(t, h, r, active, renewal,
        CanopyParameters(initiation=.003, amplification=0., latent_days=30.))
    np.testing.assert_allclose(diluted.expressed[0, 31, 0],
                               .5*baseline.expressed[0, 31, 0], atol=1e-12)


def assessment_data():
    t, h, r, active, renewal = forcing()
    target = pd.DataFrame(dict(field_index=[0], field_id=['a'], coordinate_year=['site|2018'],
        endpoint_series=['a|F1'], date=pd.to_datetime(['2018-06-01']),
        day_index=[90], leaf_index=[0], value=[30.],
        metric=['infection_percent_unspecified_basis']))
    data = FieldData(t, h, r, active, renewal,
        pd.DataFrame(dict(field_index=[0], forcing_days=[100])), target, pd.DataFrame())
    data.maximum_temperature = t+6.
    accumulated = np.arange(1., 101.)[None, :]*18
    thresholds = {10:10., 31:100., 51:200., 85:2000.}
    fitted = dict(parameters=dict(initiation=.03, amplification=0., latent_days=20.),
        host_parameters=dict(flag_fraction=.6, leaf_interval=120., expansion_units=100.))
    return data, accumulated, thresholds, fitted


def test_saved_duration_fit_applies_its_declared_weather_operator():
    data, accumulated, thresholds, fitted = assessment_data()
    fitted.update(weather_operator='duration_proxy', rain_rate_mm_hour=.6)
    driver = duration_exposure(data.temperature, data.maximum_temperature,
                               data.humidity, data.rain, .6)['exposure']
    expected, _ = predict_canopy(data, accumulated, thresholds, fitted, driver)
    actual, _ = predict_canopy(data, accumulated, thresholds, fitted)
    np.testing.assert_allclose(actual, expected, atol=1e-12)


@pytest.mark.parametrize('column,value', [('field_index', -1), ('field_index', 1),
    ('leaf_index', -1), ('leaf_index', 7), ('day_index', -1), ('day_index', 101), ('day_index', 5.9)])
def test_invalid_assessment_indices_are_rejected(column, value):
    data, accumulated, thresholds, fitted = assessment_data()
    data.targets[column] = value
    with pytest.raises(ValueError, match='indices'):
        predict_canopy(data, accumulated, thresholds, fitted)


def test_fractional_calibration_row_indices_are_rejected():
    data, accumulated, thresholds, fitted = assessment_data()
    with pytest.raises(ValueError, match='indices'):
        fit_canopy(data, accumulated, thresholds, [.9], fitted['host_parameters'], False)


def test_extreme_valid_renewal_does_not_underflow_the_pending_queue():
    t, h, r, active, renewal = forcing(days=110)
    renewal[:] = .999
    result = simulate_canopy(t, h, r, active, renewal, CanopyParameters())
    assert np.isfinite(result.state).all()
    np.testing.assert_allclose(result.state.sum(axis=-1), 1., atol=1e-12)
