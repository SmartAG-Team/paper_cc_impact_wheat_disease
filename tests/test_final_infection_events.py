"""Infection timing, censored symptom brackets, and hierarchical weights."""
import importlib

import numpy as np
import pandas as pd
import pytest

from models.calibration.seasonal_septoria.infection_events import symptom_brackets, onset_distance, equal_hierarchy_weights


def event_module():
    try:
        return importlib.import_module("models.seasonal_septoria.infection_events")
    except ModuleNotFoundError:
        pytest.fail("The isolated infection-event model is not implemented")


def test_dry_weather_cannot_produce_infection_even_with_available_host():
    m = event_module()
    t = np.full((2, 12), 18.)
    active = np.ones((2, 12, 3), bool)
    result = m.simulate_events(t, np.zeros_like(t), active, m.EventParameters(.1, 1, 2.))
    assert result.infection_day.tolist() == [[-1, -1, -1], [-1, -1, -1]]
    assert result.symptom_day.tolist() == [[-1, -1, -1], [-1, -1, -1]]
    assert not result.infected.any()
    assert not result.symptomatic.any()


def test_dose_resets_after_declared_dry_gap_and_ignores_inactive_host():
    m = event_module()
    t = np.full((1, 8), 18.)
    exposure = np.array([[.4, 0., .4, 0., 0., 0., .7, 0.]])
    active = np.ones((1, 8, 2), bool)
    active[:, :6, 1] = False
    p = m.EventParameters(.7, 1, 1.)
    reset = m.simulate_events(t, exposure, active, p)
    assert reset.infection_day.tolist() == [[7, 7]]
    assert reset.symptom_day.tolist() == [[8, 8]]
    retained = m.simulate_events(t, exposure, active, m.EventParameters(.7, 3, 1.))
    assert retained.infection_day.tolist() == [[3, 7]]


def test_delay_starts_after_infection_and_uses_positive_temperature_only():
    m = event_module()
    t = np.array([[18., 9., -4., 18., 9., 18.]])
    active = np.ones((1, 6, 1), bool)
    result = m.simulate_events(t, np.ones_like(t), active, m.EventParameters(.5, 1, 2.))
    assert result.infection_day.tolist() == [[1]]
    assert result.symptom_day.tolist() == [[5]]
    assert result.symptomatic[0, :, 0].tolist() == [False, False, False, False, False, True, True]


def test_future_weather_cannot_rewrite_past_events():
    m = event_module()
    t = np.full((1, 12), 18.)
    exposure = np.full_like(t, .2)
    active = np.ones((1, 12, 2), bool)
    p = m.EventParameters(.5, 3, 3.)
    first = m.simulate_events(t, exposure, active, p)
    t[:, 6:] = -5.
    exposure[:, 6:] = 0.
    second = m.simulate_events(t, exposure, active, p)
    np.testing.assert_array_equal(first.infected[:, :7], second.infected[:, :7])
    np.testing.assert_array_equal(first.symptomatic[:, :7], second.symptomatic[:, :7])


def test_phenology_comparator_infects_at_host_availability_and_keeps_same_delay():
    m = event_module()
    t = np.full((1, 8), 18.)
    active = np.zeros((1, 8, 1), bool)
    active[:, 2:, :] = True
    result = m.simulate_events(t, np.zeros_like(t), active, m.EventParameters(.5, 1, 2.),
                               phenology_only=True)
    assert result.infection_day.tolist() == [[3]]
    assert result.symptom_day.tolist() == [[5]]


def sign_histories():
    # Hand-derived brackets: A (2,5], B (-infinity,4], C (6,infinity).
    return pd.DataFrame([
        dict(field_id="A", field_index=0, coordinate_year="X|2017", leaf_index=0,
             endpoint_series="A|F1", day_index=day, value=value)
        for day, value in [(2, 0.), (5, 12.), (8, 3.)]
    ] + [dict(field_id="B", field_index=1, coordinate_year="X|2017", leaf_index=0,
              endpoint_series="B|F1", day_index=4, value=1.)] + [
        dict(field_id="C", field_index=2, coordinate_year="Y|2018", leaf_index=0,
             endpoint_series="C|F1", day_index=day, value=0.) for day in (3, 6)
    ])


def test_brackets_preserve_censoring_and_ignore_positive_magnitudes():
    m = event_module()
    frame = sign_histories()
    brackets = symptom_brackets(frame)
    assert brackets.censoring.tolist() == ["two_sided", "left", "right"]
    assert brackets.lower_day.fillna(-1).tolist() == [2., -1., 6.]
    assert brackets.upper_day.fillna(-1).tolist() == [5., 4., -1.]
    altered = frame.copy()
    altered.loc[altered.value.gt(0), "value"] = [99., .001, 78.]
    pd.testing.assert_frame_equal(brackets, symptom_brackets(altered))
    assert onset_distance(brackets, np.array([3, 2, -1]), np.array([10, 10, 10])).tolist() == [0., 0., 0.]
    # Never assigning an onset is meaningfully worse than crossing late.
    assert onset_distance(brackets, np.array([-1, -1, 4]), np.array([10, 10, 10])).tolist() == [30., 30., 3.]


def test_equal_hierarchy_weights_prevent_dense_histories_dominating():
    m = event_module()
    frame = pd.DataFrame(dict(coordinate_year=["X", "X", "X", "Y"],
        field_id=["A", "A", "B", "C"], leaf_index=[0, 1, 0, 0]))
    assert equal_hierarchy_weights(frame).tolist() == [.125, .125, .25, .5]


def test_invalid_event_inputs_are_rejected():
    m = event_module()
    t = np.full((1, 2), 18.)
    active = np.ones((1, 2, 1), bool)
    for p in [m.EventParameters(0., 1, 2.), m.EventParameters(.5, 0, 2.),
              m.EventParameters(.5, 1, 0.)]:
        with pytest.raises(ValueError):
            m.simulate_events(t, np.zeros_like(t), active, p)
    with pytest.raises(ValueError):
        m.simulate_events(t, np.full_like(t, -1.), active, m.EventParameters())
    with pytest.raises(ValueError):
        symptom_brackets(sign_histories().assign(value=np.nan))
