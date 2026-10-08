"""First effective infection and symptom detection use signs, not severity."""

import importlib

import numpy as np
import pandas as pd
import pytest


def event_module():
    try:
        return importlib.import_module("model.seasonal_septoria.infection_events")
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
    brackets = m.symptom_brackets(frame)
    assert brackets.censoring.tolist() == ["two_sided", "left", "right"]
    assert brackets.lower_day.fillna(-1).tolist() == [2., -1., 6.]
    assert brackets.upper_day.fillna(-1).tolist() == [5., 4., -1.]
    altered = frame.copy()
    altered.loc[altered.value.gt(0), "value"] = [99., .001, 78.]
    pd.testing.assert_frame_equal(brackets, m.symptom_brackets(altered))
    assert m.onset_distance(brackets, np.array([3, 2, -1]), np.array([10, 10, 10])).tolist() == [0., 0., 0.]
    # Never assigning an onset is meaningfully worse than crossing late.
    assert m.onset_distance(brackets, np.array([-1, -1, 4]), np.array([10, 10, 10])).tolist() == [30., 30., 3.]


def test_equal_hierarchy_weights_prevent_dense_histories_dominating():
    m = event_module()
    frame = pd.DataFrame(dict(coordinate_year=["X", "X", "X", "Y"],
        field_id=["A", "A", "B", "C"], leaf_index=[0, 1, 0, 0]))
    assert m.equal_hierarchy_weights(frame).tolist() == [.125, .125, .25, .5]


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
        m.symptom_brackets(sign_histories().assign(value=np.nan))


def runner_module():
    try:
        return importlib.import_module("analysis.paper_study.infection_priority_20261006.event_model.run")
    except ModuleNotFoundError:
        pytest.fail("The isolated nested event runner is not implemented")


def test_selection_uses_two_sided_brackets_and_is_invariant_to_positive_magnitude():
    r = runner_module()
    m = event_module()
    labels = sign_histories()
    brackets = m.symptom_brackets(labels)
    # A drives selection; the left/right cases remain diagnostic.
    days = {"late": np.array([7, 3, -1]), "compatible": np.array([4, -1, 1])}
    scored = r.rank_candidates(brackets, days, np.array([10, 10, 10]))
    assert scored[0]["candidate"] == "compatible"
    assert scored[0]["primary_distance_days"] == 0.
    assert scored[1]["primary_distance_days"] == 2.
    labels.loc[labels.value.gt(0), "value"] = [100., .00001, 53.]
    assert scored == r.rank_candidates(m.symptom_brackets(labels), days, np.array([10, 10, 10]))


def test_event_prediction_requires_redacted_values_and_ignores_source_stage():
    from model.seasonal_septoria.field_data import FieldData

    r = runner_module()
    t = np.full((1, 10), 18.)
    active = np.ones((1, 10, 8), bool)
    target = pd.DataFrame(dict(field_index=[0], day_index=[9], leaf_index=[0],
        value=[np.nan], stage_from=[31.]))
    metadata = pd.DataFrame(dict(field_index=[0], forcing_days=[10]))
    data = FieldData(t, t, t, active, np.zeros_like(active, float), metadata, target, pd.DataFrame())
    data.maximum_temperature = t.copy()
    accumulated = np.arange(1., 11.)[None, :]*18.
    fitted = dict(host_parameters=dict(flag_fraction=.3, leaf_interval=120., expansion_units=100.),
        parameters=dict(dose_threshold=.1, dry_gap_days=1, delay_reference_days=2.),
        weather_preprocessing=dict(rain_rate_mm_hour=.6))
    thresholds = {10:1., 31:10., 51:20., 85:500.}
    first = r.predict_event_record(data, accumulated, thresholds, fitted)
    stripped = r.forcing_only_prediction_copy(data)
    assert stripped.maximum_temperature.tolist() == t.tolist()
    assert stripped.targets.columns.tolist() == ["field_index", "day_index", "leaf_index", "value"]
    np.testing.assert_array_equal(first.symptom_day,
        r.predict_event_record(stripped, accumulated, thresholds, fitted).symptom_day)
    data.targets.stage_from = 99.
    second = r.predict_event_record(data, accumulated, thresholds, fitted)
    np.testing.assert_array_equal(first.symptom_day, second.symptom_day)
    data.targets.value = 50.
    with pytest.raises(ValueError, match="redacted"):
        r.predict_event_record(data, accumulated, thresholds, fitted)


def test_detection_metrics_distinguish_false_positive_and_missed_positive():
    r = runner_module()
    frame = pd.DataFrame(dict(coordinate_year=["X"]*4, field_id=["A"]*4,
        leaf_index=[0, 0, 1, 1], observed_positive=[False, True, False, True],
        predicted_positive=[True, True, False, False]))
    result = r.detection_metrics(frame)
    assert result["true_positive"] == 1
    assert result["false_positive"] == 1
    assert result["true_negative"] == 1
    assert result["false_negative"] == 1
    assert result["sensitivity"] == .5
    assert result["specificity"] == .5
    assert result["weighted_accuracy"] == .5


def test_archived_membership_accepts_storage_dtypes_but_rejects_changed_keys():
    r = runner_module()
    fresh = pd.DataFrame(dict(global_target_index=[0], field_id=["A"], endpoint_series=["A|F1"],
        date=pd.to_datetime(["2018-05-01"])))
    stored = fresh.copy()
    stored["field_id"] = stored.field_id.astype(pd.StringDtype(na_value=pd.NA))
    r.assert_same_target_membership(fresh, stored)
    stored.loc[0, "field_id"] = "B"
    with pytest.raises(AssertionError):
        r.assert_same_target_membership(fresh, stored)


def test_magnitude_invariance_audit_accepts_integer_binary_labels():
    r = runner_module()
    labels = sign_histories()
    labels["value"] = labels.value.gt(0).astype(int)
    altered = r.perturb_positive_magnitudes(labels)
    assert altered.value.gt(0).tolist() == [False, True, True, True, False, False]
    assert altered.loc[altered.value.gt(0), "value"].tolist() == [.00001, 50.000005, 100.]
    assert labels.value.tolist() == [0, 1, 1, 1, 0, 0]
