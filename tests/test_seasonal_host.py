import numpy as np
import pytest

from model.seasonal_septoria.host import cohort_inputs, sowing_date_from_calendar


def test_cross_year_sowing_uses_calendar_order_and_actual_year_length():
    assert str(sowing_date_from_calendar(2019, 280, 200).date()) == '2018-10-07'
    assert str(sowing_date_from_calendar(2020, 90, 250).date()) == '2020-03-30'
    with pytest.raises(ValueError):
        sowing_date_from_calendar(2019, 0, 200)


def test_cohorts_use_previous_day_development_and_new_leaves_start_clean():
    # Synthetic thresholds: emergence100, stem300, heading500, softdough1000.
    thresholds = {10:100.,31:300.,51:500.,85:1000.}
    accumulated = np.array([[50., 100., 300., 360., 500.]])
    temperature = np.full_like(accumulated, 10.)
    active, renewal = cohort_inputs(accumulated, temperature, thresholds)
    assert active.shape == (1,5,8)
    assert not active[0,0].any()
    assert not active[0,1].any()
    assert active[0,2,7]
    assert not active[0,3,0]  # flag cohort starts at360, visible next day
    assert active[0,4,0]
    assert np.count_nonzero(renewal[:,:,:7]) == 0
    assert renewal[0,2,7] > 0


def test_no_host_renewal_occurs_in_an_inactive_cohort():
    active, renewal = cohort_inputs(np.zeros((2,4)), np.full((2,4), 20.),
                                   {10:100.,31:300.,51:500.,85:1000.})
    assert not active.any()
    assert not renewal.any()


def test_accumulation_must_be_finite_nonnegative_and_monotonic():
    with pytest.raises(ValueError):
        cohort_inputs(np.array([[1.,0.]]), np.array([[18.,18.]]),
                      {10:100.,31:300.,51:500.,85:1000.})
