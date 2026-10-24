"""Explicit crop-calendar and phenology assumptions for leaf cohorts."""

import calendar

import numpy as np
import pandas as pd


def sowing_date_from_calendar(harvest_year, sowing_doy, maturity_doy):
    """Map calendar DOY to a harvest season without inventing an observed date."""
    if (not all(np.isfinite(x) for x in (harvest_year, sowing_doy, maturity_doy))
            or int(harvest_year) != harvest_year
            or not 1 <= sowing_doy <= 366 or not 1 <= maturity_doy <= 366):
        raise ValueError('A harvest year and calendar days1–366 are required.')
    sowing_year = int(harvest_year)-(sowing_doy > maturity_doy)
    doy = int(round(sowing_doy))
    if doy > (366 if calendar.isleap(sowing_year) else 365):
        raise ValueError('Calendar day is outside the actual sowing year.')
    return pd.Timestamp(year=sowing_year, month=1, day=1)+pd.Timedelta(days=doy-1)


def cohort_inputs(accumulation, temperature, thresholds, flag_fraction=.3,
                  juvenile_turnover_degree_days=300.):
    """Approximate final seven leaf cohorts and a renewing juvenile reservoir.

    Final leaf3 appears at the model's stage31 threshold. Flag appearance is
    interpolated a fraction of the stage31–51 developmental interval. Two
    equal increments connect leaves3,2,1; earlier final leaf cohorts use the
    same increment. These cohort timings are structural approximations, not
    extra events provided or validated by the copied phenology model.
    """
    accumulated, t = np.asarray(accumulation, float), np.asarray(temperature, float)
    events = np.asarray([thresholds[x] for x in (10,31,51,85)], float)
    if (accumulated.ndim != 2 or accumulated.size == 0 or accumulated.shape != t.shape
            or not np.isfinite(accumulated).all() or not np.isfinite(t).all()
            or np.any(accumulated < 0) or np.any(np.diff(accumulated,axis=1) < -1e-10)
            or not np.isfinite(events).all() or np.any(np.diff(events) <= 0)
            or not np.isfinite(flag_fraction) or not 0 < flag_fraction < 1
            or not np.isfinite(juvenile_turnover_degree_days)
            or juvenile_turnover_degree_days <= 0):
        raise ValueError('Invalid developmental accumulation, thresholds or host parameters.')
    flag = events[1]+flag_fraction*(events[2]-events[1])
    increment = (flag-events[1])/2
    birth = np.maximum(events[0], flag-np.arange(7)*increment)
    birth = np.append(birth, events[0])
    prior = np.column_stack([np.zeros(len(accumulated)), accumulated[:,:-1]])
    active = prior[:,:,None] >= birth[None,None,:]
    renewal = np.zeros(active.shape)
    renewal[:,:,7] = (-np.expm1(-np.maximum(t,0)/juvenile_turnover_degree_days))*active[:,:,7]
    return active, renewal
