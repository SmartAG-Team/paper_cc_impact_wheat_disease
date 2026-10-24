"""Meteorologically constrained wetness-exposure proxies from daily variables.

These are inferred duration indices. They do not reconstruct observed hourly
weather, canopy humidity, leaf wetness or the temporal overlap of rain and dew.
"""

import numpy as np


def _daily_inputs(temperature, maximum_temperature, humidity, precipitation=None):
    values = [np.asarray(x, float) for x in (temperature, maximum_temperature, humidity)]
    if precipitation is not None:
        values.append(np.asarray(precipitation, float))
    values = np.broadcast_arrays(*values)
    t, tx, rh = values[:3]
    if (not all(np.isfinite(x).all() for x in values) or not t.size
            or np.any(tx < t-.001) or np.any((rh < 0) | (rh > 100))
            or np.any(t < -80) or np.any(tx > 80)
            or (precipitation is not None and np.any(values[3] < 0))):
        raise ValueError('Finite compatible daily temperature, humidity and precipitation required.')
    return values


def _temperature_cycle(t, tx):
    phases = 2*np.pi*(np.arange(24)+.5)/24
    return t[..., None]+np.maximum(tx-t, 0)[..., None]*np.cos(phases)


def humidity_cycle(temperature, maximum_temperature, mean_humidity):
    """Infer a bounded 24-bin humidity profile matching daily mean RH.

    Temperature follows a symmetric diurnal cycle using the supplied mean and
    maximum. Vapour pressure is constant over that synthetic day and inferred
    by bisection after saturation clipping. Neither assumption is an hourly
    weather observation; their duration error needs independent verification.
    """
    t, tx, rh = _daily_inputs(temperature, maximum_temperature, mean_humidity)
    hourly = _temperature_cycle(t, tx)
    saturated = np.exp(17.625*hourly/(243.04+hourly))
    lower, upper = np.zeros(t.shape), saturated.max(axis=-1)
    for _ in range(30):
        vapor = (lower+upper)/2
        implied = np.minimum(vapor[..., None]/saturated, 1.).mean(axis=-1)
        below = implied < rh/100.
        lower = np.where(below, vapor, lower)
        upper = np.where(below, upper, vapor)
    vapor = (lower+upper)/2
    result = 100*np.minimum(vapor[..., None]/saturated, 1.)
    return np.where(rh[..., None] == 0., 0.,
                    np.where(rh[..., None] == 100., 100., result))


def duration_exposure(temperature, maximum_temperature, mean_humidity, precipitation,
                      rain_rate_mm_hour=.6):
    """Combine temperature-weighted humid duration and a daily rain-duration proxy.

    The expected rain overlap assumes temporal independence. Rain intensity is
    estimated from calibration meteorology; disease values are unnecessary.
    The same four daily variables are available in the climate projections.
    """
    t, tx, rh, rain = _daily_inputs(temperature, maximum_temperature, mean_humidity, precipitation)
    if not np.isfinite(rain_rate_mm_hour) or rain_rate_mm_hour <= 0:
        raise ValueError('Positive effective rainfall intensity required.')
    hourly = _temperature_cycle(t, tx)
    humidity = humidity_cycle(t, tx, rh)
    humid = humidity >= 90.-1e-7
    rain_fraction = -np.expm1(-rain/(24*rain_rate_mm_hour))
    temperature_response = np.exp(-.5*((hourly-18.)/8.)**2)
    exposure = (temperature_response*(humid+(~humid)*rain_fraction[..., None])).mean(axis=-1)
    return dict(exposure=exposure, humid_hours=humid.sum(axis=-1).astype(float),
                rain_hours=24*rain_fraction)
