"""Causal daily adapter to the frozen climate process equations.

The climate census wrapper requires eventual crop completion for a seasonal
output. This adapter also returns valid in-progress daily trajectories. It
does not change any frozen equation, source file or whole-season result.
"""
import numpy as np
import pandas as pd

from analysis.final_climate.kernel import FIELDS
from model.seasonal_septoria.regional import phenology_window, tpv_accumulation
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.overwinter import OverwinterParameters, simulate_overwinter
from model.seasonal_septoria.wetness import duration_exposure


def simulate_prefix(dates, weather, sowing, latitude, phenology, fit, thresholds):
    dates = pd.DatetimeIndex(dates)
    raw = {name: np.asarray(weather[name], float) for name in FIELDS}
    t = raw['tmean_c']; sowing = np.asarray(sowing, dtype='datetime64[D]')
    latitude = np.asarray(latitude, float)
    if (t.ndim != 2 or not t.size or any(a.shape != t.shape for a in raw.values())
            or len(dates) != t.shape[1] or not dates.equals(pd.date_range(dates[0], dates[-1]))
            or sowing.shape != (len(t),) or latitude.shape != (len(t),) or np.isnat(sowing).any()
            or (sowing < dates[0].to_datetime64().astype('datetime64[D]')).any()):
        raise ValueError('Consecutive dated four-weather forcing covering each sowing required')
    included = dates.to_numpy(dtype='datetime64[D]')[None] >= sowing[:, None]
    finite = np.logical_and.reduce([np.isfinite(a) for a in raw.values()])
    finite &= ((raw['tmax_c'] >= t-.001) & (raw['rh_mean_pct'] >= 0)
        & (raw['rh_mean_pct'] <= 100) & (raw['precipitation_mm'] >= 0)
        & (t >= -80) & (raw['tmax_c'] <= 80))
    unsupported = included & finite & (t > 40)
    invalid = included & (~finite | unsupported)
    first_gap = np.where(invalid.any(axis=1), invalid.argmax(axis=1), t.shape[1])
    prefix = included & (np.arange(t.shape[1])[None] < first_gap[:, None])
    safe = {name: np.where(finite & (t <= 40), value, 0.) for name, value in raw.items()}
    mask, complete, endpoint, _ = phenology_window(safe['tmean_c'], safe['tmax_c'], latitude,
        dates.dayofyear, prefix, phenology, float(thresholds[85]), True)
    # Missing/unsupported future forcing terminates a prefix rather than
    # retrospectively invalidating every earlier state in that field.
    temperature = np.where(mask, safe['tmean_c'], 0.)
    maximum = np.where(mask, safe['tmax_c'], 0.)
    accumulation = tpv_accumulation(temperature, maximum, latitude, dates.dayofyear, mask, phenology, True)
    host = leaf_host(accumulation, temperature, thresholds,
        rank_spacing_units=fit['rank_spacing_units'], forcing_mask=mask,
        juvenile_policy=fit.get('juvenile_policy', 'handover_31_39'))
    rain = np.where(mask, safe['precipitation_mm'], 0.)
    exposure = duration_exposure(temperature, maximum, np.where(mask, safe['rh_mean_pct'], 0.), rain,
        fit['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    trajectory = simulate_overwinter(temperature, exposure, rain, host.active, host.renewal, host.area, mask,
        OverwinterParameters(**fit['parameters']), initial_local_source=1.,
        imported_pressure=np.full_like(temperature, fit['constant_imported_pressure']), detection_fraction=.001)
    if np.max(np.abs(trajectory.state.sum(axis=-1)-1.)) > 1e-10:
        raise AssertionError('Daily-prefix tissue mass failed')
    first_is_hot = np.array([bool(unsupported[i, j]) if j < t.shape[1] else False
        for i, j in enumerate(first_gap)])
    status = np.where(complete, 'complete', np.where(first_is_hot, 'unsupported_temperature',
        np.where(first_gap < t.shape[1], 'missing_weather', 'in_progress')))
    return dict(trajectory=trajectory, host=host, accumulation=accumulation,
        mask=mask, complete=complete, endpoint=endpoint, status=status, first_gap=first_gap)
