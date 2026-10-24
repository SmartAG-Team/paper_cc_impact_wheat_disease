"""Causal daily-weather event summaries and fixed process-state predictors.

Humid/rain hours are reanalysis duration statistics, not measured leaf wetness.
The process trajectory is a generic continuously active cohort, not a verified
anatomical leaf. Its parameters are fixed defaults, never fitted to these labels.
"""
import numpy as np
import pandas as pd

from model.seasonal_septoria.core import Parameters, simulate_season
from model.seasonal_septoria.wetness import duration_exposure


def _longest_run(mask):
    best = run = 0
    for value in mask:
        run = run+1 if value else 0
        best = max(best, run)
    return best


def build_features(weather, fields, queries, config=None):
    config = config or {}
    cutoff = config.get('weather_cutoff_days', 0)
    rain_variable = config.get('rain_variable', 'rain_mm')
    if cutoff not in [0, 1] or rain_variable not in ['rain_mm', 'precipitation_mm']:
        raise ValueError('Valid weather cutoff and declared rain variable required')
    windows = config.get('window_days', [7, 14, 21, 42])
    w = weather.copy(); w['date'] = pd.to_datetime(w.date).dt.normalize()
    if w.duplicated(['location_id', 'date']).any():
        raise ValueError('Duplicate daily weather')
    field_index = fields.set_index('field_id')
    groups = dict(tuple(w.groupby('location_id', sort=False)))
    cache = {}
    for field_id, part in queries.groupby('field_id', sort=False):
        field = field_index.loc[field_id]
        start = pd.Timestamp(field.sowing_date)
        end = pd.to_datetime(part.date).max()-pd.Timedelta(days=cutoff)
        dates = pd.date_range(start, end)
        if field.site_id not in groups or not len(dates):
            raise ValueError('Missing weather support')
        daily = groups[field.site_id].set_index('date').reindex(dates)
        required = ['tmean_c', 'tmax_c', 'rh_mean_pct', rain_variable,
                    'rain_hours_gt_0_1mm', 'rh_hours_ge_90pct']
        a = daily[required].to_numpy(float)
        if not np.isfinite(a).all() or np.any(a[:, 3:] < 0) or np.any(a[:, 4:] > 24):
            raise ValueError('Incomplete or invalid daily weather')
        t, tx, rh, rain, rain_hours, humid_hours = a.T
        proxy = duration_exposure(t, tx, rh, rain)
        extra = dict(temperature=t, humidity=rh, rain=rain, rain_hours=rain_hours, humid_hours=humid_hours,
                     wet_proxy=proxy['exposure'], thermal=np.maximum(t, 0)/18)
        if config.get('include_process', True):
            active = np.ones((1, len(dates), 1), bool)
            trajectory = simulate_season(t[None], rh[None], rain[None], active,
                np.zeros(active.shape), Parameters())
            extra.update(process_damage=trajectory.damage[0, 1:, 0],
                         process_infectious=trajectory.infectious[0, 1:, 0],
                         process_latent=trajectory.state[0, 1:, 0, 1:4].sum(axis=1)
                         + trajectory.state[0, 1:, 0, 7:10].sum(axis=1))
        cache[field_id] = (start, extra)
    rows, provenance = [], []
    for row in queries.itertuples(index=False):
        start, series = cache[row.field_id]
        query_date = pd.Timestamp(row.date)
        now = query_date-pd.Timedelta(days=cutoff); day = (now-start).days
        if day < 0:
            raise ValueError('Assessment before weather start')
        record = {}
        for width in [7, 21, 42]:
            sl = slice(max(0, day-width+1), day+1)
            for variable in ['temperature', 'humidity', 'rain']:
                value = series[variable][sl].mean()
                name = f'weather_{variable}_{"log1p_" if variable == "rain" else ""}mean_{width}d'
                record[name] = np.log1p(value) if variable == 'rain' else value
        phase = 2*np.pi*(query_date.dayofyear-1)/365.2425
        record.update(weather_elapsed_days_scaled=((query_date-start).days+1)/330,
                      weather_calendar_sin=np.sin(phase), weather_calendar_cos=np.cos(phase))
        for width in windows:
            sl = slice(max(0, day-width+1), day+1)
            t, rain = series['temperature'][sl], series['rain'][sl]
            record.update({f'rain_total_{width}d_mm': rain.sum(),
                f'rain_max_{width}d_mm': rain.max(), f'rain_days_{width}d': (rain >= .1).sum(),
                f'rain_run_{width}d': _longest_run(rain >= .1),
                f'humid_hours_{width}d': series['humid_hours'][sl].sum(),
                f'rain_hours_{width}d': series['rain_hours'][sl].sum(),
                f'wet_dose_{width}d': series['wet_proxy'][sl].sum(),
                f'thermal_units_{width}d': series['thermal'][sl].sum(),
                f'temperature_std_{width}d': t.std()})
        previous = config.get('prior_dates', {}).get(int(row.global_target_index))
        previous = pd.Timestamp(previous) if previous is not None and not pd.isna(previous) else None
        first = (previous-start).days+1 if previous is not None else max(0, day-20)
        if first < 0 or first > day+1 or (previous is not None and previous >= query_date):
            raise ValueError('Invalid previous weather interval')
        gap = slice(first, day+1)
        rainy = np.flatnonzero(series['rain'][:day+1] >= .1)
        thermal_age = np.cumsum(series['thermal'][:day+1][::-1])[::-1] - series['thermal'][:day+1]
        record.update(gap_days=day-first+1, gap_rain_total_mm=series['rain'][gap].sum(),
            gap_humid_hours=series['humid_hours'][gap].sum(),
            gap_wet_dose=series['wet_proxy'][gap].sum(),
            gap_thermal_units=series['thermal'][gap].sum(),
            rain_age_days=day-rainy[-1] if len(rainy) else day+1,
            pending_infection_dose=series['wet_proxy'][:day+1][thermal_age < 20].sum(),
            matured_recent_dose=series['wet_proxy'][:day+1][(thermal_age >= 20) & (thermal_age < 40)].sum())
        for name in ['process_damage', 'process_infectious', 'process_latent']:
            if name in series:
                record[name] = series[name][day]
        rows.append(record)
        provenance.append(dict(global_target_index=int(row.global_target_index),
            weather_start=str(start.date()), weather_end=str(now.date()),
            rain_variable=rain_variable, weather_cutoff_days=cutoff,
            interval_start=str((start+pd.Timedelta(days=first)).date()),
            wetness_measurement='RH_and_precipitation_duration_inputs_plus_modeled_exposure_proxy; not_measured_canopy_leaf_wetness',
            process_host='generic_continuously_active_cohort'))
    x = pd.DataFrame(rows)
    if not np.isfinite(x.to_numpy()).all():
        raise ValueError('Finite event features required')
    return x.to_numpy(float), x.columns.tolist(), pd.DataFrame(provenance)
