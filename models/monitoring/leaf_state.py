"""Causal leaf-state proxies updated from earlier reported visible severity.

S/E1/E2/E3/V/I/R combines the original process engine's two infection origins.
All seven native leaf codes are continuously active scenario cohorts. Neither
their emergence nor latent infection is observed; these are modeling states.
"""
import numpy as np
import pandas as pd

from model.seasonal_septoria.core import Parameters

STATE_NAMES = ['state_susceptible', 'state_latent_1', 'state_latent_2',
    'state_latent_3', 'state_visible_nonsporulating', 'state_infectious',
    'state_removed', 'state_other_damage_mean', 'state_other_damage_max',
    'state_other_infectious_proximity', 'state_same_assimilated',
    'state_any_assimilated', 'state_change_from_last_same']


def _valid_state(state):
    value = np.asarray(state, float)
    if (value.ndim != 2 or not len(value) or value.shape[1] != 7
            or not np.isfinite(value).all() or np.any(value < 0)
            or not np.allclose(value.sum(axis=1), 1, atol=1e-10, rtol=0)):
        raise ValueError('Nonnegative unit-mass seven-compartment leaf states required')
    return value


def assimilate_visible(state, severity_fraction):
    """Match visible mass; preserve relative modeled states within each block."""
    original = _valid_state(state)
    y = np.asarray(severity_fraction, float)
    if y.shape != (len(original),) or not np.isfinite(y).all() or np.any((y < 0) | (y > 1)):
        raise ValueError('Aligned bounded observed severity fractions required')
    result = np.zeros_like(original)
    for block, mass, fallback in [(slice(0, 4), 1-y, 0), (slice(4, 7), y, 4)]:
        total = original[:, block].sum(axis=1)
        scale = np.divide(mass, total, out=np.zeros_like(mass), where=total > 0)
        result[:, block] = original[:, block]*scale[:, None]
        result[total == 0, fallback] = mass[total == 0]
    return result


def _advance(state, temperature, humidity, rain, p, substeps, kernel):
    current = state.copy()
    dt = 1/substeps
    thermal = max(temperature, 0)/18
    humidity_response = 1/(1+np.exp(-np.clip((humidity-p.humidity_midpoint)/p.humidity_scale, -700, 700)))
    splash = -np.expm1(-rain/p.rain_scale)
    establish = np.exp(-.5*((temperature-p.temperature_optimum)/p.temperature_width)**2)
    establish *= 1-(1-humidity_response)*(1-splash)
    q_latent = -np.expm1(-dt*3*thermal/p.latent_days)
    q_sporulate = -np.expm1(-dt*thermal/p.nonsporulating_days)
    q_removed = -np.expm1(-dt/p.infectious_days)
    for _ in range(substeps):
        infectious = kernel@current[:, 5]/kernel.sum(axis=1)
        hazard = establish*(p.alpha+p.beta*splash*infectious)
        infected = current[:, 0]*(-np.expm1(-dt*hazard))
        latent = current[:, 1:4]*q_latent
        sporulate = current[:, 4]*q_sporulate
        removed = current[:, 5]*q_removed
        updated = current.copy()
        updated[:, 0] -= infected
        updated[:, 1] += infected-latent[:, 0]
        updated[:, 2:4] += latent[:, :2]-latent[:, 1:]
        updated[:, 4] += latent[:, 2]-sporulate
        updated[:, 5] += sporulate-removed
        updated[:, 6] += removed
        current = updated
    return current


def advance_state(state, temperature, humidity, rain, parameters=None, time_step=.25):
    original = _valid_state(state)
    p = parameters or Parameters()
    values = [temperature, humidity, rain, time_step, p.alpha, p.beta, p.latent_days,
        p.nonsporulating_days, p.infectious_days, p.humidity_midpoint,
        p.humidity_scale, p.rain_scale, p.temperature_optimum,
        p.temperature_width, p.rank_distance_scale]
    if (not np.isfinite(values).all() or not 0 <= humidity <= 100 or rain < 0
            or not 0 < time_step <= 1 or min(p.alpha, p.beta) < 0 or p.latent_stages != 3
            or min(p.latent_days, p.nonsporulating_days, p.infectious_days,
                   p.humidity_scale, p.rain_scale, p.temperature_width, p.rank_distance_scale) <= 0):
        raise ValueError('Valid daily weather and process parameters required')
    ranks = np.arange(len(original))
    kernel = np.exp(-abs(ranks[:, None]-ranks[None, :])/p.rank_distance_scale)
    return _advance(original, temperature, humidity, rain, p, int(np.ceil(1/time_step)), kernel)


def build_state_features(weather, fields, observations, queries):
    """Return features before each query day's observations and forcing."""
    if 'observed_percent' in queries:
        raise ValueError('Current query labels must be withheld')
    required = ['global_target_index', 'field_id', 'site_id', 'source', 'leaf_index', 'date']
    for frame in [queries, observations]:
        if frame[required].isna().any().any() or frame.global_target_index.duplicated().any():
            raise ValueError('Complete unique observation and query identities required')
        if not frame.leaf_index.isin(range(7)).all():
            raise ValueError('Native integer leaf codes zero through six required')
    if not observations.observed_percent.between(0, 100).all():
        raise ValueError('Finite bounded observation severities required')
    obs = observations.copy(); q = queries.copy(); w = weather.copy()
    for frame in [obs, q, w]:
        frame['date'] = pd.to_datetime(frame.date, errors='raise', format='mixed').dt.normalize()
        if frame.date.isna().any():
            raise ValueError('Valid dates required')
    if obs.duplicated(['field_id', 'leaf_index', 'date']).any():
        raise ValueError('Duplicate field/leaf/day disease observations')
    if w.duplicated(['location_id', 'date']).any():
        raise ValueError('Duplicate daily weather')
    if fields.field_id.duplicated().any():
        raise ValueError('Unique field metadata required')
    meta = fields.set_index('field_id')
    wg = {key: g.set_index('date').sort_index() for key, g in w.groupby('location_id')}
    og = dict(tuple(obs.groupby('field_id', sort=False)))
    rows, provenance = {}, {}
    p = Parameters(); rank = np.arange(7)
    kernel = np.exp(-abs(rank[:, None]-rank[None, :])/p.rank_distance_scale)
    for field_id, queries_here in q.groupby('field_id', sort=False):
        field = meta.loc[field_id]
        start = pd.Timestamp(field.sowing_date).normalize()
        final = queries_here.date.max()
        if start >= queries_here.date.min():
            raise ValueError('At least one earlier weather day required after sowing')
        context = og.get(field_id, obs.iloc[:0])
        if (not queries_here.site_id.eq(field.site_id).all()
                or (len(context) and (not context.site_id.eq(field.site_id).all()
                    or context.source.nunique() != 1
                    or not queries_here.source.eq(context.source.iloc[0]).all()))):
            raise ValueError('Conflicting field metadata')
        context = context.loc[context.date.lt(final)]
        if context.date.lt(start).any():
            raise ValueError('Disease history before declared season start')
        dates = pd.date_range(start, final-pd.Timedelta(days=1))
        if field.site_id not in wg:
            raise ValueError('Missing field weather')
        forcing = wg[field.site_id].reindex(dates)[['tmean_c', 'rh_mean_pct', 'rain_mm']].to_numpy(float)
        if (not np.isfinite(forcing).all() or np.any((forcing[:, 1] < 0) | (forcing[:, 1] > 100))
                or np.any(forcing[:, 2] < 0)):
            raise ValueError('Incomplete or invalid earlier weather')
        by_day = dict(tuple(context.groupby('date', sort=False)))
        by_query = dict(tuple(queries_here.groupby('date', sort=False)))
        state = np.zeros((7, 7)); state[:, 0] = 1
        seen = np.zeros(7, bool); last = np.zeros(7); prior_ids = []
        latest = None
        for offset, day in enumerate(pd.date_range(start, final)):
            if day in by_query:
                damage = state[:, 4:].sum(axis=1)
                for row in by_query[day].itertuples(index=False):
                    code = int(row.leaf_index); other = rank != code
                    proximity = kernel[code, other]
                    features = np.r_[state[code], damage[other].mean(), damage[other].max(),
                        np.average(state[other, 5], weights=proximity),
                        float(seen[code]), float(seen.any()), damage[code]-last[code] if seen[code] else 0.]
                    rows[int(row.global_target_index)] = features
                    provenance[int(row.global_target_index)] = dict(
                        global_target_index=int(row.global_target_index), prior_target_ids=prior_ids.copy(),
                        latest_assimilated_date=str(latest.date()) if latest is not None else None,
                        weather_start=str(start.date()), weather_end=str((day-pd.Timedelta(days=1)).date()),
                        state_basis='Unvalidated S/E1/E2/E3/V/I/R scenario; continuously active native leaf codes; visible-severity updates only',
                        no_current_or_future_observation_used=True)
            if day == final:
                break
            if day in by_day:
                today = by_day[day]
                codes = today.leaf_index.to_numpy(int)
                state[codes] = assimilate_visible(state[codes], today.observed_percent.to_numpy(float)/100)
                seen[codes] = True; last[codes] = today.observed_percent.to_numpy(float)/100
                prior_ids += today.global_target_index.astype(int).tolist(); latest = day
            state = _advance(state, *forcing[offset], p, 4, kernel)
            _valid_state(state)
    ids = queries.global_target_index.astype(int).tolist()
    x = np.asarray([rows[i] for i in ids])
    if x.shape != (len(queries), len(STATE_NAMES)) or not np.isfinite(x).all():
        raise ValueError('Complete finite state features required')
    return x, STATE_NAMES.copy(), pd.DataFrame([provenance[i] for i in ids])
