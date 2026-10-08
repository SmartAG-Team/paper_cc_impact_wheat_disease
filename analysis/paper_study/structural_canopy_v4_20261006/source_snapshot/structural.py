"""Development-indexed canopy dynamics for an effective INFECT score.

Normalized states describe an empirical score capacity, not measured leaf
area or source-resolved infection routes. Delayed expression and subsequent
amplification are separate processes. Host growth dilutes affected capacity.
"""

from dataclasses import asdict, dataclass

import numpy as np
from numba import njit
from scipy.optimize import least_squares

from .refinement import calibration_weights


@dataclass(frozen=True)
class CanopyParameters:
    initiation: float = .003
    amplification: float = .05
    latent_days: float = 30.


@dataclass
class CanopyTrajectory:
    state: np.ndarray
    expressed: np.ndarray
    initiation_flow: np.ndarray
    amplification_flow: np.ndarray


def canopy_host(accumulation, temperature, thresholds, *, flag_fraction=.6,
                leaf_interval=120., expansion_units=100.):
    """Separate flag timing, leaf spacing and dilution by unfolding capacity.

    Spacing and expansion use the transferred development index. They are
    effective canopy parameters and are not measured thermal phyllochrons.
    The juvenile reservoir retains the original 300°C-day renewal assumption.
    Natural senescence is not recoded as an infection score or numerical zero.
    """
    accumulated, t = np.asarray(accumulation, float), np.asarray(temperature, float)
    events = np.asarray([thresholds[x] for x in (10, 31, 51, 85)], float)
    if (accumulated.ndim != 2 or not accumulated.size or t.shape != accumulated.shape
            or not np.isfinite(accumulated).all() or not np.isfinite(t).all()
            or np.any(accumulated < 0) or np.any(np.diff(accumulated, axis=1) < -1e-10)
            or not np.isfinite(events).all() or np.any(np.diff(events) <= 0)
            or not np.isfinite(flag_fraction) or not 0 < flag_fraction < 1
            or not np.isfinite(leaf_interval) or leaf_interval <= 0
            or not np.isfinite(expansion_units) or expansion_units < 0):
        raise ValueError('Valid development trajectories and canopy parameters required.')
    flag = events[1]+flag_fraction*(events[2]-events[1])
    births = np.maximum(events[0], flag-np.arange(7)*leaf_interval)
    births = np.append(births, events[0])
    prior = np.column_stack([np.zeros(len(accumulated)), accumulated[:, :-1]])
    active = prior[:, :, None] >= births[None, None, :]
    if expansion_units:
        area = np.clip((accumulated[:, :, None]-births[None, None, :])/expansion_units, 0., 1.)
        area *= active
        active &= area > 0
    else:
        area = active.astype(float)
    previous_area = np.concatenate([np.zeros_like(area[:, :1]), area[:, :-1]], axis=1)
    renewal = np.divide(np.maximum(area-previous_area, 0.), area,
                        out=np.zeros_like(area), where=area > 0)
    renewal[:, :, 7] = (-np.expm1(-np.maximum(t, 0)/300.))*active[:, :, 7]
    renewal[~active] = 0.
    return active, renewal, area, births


@njit(cache=True)
def _canopy_integrate(t, h, rain, active, renewal, initiation, amplification,
                      latent_days, substeps):
    fields, days, leaves = active.shape
    state = np.zeros((fields, days+1, leaves, 3))
    state[..., 0] = 1.
    external = np.zeros((fields, days, leaves))
    expansion = np.zeros_like(external)
    dt = 1./substeps
    for field in range(fields):
        current = state[field, 0].copy()
        # A FIFO thermal queue retains each exposure until its declared delay.
        # Cumulative retention applies leaf dilution lazily to queued cohorts.
        queue = np.zeros((days*substeps, leaves))
        maturity_heat = np.zeros(days*substeps)
        retained = np.ones(leaves)
        head, tail, heat = 0, 0, 0.
        for day in range(days):
            thermal = max(t[field, day], 0.)/18.
            temperature_response = np.exp(-.5*((t[field, day]-18.)/8.)**2)
            humidity = 1./(1+np.exp(-max(-700., min(700., (h[field, day]-85.)/5.))))
            water = -np.expm1(-rain[field, day]/2.)
            establish = temperature_response*(1-(1-humidity)*(1-water))
            for leaf in range(leaves):
                if not active[field, day, leaf]:
                    current[leaf, :] = 0.; current[leaf, 0] = 1.
                    queue[head:tail, leaf] = 0.
                    retained[leaf] = 1.
                else:
                    replacement = renewal[field, day, leaf]
                    if replacement == 1:
                        queue[head:tail, leaf] = 0.
                        retained[leaf] = 1.
                    else:
                        retained[leaf] *= 1-replacement
                    current[leaf, 1:] *= 1-replacement
                    current[leaf, 0] = 1-current[leaf, 1]-current[leaf, 2]
            for _ in range(substeps):
                updated = current.copy()
                for leaf in range(leaves):
                    if not active[field, day, leaf]:
                        continue
                    initial_hazard = initiation*establish
                    expansion_hazard = amplification*thermal*current[leaf, 2]
                    total = initial_hazard+expansion_hazard
                    consumed = current[leaf, 0]*(-np.expm1(-total*dt))
                    initial_flow, expansion_flow = 0., 0.
                    if total > 0:
                        initial_flow = consumed*initial_hazard/total
                        expansion_flow = consumed*expansion_hazard/total
                    updated[leaf, 0] -= consumed
                    updated[leaf, 1] += initial_flow
                    updated[leaf, 2] += expansion_flow
                    queue[tail, leaf] = initial_flow/retained[leaf]
                    external[field, day, leaf] += initial_flow
                    expansion[field, day, leaf] += expansion_flow
                heat += thermal*dt
                while head < tail and maturity_heat[head] <= heat+1e-10:
                    for leaf in range(leaves):
                        flow = queue[head, leaf]*retained[leaf]
                        updated[leaf, 1] -= flow
                        updated[leaf, 2] += flow
                    head += 1
                maturity_heat[tail] = heat+latent_days
                tail += 1
                current = updated
            state[field, day+1] = current
    return state, external, expansion


def simulate_canopy(temperature, rh_mean, rain, host_active, host_renewal,
                    parameters, time_step=.25):
    """Express delayed exposure and subsequent score growth without assimilation."""
    t, h, r = [np.asarray(value, float) for value in (temperature, rh_mean, rain)]
    raw_active, renewal = np.asarray(host_active), np.asarray(host_renewal, float)
    p = parameters
    if (t.ndim != 2 or not t.size or h.shape != t.shape or r.shape != t.shape
            or raw_active.ndim != 3 or raw_active.shape[:2] != t.shape
            or raw_active.shape[2] == 0 or renewal.shape != raw_active.shape
            or not all(np.isfinite(value).all() for value in (t, h, r, renewal))
            or np.any((h < 0) | (h > 100)) or np.any(r < 0)
            or np.any((raw_active != 0) & (raw_active != 1))
            or np.any((renewal < 0) | (renewal > 1))
            or np.any(renewal[raw_active == 0] != 0)):
        raise ValueError('Valid daily forcing and normalized canopy inputs required.')
    values = (p.initiation, p.amplification, p.latent_days, time_step)
    if (not np.isfinite(values).all() or p.initiation < 0 or p.amplification < 0
            or p.latent_days <= 0 or not 0 < time_step <= 1):
        raise ValueError('Nonnegative hazards, positive delay and valid integration step required.')
    state, external, expansion = _canopy_integrate(t, h, r, raw_active.astype(bool), renewal,
        p.initiation, p.amplification, p.latent_days,
        int(np.ceil(1/time_step)))
    return CanopyTrajectory(state, state[..., -1], external, expansion)


def predict_canopy(data, accumulation, thresholds, fitted):
    active, renewal, _, _ = canopy_host(accumulation, data.temperature, thresholds,
                                       **fitted['host_parameters'])
    trajectory = simulate_canopy(data.temperature, data.humidity, data.rain, active,
                                 renewal, CanopyParameters(**fitted['parameters']))
    target = data.targets
    idx = (target.field_index.to_numpy(int), target.day_index.to_numpy(int),
           target.leaf_index.to_numpy(int))
    return 100*trajectory.expressed[idx], trajectory


def fit_canopy(data, accumulation, thresholds, target_indices, host_parameters,
               amplification_free=True, latent_days=30., final_weight=.5):
    """Fit selected complete field-seasons without reading withheld scores."""
    indices = np.asarray(target_indices, int)
    if (indices.ndim != 1 or not indices.size or len(np.unique(indices)) != len(indices)
            or np.any(indices < 0) or np.any(indices >= len(data.targets))):
        raise ValueError('Unique nonempty calibration indices required.')
    target = data.targets.iloc[indices]
    fields = np.unique(target.field_index.to_numpy(int))
    membership = np.flatnonzero(data.targets.field_index.isin(fields))
    if not np.array_equal(np.sort(indices), membership):
        raise ValueError('Calibration cannot split a field-season.')
    if not target.metric.eq('infection_percent_unspecified_basis').all():
        raise ValueError('Structural INFECT calibration requires source percentage proxies.')
    observed = target.value.to_numpy(float)
    if not np.isfinite(observed).all() or np.any((observed < 0) | (observed > 100)):
        raise ValueError('Finite percentage calibration observations required.')
    mapped = {field:index for index, field in enumerate(fields)}
    field = target.field_index.map(mapped).to_numpy(int)
    day, leaf = target.day_index.to_numpy(int), target.leaf_index.to_numpy(int)
    active, renewal, _, _ = canopy_host(accumulation[fields], data.temperature[fields],
                                       thresholds, **host_parameters)
    forcing = (data.temperature[fields], data.humidity[fields], data.rain[fields], active, renewal)
    weight = calibration_weights(target, final_weight)

    def unpack(x):
        return CanopyParameters(initiation=float(10**x[0]),
            amplification=float(10**x[1]) if amplification_free else 0., latent_days=latent_days)

    def residual(x):
        result = simulate_canopy(*forcing, unpack(x))
        return np.sqrt(weight)*(100*result.expressed[field, day, leaf]-observed)

    lower, upper = ([-6., -5.], [-1., np.log10(.3)]) if amplification_free else ([-6.], [-1.])
    records = []
    for log_initial, log_expansion in [(-2.5, -1.3), (-1.5, -2.), (-3.5, -1.)]:
        start = [log_initial, log_expansion] if amplification_free else [log_initial]
        result = least_squares(residual, start, bounds=(lower, upper), max_nfev=150,
                               ftol=1e-7, xtol=1e-7, gtol=1e-7)
        records.append(dict(start=start, parameters=asdict(unpack(result.x)),
            weighted_sse=float(np.square(result.fun).sum()), success=bool(result.success),
            evaluations=int(result.nfev), termination=str(result.message)))
    success = [i for i, row in enumerate(records) if row['success'] and np.isfinite(row['weighted_sse'])]
    if not success:
        raise RuntimeError('No structural calibration start converged.')
    winner = min(success, key=lambda i: records[i]['weighted_sse'])
    return dict(parameters=records[winner]['parameters'], host_parameters=host_parameters,
        amplification_free=bool(amplification_free), final_weight=final_weight,
        target_indices=indices.tolist(), field_indices=fields.tolist(),
        calibration_field_ids=sorted(target.field_id.unique().tolist()),
        training_rmse=float(np.sqrt(records[winner]['weighted_sse']/weight.sum())),
        multistart_records=records, winning_start=winner, calibration_only=True,
        observed_disease_assimilated=False, observation='source_INFECT_percentage_proxy')
