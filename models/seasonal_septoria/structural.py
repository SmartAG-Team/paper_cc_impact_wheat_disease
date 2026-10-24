"""Development-indexed canopy dynamics for an effective INFECT score.

Normalized states describe an empirical score capacity, not measured leaf
area or source-resolved infection routes. Delayed expression and subsequent
amplification are separate processes. Host growth dilutes affected capacity.
"""

from dataclasses import dataclass

import numpy as np
from numba import njit



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
def _canopy_integrate(t, exposure, active, renewal, initiation, amplification,
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
            establish = exposure[field, day]
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
                        if retained[leaf] < 1e-100:
                            queue[head:tail, leaf] *= retained[leaf]
                            retained[leaf] = 1.
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
                    parameters, time_step=.25, environmental_response=None):
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
    if environmental_response is None:
        humid = 1./(1+np.exp(-np.clip((h-85.)/5., -700., 700.)))
        water = -np.expm1(-r/2.)
        exposure = np.exp(-.5*((t-18.)/8.)**2)*(1-(1-humid)*(1-water))
    else:
        exposure = np.asarray(environmental_response, float)
        if (exposure.shape != t.shape or not np.isfinite(exposure).all()
                or np.any((exposure < 0) | (exposure > 1))):
            raise ValueError('Environmental exposure must match daily forcing and lie in [0,1].')
    state, external, expansion = _canopy_integrate(t, exposure, raw_active.astype(bool), renewal,
        p.initiation, p.amplification, p.latent_days,
        int(np.ceil(1/time_step)))
    return CanopyTrajectory(state, state[..., -1], external, expansion)


