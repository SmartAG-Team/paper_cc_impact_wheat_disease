"""Origin-aware, multi-stage Septoria simulator.

Initial, external and secondary infection origins remain separate. The model
tracks tissue fractions, not measured spore numbers. See SPEC.txt for scope.
"""
from dataclasses import dataclass
import numpy as np
from numba import njit


@dataclass(frozen=True)
class Parameters:
    alpha: float = 0.0
    beta: float = 0.0
    latent_days: float = 20.0
    nonsporulating_days: float = 3.0
    infectious_days: float = 21.0
    latent_stages: int = 3
    rank_distance_scale: float = 1.0
    weather_response: bool = True


@dataclass
class Trajectory:
    state: np.ndarray
    damage: np.ndarray
    pycnidia: np.ndarray
    infectious: np.ndarray
    origin_total: np.ndarray
    primary_flux: np.ndarray
    secondary_flux: np.ndarray


@njit(cache=True)
def _integrate(initial, temperature, humidity, rain, alpha, beta, latent_days,
               nonsporulating_days, infectious_days, stages, kernel,
               weather_response, substeps, active):
    episodes, leaves, compartments = initial.shape
    days = temperature.shape[1]
    state = np.empty((episodes, days+1, leaves, compartments))
    primary = np.zeros((episodes, days, leaves))
    secondary = np.zeros_like(primary)
    state[:, 0] = initial
    dt = 1.0/substeps
    block = stages+3
    q_removed = -np.expm1(-dt/infectious_days)
    for episode in range(episodes):
        current = initial[episode].copy()
        for day in range(days):
            thermal = max(temperature[episode, day], 0.0)/18.0
            establish = 1.0
            splash = 1.0
            if weather_response:
                establish = (np.exp(-.5*((temperature[episode, day]-18.0)/8.0)**2)
                             * humidity[episode, day]/24.0)
                splash = -np.expm1(-rain[episode, day]/2.0)
            q_latent = -np.expm1(-dt*stages*thermal/latent_days)
            q_sporulate = -np.expm1(-dt*thermal/nonsporulating_days)
            for _ in range(substeps):
                infectious = np.zeros(leaves)
                for leaf in range(leaves):
                    for origin in range(3):
                        infectious[leaf] += current[leaf, 1+origin*block+stages+1]
                updated = current.copy()
                for leaf in range(leaves):
                    if not active[episode, leaf]:
                        continue
                    imported_infectious = 0.0
                    for other in range(leaves):
                        imported_infectious += kernel[episode, leaf, other]*infectious[other]
                    primary_hazard = alpha*establish
                    secondary_hazard = beta*establish*splash*imported_infectious
                    hazard = primary_hazard+secondary_hazard
                    newly_infected = current[leaf, 0]*(-np.expm1(-hazard*dt))
                    new_primary = 0.0
                    new_secondary = 0.0
                    if hazard > 0:
                        new_primary = newly_infected*primary_hazard/hazard
                        new_secondary = newly_infected*secondary_hazard/hazard
                    updated[leaf, 0] -= newly_infected
                    updated[leaf, 1+block] += new_primary
                    updated[leaf, 1+2*block] += new_secondary
                    primary[episode, day, leaf] += new_primary
                    secondary[episode, day, leaf] += new_secondary
                    for origin in range(3):
                        start = 1+origin*block
                        for stage in range(stages):
                            transitioned = current[leaf, start+stage]*q_latent
                            updated[leaf, start+stage] -= transitioned
                            updated[leaf, start+stage+1] += transitioned
                        visible = start+stages
                        to_sporulating = current[leaf, visible]*q_sporulate
                        to_removed = current[leaf, visible+1]*q_removed
                        updated[leaf, visible] -= to_sporulating
                        updated[leaf, visible+1] += to_sporulating-to_removed
                        updated[leaf, visible+2] += to_removed
                current = updated
            state[episode, day+1] = current
    return state, primary, secondary


def simulate(initial_visible, temperature, humidity_hours, rain, parameters,
             initial_latent=0.0, time_step=.25, leaf_ranks=None, leaf_active=None):
    """Simulate each episode using daily forcing before its target assessments.

    Visible initial tissue starts in infectious tissue with unresolved origin.
    Initial latent tissue is a fraction of unaffected tissue, uniformly split
    across latent stages to represent unresolved infection ages. No later
    observations are assimilated. Optional ranks define the transport kernel.
    """
    initial_visible = np.asarray(initial_visible, dtype=float)
    temperature, humidity_hours, rain = [np.asarray(x, dtype=float)
                                        for x in (temperature, humidity_hours, rain)]
    p = parameters
    if initial_visible.ndim != 2 or temperature.ndim != 2:
        raise ValueError('Initial fractions and daily forcing require two dimensions.')
    if temperature.shape != humidity_hours.shape or temperature.shape != rain.shape:
        raise ValueError('Daily forcing shapes differ.')
    if len(initial_visible) != len(temperature) or initial_visible.shape[1] < 1:
        raise ValueError('Episode counts or leaf counts differ.')
    if not all(np.isfinite(x).all() for x in (initial_visible, temperature, humidity_hours, rain)):
        raise ValueError('Initial state or daily forcing contains missing/nonfinite values.')
    if np.any((initial_visible < 0) | (initial_visible > 1)):
        raise ValueError('Visible fractions must lie in[0,1].')
    if np.any((humidity_hours < 0) | (humidity_hours > 24)) or np.any(rain < 0):
        raise ValueError('Humidity hours or rainfall are outside their physical ranges.')
    if not np.isfinite(initial_latent) or not 0 <= initial_latent <= 1:
        raise ValueError('Initial latent fraction must lie in[0,1].')
    if not all(np.isfinite(x) and x >= 0 for x in (p.alpha, p.beta)):
        raise ValueError('Transmission coefficients must be nonnegative and finite.')
    if not all(np.isfinite(x) and x > 0 for x in
               (p.latent_days, p.nonsporulating_days, p.infectious_days, p.rank_distance_scale)):
        raise ValueError('Process durations and rank-distance scale must be positive and finite.')
    if not isinstance(p.latent_stages, (int, np.integer)) or p.latent_stages < 1:
        raise ValueError('Latent stage count must be a positive integer.')
    if not np.isfinite(time_step) or not 0 < time_step <= 1:
        raise ValueError('Time step must lie in(0,1].')
    substeps = int(np.ceil(1/time_step))
    n, leaves = initial_visible.shape
    ranks = np.arange(1, leaves+1, dtype=float) if leaf_ranks is None else np.asarray(leaf_ranks, float)
    if ranks.shape != (leaves,) or not np.isfinite(ranks).all() or len(np.unique(ranks)) != leaves:
        raise ValueError('Leaf ranks must be unique, finite and match the leaf axis.')
    active = np.ones((n,leaves), dtype=bool) if leaf_active is None else np.asarray(leaf_active, dtype=bool)
    if active.shape != (n,leaves) or not active.any(axis=1).all():
        raise ValueError('Each episode needs at least one active leaf and a matching active mask.')
    if np.any(initial_visible[~active] != 0):
        raise ValueError('Inactive leaves cannot have visible disease.')
    base_kernel = np.exp(-abs(ranks[:, None]-ranks[None, :])/p.rank_distance_scale)
    kernel = np.tile(base_kernel[None,:,:], (n,1,1))*active[:,None,:]
    kernel /= kernel.sum(axis=2, keepdims=True)
    block = p.latent_stages+3
    initial = np.zeros((n, leaves, 1+3*block))
    latent = (1-initial_visible)*initial_latent*active
    initial[..., 0] = 1-initial_visible-latent
    initial[..., 1:1+p.latent_stages] = latent[..., None]/p.latent_stages
    initial[..., 1+p.latent_stages+1] = initial_visible
    state, primary, secondary = _integrate(
        initial, temperature, humidity_hours, rain, p.alpha, p.beta, p.latent_days,
        p.nonsporulating_days, p.infectious_days, p.latent_stages, kernel,
        p.weather_response, substeps, active)
    damage = np.zeros(state.shape[:3])
    pycnidia = np.zeros_like(damage)
    infectious = np.zeros_like(damage)
    origin_total = np.zeros((*damage.shape, 3))
    for origin in range(3):
        start = 1+origin*block
        visible = start+p.latent_stages
        damage += state[..., visible:visible+3].sum(axis=-1)
        pycnidia += state[..., visible+1:visible+3].sum(axis=-1)
        infectious += state[..., visible+1]
        origin_total[..., origin] = state[..., start:start+block].sum(axis=-1)
    return Trajectory(state, damage, pycnidia, infectious, origin_total, primary, secondary)
