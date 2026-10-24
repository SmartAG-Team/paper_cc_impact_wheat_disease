"""Disease fractions from crop establishment with daily climate forcing."""

from dataclasses import dataclass

import numpy as np
from numba import njit


@dataclass(frozen=True)
class Parameters:
    alpha: float = .01
    beta: float = 1.
    latent_days: float = 20.
    latent_stages: int = 3
    nonsporulating_days: float = 3.
    infectious_days: float = 21.
    humidity_midpoint: float = 85.
    humidity_scale: float = 5.
    rain_scale: float = 2.
    temperature_optimum: float = 18.
    temperature_width: float = 8.
    rank_distance_scale: float = 2.


@dataclass
class Trajectory:
    state: np.ndarray
    damage: np.ndarray
    pycnidia: np.ndarray
    infectious: np.ndarray
    origin_total: np.ndarray
    primary_flow: np.ndarray
    secondary_flow: np.ndarray


@njit(cache=True)
def _integrate(temperature, humidity, rain, active, renewal, kernel,
               coefficients, stages, substeps):
    (alpha, beta, latent_days, nonsporulating_days, infectious_days,
     humidity_midpoint, humidity_scale, rain_scale, optimum, width) = coefficients
    episodes, days, leaves = active.shape
    block = stages+3
    state = np.zeros((episodes, days+1, leaves, 1+2*block))
    state[..., 0] = 1.
    primary = np.zeros((episodes, days, leaves))
    secondary = np.zeros_like(primary)
    dt = 1./substeps
    q_removed = -np.expm1(-dt/infectious_days)
    for episode in range(episodes):
        current = state[episode, 0].copy()
        for day in range(days):
            for leaf in range(leaves):
                if not active[episode, day, leaf]:
                    current[leaf, :] = 0.
                    current[leaf, 0] = 1.
                else:
                    retained = 0.
                    for compartment in range(1, current.shape[1]):
                        current[leaf, compartment] *= 1-renewal[episode, day, leaf]
                        retained += current[leaf, compartment]
                    current[leaf, 0] = 1-retained
            thermal = max(temperature[episode, day], 0.)/18.
            thermal_establishment = np.exp(-.5*((temperature[episode, day]-optimum)/width)**2)
            humidity_response = 1./(1+np.exp(-max(-700., min(700.,
                (humidity[episode, day]-humidity_midpoint)/humidity_scale))))
            splash = -np.expm1(-rain[episode, day]/rain_scale)
            establish = thermal_establishment*(1-(1-humidity_response)*(1-splash))
            q_latent = -np.expm1(-dt*stages*thermal/latent_days)
            q_sporulate = -np.expm1(-dt*thermal/nonsporulating_days)
            for _ in range(substeps):
                infectious = np.zeros(leaves)
                for leaf in range(leaves):
                    if active[episode, day, leaf]:
                        infectious[leaf] = current[leaf, 1+stages+1]+current[leaf, 1+block+stages+1]
                updated = current.copy()
                for leaf in range(leaves):
                    if not active[episode, day, leaf]:
                        continue
                    imported, denominator = 0., 0.
                    for other in range(leaves):
                        if active[episode, day, other]:
                            imported += kernel[leaf, other]*infectious[other]
                            denominator += kernel[leaf, other]
                    primary_hazard = alpha*establish
                    secondary_hazard = beta*establish*splash*imported/denominator
                    total_hazard = primary_hazard+secondary_hazard
                    newly_infected = current[leaf, 0]*(-np.expm1(-total_hazard*dt))
                    p_flow, s_flow = 0., 0.
                    if total_hazard > 0:
                        p_flow = newly_infected*primary_hazard/total_hazard
                        s_flow = newly_infected*secondary_hazard/total_hazard
                    updated[leaf, 0] -= newly_infected
                    updated[leaf, 1] += p_flow
                    updated[leaf, 1+block] += s_flow
                    primary[episode, day, leaf] += p_flow
                    secondary[episode, day, leaf] += s_flow
                    for origin in range(2):
                        start = 1+origin*block
                        for latent in range(stages):
                            transfer = current[leaf, start+latent]*q_latent
                            updated[leaf, start+latent] -= transfer
                            updated[leaf, start+latent+1] += transfer
                        visible = start+stages
                        to_infectious = current[leaf, visible]*q_sporulate
                        to_removed = current[leaf, visible+1]*q_removed
                        updated[leaf, visible] -= to_infectious
                        updated[leaf, visible+1] += to_infectious-to_removed
                        updated[leaf, visible+2] += to_removed
                current = updated
            state[episode, day+1] = current
    return state, primary, secondary


def simulate_season(temperature, rh_mean, rain, host_active, host_renewal,
                    parameters, time_step=.25, leaf_ranks=None):
    """Simulate from susceptible host without assimilating disease observations.

    Forcing is daily mean temperature(°C), daily mean relative humidity(%) and
    precipitation(mm). Humidity response is a fitted daily proxy, not humid
    hours or leaf wetness. Host renewal denotes the daily newly replaced share
    of a normalized leaf cohort. The state at index d+1 follows forcing day d.
    Inactive cohorts are susceptible placeholders and cannot participate in
    infection or transport.
    """
    t, h, r = [np.asarray(x, float) for x in (temperature, rh_mean, rain)]
    active_raw = np.asarray(host_active)
    active = active_raw.astype(bool)
    renewal = np.asarray(host_renewal, float)
    p = parameters
    if (t.ndim != 2 or t.size == 0 or t.shape != h.shape or t.shape != r.shape
            or active.ndim != 3 or active.shape[:2] != t.shape
            or active.shape[2] == 0 or renewal.shape != active.shape):
        raise ValueError('Daily weather and cohort input shapes do not match.')
    if (not all(np.isfinite(x).all() for x in (t, h, r, renewal))
            or np.any((active_raw != 0) & (active_raw != 1))
            or np.any((h < 0) | (h > 100)) or np.any(r < 0)
            or np.any((renewal < 0) | (renewal > 1)) or np.any(renewal[~active] != 0)):
        raise ValueError('Weather or host fractions are missing or outside their physical ranges.')
    coefficients = (p.alpha, p.beta, p.latent_days, p.nonsporulating_days,
                    p.infectious_days, p.humidity_midpoint, p.humidity_scale,
                    p.rain_scale, p.temperature_optimum, p.temperature_width)
    positive = (p.latent_days, p.nonsporulating_days, p.infectious_days,
                p.humidity_scale, p.rain_scale, p.temperature_width, p.rank_distance_scale)
    if (not np.isfinite(coefficients).all() or p.alpha < 0 or p.beta < 0
            or not all(np.isfinite(x) and x > 0 for x in positive)
            or not 0 <= p.humidity_midpoint <= 100
            or not isinstance(p.latent_stages, (int, np.integer)) or p.latent_stages < 1
            or not np.isfinite(time_step) or not 0 < time_step <= 1):
        raise ValueError('Model coefficients or time step are outside their valid ranges.')
    leaves = active.shape[-1]
    ranks = np.arange(leaves) if leaf_ranks is None else np.asarray(leaf_ranks, float)
    if ranks.shape != (leaves,) or not np.isfinite(ranks).all() or np.unique(ranks).size != leaves:
        raise ValueError('Leaf ranks must be finite, unique and match the cohort count.')
    kernel = np.exp(-abs(ranks[:, None]-ranks[None, :])/p.rank_distance_scale)
    state, primary, secondary = _integrate(t, h, r, active, renewal, kernel,
                                          coefficients, p.latent_stages,
                                          int(np.ceil(1/time_step)))
    damage = np.zeros(state.shape[:3])
    pycnidia = np.zeros_like(damage)
    infectious = np.zeros_like(damage)
    origin_total = np.zeros((*damage.shape, 2))
    block = p.latent_stages+3
    for origin in range(2):
        start = 1+origin*block
        visible = start+p.latent_stages
        damage += state[..., visible:visible+3].sum(axis=-1)
        pycnidia += state[..., visible+1:visible+3].sum(axis=-1)
        infectious += state[..., visible+1]
        origin_total[..., origin] = state[..., start:start+block].sum(axis=-1)
    return Trajectory(state, damage, pycnidia, infectious, origin_total, primary, secondary)
