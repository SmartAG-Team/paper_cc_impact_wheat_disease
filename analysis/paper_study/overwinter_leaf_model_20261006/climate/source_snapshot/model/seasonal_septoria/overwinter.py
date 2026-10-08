"""Seasonal residue and living-canopy states for computational STB forecasting.

Relative source potential and normalized affected tissue are model states;
neither is a measured spore count or a calibrated healthy leaf-area loss.
Numerical kernels are shared by the daily object and batch field adapter.
"""

from dataclasses import asdict, dataclass
from uuid import uuid4

import numpy as np
from numba import njit


@dataclass(frozen=True)
class OverwinterParameters:
    """Effective scenario parameters; reference durations use the 18 C clock."""

    primary_scale: float = .01
    secondary_scale: float = 1.
    latent_reference_days: float = 20.
    latent_stages: int = 3
    nonsporulating_reference_days: float = 3.
    infectious_reference_days: float = 21.
    residue_maturation_reference_days: float = 30.
    residue_decay_reference_days: float = 90.
    initial_ready_fraction: float = .5
    rank_distance_scale: float = 2.
    rain_scale_mm: float = 2.
    local_airborne_fraction: float = .1
    contact_fraction: float = .2

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(value) for value in values.values()):
            raise ValueError('Disease parameters must be finite.')
        if (self.primary_scale < 0 or self.secondary_scale < 0
                or not isinstance(self.latent_stages, (int, np.integer))
                or self.latent_stages < 1):
            raise ValueError('Nonnegative establishment scales and integer latent stages required.')
        positive = [self.latent_reference_days, self.nonsporulating_reference_days,
            self.infectious_reference_days, self.residue_maturation_reference_days,
            self.residue_decay_reference_days, self.rank_distance_scale, self.rain_scale_mm]
        if min(positive) <= 0:
            raise ValueError('Disease duration and transfer scales must be positive.')
        if any(not 0 <= value <= 1 for value in [self.initial_ready_fraction,
                self.local_airborne_fraction, self.contact_fraction]):
            raise ValueError('Source and contact fractions must lie in [0,1].')

    def vector(self):
        return np.array([self.primary_scale, self.secondary_scale,
            self.latent_reference_days, self.nonsporulating_reference_days,
            self.infectious_reference_days, self.residue_maturation_reference_days,
            self.residue_decay_reference_days, self.rank_distance_scale,
            self.rain_scale_mm, self.local_airborne_fraction, self.contact_fraction])


@njit(cache=True)
def _advance_day(state, residue, t, exposure, rain, active, renewal, area,
                 imported_pressure, coefficients, stages, substeps):
    """A proposal from beginning-of-day states; no caller state is mutated."""
    (alpha, beta, latency, presporulation, infectious_life, maturation,
     residue_life, distance, rain_scale, airborne_share, contact_share) = coefficients
    current, reservoir = state.copy(), residue.copy()
    leaves = len(active)
    flows = np.zeros((4, leaves))
    for leaf in range(leaves):
        if not active[leaf]:
            current[leaf] = 0.
            current[leaf, 0] = 1.
        else:
            current[leaf, 1:] *= 1-renewal[leaf]
            current[leaf, 0] = 1-current[leaf, 1:].sum()
    dt = 1./substeps
    thermal = max(t, 0.)/18.
    splash = -np.expm1(-rain/rain_scale)
    q_latent = -np.expm1(-dt*stages*thermal/latency)
    q_sporulation = -np.expm1(-dt*thermal/presporulation)
    q_removal = -np.expm1(-dt*thermal/infectious_life)
    ready_decay = np.exp(-dt*thermal/residue_life)
    maturation_rate, decay_rate = thermal*exposure/maturation, thermal/residue_life
    immature_survival = np.exp(-dt*(maturation_rate+decay_rate))
    mature_transfer_fraction = (-np.expm1(-dt*maturation_rate))*ready_decay
    for _ in range(substeps):
        updated = current.copy()
        total_area = 0.
        for donor in range(leaves):
            if active[donor]:
                total_area += area[donor]
        for leaf in range(leaves):
            if not active[leaf]:
                continue
            nearby, contact = 0., 0.
            for donor in range(leaves):
                if not active[donor]:
                    continue
                gap = abs(leaf-donor)
                infectious = current[donor, stages+2]*area[donor]
                nearby += np.exp(-gap/distance)*infectious
                if gap == 1:
                    contact += infectious
            if total_area > 0:
                nearby /= total_area
                contact /= total_area
            # Lower leaves lie nearer local surface residue. A declared local
            # airborne share and explicit imported forcing are separate routes.
            ground_proximity = np.exp(-(leaves-1-leaf)/distance)
            local_hazard = alpha*exposure*reservoir[1]*(
                (1-airborne_share)*splash*ground_proximity+airborne_share)
            imported_hazard = alpha*exposure*imported_pressure
            splash_hazard = beta*exposure*splash*nearby
            contact_hazard = beta*exposure*contact_share*(1-area[leaf])*contact
            # The eight-slot host contract reserves slot7 for an early-crop
            # aggregate. Its withdrawal is not the unfolding of a leaf.
            if leaves == 8 and leaf == 7:
                contact_hazard = 0.
            total = local_hazard+imported_hazard+splash_hazard+contact_hazard
            newly_infected = current[leaf, 0]*(-np.expm1(-dt*total))
            updated[leaf, 0] -= newly_infected
            updated[leaf, 1] += newly_infected
            if total > 0:
                flows[0, leaf] += newly_infected*local_hazard/total
                flows[1, leaf] += newly_infected*imported_hazard/total
                flows[2, leaf] += newly_infected*splash_hazard/total
                flows[3, leaf] += newly_infected*contact_hazard/total
            for latent in range(stages):
                transfer = current[leaf, 1+latent]*q_latent
                updated[leaf, 1+latent] -= transfer
                updated[leaf, 2+latent] += transfer
            visible = stages+1
            become_infectious = current[leaf, visible]*q_sporulation
            finish_infectious = current[leaf, visible+1]*q_removal
            updated[leaf, visible] -= become_infectious
            updated[leaf, visible+1] += become_infectious-finish_infectious
            updated[leaf, visible+2] += finish_infectious
        # Equal decay acts on both compartments during maturation; the total
        # potential therefore follows exactly exp(-thermal*dt/residue_life).
        transferred = reservoir[0]*mature_transfer_fraction
        reservoir[0] *= immature_survival
        reservoir[1] = reservoir[1]*ready_decay+transferred
        current = updated
    return current, reservoir, flows


def _readonly(value):
    value = np.array(value, dtype=float, copy=True)
    value.setflags(write=False)
    return value


@dataclass(frozen=True)
class DiseaseRates:
    """Daily integrated proposals, committed once by their owning model."""

    owner: str
    epoch: int
    next_tissue: np.ndarray
    next_residue: np.ndarray
    local_flow: np.ndarray
    imported_flow: np.ndarray
    splash_flow: np.ndarray
    contact_flow: np.ndarray


class OverwinterModel:
    """Stateful disease component with explicit calculation/integration phases."""

    def __init__(self, parameters=OverwinterParameters(), *, leaf_count=8,
                 initial_local_source=1., time_step=.25):
        if (not isinstance(parameters, OverwinterParameters)
                or not isinstance(leaf_count, (int, np.integer)) or leaf_count < 1
                or not np.isfinite(initial_local_source) or initial_local_source < 0
                or not np.isfinite(time_step) or not 0 < time_step <= 1):
            raise ValueError('Valid disease parameters, source potential and time step required.')
        self.parameters = parameters
        self.leaf_count = int(leaf_count)
        self.initial_local_source = float(initial_local_source)
        self.time_step = float(time_step)
        self._owner = uuid4().hex
        self._epoch = 0
        self._state = np.zeros((self.leaf_count, parameters.latent_stages+4))
        self._state[:, 0] = 1.
        self._residue = np.array([1-parameters.initial_ready_fraction,
            parameters.initial_ready_fraction])*initial_local_source

    @property
    def state(self):
        return _readonly(self._state)

    @property
    def residue(self):
        return _readonly(self._residue)

    @property
    def damage(self):
        return _readonly(self._state[:, -3:].sum(axis=-1))

    @property
    def infectious(self):
        return _readonly(self._state[:, -2])

    def calc_rates(self, temperature, exposure, rain, host_active, host_renewal,
                   host_area, imported_pressure=0.):
        forcing = np.asarray([temperature, exposure, rain, imported_pressure], float)
        active, renewal, area = np.asarray(host_active), np.asarray(host_renewal, float), np.asarray(host_area, float)
        if (forcing.shape != (4,) or not np.isfinite(forcing).all()
                or not 0 <= exposure <= 1 or rain < 0 or imported_pressure < 0
                or active.shape != (self.leaf_count,) or renewal.shape != active.shape or area.shape != active.shape
                or np.any((active != 0) & (active != 1))
                or not np.isfinite(renewal).all() or not np.isfinite(area).all()
                or np.any((renewal < 0) | (renewal > 1)) or np.any((area < 0) | (area > 1))
                or np.any(renewal[active == 0] != 0) or np.any(area[active == 0] != 0)
                or np.any(area[active == 1] <= 0)):
            raise ValueError('Valid daily forcing and active, normalized leaf areas required.')
        state, residue, flow = _advance_day(self._state, self._residue, float(temperature),
            float(exposure), float(rain), active.astype(bool), renewal, area, float(imported_pressure),
            self.parameters.vector(), self.parameters.latent_stages, int(np.ceil(1/self.time_step)))
        return DiseaseRates(self._owner, self._epoch, _readonly(state), _readonly(residue),
                            *[_readonly(row) for row in flow])

    def integrate(self, rates):
        if not isinstance(rates, DiseaseRates) or rates.owner != self._owner or rates.epoch != self._epoch:
            raise ValueError('A rate proposal must belong to this model and its current state.')
        tissue, residue = np.asarray(rates.next_tissue), np.asarray(rates.next_residue)
        flow = [np.asarray(getattr(rates, name)) for name in (
            'local_flow', 'imported_flow', 'splash_flow', 'contact_flow')]
        if (tissue.shape != self._state.shape or residue.shape != (2,)
                or not np.isfinite(tissue).all() or not np.isfinite(residue).all()
                or np.any(tissue < -1e-12) or np.any(residue < 0)
                or not np.allclose(tissue.sum(axis=-1), 1., rtol=0., atol=1e-10)
                or any(a.shape != (self.leaf_count,) or not np.isfinite(a).all()
                       or np.any(a < 0) for a in flow)):
            raise ValueError('The rate proposal contains invalid tissue, residue or pathway states.')
        self._state = rates.next_tissue.copy()
        self._residue = rates.next_residue.copy()
        self._epoch += 1

    def snapshot(self):
        return dict(schema_version=1, parameters=asdict(self.parameters), leaf_count=self.leaf_count,
            initial_local_source=self.initial_local_source, time_step=self.time_step, epoch=self._epoch,
            tissue_state=self._state.tolist(), residue_state=self._residue.tolist())

    def restore(self, snapshot):
        identity = {name: snapshot.get(name) for name in ('schema_version', 'parameters', 'leaf_count',
                    'initial_local_source', 'time_step')}
        expected = {name: self.snapshot()[name] for name in identity}
        state = np.asarray(snapshot.get('tissue_state'), float)
        residue = np.asarray(snapshot.get('residue_state'), float)
        epoch = snapshot.get('epoch')
        if (identity != expected or state.shape != self._state.shape or residue.shape != (2,)
                or not np.isfinite(state).all() or not np.isfinite(residue).all()
                or state.min() < -1e-12 or residue.min() < 0
                or not np.allclose(state.sum(axis=-1), 1., rtol=0., atol=1e-10)
                or not isinstance(epoch, int) or epoch < 0):
            raise ValueError('Checkpoint state or disease configuration is invalid.')
        self._state, self._residue, self._epoch = state.copy(), residue.copy(), epoch
        # Old proposals cannot be committed after a restore, even at equal epoch.
        self._owner = uuid4().hex


@dataclass
class OverwinterTrajectory:
    state: np.ndarray
    residue: np.ndarray
    damage: np.ndarray
    infectious: np.ndarray
    local_flow: np.ndarray
    imported_flow: np.ndarray
    splash_flow: np.ndarray
    contact_flow: np.ndarray
    infection_day: np.ndarray
    symptom_day: np.ndarray

    @property
    def primary_flow(self):
        return self.local_flow+self.imported_flow

    @property
    def secondary_flow(self):
        return self.splash_flow+self.contact_flow


@njit(cache=True)
def _batch(t, exposure, rain, active, renewal, area, valid, imported, initial,
           coefficients, ready_fraction, stages, substeps):
    fields, days, leaves = active.shape
    state = np.zeros((fields, days+1, leaves, stages+4))
    state[..., 0] = 1.
    residue = np.zeros((fields, days+1, 2))
    flows = np.zeros((4, fields, days, leaves))
    for field in range(fields):
        current = state[field, 0].copy()
        reservoir = np.array([initial[field]*(1-ready_fraction), initial[field]*ready_fraction])
        residue[field, 0] = reservoir
        for day in range(days):
            if valid[field, day]:
                current, reservoir, flow = _advance_day(current, reservoir, t[field, day],
                    exposure[field, day], rain[field, day], active[field, day], renewal[field, day],
                    area[field, day], imported[field, day], coefficients, stages, substeps)
                flows[:, field, day] = flow
            state[field, day+1], residue[field, day+1] = current, reservoir
    return state, residue, flows


def _first_crossing(values, cutoff, valid):
    mask = np.concatenate([np.zeros((len(valid), 1), bool), valid], axis=1)
    crossing = (values >= cutoff) & mask[:, :, None]
    return np.where(crossing.any(axis=1), crossing.argmax(axis=1), -1)


def simulate_overwinter(temperature, exposure, rain, host_active, host_renewal,
                        host_area, forcing_mask, parameters=OverwinterParameters(), *,
                        initial_local_source=1., imported_pressure=None,
                        time_step=.25, detection_fraction=.001):
    """Batch adapter to the same daily kernel, with bounded field forcing."""
    t, e, rain = [np.asarray(v, float) for v in (temperature, exposure, rain)]
    active, valid = np.asarray(host_active), np.asarray(forcing_mask)
    renewal, area = np.asarray(host_renewal, float), np.asarray(host_area, float)
    if (t.ndim != 2 or not t.size or e.shape != t.shape or rain.shape != t.shape or valid.shape != t.shape
            or active.ndim != 3 or active.shape[:2] != t.shape or not active.shape[2]
            or renewal.shape != active.shape or area.shape != active.shape
            or not all(np.isfinite(v).all() for v in (t, e, rain, renewal, area))
            or np.any((active != 0) & (active != 1)) or np.any((valid != 0) & (valid != 1))
            or np.any((e < 0) | (e > 1)) or np.any(rain < 0)
            or np.any((renewal < 0) | (renewal > 1)) or np.any((area < 0) | (area > 1))
            or np.any(renewal[(active == 0) & valid[:, :, None]] != 0)
            or np.any(area[(active == 0) & valid[:, :, None]] != 0)
            or np.any(area[(active == 1) & valid[:, :, None]] <= 0)
            or not isinstance(parameters, OverwinterParameters)
            or not np.isfinite(time_step) or not 0 < time_step <= 1
            or not np.isfinite(detection_fraction) or not 0 < detection_fraction <= 1):
        raise ValueError('Valid weather, leaf states, forcing masks and integration settings required.')
    initial = np.broadcast_to(np.asarray(initial_local_source, float), (len(t),)).copy()
    imported = np.zeros_like(t) if imported_pressure is None else np.asarray(imported_pressure, float)
    if (imported.shape != t.shape or not np.isfinite(imported).all() or np.any(imported < 0)
            or not np.isfinite(initial).all() or np.any(initial < 0)):
        raise ValueError('Nonnegative local source and daily imported forcing required.')
    state, residue, flows = _batch(t, e, rain, active.astype(bool), renewal, area, valid.astype(bool),
        imported, initial, parameters.vector(), parameters.initial_ready_fraction,
        parameters.latent_stages, int(np.ceil(1/time_step)))
    damage = state[..., -3:].sum(axis=-1)
    affected = state[..., 1:].sum(axis=-1)
    return OverwinterTrajectory(state, residue, damage, state[..., -2], *flows,
        _first_crossing(affected, detection_fraction, valid),
        _first_crossing(damage, detection_fraction, valid))
