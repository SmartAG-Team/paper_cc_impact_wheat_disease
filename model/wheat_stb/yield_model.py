"""Conditional daily HAD3 diagnostics; disabled unless explicitly configured."""

from dataclasses import asdict, dataclass
from datetime import date
import hashlib
import json
from uuid import uuid4

import numpy as np

from .config import YieldParameters
from .weather import WeatherDay, iso_date


@dataclass(frozen=True)
class YieldState:
    reference_had3: float = 0.
    lost_had3: float = 0.
    window_start_date: date | None = None
    window_end_date: date | None = None
    day_count: int = 0


@dataclass(frozen=True)
class YieldRates:
    owner: str
    epoch: int
    next_state: YieldState


class ConditionalYieldModel:
    def __init__(self, parameters: YieldParameters, *, anthesis_threshold: float, end_threshold: float):
        self.parameters = parameters
        self.anthesis_threshold, self.end_threshold = float(anthesis_threshold), float(end_threshold)
        if not 0 < self.anthesis_threshold < self.end_threshold:
            raise ValueError('Ordered flowering and terminal development thresholds required.')
        self._state, self._owner = YieldState(), uuid4().hex
        payload = dict(parameters=asdict(parameters), anthesis_threshold=self.anthesis_threshold,
                       end_threshold=self.end_threshold)
        self.identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def state(self):
        return self._state

    def calc_rates(self, day: WeatherDay, accumulation: float, host_area, affected_fraction,
                   crop_included: bool) -> YieldRates:
        s, p = self.state, self.parameters
        start, end = s.window_start_date, s.window_end_date
        if crop_included and start is None and accumulation >= self.anthesis_threshold:
            start = day.date
        within = crop_included and start is not None and end is None
        if within and accumulation >= self.end_threshold:
            end = day.date
        reference, lost = s.reference_had3, s.lost_had3
        if p.mode != 'disabled' and within:
            if p.mode == 'external_functional_loss':
                if day.reference_leaf_area_index is None:
                    raise ValueError('Every in-window day requires external reference LAI and functional loss.')
                area, fraction = np.asarray(day.reference_leaf_area_index), np.asarray(day.functional_loss_fraction)
            else:
                area = p.standardized_upper3_lai*np.asarray(host_area, float)[:3]/3.
                fraction = np.asarray(affected_fraction, float)[:3]
            if (area.shape != (3,) or fraction.shape != (3,)
                    or not np.isfinite(area).all() or not np.isfinite(fraction).all()
                    or np.any(area < 0) or np.any((fraction < -1e-12) | (fraction > 1+1e-12))):
                raise ValueError('Valid area and functional-loss fractions required for conditional HAD.')
            reference += float(area.sum())
            lost += float((area*np.clip(fraction, 0., 1.)).sum())
        return YieldRates(self._owner, s.day_count,
            YieldState(reference, lost, start, end, s.day_count+1))

    def integrate(self, rates):
        if not isinstance(rates, YieldRates) or rates.owner != self._owner or rates.epoch != self.state.day_count:
            raise ValueError('Yield rates must belong to this instance and its current state.')
        self._validate_state(rates.next_state)
        if rates.next_state.day_count != self.state.day_count+1:
            raise ValueError('Yield rate epoch is invalid.')
        self._state = rates.next_state
        return self._state

    @staticmethod
    def _validate_state(state):
        if (not np.isfinite([state.reference_had3, state.lost_had3]).all()
                or state.reference_had3 < 0 or state.lost_had3 < 0
                or state.lost_had3 > state.reference_had3+1e-10
                or not isinstance(state.day_count, int) or state.day_count < 0
                or state.window_end_date is not None and (state.window_start_date is None
                    or state.window_end_date < state.window_start_date)):
            raise ValueError('Invalid conditional yield/HAD checkpoint state.')

    def snapshot(self):
        values = asdict(self.state)
        for key in ['window_start_date', 'window_end_date']:
            values[key] = None if values[key] is None else values[key].isoformat()
        return dict(schema_version=1, config_identity=self.identity, state=values)

    def restore(self, snapshot):
        if snapshot.get('schema_version') != 1 or snapshot.get('config_identity') != self.identity:
            raise ValueError('Conditional yield checkpoint configuration identity differs.')
        values = dict(snapshot['state'])
        for key in ['window_start_date', 'window_end_date']:
            values[key] = None if values[key] is None else iso_date(values[key])
        state = YieldState(**values)
        self._validate_state(state)
        self._state, self._owner = state, uuid4().hex

    def summary(self):
        s, p = self.state, self.parameters
        complete = s.window_start_date is not None and s.window_end_date is not None
        enabled_complete = complete and p.mode != 'disabled'
        absolute = None if not enabled_complete else tuple(s.lost_had3*b for b in p.coefficients_t_ha_per_glai_day)
        relative = None if absolute is None or p.reference_yield_t_ha is None else tuple(100*x/p.reference_yield_t_ha for x in absolute)
        exceeds = None if absolute is None or p.reference_yield_t_ha is None else tuple(x > p.reference_yield_t_ha for x in absolute)
        return dict(complete_grain_fill_window=complete, yield_area_mode=p.mode,
            reference_had3=None if not enabled_complete else s.reference_had3,
            lost_had3=None if not enabled_complete else s.lost_had3,
            conditional_yield_loss_t_ha=absolute, conditional_relative_loss_percent=relative,
            exceeds_reference_yield=exceeds, actual_field_yield_forecast=None,
            coefficient_range_is_confidence_interval=False, locally_yield_calibrated=False,
            yield_window='inclusive flowering proxy to BBCH85; conditional source-window adaptation',
            functional_conversion=('modeled symptomatic/damage fraction as functional-loss scenario; not measured green-area loss'
                if p.mode == 'standardized_model_proxy' else 'caller-supplied external functional-loss fractions'))
