"""Incremental donor T-P-V calculation with previous-day development gates."""

from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
from uuid import uuid4

import numpy as np

from .config import TPVParameters
from .weather import WeatherDay, iso_date


@dataclass(frozen=True)
class TPVState:
    cumulative_gdd: float = 0.
    cumulative_tpp: float = 0.
    cumulative_tpv: float = 0.
    cumulative_vernalization: float = 0.
    day_count: int = 0
    last_date: date | None = None


@dataclass(frozen=True)
class TPVRates:
    owner: str
    epoch: int
    date: date
    gdd: float
    tpp: float
    tpv: float
    next_vernalization: float
    daylength_hours: float
    forcing: WeatherDay


class PhenologyModel:
    def __init__(self, parameters: TPVParameters, *, latitude: float, sowing_date: date,
                 crop_end_date: date | None = None, vernalization_required: bool = True):
        if not isinstance(parameters, TPVParameters) or not np.isfinite(latitude) or abs(latitude) > 90:
            raise ValueError('Typed T-P-V parameters and valid latitude required.')
        self.parameters, self.latitude = parameters, float(latitude)
        self.sowing_date = iso_date(sowing_date)
        self.crop_end_date = None if crop_end_date is None else iso_date(crop_end_date)
        self.vernalization_required = bool(vernalization_required)
        self._state = TPVState()
        self._owner = uuid4().hex
        payload = dict(parameters=asdict(parameters), latitude=self.latitude,
            sowing_date=self.sowing_date.isoformat(),
            crop_end_date=None if self.crop_end_date is None else self.crop_end_date.isoformat(),
            vernalization_required=self.vernalization_required)
        self.identity = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    @property
    def state(self):
        return self._state

    def included(self, day: date) -> bool:
        return day >= self.sowing_date and (self.crop_end_date is None or day <= self.crop_end_date)

    def calc_rates(self, day: WeatherDay) -> TPVRates:
        if not isinstance(day, WeatherDay):
            raise ValueError('Validated WeatherDay required.')
        if self.state.last_date is not None and day.date != self.state.last_date+timedelta(days=1):
            raise ValueError('Phenology forcing dates must be consecutive.')
        p, s, t, tx = self.parameters, self.state, day.tmean_c, day.tmax_c
        included = self.included(day.date)
        gdd = (20-2*(t-30) if t > 30 else float(np.clip(t, 0, 20))) if included else 0.
        doy = day.date.timetuple().tm_yday
        declination = .4093*np.sin(2*np.pi*(doy-81)/365)
        daylength = float(24/np.pi*np.arccos(np.clip(
            -np.tan(np.radians(self.latitude))*np.tan(declination), -1., 1.)))
        pp_factor = float(np.clip(1-.09*(16-daylength), 0., 1.))
        pp_active = included and p.photoperiod_onset_gdd <= s.cumulative_gdd < p.photoperiod_stop_gdd
        tpp = gdd*(pp_factor if pp_active else 1.)
        vernalization, factor = s.cumulative_vernalization, 1.
        if self.vernalization_required:
            active = included and s.cumulative_tpp >= p.vernalization_onset_tpp
            sensitive = s.cumulative_tpp < p.vernalization_stop_tpp
            updated = vernalization+(float(np.interp(t, [-4, 0, 10, 16], [0., 1., 1., 0.])) if sensitive else 0.)
            if updated < 10 and tx > 30:
                updated = max(0., vernalization-.5*(tx-30))
            if active:
                vernalization = updated
            if active and sensitive:
                factor = .3+.7*min(vernalization/40, 1.)
        return TPVRates(self._owner, s.day_count, day.date, gdd, tpp, tpp*factor, vernalization, daylength, day)

    def integrate(self, rates: TPVRates) -> TPVState:
        if not isinstance(rates, TPVRates) or rates.owner != self._owner:
            raise ValueError('Phenology rates belong to another owner/instance.')
        if rates.epoch != self.state.day_count:
            raise ValueError('Phenology rates are stale for the current state.')
        if rates != self.calc_rates(rates.forcing):
            raise ValueError('Phenology rate proposal does not match its validated forcing.')
        s = self.state
        self._state = TPVState(s.cumulative_gdd+rates.gdd, s.cumulative_tpp+rates.tpp,
            s.cumulative_tpv+rates.tpv, rates.next_vernalization, s.day_count+1, rates.date)
        return self.state

    def snapshot(self):
        values = asdict(self.state)
        values['last_date'] = None if self.state.last_date is None else self.state.last_date.isoformat()
        return dict(schema_version=1, config_identity=self.identity, state=values)

    def restore(self, snapshot):
        if snapshot.get('schema_version') != 1 or snapshot.get('config_identity') != self.identity:
            raise ValueError('Phenology checkpoint configuration identity differs.')
        values = dict(snapshot['state'])
        if (not np.isfinite([values[k] for k in ['cumulative_gdd', 'cumulative_tpp',
                'cumulative_tpv', 'cumulative_vernalization']]).all()
                or min(values[k] for k in ['cumulative_gdd', 'cumulative_tpp',
                    'cumulative_tpv', 'cumulative_vernalization']) < 0
                or not isinstance(values['day_count'], int) or values['day_count'] < 0):
            raise ValueError('Invalid phenology checkpoint state.')
        if (values['cumulative_tpv'] > values['cumulative_tpp']+1e-10
                or values['cumulative_tpp'] > values['cumulative_gdd']+1e-10):
            raise ValueError('Phenology checkpoint clocks must obey TPV <= TPP <= GDD.')
        values['last_date'] = None if values['last_date'] is None else iso_date(values['last_date'])
        if (values['last_date'] is None) != (values['day_count'] == 0):
            raise ValueError('Phenology checkpoint date and epoch disagree.')
        self._state, self._owner = TPVState(**values), uuid4().hex
