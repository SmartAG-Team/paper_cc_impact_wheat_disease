"""Explicit immutable runtime parameters; calibration remains external."""

from dataclasses import asdict, dataclass, field
from datetime import date
import hashlib
import json
from pathlib import Path

import numpy as np

from model.seasonal_septoria.overwinter import OverwinterParameters
from .weather import iso_date

def _json(path):
    return json.loads(Path(path).read_text())


@dataclass(frozen=True)
class TPVParameters:
    photoperiod_onset_gdd: float
    photoperiod_stop_gdd: float
    vernalization_onset_tpp: float
    vernalization_stop_tpp: float

    def __post_init__(self):
        if (not np.isfinite(list(asdict(self).values())).all()
                or not 0 <= self.photoperiod_onset_gdd < self.photoperiod_stop_gdd
                or not 0 <= self.vernalization_onset_tpp < self.vernalization_stop_tpp):
            raise ValueError('Ordered finite nonnegative T-P-V development gates required.')

    @classmethod
    def from_dict(cls, values):
        try:
            return cls(**{key: float(value) for key, value in values.items()})
        except (TypeError, KeyError) as error:
            raise ValueError('All explicit T-P-V parameter fields are required.') from error


@dataclass(frozen=True)
class LeafParameters:
    stage_thresholds: tuple[tuple[int, float], ...]
    rank_spacing_units: float = 120.
    threshold_source_sha256: str = ''
    threshold_status: str = 'caller-supplied stage scenario'
    juvenile_policy: str = 'handover_31_39'

    def __post_init__(self):
        values = tuple(sorted((int(k), float(v)) for k, v in self.stage_thresholds))
        keys = [key for key, _ in values]
        thresholds = [value for _, value in values]
        if (len(set(keys)) != len(keys) or not {10, 31, 32, 37, 39, 51, 65, 85}.issubset(keys)
                or not np.isfinite(thresholds).all() or min(thresholds) <= 0
                or np.any(np.diff(thresholds) <= 0)
                or not np.isfinite(self.rank_spacing_units) or self.rank_spacing_units <= 0
                or self.juvenile_policy not in ['handover_31_39', 'persistent']):
            raise ValueError('Ordered complete stage thresholds and positive effective rank spacing required.')
        object.__setattr__(self, 'stage_thresholds', values)

@dataclass(frozen=True)
class YieldParameters:
    mode: str = 'disabled'
    coefficients_t_ha_per_glai_day: tuple[float, ...] = (.0141, .018, .0207)
    reference_yield_t_ha: float | None = None
    standardized_upper3_lai: float = 1.

    def __post_init__(self):
        object.__setattr__(self, 'coefficients_t_ha_per_glai_day', tuple(float(x) for x in self.coefficients_t_ha_per_glai_day))
        if self.mode not in ['disabled', 'external_functional_loss', 'standardized_model_proxy']:
            raise ValueError('Unsupported yield tracking mode.')
        if (not len(self.coefficients_t_ha_per_glai_day)
                or not np.isfinite(self.coefficients_t_ha_per_glai_day).all()
                or min(self.coefficients_t_ha_per_glai_day) < 0
                or not np.isfinite(self.standardized_upper3_lai) or self.standardized_upper3_lai <= 0):
            raise ValueError('Finite nonnegative yield slopes and positive scenario LAI required.')
        if self.reference_yield_t_ha is not None:
            if (self.mode == 'disabled' or not np.isfinite(self.reference_yield_t_ha)
                    or self.reference_yield_t_ha <= 0):
                raise ValueError('Independent positive reference yield requires enabled conditional HAD tracking.')


@dataclass(frozen=True)
class EngineConfig:
    latitude: float
    sowing_date: date
    phenology: TPVParameters
    leaf: LeafParameters
    disease: OverwinterParameters = field(default_factory=OverwinterParameters)
    yield_model: YieldParameters = field(default_factory=YieldParameters)
    field_id: str = 'field'
    crop_end_date: date | None = None
    vernalization_required: bool = True
    rain_rate_mm_hour: float = .46
    initial_local_source: float = 1.
    background_imported_pressure: float = 0.
    detection_fraction: float = .001
    time_step: float = .25
    disease_fit_source_sha256: str = ''
    parameter_status: str = 'disease coefficients are explicit assumptions; not jointly field-calibrated'
    schema_version: int = 1

    def __post_init__(self):
        object.__setattr__(self, 'sowing_date', iso_date(self.sowing_date))
        if self.crop_end_date is not None:
            object.__setattr__(self, 'crop_end_date', iso_date(self.crop_end_date))
        if not np.isfinite(self.detection_fraction) or not 0 < self.detection_fraction <= 1:
            raise ValueError('A finite detection fraction in (0,1] is required.')
        if (self.schema_version != 1 or not np.isfinite(self.latitude) or abs(self.latitude) > 90
                or not self.field_id or not isinstance(self.vernalization_required, bool)
                or not np.isfinite(self.rain_rate_mm_hour) or self.rain_rate_mm_hour <= 0
                or not np.isfinite(self.initial_local_source) or self.initial_local_source < 0
                or not np.isfinite(self.background_imported_pressure) or self.background_imported_pressure < 0
                or not np.isfinite(self.time_step) or not 0 < self.time_step <= 1
                or self.crop_end_date is not None and self.crop_end_date < self.sowing_date):
            raise ValueError('Valid crop calendar, latitude, rainfall rate and disease integration configuration required.')
        if not isinstance(self.phenology, TPVParameters) or not isinstance(self.leaf, LeafParameters):
            raise ValueError('Typed immutable phenology and leaf parameters required.')
        if not isinstance(self.disease, OverwinterParameters) or not isinstance(self.yield_model, YieldParameters):
            raise ValueError('Typed immutable disease and yield parameters required.')

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if not {'phenology', 'leaf'}.issubset(values):
            raise ValueError('Explicit phenology and leaf parameters are required; runtime loads no calibration defaults.')
        values['phenology'] = TPVParameters.from_dict(values['phenology'])
        leaf = dict(values['leaf'])
        leaf['stage_thresholds'] = tuple((int(k), float(v)) for k, v in dict(leaf['stage_thresholds']).items())
        values['leaf'] = LeafParameters(**leaf)
        values['disease'] = OverwinterParameters(**values.get('disease', {}))
        values['yield_model'] = YieldParameters(**values.get('yield_model', {}))
        try:
            return cls(**values)
        except TypeError as error:
            raise ValueError('Unknown or missing engine configuration fields.') from error

    @classmethod
    def from_json(cls, path):
        return cls.from_dict(_json(path))

    def to_dict(self):
        values = asdict(self)
        values['sowing_date'] = self.sowing_date.isoformat()
        values['crop_end_date'] = None if self.crop_end_date is None else self.crop_end_date.isoformat()
        values['leaf']['stage_thresholds'] = {str(k): v for k, v in self.leaf.stage_thresholds}
        return values

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
