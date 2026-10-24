"""Daily one-field component composition and checkpoint-safe lifecycle."""

from dataclasses import asdict, dataclass, replace
from datetime import date
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Protocol
from uuid import uuid4

import numpy as np
import pandas as pd

from model.seasonal_septoria.overwinter import DiseaseRates, OverwinterModel
from model.seasonal_septoria.wetness import duration_exposure
from ._version import __version__
from .config import EngineConfig
from .leaf import LeafCanopyModel
from .phenology import PhenologyModel, TPVRates
from .weather import WeatherDay, WeatherProvider, iso_date
from .yield_model import ConditionalYieldModel, YieldRates


class DiseaseBackend(Protocol):
    @property
    def state(self) -> np.ndarray: ...
    @property
    def residue(self) -> np.ndarray: ...
    def calc_rates(self, temperature, exposure, rain, host_active, host_renewal,
                   host_area, imported_pressure=0.) -> DiseaseRates: ...
    def integrate(self, rates: DiseaseRates) -> None: ...
    def snapshot(self) -> dict: ...
    def restore(self, snapshot: dict) -> None: ...


def runtime_source_hashes() -> dict[str, str]:
    package = Path(__file__).resolve().parent
    paths = list(package.glob('*.py'))
    shared = package.parent/'seasonal_septoria'
    paths += [shared/name for name in ['overwinter.py', 'leaf_phenology.py', 'wetness.py']]
    return {str(path.relative_to(package.parent.parent)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


@dataclass(frozen=True)
class DailySimulationRates:
    owner: str
    epoch: int
    weather: WeatherDay
    exposure: float
    imported_pressure: float
    phenology: TPVRates
    leaf: object
    disease: DiseaseRates
    yield_model: YieldRates


@dataclass(frozen=True)
class DailyOutput:
    date: date
    field_id: str
    cumulative_gdd: float
    cumulative_tpp: float
    cumulative_tpv: float
    cumulative_vernalization: float
    predicted_stage_code: int
    modeled_visible_final_leaf_count: int
    modeled_unfolded_final_leaf_count: int
    exposure: float
    imported_pressure: float
    residue_unready: float
    residue_competent: float
    crop_included: bool
    host_area: tuple[float, ...]
    host_active: tuple[bool, ...]
    susceptible_fraction: tuple[float, ...]
    latent_fraction: tuple[float, ...]
    symptomatic_fraction: tuple[float, ...]
    affected_fraction: tuple[float, ...]
    infectious_fraction: tuple[float, ...]
    local_flow: tuple[float, ...]
    imported_flow: tuple[float, ...]
    splash_flow: tuple[float, ...]
    contact_flow: tuple[float, ...]
    cumulative_reference_had3: float | None
    cumulative_lost_had3: float | None

    def to_dict(self):
        values = asdict(self)
        values['date'] = self.date.isoformat()
        return values

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        values['date'] = iso_date(values['date'])
        for key in ['host_area', 'host_active', 'susceptible_fraction', 'latent_fraction',
                    'symptomatic_fraction', 'affected_fraction', 'infectious_fraction',
                    'local_flow', 'imported_flow', 'splash_flow', 'contact_flow']:
            values[key] = tuple(values[key])
        output = cls(**values)
        numeric = [output.cumulative_gdd, output.cumulative_tpp, output.cumulative_tpv,
            output.cumulative_vernalization, output.exposure, output.imported_pressure,
            output.residue_unready, output.residue_competent]
        fractions = [output.susceptible_fraction, output.latent_fraction,
                     output.symptomatic_fraction, output.affected_fraction, output.infectious_fraction]
        arrays = [output.host_area, *fractions,
                  output.local_flow, output.imported_flow, output.splash_flow, output.contact_flow]
        if (not np.isfinite(numeric).all() or min(numeric) < 0 or not 0 <= output.exposure <= 1
                or any(len(row) != 8 or not np.isfinite(row).all() or min(row) < -1e-12 for row in arrays)
                or type(output.crop_included) is not bool or len(output.host_active) != 8
                or any(value not in [True, False] for value in output.host_active)
                or any(max(row) > 1+1e-10 for row in [output.host_area, *fractions])):
            raise ValueError('Invalid archived daily output.')
        susceptible, latent, symptomatic, affected, infectious = map(np.asarray, fractions)
        if (not np.allclose(susceptible+latent+symptomatic, 1., rtol=0., atol=1e-10)
                or not np.allclose(affected, latent+symptomatic, rtol=0., atol=1e-10)
                or np.any(infectious > symptomatic+1e-10)):
            raise ValueError('Archived daily disease compartment fractions disagree.')
        for key in ['cumulative_reference_had3', 'cumulative_lost_had3']:
            value = getattr(output, key)
            if value is not None and (not np.isfinite(value) or value < 0):
                raise ValueError('Invalid archived daily HAD output.')
        return output


@dataclass(frozen=True)
class SimulationResults:
    daily: tuple[DailyOutput, ...]
    summary: object
    provenance: object

    def daily_frame(self):
        records = []
        layers = [f'F{i}' for i in range(1, 8)]+['juvenile']
        vectors = ['host_area', 'host_active', 'susceptible_fraction', 'latent_fraction',
                   'symptomatic_fraction', 'affected_fraction', 'infectious_fraction',
                   'local_flow', 'imported_flow', 'splash_flow', 'contact_flow']
        for day in self.daily:
            record = day.to_dict()
            for variable in vectors:
                values = record.pop(variable)
                record.update({f'{variable}_{layer}': value for layer, value in zip(layers, values)})
            records.append(record)
        return pd.DataFrame(records)

    def write(self, output_directory):
        dest = Path(output_directory)
        if dest.exists():
            raise FileExistsError('Simulation output directory must be new.')
        dest.mkdir(parents=True)
        self.daily_frame().to_csv(dest/'daily.csv', index=False)
        for name, values in [('summary', self.summary), ('provenance', self.provenance)]:
            (dest/f'{name}.json').write_text(json.dumps(dict(values), indent=2, allow_nan=False)+'\n')
        return dest


class WheatSTBSimulation:
    """A deterministic one-field simulation with explicit end-of-day output.

    All daily proposals are calculated before component integration. Host
    forcing is today's pure development proposal with yesterday's leaf state;
    yield diagnostics consume the disease proposal's end-day affected fraction.
    This sequencing is explicit and does not assimilate observed disease.
    """

    def __init__(self, config: EngineConfig, weather: WeatherProvider):
        if not isinstance(config, EngineConfig) or not isinstance(weather, WeatherProvider):
            raise ValueError('Typed immutable configuration and WeatherProvider required.')
        if weather[0].date > config.sowing_date:
            raise ValueError('Fresh-state weather coverage must begin at or before sowing.')
        self.config, self.weather = config, weather
        self._source_hashes = runtime_source_hashes()
        self._owner, self._cursor, self._status = uuid4().hex, 0, 'created'
        self._outputs = []
        self._make_components()

    def _components(self):
        c = self.config
        phenology = PhenologyModel(c.phenology, latitude=c.latitude, sowing_date=c.sowing_date,
            crop_end_date=c.crop_end_date, vernalization_required=c.vernalization_required)
        leaf = LeafCanopyModel(c.leaf)
        disease = OverwinterModel(c.disease, leaf_count=8,
            initial_local_source=c.initial_local_source, time_step=c.time_step)
        thresholds = dict(c.leaf.stage_thresholds)
        yield_model = ConditionalYieldModel(c.yield_model,
            anthesis_threshold=thresholds[65], end_threshold=thresholds[85])
        return phenology, leaf, disease, yield_model

    def _make_components(self):
        self.phenology, self.leaf, self.disease, self.yield_model = self._components()

    @property
    def outputs(self):
        return tuple(self._outputs)

    @property
    def next_date(self):
        return None if self._cursor == len(self.weather) else self.weather[self._cursor].date

    def initialize(self):
        if self._status != 'created':
            raise ValueError('A simulation can initialize only once.')
        self._status = 'running'
        return self

    def calc_rates(self):
        if self._status == 'created':
            self.initialize()
        if self._status != 'running' or self._cursor >= len(self.weather):
            raise ValueError('No current forcing day is available for calculation.')
        day, c = self.weather[self._cursor], self.config
        pheno = self.phenology.calc_rates(day)
        accumulation = self.phenology.state.cumulative_tpv+pheno.tpv
        crop_included = self.phenology.included(day.date)
        leaf = self.leaf.calc_rates(accumulation, day.tmean_c, crop_included)
        if day.exposure_override is None:
            exposure = float(duration_exposure(np.array([[day.tmean_c]]), np.array([[day.tmax_c]]),
                np.array([[day.rh_mean_pct]]), np.array([[day.precipitation_mm]]),
                c.rain_rate_mm_hour)['exposure'][0, 0])
        else:
            exposure = day.exposure_override
        imported = c.background_imported_pressure if day.imported_pressure is None else day.imported_pressure
        if crop_included:
            disease = self.disease.calc_rates(day.tmean_c, exposure, day.precipitation_mm,
                leaf.active, leaf.renewal, leaf.area, imported)
        else:
            # The batch forcing-mask contract freezes the whole disease state,
            # including residue, outside the explicitly declared crop window.
            disease = self.disease.calc_rates(0., 0., 0., np.zeros(8, bool),
                np.zeros(8), np.zeros(8), 0.)
            disease = replace(disease, next_tissue=self.disease.state,
                              next_residue=self.disease.residue)
        symptomatic = disease.next_tissue[:, -3:].sum(axis=-1)
        yield_rates = self.yield_model.calc_rates(day, accumulation, leaf.area, symptomatic, crop_included)
        return DailySimulationRates(self._owner, self._cursor, day, exposure, imported,
                                    pheno, leaf, disease, yield_rates)

    def integrate(self, rates):
        if not isinstance(rates, DailySimulationRates) or rates.owner != self._owner:
            raise ValueError('Daily rates belong to another simulation instance.')
        if self._status != 'running' or rates.epoch != self._cursor:
            raise ValueError('Daily rates are stale for the current simulation state.')
        if rates.weather != self.weather[self._cursor]:
            raise ValueError('Daily rates differ from the current weather forcing.')
        components = [self.phenology, self.leaf, self.disease, self.yield_model]
        before = [component.snapshot() for component in components]
        try:
            pheno = self.phenology.integrate(rates.phenology)
            leaf = self.leaf.integrate(rates.leaf)
            self.disease.integrate(rates.disease)
            yield_state = self.yield_model.integrate(rates.yield_model)
            tissue, residue = self.disease.state, self.disease.residue
            enabled = self.config.yield_model.mode != 'disabled'
            output = DailyOutput(rates.weather.date, self.config.field_id,
                pheno.cumulative_gdd, pheno.cumulative_tpp, pheno.cumulative_tpv,
                pheno.cumulative_vernalization, leaf.predicted_stage_code,
                leaf.modeled_visible_final_leaf_count, leaf.modeled_unfolded_final_leaf_count,
                rates.exposure, rates.imported_pressure, float(residue[0]), float(residue[1]),
                self.phenology.included(rates.weather.date),
                tuple(float(x) for x in leaf.area), tuple(bool(x) for x in leaf.active),
                tuple(float(x) for x in tissue[:, 0]),
                tuple(float(x) for x in tissue[:, 1:-3].sum(axis=-1)),
                tuple(float(x) for x in tissue[:, -3:].sum(axis=-1)),
                tuple(float(x) for x in tissue[:, 1:].sum(axis=-1)),
                tuple(float(x) for x in tissue[:, -2]),
                *[tuple(float(x) for x in getattr(rates.disease, name)) for name in
                  ['local_flow', 'imported_flow', 'splash_flow', 'contact_flow']],
                yield_state.reference_had3 if enabled else None,
                yield_state.lost_had3 if enabled else None)
            DailyOutput.from_dict(output.to_dict())
        except Exception:
            for component, snapshot in zip(components, before):
                component.restore(snapshot)
            self._owner = uuid4().hex
            raise
        self._outputs.append(output)
        self._cursor += 1
        return output

    def step(self):
        return self.integrate(self.calc_rates())

    def run(self, days: int | None = None):
        if self._status == 'created':
            self.initialize()
        if self._status != 'running':
            raise ValueError('A finalized simulation cannot run again.')
        if days is not None and (not isinstance(days, int) or isinstance(days, bool) or days < 0):
            raise ValueError('A nonnegative integer number of simulation days is required.')
        remaining = len(self.weather)-self._cursor
        for _ in range(remaining if days is None else min(days, remaining)):
            self.step()
        return self.outputs

    def finalize(self):
        if self._status == 'created':
            self.initialize()
        self._status = 'finalized'
        summary = dict(model_version=__version__, field_id=self.config.field_id,
            days_simulated=self._cursor, weather_exhausted=self._cursor == len(self.weather),
            state_boundary='end of each supplied forcing day',
            disease_state_units='normalized affected tissue fractions; not observed leaf-area loss',
            residue_units='relative local source potential; not measured spores',
            no_observation_assimilation=True, parameter_status=self.config.parameter_status,
            **self._event_summary(),
            **self.yield_model.summary())
        provenance = dict(configuration=self.config.to_dict(), configuration_sha256=self.config.identity,
            weather_source=self.weather.source, weather_source_sha256=self.weather.source_sha256,
            consumed_weather_sha256=self.weather.prefix_identity(self._cursor),
            runtime_source_sha256=dict(self._source_hashes), machine_readable_version=__version__,
            requested_time_step_days=self.config.time_step,
            numerical_time_step_days=1./int(np.ceil(1./self.config.time_step)))
        return SimulationResults(self.outputs, MappingProxyType(summary), MappingProxyType(provenance))

    def _event_summary(self):
        included = [day for day in self._outputs if day.crop_included]
        labels = [f'F{i}' for i in range(1, 8)]+['juvenile']
        def first(variable, rank):
            return next((day.date.isoformat() for day in included
                         if getattr(day, variable)[rank] >= self.config.detection_fraction), None)
        return dict(detection_fraction=self.config.detection_fraction,
            detection_fraction_status='observation/detection assumption; not a biological infection threshold',
            first_infection_date_by_leaf={label: first('affected_fraction', index)
                for index, label in enumerate(labels)},
            first_symptom_date_by_leaf={label: first('symptomatic_fraction', index)
                for index, label in enumerate(labels)},
            stage_dates={str(stage): next((day.date.isoformat() for day in included
                if day.cumulative_tpv >= threshold), None)
                for stage, threshold in self.config.leaf.stage_thresholds},
            event_date_boundary='first included end-of-day state at or above declared cutoff/threshold',
            model_event_dates_are_observed_infection_dates=False)

    def snapshot(self):
        return dict(schema_version=1, model_version=__version__, configuration_sha256=self.config.identity,
            runtime_source_sha256=dict(self._source_hashes), consumed_days=self._cursor,
            consumed_weather_sha256=self.weather.prefix_identity(self._cursor),
            components={'phenology': self.phenology.snapshot(), 'leaf': self.leaf.snapshot(),
                'disease': self.disease.snapshot(), 'yield': self.yield_model.snapshot()},
            daily_outputs=[output.to_dict() for output in self._outputs])

    def restore(self, checkpoint):
        if (checkpoint.get('schema_version') != 1 or checkpoint.get('model_version') != __version__
                or checkpoint.get('configuration_sha256') != self.config.identity):
            raise ValueError('Checkpoint model version or configuration identity differs.')
        if checkpoint.get('runtime_source_sha256') != self._source_hashes:
            raise ValueError('Checkpoint runtime source version differs.')
        count = checkpoint.get('consumed_days')
        if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= len(self.weather):
            raise ValueError('Checkpoint consumed weather-prefix length is invalid.')
        if checkpoint.get('consumed_weather_sha256') != self.weather.prefix_identity(count):
            raise ValueError('Checkpoint historical weather prefix changed.')
        outputs = [DailyOutput.from_dict(value) for value in checkpoint['daily_outputs']]
        if (len(outputs) != count or any(output.date != self.weather[index].date
                or output.field_id != self.config.field_id
                or output.crop_included != self.phenology.included(output.date)
                for index, output in enumerate(outputs))):
            raise ValueError('Checkpoint daily output membership differs from its forcing prefix.')
        candidates = self._components()
        initial_states = [component.snapshot() for component in candidates]
        for component, name in zip(candidates, ['phenology', 'leaf', 'disease', 'yield']):
            component.restore(checkpoint['components'][name])
        epochs = [candidates[0].state.day_count, checkpoint['components']['leaf']['day_count'],
                  checkpoint['components']['disease']['epoch'], candidates[3].state.day_count]
        if any(value != count for value in epochs):
            raise ValueError('Checkpoint component epochs disagree with the weather prefix.')
        if count and candidates[0].state.last_date != self.weather[count-1].date:
            raise ValueError('Checkpoint phenology date differs from its forcing prefix.')
        self._validate_checkpoint_components(candidates, outputs, initial_states)
        self.phenology, self.leaf, self.disease, self.yield_model = candidates
        self._outputs, self._cursor, self._owner, self._status = outputs, count, uuid4().hex, 'running'
        return self

    def _validate_checkpoint_components(self, candidates, outputs, initial_states):
        if not outputs:
            if any(component.snapshot() != initial for component, initial
                   in zip(candidates, initial_states)):
                raise ValueError('Empty checkpoint differs from canonical initial component states.')
            return
        pheno, leaf, disease, yield_model = candidates
        last, leaf_state, y = outputs[-1], leaf.snapshot(), yield_model.state
        def agrees(actual, expected):
            return np.allclose(actual, expected, rtol=0., atol=1e-10)
        clocks = ['cumulative_gdd', 'cumulative_tpp', 'cumulative_tpv', 'cumulative_vernalization']
        if not agrees([getattr(pheno.state, key) for key in clocks],
                      [getattr(last, key) for key in clocks]):
            raise ValueError('Checkpoint development clocks disagree with terminal daily output.')
        included = [day for day in outputs if day.crop_included]
        reached = {str(stage): any(day.cumulative_tpv >= threshold for day in included)
                   for stage, threshold in self.config.leaf.stage_thresholds}
        if (not agrees(leaf_state['previous_accumulation'], last.cumulative_tpv)
                or not agrees(leaf_state['previous_area'], last.host_area)
                or leaf_state['previous_valid'] != last.crop_included
                or leaf_state['started'] != bool(included)
                or leaf_state['ended'] != bool(included and not last.crop_included)
                or leaf_state['reached'] != reached):
            raise ValueError('Checkpoint leaf state disagrees with daily development and capacity.')
        tissue, residue = disease.state, disease.residue
        if (not agrees(residue, [last.residue_unready, last.residue_competent])
                or not agrees(tissue[:, 0], last.susceptible_fraction)
                or not agrees(tissue[:, 1:-3].sum(axis=-1), last.latent_fraction)
                or not agrees(tissue[:, -3:].sum(axis=-1), last.symptomatic_fraction)
                or not agrees(tissue[:, 1:].sum(axis=-1), last.affected_fraction)
                or not agrees(tissue[:, -2], last.infectious_fraction)):
            raise ValueError('Checkpoint disease fractions/residue disagree with terminal daily output.')
        expected_had = ([0., 0.] if self.config.yield_model.mode == 'disabled' else
                        [last.cumulative_reference_had3, last.cumulative_lost_had3])
        if any(value is None for value in expected_had) or not agrees(
                [y.reference_had3, y.lost_had3], expected_had):
            raise ValueError('Checkpoint cumulative HAD disagrees with terminal daily output.')
        thresholds = dict(self.config.leaf.stage_thresholds)
        def crossing(stage):
            return next((day.date for day in included if day.cumulative_tpv >= thresholds[stage]), None)
        if y.window_start_date != crossing(65) or y.window_end_date != crossing(85):
            raise ValueError('Checkpoint yield window disagrees with its daily stage crossings.')
