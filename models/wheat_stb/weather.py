"""Immutable, consecutive daily forcing with explicit physical units."""

from collections.abc import Iterator, Sequence
import csv
from dataclasses import asdict, dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np


def iso_date(value: date | str) -> date:
    if type(value) is date:
        return value
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as error:
            raise ValueError('A calendar date in YYYY-MM-DD format is required.') from error
        if value == parsed.isoformat():
            return parsed
    raise ValueError('A calendar date in YYYY-MM-DD format is required.')


@dataclass(frozen=True)
class WeatherDay:
    date: date
    tmean_c: float
    tmax_c: float
    rh_mean_pct: float
    precipitation_mm: float
    imported_pressure: float | None = None
    exposure_override: float | None = None
    reference_leaf_area_index: tuple[float, float, float] | None = None
    functional_loss_fraction: tuple[float, float, float] | None = None

    def __post_init__(self):
        object.__setattr__(self, 'date', iso_date(self.date))
        names = ['tmean_c', 'tmax_c', 'rh_mean_pct', 'precipitation_mm']
        for name in names:
            object.__setattr__(self, name, float(getattr(self, name)))
        if not np.isfinite([getattr(self, name) for name in names]).all():
            raise ValueError('Daily forcing must be finite.')
        if self.tmean_c > 40:
            raise ValueError('The copied thermal response is unsupported above 40 C daily mean.')
        if self.tmax_c < self.tmean_c-.001:
            raise ValueError('Daily maximum temperature must not be below the mean.')
        if not 0 <= self.rh_mean_pct <= 100:
            raise ValueError('Relative humidity must be in percent within [0,100].')
        if self.precipitation_mm < 0:
            raise ValueError('Daily precipitation must be nonnegative mm.')
        if self.imported_pressure is not None:
            value = float(self.imported_pressure)
            if not np.isfinite(value) or value < 0:
                raise ValueError('Imported relative source pressure must be nonnegative.')
            object.__setattr__(self, 'imported_pressure', value)
        if self.exposure_override is not None:
            value = float(self.exposure_override)
            if not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('An explicit dimensionless exposure must lie in [0,1].')
            object.__setattr__(self, 'exposure_override', value)
        supplied = (self.reference_leaf_area_index is not None, self.functional_loss_fraction is not None)
        if supplied[0] != supplied[1]:
            raise ValueError('Reference leaf area and functional loss must be supplied together.')
        if all(supplied):
            area = tuple(float(x) for x in self.reference_leaf_area_index)
            loss = tuple(float(x) for x in self.functional_loss_fraction)
            if (len(area) != 3 or len(loss) != 3 or not np.isfinite(area+loss).all()
                    or min(area) < 0 or min(loss) < 0 or max(loss) > 1):
                raise ValueError('Three nonnegative LAI values and functional fractions in [0,1] required.')
            object.__setattr__(self, 'reference_leaf_area_index', area)
            object.__setattr__(self, 'functional_loss_fraction', loss)

    def to_dict(self) -> dict:
        values = asdict(self)
        values['date'] = self.date.isoformat()
        return values


@dataclass(frozen=True)
class WeatherProvider(Sequence[WeatherDay]):
    records: tuple[WeatherDay, ...]
    source: str = 'caller-supplied daily forcing'
    source_sha256: str | None = None

    def __post_init__(self):
        rows = tuple(self.records)
        if not rows or any(not isinstance(row, WeatherDay) for row in rows):
            raise ValueError('At least one validated WeatherDay is required.')
        if any(right.date != left.date+timedelta(days=1) for left, right in zip(rows, rows[1:])):
            raise ValueError('Weather dates must be ordered, unique and consecutive without gaps.')
        object.__setattr__(self, 'records', rows)

    def __getitem__(self, index):
        return self.records[index]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[WeatherDay]:
        return iter(self.records)

    def prefix_identity(self, count: int | None = None) -> str:
        count = len(self) if count is None else count
        if not isinstance(count, int) or not 0 <= count <= len(self):
            raise ValueError('A valid consumed weather-prefix length is required.')
        raw = json.dumps([row.to_dict() for row in self.records[:count]],
                         sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
        return hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_csv(cls, path: str | Path) -> 'WeatherProvider':
        path = Path(path)
        required = {'date', 'tmean_c', 'tmax_c', 'rh_mean_pct', 'precipitation_mm'}
        optional = {'imported_pressure', 'exposure_override',
            *[f'reference_lai_F{i}' for i in range(1, 4)],
            *[f'functional_loss_F{i}' for i in range(1, 4)]}
        with path.open(newline='') as stream:
            reader = csv.DictReader(stream)
            columns = set(reader.fieldnames or [])
            if not required.issubset(columns) or columns-required-optional:
                raise ValueError('Weather CSV has missing required or unsupported/outcome columns.')
            area_columns = {f'reference_lai_F{i}' for i in range(1, 4)}
            loss_columns = {f'functional_loss_F{i}' for i in range(1, 4)}
            if columns & (area_columns | loss_columns) and not (area_columns | loss_columns).issubset(columns):
                raise ValueError('All three reference LAI and functional-loss columns are required together.')
            rows = []
            for raw in reader:
                values = {key: raw[key] for key in required}
                values['date'] = iso_date(values['date'])
                for key in ['imported_pressure', 'exposure_override']:
                    if raw.get(key, '') != '':
                        values[key] = float(raw[key])
                if area_columns.issubset(columns):
                    entries = [raw[key] for key in sorted(area_columns | loss_columns)]
                    if any(x != '' for x in entries):
                        if not all(x != '' for x in entries):
                            raise ValueError('Partial daily reference area or functional loss is invalid.')
                        values['reference_leaf_area_index'] = tuple(float(raw[f'reference_lai_F{i}']) for i in range(1, 4))
                        values['functional_loss_fraction'] = tuple(float(raw[f'functional_loss_F{i}']) for i in range(1, 4))
                rows.append(WeatherDay(**values))
        return cls(tuple(rows), str(path.resolve()), hashlib.sha256(path.read_bytes()).hexdigest())
