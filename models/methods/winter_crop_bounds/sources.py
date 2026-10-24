"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/winter_wheat_population_20261010')
from itertools import product
import numpy as np
import pandas as pd
UNIT = 'HPRD_HUMD_EU_THS_T'
CROPS = ('C1100', 'C1110', 'C1111', 'C1112', 'C1120')

def native_country_totals(native, raster, nodata, pixel_size=1 / 12, west=-180.0, north=90.0):
    """Aggregate native production, preserving unassigned positive tonnes.

    SPAM CSV centres are rounded to six significant digits. Country identities
    come from the provider's native grid, not a new boundary overlay.
    """
    a = np.asarray(raster, dtype=float)
    if a.ndim != 2 or not np.isfinite(pixel_size) or pixel_size <= 0:
        raise ValueError('Invalid native raster geometry')
    valid = np.isfinite(a) & (a != nodata)
    if (a[valid] < 0).any():
        raise ValueError('Negative native production')
    quantities = np.where(valid, a, 0.0)
    xy = native[['x', 'y']].to_numpy(dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError('Invalid native coordinates')
    col = np.floor((xy[:, 0] - west) / pixel_size).astype(int)
    row = np.floor((north - xy[:, 1]) / pixel_size).astype(int)
    if ((row < 0) | (row >= a.shape[0]) | (col < 0) | (col >= a.shape[1])).any():
        raise ValueError('Native coordinate outside raster')
    centres = np.column_stack((west + (col + 0.5) * pixel_size, north - (row + 0.5) * pixel_size))
    if not np.allclose(xy, centres, rtol=0, atol=0.00051):
        raise ValueError('Native coordinates are not pixel centres')
    if native.grid_code.duplicated().any() or pd.DataFrame({'row': row, 'col': col}).duplicated().any():
        raise ValueError('duplicate native grid location')
    labels = native[['ADM0_NAME', 'FIPS0']].copy()
    named = labels.notna().all(axis=1) & labels.apply(lambda x: x.astype(str).str.strip().ne('')).all(axis=1)
    labels['mapped_national_total_tonnes'] = quantities[row, col]
    table = labels[named].groupby(['ADM0_NAME', 'FIPS0'], as_index=False).mapped_national_total_tonnes.sum().rename(columns={'ADM0_NAME': 'country', 'FIPS0': 'country_fips'})
    if table.country.duplicated().any() or table.country_fips.duplicated().any():
        raise ValueError('Country names and native codes are not one to one')
    global_mass = float(quantities.sum())
    mapped_mass = float(table.mapped_national_total_tonnes.sum())
    residual = global_mass - mapped_mass
    if residual < -1e-06:
        raise ValueError('Mapped production exceeds global raster mass')
    return (table, dict(global_positive_production_tonnes=global_mass, mapped_country_production_tonnes=mapped_mass, unassigned_positive_production_tonnes=max(0.0, residual), native_identity_rows=len(native), assigned_identity_rows=int(named.sum()), raster_rows=a.shape[0], raster_columns=a.shape[1], coordinate_tolerance_degrees=0.00051))

def parse_eurostat(payload):
    """Read sparse JSON-stat, preserving absent quantities and source flags."""
    ids = payload['id']
    sizes = payload['size']
    if set(ids) != {'freq', 'crops', 'strucpro', 'geo', 'time'}:
        raise ValueError('Unexpected Eurostat dimensions')
    levels = {}
    for key, size in zip(ids, sizes):
        index = payload['dimension'][key]['category']['index']
        if sorted(index.values()) != list(range(size)):
            raise ValueError('Invalid Eurostat category index')
        levels[key] = [k for k, _ in sorted(index.items(), key=lambda x: x[1])]
    if levels['strucpro'] != [UNIT]:
        raise ValueError('Expected EU-standard-humidity production unit')
    if levels['freq'] != ['A']:
        raise ValueError('Expected annual production')
    values = payload.get('value', {})
    flags = payload.get('status', {})

    def at(array, flat, default):
        return array.get(str(flat), default) if isinstance(array, dict) else array[flat]
    rows = []
    for flat, positions in enumerate(product(*(range(n) for n in sizes))):
        key = {k: levels[k][j] for k, j in zip(ids, positions)}
        if key['crops'] not in CROPS:
            continue
        value = at(values, flat, None)
        if value is not None and (not np.isfinite(value) or value < 0):
            raise ValueError('Invalid production quantity')
        rows.append(dict(geo=key['geo'], year=int(key['time']), crop=key['crops'], production_thousand_tonnes=np.nan if value is None else float(value), flag=at(flags, flat, '') or ''))
    return pd.DataFrame(rows)

def national_margins(records, crosswalk, years=(2019, 2020, 2021), tolerance_thousand_tonnes=0.15):
    """Constrain only complete, compatible national seasonal production series."""
    if records.duplicated(['geo', 'year', 'crop']).any():
        raise ValueError('duplicate national crop record')
    if crosswalk.country.duplicated().any() or crosswalk.eurostat_geo.duplicated().any():
        raise ValueError('Country crosswalk must be one to one')
    if len(set(years)) != len(years) or not years or tolerance_thousand_tonnes < 0:
        raise ValueError('Invalid reference years or arithmetic tolerance')
    required = pd.MultiIndex.from_product([years, CROPS], names=['year', 'crop'])
    out = []
    for row in crosswalk.itertuples(index=False):
        source = records[records.geo.eq(row.eurostat_geo)].set_index(['year', 'crop'])
        s = source.reindex(required)
        values = s.production_thousand_tonnes
        reasons = []
        finite = values.notna().all()
        if not finite:
            reasons.append('missing_crop_or_year')
        if values.dropna().lt(0).any() or not np.isfinite(values.dropna()).all():
            raise ValueError('Invalid production quantity')
        flags = sorted(set(s.flag.dropna().astype(str)) - {''})
        if any((set(flag.replace(' ', '')) - {'e', 'n'} for flag in flags)):
            reasons.append('unsupported_quality_flag')
        if (s.flag.fillna('').str.contains('n') & values.notna() & values.ne(0)).any():
            raise ValueError('n flag requires a reported zero quantity')
        a = values.unstack('crop')
        total = a.C1100.sum(min_count=len(years))
        winter = a.C1111.sum(min_count=len(years))
        common_residual = (a.C1111 + a.C1112 - a.C1110).abs().max()
        total_residual = (a.C1110 + a.C1120 - a.C1100).abs().max()
        if finite and (common_residual > tolerance_thousand_tonnes or total_residual > tolerance_thousand_tonnes):
            reasons.append('nonadditive_components')
        if a.C1100.isna().any() or a.C1100.le(0).any():
            reasons.append('missing_or_nonpositive_total')
        fraction = float(winter / total) if np.isfinite(total) and total > 0 else np.nan
        if np.isfinite(fraction) and (not 0 <= fraction <= 1):
            reasons.append('winter_outside_total')
        supported = not reasons
        out.append(dict(country=row.country, country_fips=row.country_fips, eurostat_geo=row.eurostat_geo, supported=supported, reason='supported' if supported else '|'.join(reasons), flags='|'.join(flags), years='|'.join(map(str, years)), observed_required_records=int(values.notna().sum()), required_records=len(required), winter_fraction=fraction, fraction_lower=fraction if supported else 0.0, fraction_upper=fraction if supported else 1.0, total_production_mean_tonnes=float(total / len(years) * 1000), winter_production_mean_tonnes=float(winter / len(years) * 1000), maximum_common_identity_residual_tonnes=float(common_residual * 1000), maximum_total_identity_residual_tonnes=float(total_residual * 1000)))
    return pd.DataFrame(out)
