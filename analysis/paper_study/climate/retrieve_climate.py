"""Resumable public daily meteorological forcing retrieval, with explicit coverage.

ERA5 is aggregated from hourly observations; NASA NEX-GDDP-CMIP6 v2 is
retrieved from the provider's THREDDS archive, including SSP126. Credentials
are reused solely through ee.Initialize and never opened or archived here.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
import xml.etree.ElementTree as ET

import ee
import numpy as np
import pandas as pd
import rasterio
import requests
import xarray as xr
from pyproj import Geod

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / 'analysis/paper_study/climate'
DATA = ROOT / 'data/paper_study/climate'
REGISTRY = ROOT / 'data/paper_study/wheat_area/europe_wheat_cells_025.parquet'
PROJECT = 'ee-gangzhaomodel'
ERA5 = 'ECMWF/ERA5/HOURLY'
TRANSFORM = [0.25, 0, -180, 0, -0.25, 90]
MODELS = ['ACCESS-CM2', 'MPI-ESM1-2-HR', 'MRI-ESM2-0']
SCENARIOS = ['ssp126', 'ssp245', 'ssp585']
FIELDS = ['tmean_c', 'tmax_c', 'precipitation_mm', 'rh_mean_pct']
FIELD_CHOICES = ['tmean_c', 'tmin_c', 'tmax_c', 'precipitation_mm', 'rh_mean_pct']
NASA_FIELDS = {'tas': 'tmean_c', 'tasmin': 'tmin_c', 'tasmax': 'tmax_c',
               'pr': 'precipitation_mm', 'hurs': 'rh_mean_pct'}
BASE = 'https://ds.nccs.nasa.gov/thredds/'
NS = {'t': 'http://www.unidata.ucar.edu/namespaces/thredds/InvCatalog/v1.0'}
PRINT_LOCK = threading.Lock()
CATALOG_LOCK = threading.Lock()
MERGE_LOCK = threading.Lock()


def utc():
    return datetime.now(timezone.utc).isoformat()


def report(value):
    with PRINT_LOCK:
        print(json.dumps(value, ensure_ascii=False), flush=True)


def safe_error(exc):
    message = re.sub(r'https?://\S+', '<redacted URL>', str(exc))
    return re.sub(r'/v1/projects/\S+', '<redacted download endpoint>', message)[:1200]


def jsonable(value):
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    tmp.write_text(json.dumps(jsonable(value), indent=2, ensure_ascii=False) + '\n')
    tmp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda: f.read(1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def registry(path):
    cells = pd.read_parquet(path).sort_values(['row', 'col']).reset_index(drop=True)
    assert len(cells) and not cells.cell_id.duplicated().any()
    assert (cells.harvested_total_ha > 0).all()
    assert np.allclose(cells.longitude, -180 + (cells.col + 0.5) * 0.25)
    assert np.allclose(cells.latitude, 90 - (cells.row + 0.5) * 0.25)
    return cells


def date_range(start, end):
    return pd.date_range(start, pd.Timestamp(end) - pd.Timedelta(days=1), freq='D')


def quality(frame, fields):
    assert not frame.duplicated(['date', 'cell_id']).any()
    values = frame[fields]
    result = {'rows': len(frame), 'days': frame.date.nunique(),
              'cells': frame.cell_id.nunique(),
              'date_min': str(frame.date.min()), 'date_max': str(frame.date.max()),
              'missing_values': {k: int(v) for k, v in values.isna().sum().items()},
              'ranges': {k: {'min': float(values[k].min()), 'max': float(values[k].max())}
                         for k in fields}}
    if any(result['missing_values'].values()):
        raise ValueError(f'Missing meteorological data: {result["missing_values"]}')
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('Nonfinite meteorological value')
    if 'tmean_c' in fields and 'tmax_c' in fields and (frame.tmean_c > frame.tmax_c + .001).any():
        raise ValueError('Daily mean temperature exceeds maximum')
    if 'tmin_c' in fields and 'tmax_c' in fields:
        if (frame.tmin_c > frame.tmax_c + 0.001).any():
            raise ValueError('Daily minimum temperature exceeds maximum')
    if 'rh_mean_pct' in fields:
        if not frame.rh_mean_pct.between(0, 100).all():
            raise ValueError('Harmonized relative humidity outside 0–100%')
    if 'precipitation_mm' in fields and (frame.precipitation_mm < 0).any():
        raise ValueError('Harmonized precipitation is negative')
    return result


def commit_frame(path, frame, meta, fields):
    meta['quality'] = quality(frame, fields)
    if shutil.disk_usage(ROOT).free < 10_000_000_000:
        raise OSError('Less than 10GB free disk; retrieval paused before writing another climate partition')
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp.parquet')
    frame.to_parquet(tmp, index=False, compression='zstd')
    tmp.replace(path)
    meta['parquet_sha256'] = sha256(path)
    meta['parquet_bytes'] = path.stat().st_size
    meta['retrieved_utc'] = utc()
    write_json(path.with_suffix('.json'), meta)
    return meta


def done(path, fields=None):
    provenance = path.with_suffix('.json')
    if not path.exists() or not provenance.exists():
        return False
    info = json.loads(provenance.read_text())
    if fields is not None:
        if not set(fields).issubset(info.get('quality', {}).get('missing_values', {})):
            return False
        if info.get('registry_sha256') != sha256(REGISTRY):
            return False
    return info.get('parquet_sha256') == sha256(path)


def era5_daily_image(day, fields):
    start = ee.Date(day.strftime('%Y-%m-%d'))
    end = start.advance(1, 'day')
    hourly = ee.ImageCollection(ERA5).filterDate(start, end)
    native_projection = ee.Image(hourly.first()).select('temperature_2m').projection()
    t = hourly.select('temperature_2m')
    images = {'tmean_c': t.mean().subtract(273.15),
              'tmin_c': t.min().subtract(273.15),
              'tmax_c': t.max().subtract(273.15)}

    def rh(image):
        temp = image.select('temperature_2m').subtract(273.15)
        dew = image.select('dewpoint_temperature_2m').subtract(273.15)
        return (dew.multiply(17.625).divide(dew.add(243.04))
                .subtract(temp.multiply(17.625).divide(temp.add(243.04)))
                .exp().multiply(100).clamp(0, 100).rename('rh_mean_pct'))

    images['rh_mean_pct'] = hourly.map(rh).mean()
    # Each accumulation ends at its timestamp. Day D needs D01:00 through D+1 00:00.
    precip = (ee.ImageCollection(ERA5)
              .filterDate(start.advance(1, 'hour'), end.advance(1, 'hour'))
              .select('total_precipitation').sum().multiply(1000))
    images['precipitation_mm'] = precip
    return (ee.Image.cat([images[f].rename(f) for f in fields])
            .setDefaultProjection(native_projection)
            .resample('bilinear').toFloat()
            .set('system:index', day.strftime('%Y%m%d'))
            .set('system:time_start', start.millis()))


def era5_job(start, end, cells, fields, outdir):
    name = f'{start}_{end}'
    path = outdir / f'{name}.parquet'
    if done(path, fields):
        return {'status': 'existing', 'job': name}
    days = date_range(start, end)
    west = float(cells.west.min()); east = float(cells.east.max())
    south = float(cells.south.min()); north = float(cells.north.max())
    pixels = int(round((east-west)/0.25)) * int(round((north-south)/0.25))
    estimated = pixels * len(days) * len(fields) * 4
    if estimated > 29_000_000:
        raise ValueError(f'Untiled ERA5 job exceeds safe 32MB limit ({estimated} bytes)')
    collection = ee.ImageCollection(ERA5).filterDate(start, pd.Timestamp(end).strftime('%Y-%m-%dT01:00:00'))
    counts = collection.size().getInfo()
    if counts != len(days) * 24 + 1:
        raise ValueError(f'ERA5 hourly coverage {counts} != {len(days)*24+1}')
    source_projection = collection.first().select('temperature_2m').projection().getInfo()
    images = [era5_daily_image(day, fields) for day in days]
    image = ee.ImageCollection.fromImages(images).toBands()
    # Paint positive-wheat centres on the common grid so other cells are masked.
    points = ee.FeatureCollection([ee.Feature(ee.Geometry.Point([float(x), float(y)]))
                                  for x, y in zip(cells.longitude, cells.latitude)])
    mask = (ee.Image.constant(0).toByte()
            .reproject(crs='EPSG:4326', crsTransform=TRANSFORM).paint(points, 1))
    image = image.updateMask(mask).unmask(-9999)
    payload = {'format': 'GEO_TIFF', 'filePerBand': False, 'crs': 'EPSG:4326',
               'crs_transform': TRANSFORM,
               'region': ee.Geometry.Rectangle([west, south, east, north], geodesic=False)}
    url = image.getDownloadURL(payload)
    response = requests.get(url, timeout=(30, 420))
    if not response.ok:
        if response.status_code >= 500 or response.status_code == 429:
            raise requests.HTTPError(f'ERA5 raster HTTP {response.status_code}')
        raise RuntimeError(f'ERA5 raster HTTP {response.status_code}')
    raw_checksum = hashlib.sha256(response.content).hexdigest()
    with rasterio.MemoryFile(response.content) as memory:
        with memory.open() as src:
            assert src.count == len(days) * len(fields) and src.crs.to_epsg() == 4326
            assert abs(src.transform.a - .25) < 1e-12 and abs(src.transform.e + .25) < 1e-12
            cols, rows = (~src.transform) * (cells.longitude.to_numpy(), cells.latitude.to_numpy())
            cols = np.floor(cols).astype(int); rows = np.floor(rows).astype(int)
            array = src.read()[:, rows, cols].reshape(len(days), len(fields), len(cells))
    frame = pd.DataFrame({'date': np.repeat(days.strftime('%Y-%m-%d'), len(cells)),
                          'cell_id': np.tile(cells.cell_id.values, len(days))})
    for i, field in enumerate(fields):
        frame[field] = array[:, i, :].ravel()
    if (frame[fields] == -9999).any().any():
        raise ValueError('Positive-wheat registry centre absent from ERA5 raster mask')
    adjustments = {}
    if 'precipitation_mm' in fields:
        negatives = frame.precipitation_mm < 0
        if negatives.any():
            adjustments['precipitation_negative_cells'] = int(negatives.sum())
            adjustments['precipitation_raw_min_mm'] = float(frame.precipitation_mm.min())
            if frame.precipitation_mm.min() < -0.01:
                raise ValueError('Negative ERA5 daily precipitation beyond 0.01mm numerical tolerance')
            frame['precipitation_mm_raw'] = frame.precipitation_mm
            frame['precipitation_mm'] = frame.precipitation_mm.clip(lower=0)
    meta = {'source_collection': ERA5, 'source_catalog':
            'https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_HOURLY',
            'window_start': start, 'window_end_exclusive': end, 'hourly_images': counts,
            'source_projection': source_projection, 'target_transform': TRANSFORM,
            'resampling': 'Daily continuous fields bilinearly interpolated from native ERA5 grid to common centres',
            'temperature': 'Mean/minimum/maximum of 24 hourly 2m temperatures at 00–23 UTC; K minus 273.15',
            'relative_humidity': 'Daily arithmetic mean of 24 hourly values: 100exp(17.625Td/(243.04+Td)-17.625T/(243.04+T)), T/Td in °C; hourly RH clipped to 0–100%',
            'precipitation': 'Sum of 24 one-hour accumulations ending 01 UTC on day D through 00 UTC on D+1; m×1000; total rain+snow water equivalent',
            'units': {'tmean_c': 'degC', 'tmin_c': 'degC', 'tmax_c': 'degC',
                      'precipitation_mm': 'mm/day', 'rh_mean_pct': '%'},
            'attribution': 'Contains modified Copernicus Climate Change Service information; supplied via Google Earth Engine',
            'license_url': 'https://apps.ecmwf.int/datasets/licences/copernicus/',
            'registry_path': str(REGISTRY.relative_to(ROOT)), 'registry_sha256': sha256(REGISTRY),
            'download_bytes': len(response.content), 'download_sha256': raw_checksum,
            'download_URL_policy': 'Ephemeral signed URL excluded from provenance',
            'raw_raster_retention': 'Transient in memory; extracted positive-wheat cell values archived',
            'numeric_adjustments': adjustments}
    return commit_frame(path, frame, meta, fields) | {'status': 'downloaded', 'job': name}


def nasa_catalog(model, scenario, variable):
    # Confirm the actual realization and version; never manufacture a filename.
    realization = 'r1i1p1f1'
    cache = DATA / 'catalogs' / f'{model}_{scenario}_{realization}_{variable}.xml'
    url = BASE + f'catalog/AMES/NEX/GDDP-CMIP6/{model}/{scenario}/{realization}/{variable}/catalog.xml'
    with CATALOG_LOCK:
        if not cache.exists():
            response = requests.get(url, timeout=(20, 90))
            if not response.ok:
                raise RuntimeError(f'NASA catalog HTTP{response.status_code}: {model}/{scenario}/{variable}')
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_bytes(response.content)
        parsed = ET.fromstring(cache.read_bytes())
    paths = [d.attrib['urlPath'] for d in parsed.findall('.//t:dataset', NS) if 'urlPath' in d.attrib]
    return paths, url, cache


def nasa_source(model, scenario, variable, year):
    paths, catalog_url, catalog_path = nasa_catalog(model, scenario, variable)
    year_pattern = re.compile(rf'_{year}(?:_v[\d.]+)?\.nc$')
    candidates = [p for p in paths if year_pattern.search(p)]
    v2 = [p for p in candidates if p.endswith('_v2.0.nc')]
    if len(v2) != 1:
        raise ValueError(f'Expected exactly one native NASA v2.0 file, got{len(v2)}: {model}/{scenario}/{variable}/{year}')
    return v2[0], catalog_url, catalog_path


def prepare_nasa_alignment(model, cells, variables, outdir):
    """Freeze one model-specific, jointly valid land neighbour across all periods."""
    path = outdir / 'coastal_alignment' / f'{model}.parquet'
    meta_path = path.with_suffix('.json')
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get('registry_sha256') == sha256(REGISTRY) and meta.get('variables') == variables:
            return pd.read_parquet(path)
    grid = None
    joint_valid = None
    probes = []
    for variable in variables:
        source, _, _ = nasa_source(model, 'historical', variable, 1990)
        params = {'var': variable, 'north': float(cells.latitude.max()+.5),
                  'south': float(cells.latitude.min()-.5), 'west': float(cells.longitude.min()-.5),
                  'east': float(cells.longitude.max()+.5), 'horizStride': 1,
                  'time_start': '1990-01-01T12:00:00Z', 'time_end': '1990-01-01T12:00:00Z',
                  'accept': 'netcdf4', 'addLatLon': 'true'}
        response = requests.get(BASE+'ncss/grid/'+source, params=params, timeout=(30, 180))
        if not response.ok:
            raise RuntimeError(f'NASA joint land-mask probe HTTP{response.status_code}')
        with xr.open_dataset(BytesIO(response.content), engine='h5netcdf') as ds:
            lon = (ds.lon.values+180) % 360-180
            lat = ds.lat.values
            valid = np.isfinite(ds[variable].values[0])
            if grid is None:
                grid = (lat, lon)
                joint_valid = valid
            else:
                assert np.allclose(grid[0], lat) and np.allclose(grid[1], lon)
                joint_valid &= valid
            probes.append({'variable': variable, 'native_path': source,
                           'query': params, 'download_bytes': len(response.content),
                           'download_sha256': hashlib.sha256(response.content).hexdigest()})
    lat, lon = grid
    land_centres = {(int(round((90-y)/.25-.5)), int(round((x+180)/.25-.5))): (float(y), float(x))
                    for i, y in enumerate(lat) for j, x in enumerate(lon) if joint_valid[i, j]}
    geod = Geod(ellps='WGS84')
    rows = []
    for cell in cells.itertuples(index=False):
        key = (int(cell.row), int(cell.col))
        chosen = key if key in land_centres else None
        angular_distance = 0.0
        if chosen is None:
            candidates = []
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    distance = .25*np.hypot(dr, dc)
                    candidate = (key[0]+dr, key[1]+dc)
                    if distance <= .5+1e-12 and candidate in land_centres:
                        candidates.append((float(distance), candidate[0], candidate[1]))
            if candidates:
                angular_distance, sr, sc = min(candidates)
                chosen = (sr, sc)
        if chosen is None:
            source_lat = source_lon = distance_km = np.nan
            source_row = source_col = -1
        else:
            source_lat, source_lon = land_centres[chosen]
            source_row, source_col = chosen
            distance_km = geod.inv(float(cell.longitude), float(cell.latitude), source_lon, source_lat)[2]/1000
        rows.append({'cell_id': cell.cell_id, 'row': cell.row, 'col': cell.col,
                     'latitude': cell.latitude, 'longitude': cell.longitude,
                     'harvested_total_ha': cell.harvested_total_ha,
                     'native_joint_land_masked': key not in land_centres,
                     'coastal_imputed': chosen is not None and chosen != key,
                     'eligible': chosen is not None, 'source_row': source_row, 'source_col': source_col,
                     'source_latitude': source_lat, 'source_longitude': source_lon,
                     'distance_degrees': angular_distance if chosen is not None else np.nan,
                     'distance_km': distance_km})
    alignment = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    alignment.to_parquet(path, index=False, compression='zstd')
    alignment.to_csv(path.with_suffix('.csv'), index=False)
    meta = {'model': model, 'registry_sha256': sha256(REGISTRY), 'variables': variables,
            'reference': 'Joint nonmissing native fields on historical 1990-01-01; one frozen lookup across historical and all future SSPs',
            'radius': 'Euclidean distance in latitude/longitude degrees ≤0.5; tie broken by source row then column',
            'source_grid': 'Native 0.25° NASA NEX-GDDP-CMIP6 v2.0',
            'probes': probes, 'total_wheat_cells': len(alignment),
            'native_land_masked_cells': int(alignment.native_joint_land_masked.sum()),
            'imputed_cells': int(alignment.coastal_imputed.sum()),
            'imputed_harvested_ha': float(alignment.loc[alignment.coastal_imputed, 'harvested_total_ha'].sum()),
            'excluded_cells': int((~alignment.eligible).sum()),
            'excluded_harvested_ha': float(alignment.loc[~alignment.eligible, 'harvested_total_ha'].sum()),
            'eligible_harvested_ha': float(alignment.loc[alignment.eligible, 'harvested_total_ha'].sum()),
            'mapping_sha256': sha256(path), 'checked_utc': utc()}
    write_json(meta_path, meta)
    report({'source': 'nasa_alignment', **{k: meta[k] for k in ['model','total_wheat_cells','native_land_masked_cells','imputed_cells','excluded_cells','imputed_harvested_ha','excluded_harvested_ha']}})
    return alignment


def nasa_quarter_subsets(url, source, params, model, scenario, variable, year, outdir):
    datasets = []
    parts = []
    paths = []
    aggregate_hash = hashlib.sha256()
    for quarter in range(1, 5):
        start = pd.Timestamp(year=year, month=3*quarter-2, day=1)
        end = start+pd.offsets.QuarterEnd()
        query = params | {'time_start': f'{start:%Y-%m-%d}T12:00:00Z',
                          'time_end': f'{end:%Y-%m-%d}T12:00:00Z'}
        cache = outdir/'ncss_cache'/model/scenario/variable/f'{year}_Q{quarter}.nc'
        cached_meta = cache.with_suffix('.json')
        cached = False
        if cache.exists() and cached_meta.exists():
            info = json.loads(cached_meta.read_text())
            cached = info.get('query') == query and info.get('sha256') == sha256(cache)
        if not cached:
            response = requests.get(url, params=query, timeout=(30, 180))
            if not response.ok:
                if response.status_code >= 500 or response.status_code == 429:
                    raise requests.HTTPError(f'NASA NCSS quarter HTTP{response.status_code}: {model}/{scenario}/{variable}/{year}/Q{quarter}')
                raise RuntimeError(f'NASA NCSS quarter HTTP{response.status_code}')
            content = response.content
            cache.parent.mkdir(parents=True, exist_ok=True)
            temp = cache.with_suffix('.tmp.nc')
            temp.write_bytes(content); temp.replace(cache)
            info = {'native_source_path': source, 'query': query, 'bytes': len(content),
                    'sha256': sha256(cache), 'retrieved_utc': utc()}
            write_json(cached_meta, info)
        content = cache.read_bytes()
        aggregate_hash.update(content)
        with xr.open_dataset(BytesIO(content), engine='h5netcdf') as ds:
            expected = pd.date_range(start, end).strftime('%Y-%m-%d').tolist()
            actual = pd.DatetimeIndex(ds.time.values).strftime('%Y-%m-%d').tolist()
            if actual != expected:
                raise ValueError(f'NASA quarter dates differ: {model}/{scenario}/{year}/Q{quarter}')
            assert ds.attrs.get('version') == '2.0' and ds.attrs.get('cmip6_source_id') == model
            assert ds.attrs.get('scenario') == scenario
            datasets.append(ds.load())
            parts.append(info | {'native_global_attributes': jsonable(ds.attrs),
                                 'native_calendar': ds.time.encoding.get('calendar')})
        paths.extend([cache, cached_meta])
    combined = xr.concat(datasets, dim='time', combine_attrs='override')
    return combined, parts, aggregate_hash.hexdigest(), paths


def nasa_variable(model, scenario, variable, year, cells, outdir, alignment=None):
    field = NASA_FIELDS[variable]
    path = outdir / 'variables' / model / scenario / variable / f'{year}.parquet'
    if done(path, [field]):
        return {'status': 'existing', 'job': f'{model}/{scenario}/{variable}/{year}'}
    source, catalog_url, catalog_path = nasa_source(model, scenario, variable, year)
    url = BASE + 'ncss/grid/' + source
    # Exact source-centre boundaries prevent THREDDS axis shifts at the prime meridian.
    if alignment is not None:
        selected = alignment.loc[alignment.eligible].copy()
        cells = cells.merge(selected[['cell_id','source_latitude','source_longitude','source_row','source_col','coastal_imputed']],
                            on='cell_id', how='inner', validate='one_to_one')
        source_latitudes = cells.source_latitude
        source_longitudes = cells.source_longitude
    else:
        source_latitudes = cells.latitude
        source_longitudes = cells.longitude
    params = {'var': variable, 'north': float(max(cells.latitude.max(), source_latitudes.max())),
              'south': float(min(cells.latitude.min(), source_latitudes.min())),
              'west': float(min(cells.longitude.min(), source_longitudes.min())),
              'east': float(max(cells.longitude.max(), source_longitudes.max())), 'horizStride': 1,
              'time_start': f'{year}-01-01T12:00:00Z',
              'time_end': f'{year}-12-31T12:00:00Z',
              'accept': 'netcdf4', 'addLatLon': 'true'}
    combined, subsets, raw_checksum, cache_paths = nasa_quarter_subsets(
        url, source, params, model, scenario, variable, year, outdir)
    with combined as dataset:
        assert dataset.attrs.get('version') == '2.0'
        assert dataset.attrs.get('cmip6_source_id') == model
        assert dataset.attrs.get('scenario') == scenario
        lon = (dataset.lon.values + 180) % 360 - 180
        lat = dataset.lat.values
        # Every native centre must match the common grid to1e-8 degrees.
        assert np.allclose((lon + 179.875) / .25, np.round((lon + 179.875) / .25), atol=1e-8)
        assert np.allclose((89.875 - lat) / .25, np.round((89.875 - lat) / .25), atol=1e-8)
        lon_lookup = {round(float(v), 6): i for i, v in enumerate(lon)}
        lat_lookup = {round(float(v), 6): i for i, v in enumerate(lat)}
        ci = np.array([lon_lookup[round(float(v), 6)] for v in source_longitudes])
        ri = np.array([lat_lookup[round(float(v), 6)] for v in source_latitudes])
        values = dataset[variable].values[:, ri, ci]
        original_ci = np.array([lon_lookup[round(float(v), 6)] for v in cells.longitude])
        original_ri = np.array([lat_lookup[round(float(v), 6)] for v in cells.latitude])
        native_missing = ~np.isfinite(dataset[variable].values[:, original_ri, original_ci])
        dates = pd.DatetimeIndex(dataset.time.values).strftime('%Y-%m-%d')
        expected = pd.date_range(f'{year}-01-01', f'{year}-12-31').strftime('%Y-%m-%d')
        if list(dates) != list(expected):
            raise ValueError(f'NASA missing/duplicated dates: {model}/{scenario}/{year}')
        native_attrs = jsonable(dataset.attrs)
        variable_attrs = jsonable(dataset[variable].attrs)
        calendar = dataset.time.encoding.get('calendar')
    frame = pd.DataFrame({'date': np.repeat(dates, len(cells)),
                          'cell_id': np.tile(cells.cell_id.values, len(dates))})
    frame[f'{field}_native_missing'] = native_missing.ravel()
    if alignment is not None and variable == 'tas':
        frame['coastal_imputed'] = np.tile(cells.coastal_imputed.values, len(dates))
        frame['climate_source_row'] = np.tile(cells.source_row.values, len(dates)).astype('int16')
        frame['climate_source_col'] = np.tile(cells.source_col.values, len(dates)).astype('int16')
    raw = values.ravel().astype('float32')
    adjustments = {}
    if variable.startswith('tas'):
        assert variable_attrs.get('units') == 'K'
        frame[field] = raw - np.float32(273.15)
    elif variable == 'pr':
        assert variable_attrs.get('units') in ['kg m-2 s-1', 'kg/m2/s', 'kg m**-2 s**-1']
        converted = raw * np.float32(86400)
        if np.nanmin(converted) < -0.01:
            raise ValueError('NASA precipitation < -0.01mm/day')
        frame[field] = np.clip(converted, 0, None)
        if (converted < 0).any():
            frame['precipitation_mm_raw'] = converted
            adjustments['negative_precipitation_values'] = int((converted < 0).sum())
    elif variable == 'hurs':
        assert variable_attrs.get('units') == '%'
        frame['rh_mean_pct_raw'] = raw
        frame[field] = np.clip(raw, 0, 100)
        adjustments['relative_humidity_outside_0_100_values'] = int(((raw < 0) | (raw > 100)).sum())
    meta = {'model': model, 'scenario': scenario, 'year': year,
            'version': '2.0', 'native_source_path': source,
            'source_catalog': catalog_url, 'source_catalog_sha256': sha256(catalog_path),
            'ncss_query': params, 'native_global_attributes': native_attrs,
            'native_variable_attributes': variable_attrs, 'native_calendar': calendar,
            'target_transform': TRANSFORM, 'resampling': 'None; native 0.25° centres match common grid',
            'registry_path': str(REGISTRY.relative_to(ROOT)), 'registry_sha256': sha256(REGISTRY),
            'coastal_alignment_provenance': str((outdir/'coastal_alignment'/f'{model}.json').relative_to(ROOT)) if alignment is not None else None,
            'native_missing_values': int(native_missing.sum()),
            'unit_conversion': 'K−273.15 for temperature; kg m−2 s−1×86400 for precipitation; relative humidity%',
            'native_file_license': native_attrs.get('cmip6_license'),
            'license_provenance': 'NASA 2023 technical note and EE catalog state blanket CC0 from September 2022; native v2 attributes and NASA v2 technical note retain original CMIP6 terms. Both source statements preserved; no automatic override of native file license.',
            'license_sources': ['https://www.nccs.nasa.gov/sites/default/files/NEX-GDDP-CMIP6-Tech_Note.pdf',
                                'https://www.nccs.nasa.gov/sites/default/files/NEX-GDDP-CMIP6-v2-Tech_Note.pdf',
                                'https://www.nccs.nasa.gov/wp-content/uploads/2025/06/NEX-GDDP-CMIP6-v2-Tech_Note.pdf'],
            'provider_technical_note': 'https://www.nccs.nasa.gov/wp-content/uploads/2025/06/NEX-GDDP-CMIP6-v2-Tech_Note.pdf',
            'provider_bias_correction_reference': 'GMFD / Princeton Global Forcing daily observational reference, 1960–2014; not ERA5',
            'download_sha256': raw_checksum, 'download_sha256_definition': 'SHA256 of chronological NCSS quarter-file bytes concatenated',
            'download_bytes': sum(part['bytes'] for part in subsets), 'quarter_subsets': subsets,
            'raw_subset_retention': 'Resumable local quarter cache deleted after annual selected-cell Parquet and complete native metadata are committed',
            'numeric_adjustments': adjustments}
    result = commit_frame(path, frame, meta, [field])
    for cached_path in cache_paths:
        cached_path.unlink(missing_ok=True)
    return result | {'status': 'downloaded', 'job': f'{model}/{scenario}/{variable}/{year}'}


def year_plan():
    result = []
    for model in MODELS:
        result.extend((model, 'historical', year) for year in range(1990, 2015))
        for scenario in SCENARIOS:
            years = list(range(2015, 2021)) + list(range(2030, 2061)) + list(range(2070, 2101))
            result.extend((model, scenario, year) for year in years)
    return result


def merge_nasa_year(model, scenario, year, fields, outdir):
    destination = outdir / 'daily' / model / scenario / f'{year}.parquet'
    if done(destination, fields):
        return True
    frames = []
    for variable, field in NASA_FIELDS.items():
        if field not in fields:
            continue
        path = outdir / 'variables' / model / scenario / variable / f'{year}.parquet'
        if not done(path, [field]):
            return False
        frames.append(pd.read_parquet(path))
    frame = frames[0]
    for other in frames[1:]:
        frame = frame.merge(other, on=['date', 'cell_id'], how='outer', validate='one_to_one')
    meta = {'model': model, 'scenario': scenario, 'year': year, 'version': '2.0',
            'fields': fields, 'registry_sha256': sha256(REGISTRY),
            'source_variable_provenance': [str((outdir/'variables'/model/scenario/v/f'{year}.json').relative_to(ROOT))
                                           for v, f in NASA_FIELDS.items() if f in fields],
            'baseline_convention': 'Historical 1990–2014 plus scenario-specific 2015–2020; harvest 1991–2020',
            'future_convention': 'Harvest 2031–2060 and 2071–2100 plus antecedent 2030/2070'}
    commit_frame(destination, frame, meta, fields)
    return True


def update_coverage(source, outdir, expected_jobs):
    manifests = [json.loads(p.read_text()) for p in outdir.rglob('*.json')]
    records = [m for m in manifests if 'quality' in m]
    total = sum(m['parquet_bytes'] for m in records)
    summary = {'updated_utc': utc(), 'source': source, 'expected_jobs': expected_jobs,
               'completed_files': len(records), 'parquet_bytes': total,
               'available_disk_bytes': shutil.disk_usage(ROOT).free,
               'actual_files': [{k: m.get(k) for k in ['model','scenario','year','window_start',
                                'window_end_exclusive','parquet_bytes','quality']} for m in records],
               'coverage_status': 'Complete' if len(records) == expected_jobs else 'Incomplete'}
    write_json(OUT / f'{source}_coverage.json', summary)
    return summary


def run_jobs(jobs, fn, workers, source, outdir):
    errors = []
    completed = 0
    def retry(job):
        for attempt in range(3):
            try:
                return fn(job)
            except (requests.RequestException, ee.EEException) as exc:
                if attempt == 2:
                    raise
                report({'source': source, 'status': 'retry', 'job': list(job),
                        'attempt': attempt+1, 'error': safe_error(exc)})
                time.sleep(3 * (attempt+1))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        future_jobs = {pool.submit(retry, job): job for job in jobs}
        for future in as_completed(future_jobs):
            job = future_jobs[future]
            try:
                result = future.result()
                completed += 1
                report({'source': source, 'completed': completed, 'total': len(jobs),
                        'status': result.get('status'), 'job': result.get('job'),
                        'bytes': result.get('parquet_bytes'), 'quality': result.get('quality')})
            except Exception as exc:
                error = {'job': list(job), 'error': safe_error(exc), 'utc': utc()}
                errors.append(error)
                report({'source': source, 'status': 'failed', **error})
            if (completed + len(errors)) % 10 == 0:
                update_coverage(source, outdir, len(jobs))
    write_json(OUT / f'{source}_errors.json', errors)
    update_coverage(source, outdir, len(jobs))
    return len(errors)


def main():
    global REGISTRY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', choices=['era5', 'nasa', 'coverage'])
    parser.add_argument('--registry', type=Path, default=REGISTRY)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--start', default='1990-01-01')
    parser.add_argument('--end', default='2021-01-01', help='Exclusive ERA5 end')
    parser.add_argument('--fields', nargs='+', choices=FIELD_CHOICES, default=FIELDS)
    parser.add_argument('--models', nargs='+', default=MODELS)
    parser.add_argument('--scenarios', nargs='+', default=SCENARIOS + ['historical'])
    parser.add_argument('--years', nargs='+', type=int)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    REGISTRY = args.registry.resolve()
    OUT.mkdir(parents=True, exist_ok=True); DATA.mkdir(parents=True, exist_ok=True)
    cells = registry(args.registry)
    if args.source == 'era5':
        ee.Initialize(project=PROJECT); ee.data.setDeadline(420000)
        outdir = DATA / 'era5_daily'
        dates = date_range(args.start, args.end)
        # Monthtiles are shortened automatically to keep worst-case raw raster<29MB.
        pixels = int(round((cells.east.max()-cells.west.min())/.25)) * int(round((cells.north.max()-cells.south.min())/.25))
        days_per_tile = max(1, min(31, 29_000_000 // (pixels * len(args.fields) * 4)))
        jobs = []
        for _, monthly in pd.Series(dates, index=dates).groupby(dates.to_period('M')):
            for offset in range(0, len(monthly), days_per_tile):
                part = monthly.iloc[offset:offset+days_per_tile]
                jobs.append((part.iloc[0].strftime('%Y-%m-%d'), (part.iloc[-1]+pd.Timedelta(days=1)).strftime('%Y-%m-%d')))
        if args.limit:
            jobs = jobs[:args.limit]
        errors = run_jobs(jobs, lambda job: era5_job(*job, cells, args.fields, outdir), args.workers, 'era5', outdir)
    elif args.source == 'nasa':
        outdir = DATA / 'nasa_v2'
        years = [j for j in year_plan() if j[0] in args.models and j[1] in args.scenarios and (args.years is None or j[2] in args.years)]
        # Verify consecutive years in both future windows first, then finish
        # all historical/scenario baseline years needed for forcing alignment.
        priority_years = [1990, 2030, 1991, 2031, 2070, 2071]
        priority = {year: i for i, year in enumerate(priority_years)}
        years.sort(key=lambda job: (priority.get(job[2], len(priority)+(job[2] > 2020)), job[2], job[0], job[1]))
        variables = [v for v, f in NASA_FIELDS.items() if f in args.fields]
        alignments = {model: prepare_nasa_alignment(model, cells, variables, outdir) for model in args.models}
        jobs = [(m, s, v, y) for m, s, y in years for v in variables]
        if args.limit:
            jobs = jobs[:args.limit]
        def retrieve(job):
            result = nasa_variable(*job, cells, outdir, alignment=alignments[job[0]])
            with MERGE_LOCK:
                merge_nasa_year(job[0], job[1], job[3], args.fields, outdir)
            return result
        errors = run_jobs(jobs, retrieve, args.workers, 'nasa', outdir/'variables')
        for model, scenario, year in years:
            merge_nasa_year(model, scenario, year, args.fields, outdir)
        update_coverage('nasa_daily', outdir/'daily', len(years))
    else:
        update_coverage('era5', DATA/'era5_daily', 372)
        update_coverage('nasa', DATA/'nasa_v2/variables', len(year_plan())*len(args.fields))
        update_coverage('nasa_daily', DATA/'nasa_v2/daily', len(year_plan()))
        errors = 0
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
