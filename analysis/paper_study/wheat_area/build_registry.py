"""Conserve source wheat hectares on an edge-aligned 0.25-degree grid."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from pyproj import Geod
import shapely
from shapely.geometry import box
import xarray as xr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / 'data/paper_study/wheat_area'
GEOD = Geod(ellps='WGS84')
TRANS = ['Russian Federation', 'Turkey', 'Kazakhstan', 'Georgia', 'Azerbaijan']
NAMES = {'Russia': 'Russian Federation', 'United Kingdom': 'United Kingdom of Great Britain and Northern Ireland',
         'Moldova': 'Republic of Moldova', 'Republic of Serbia': 'Serbia'}
BANDS = {'harvested_total_ha': ('H', 'A'), 'harvested_irrigated_ha': ('H', 'I'),
         'harvested_rainfed_ha': ('H', 'R'), 'physical_total_ha': ('A', 'A'),
         'physical_irrigated_ha': ('A', 'I'), 'physical_rainfed_ha': ('A', 'R')}


def area(geom):
    return abs(GEOD.geometry_area_perimeter(geom)[0]) / 10000


def cell_area(north, size):
    return abs(GEOD.polygon_area_perimeter([0, size, size, 0],
                   [north, north, north-size, north-size])[0]) / 10000


def run():
    HERE.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_parquet(DATA / 'spam2020_wheat/source_wheat_country_metadata.parquet')
    assert not metadata.grid_code.duplicated().any()
    metadata['source_row'] = metadata.grid_code // 4320
    metadata['source_col'] = metadata.grid_code % 4320
    # grid_code is independently verified as zero-based row-major source-pixel identity.
    assert np.max(abs(metadata.x - (-180 + (metadata.source_col+.5)/12))) < .00051
    assert np.max(abs(metadata.y - (90 - (metadata.source_row+.5)/12))) < .00051
    countries = gpd.read_file(ROOT / 'data/geography/ne_110m_admin_0_countries.zip')
    physical = gpd.read_file(DATA / 'ne_50m_geography_regions_polys.zip')
    continent = physical.loc[physical.NAME.eq('EUROPE')].geometry.iloc[0]
    continent = shapely.make_valid(continent)
    ordinary = {NAMES.get(name, name) for name in countries.loc[countries.CONTINENT.eq('Europe'), 'ADMIN']}
    ordinary -= set(TRANS)
    ordinary |= {'Malta', 'Andorra', 'Liechtenstein', 'Monaco', 'San Marino', 'Holy See'}
    selected_names = ordinary | set(TRANS)
    source = metadata.loc[metadata.ADM0_NAME.isin(selected_names) & metadata.whea.gt(0) &
                          metadata.x.between(-35, 75) & metadata.y.between(30, 90)].copy()
    source['west'] = -180 + source.source_col/12
    source['east'] = source.west + 1/12
    source['north'] = 90 - source.source_row/12
    source['south'] = source.north - 1/12
    source['europe_fraction'] = 1.
    source['mask_rule'] = 'source_country_membership'
    split = source.ADM0_NAME.isin(TRANS)
    pixels = shapely.box(source.loc[split, 'west'].to_numpy(), source.loc[split, 'south'].to_numpy(),
                         source.loc[split, 'east'].to_numpy(), source.loc[split, 'north'].to_numpy())
    covered = shapely.covered_by(pixels, continent)
    intersects = shapely.intersects(pixels, continent)
    fraction = np.zeros(len(pixels))
    fraction[covered] = 1.
    boundary = np.flatnonzero(intersects & ~covered)
    for i in boundary:
        fraction[i] = np.clip(area(shapely.intersection(pixels[i], continent)) / area(pixels[i]), 0, 1)
    source.loc[split, 'europe_fraction'] = fraction
    source.loc[split, 'mask_rule'] = 'Natural_Earth_physical_Europe_polygon_overlap'
    excluded_trans_area_csv = float((source.loc[split, 'whea']*(1-source.loc[split, 'europe_fraction'])).sum())
    source = source.loc[source.europe_fraction.gt(0)].reset_index(drop=True)
    arrays, valid = {}, {}
    pixel_area = {int(row): cell_area(90-row/12, 1/12) for row in source.source_row.unique()}
    source['source_pixel_geometric_area_ha'] = source.source_row.map(pixel_area)
    max_csv_difference = 0.
    for name, (variable, technology) in BANDS.items():
        path = DATA / f'spam2020_wheat/spam2020_V2r2_global_{variable}_WHEA_{technology}.tif'
        with rasterio.open(path) as raster:
            assert raster.crs.to_epsg() == 4326 and raster.shape == (2160, 4320)
            np.testing.assert_allclose(tuple(raster.transform)[:6], [1/12,0,-180,0,-1/12,90], atol=1e-12)
            array = raster.read(1)
            mask = raster.read_masks(1).astype(bool)
            values = array[source.source_row.to_numpy(int), source.source_col.to_numpy(int)].astype(float)
            pixel_valid = mask[source.source_row.to_numpy(int), source.source_col.to_numpy(int)]
            source[f'{name}_source_raster_valid'] = pixel_valid
            if technology == 'A':
                assert pixel_valid.all()
            else:
                # Technology rasters omit whole pixels where that production system is absent.
                # Preserve raw validity and check the A = I + R identity before accepting zeros.
                values = np.where(pixel_valid, values, 0.)
            assert (values >= 0).all()
            source[name] = values
            arrays[name], valid[name] = array, mask
            if name == 'harvested_total_ha':
                max_csv_difference = float(abs(values-source.whea).max())
                assert max_csv_difference < .002
    physical_excess = source.physical_total_ha - source.source_pixel_geometric_area_ha
    technology_identity = {}
    for prefix in ['harvested', 'physical']:
        difference = source[f'{prefix}_total_ha'] - source[f'{prefix}_irrigated_ha'] - source[f'{prefix}_rainfed_ha']
        missing = ~source[f'{prefix}_irrigated_ha_source_raster_valid'] | ~source[f'{prefix}_rainfed_ha_source_raster_valid']
        assert not ((~source[f'{prefix}_irrigated_ha_source_raster_valid']) & (~source[f'{prefix}_rainfed_ha_source_raster_valid']) & source[f'{prefix}_total_ha'].gt(0)).any()
        assert difference.loc[missing].abs().max() < .11
        technology_identity[prefix] = dict(maximum_absolute_residual_ha=float(difference.abs().max()),
            maximum_absolute_residual_when_a_technology_is_absent_ha=float(difference.loc[missing].abs().max()),
            total_residual_ha=float(difference.sum()),structural_zero_pixels=int(missing.sum()))
    source['physical_area_exceeds_geometric_pixel'] = physical_excess.gt(.01)
    source['row'], source['col'] = source.source_row//3, source.source_col//3
    weighted = source.copy()
    for name in BANDS:
        weighted[name] *= weighted.europe_fraction
    grouped = weighted.groupby(['row', 'col'])[list(BANDS)].sum()
    registry = grouped.reset_index()
    registry['cell_id'] = [f'g025_r{r:03d}_c{c:04d}' for r,c in zip(registry.row,registry.col)]
    registry['west'], registry['north'] = -180+registry.col*.25, 90-registry.row*.25
    registry['east'], registry['south'] = registry.west+.25, registry.north-.25
    registry['longitude'], registry['latitude'] = registry.west+.125, registry.north-.125
    areas = {int(r): cell_area(90-r*.25,.25) for r in registry.row.unique()}
    registry['grid_cell_geometric_area_ha'] = registry.row.map(areas)
    registry['wheat_area_weight'] = registry.harvested_total_ha / registry.harvested_total_ha.sum()
    registry['physical_wheat_area_weight'] = registry.physical_total_ha / registry.physical_total_ha.sum()
    registry['crop_season'] = 'unallocated_wheat'
    registry['area_reference_year'] = 2020
    registry['source_crop'] = 'WHEA_all_wheat'
    for key, output in [(valid['harvested_total_ha'], 'source_valid_pixels'),
                        (valid['harvested_total_ha'] & (arrays['harvested_total_ha'] == 0), 'source_zero_wheat_pixels')]:
        count = key.reshape(720,3,1440,3).sum(axis=(1,3))
        registry[output] = count[registry.row,registry.col]
    registry['source_nodata_pixels'] = 9 - registry.source_valid_pixels
    country_cells = weighted.groupby(['row','col','ADM0_NAME','FIPS0'])[list(BANDS)].sum().reset_index()
    country_cells['cell_id'] = [f'g025_r{r:03d}_c{c:04d}' for r,c in zip(country_cells.row,country_cells.col)]
    dominant = country_cells.sort_values('harvested_total_ha').groupby(['row','col']).tail(1)
    registry = registry.merge(dominant[['row','col','ADM0_NAME','FIPS0']].rename(columns={
        'ADM0_NAME':'dominant_source_country','FIPS0':'dominant_source_country_fips0'}), on=['row','col'],validate='one_to_one')
    countries_summary = weighted.groupby(['ADM0_NAME','FIPS0'])[list(BANDS)].sum().reset_index()
    source.to_parquet(DATA / 'europe_source_wheat_pixels.parquet',index=False)
    registry.to_parquet(DATA / 'europe_wheat_cells_025.parquet',index=False)
    registry.to_csv(DATA / 'europe_wheat_cells_025.csv',index=False)
    country_cells.to_parquet(DATA / 'europe_cell_country_wheat_areas.parquet',index=False)
    countries_summary.to_csv(DATA / 'europe_country_wheat_area.csv',index=False)
    calendar_frames, calendar_metadata = [], []
    for crop, season in [('wwh','winter_wheat'),('swh','spring_wheat')]:
        for system, technology in [('rf','rainfed'),('ir','irrigated')]:
            path = DATA / f'{crop}_{system}_ggcmi_crop_calendar_phase3_v1.01.nc4'
            with xr.open_dataset(path) as dataset:
                np.testing.assert_allclose(dataset.lat,89.75-np.arange(360)*.5)
                np.testing.assert_allclose(dataset.lon,-179.75+np.arange(720)*.5)
                cal = registry[['cell_id','row','col']].copy()
                rr,cc = registry.row.to_numpy()//2,registry.col.to_numpy()//2
                for old,new in [('planting_day','planting_doy'),('maturity_day','maturity_doy'),
                                ('growing_season_length','growing_season_length_days'),('data_source_used','provider_source_index')]:
                    cal[new] = dataset[old].to_numpy()[rr,cc]
                cal['crop_season'],cal['water_system'] = season,technology
                cal['calendar_valid'] = cal.planting_doy.between(1,366)&cal.maturity_doy.between(1,366)
                cal['calendar_resolution_deg'] = .5
                cal['calendar_terminal_stage'] = 'provider_maturity_not_observed_harvest'
                cal['winter_spring_area_allocation'] = 'unavailable_from_SPAM'
                cal['source_file'] = path.name
                calendar_frames.append(cal)
                calendar_metadata.append(dict(file=path.name,attributes=dict(dataset.attrs),
                    variables={n:dict(dataset[n].attrs) for n in dataset.data_vars},
                    valid_calendar_cells=int(cal.calendar_valid.sum()),registry_cells=len(cal)))
    calendars = pd.concat(calendar_frames,ignore_index=True)
    calendars.to_parquet(DATA / 'europe_wheat_calendar_scenarios.parquet',index=False)
    calendars.to_csv(DATA / 'europe_wheat_calendar_scenarios.csv.gz',index=False)
    (HERE / 'calendar_metadata.json').write_text(json.dumps(calendar_metadata,indent=2)+'\n')
    # Map geometry is supplementary; source administrative ownership drives ordinary-country area inclusion.
    visual = []
    for record in countries.itertuples():
        name=NAMES.get(record.ADMIN,record.ADMIN)
        if name in ordinary:
            geometry=record.geometry.intersection(box(-35,30,75,90))
        elif name in TRANS:
            geometry=record.geometry.intersection(continent)
        else:continue
        if not geometry.is_empty: visual.append(dict(country=name,geometry=geometry))
    gpd.GeoDataFrame(visual,crs='EPSG:4326').to_file(DATA/'europe_operational_domain.geojson',driver='GeoJSON')
    scope=dict(grid_crs='EPSG:4326',grid_transform=[.25,0,-180,0,-.25,90],cell_id='g025_r{row:03d}_c{col:04d}',
        row_origin='north',col_origin='west',source_grid_resolution_deg=1/12,source_grid_transform=[1/12,0,-180,0,-1/12,90],
        area_aggregation='Exact aligned 3x3 native pixel footprint sums; transcontinental pixels weighted by ellipsoidal polygon-overlap fraction',
        source_country_ownership='SPAM CSV ADM0_NAME and FIPS0; FIPS0 is not ISO',
        core_country_rule='Natural Earth CONTINENT=Europe except Russia, supplemented by European microstates present in source',
        transcontinental_countries=TRANS,transcontinental_rule='Overlap with Natural Earth 50m physical EUROPE label polygon',
        country_names=countries_summary.ADM0_NAME.tolist(),source_country_extent_filter=[-35,30,75,90],
        cyprus='Excluded from primary geographical domain; political European extension requires separate declared weighting',
        arctic='No72N cutoff; included if source has positive wheat area',
        boundaries='Cartographic approximation, not a surveyed continental divide; physical-region labels have provider accuracy caveats',
        small_states='Source ownership preserves positive pixels even where110m country geometry omits a state; absent source statistics remain unavailable',
        coastal_area_rule='Ordinary-country native crop hectares retained using source country membership, without coastline centroid clipping',
        terminal_calendar='GGCMI maturity, not field-observed harvest',season_area_weights='Winter/spring hectares unresolved; scenarios must not be added as area partitions')
    (HERE / 'grid_and_scope.json').write_text(json.dumps(scope,indent=2)+'\n')
    totals={name:dict(included_source_ha=float((source[name]*source.europe_fraction).sum()),aggregated_ha=float(registry[name].sum()),
        absolute_difference_ha=float(abs((source[name]*source.europe_fraction).sum()-registry[name].sum()))) for name in BANDS}
    assert all(v['absolute_difference_ha']<1e-6 for v in totals.values())
    receipt=dict(status='source_area_conserved',completed_utc=datetime.now(timezone.utc).isoformat(),
        registry_cells=len(registry),included_positive_source_pixels=len(source),countries_with_positive_wheat=len(countries_summary),
        totals=totals,source_csv_raster_max_difference_ha=max_csv_difference,
        partially_clipped_positive_source_pixels=int(source.europe_fraction.between(0,1,inclusive='neither').sum()),
        transcontinental_excluded_harvested_area_within_search_extent_ha=excluded_trans_area_csv,
        source_physical_area_exceeds_geometric_pixel_count=int(source.physical_area_exceeds_geometric_pixel.sum()),
        maximum_source_physical_area_excess_ha=float(physical_excess.max()),
        area_weight_sum=float(registry.wheat_area_weight.sum()),season_allocated=False,
        source_raster_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (DATA/'spam2020_wheat').glob('*.tif')})
    receipt['technology_area_identity'] = technology_identity
    (HERE/'aggregation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],cells=len(registry),source_pixels=len(source),countries=len(countries_summary),
        harvested_ha=registry.harvested_total_ha.sum(),physical_ha=registry.physical_total_ha.sum()),indent=2))


if __name__=='__main__':run()
