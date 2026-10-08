"""Independently audit archived source bytes, native membership, area sums and calendars."""
from pathlib import Path
import hashlib
import json
import zipfile
from datetime import datetime, timezone

import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Geod
import rasterio
import shapely
import xarray as xr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/wheat_area'
GEOD=Geod(ellps='WGS84')
RULES={'harvested_total_ha':('H','A'),'harvested_irrigated_ha':('H','I'),
       'harvested_rainfed_ha':('H','R'),'physical_total_ha':('A','A'),
       'physical_irrigated_ha':('A','I'),'physical_rainfed_ha':('A','R')}
TRANS={'Russian Federation','Turkey','Kazakhstan','Georgia','Azerbaijan'}
ALIASES={'Russia':'Russian Federation','United Kingdom':'United Kingdom of Great Britain and Northern Ireland',
         'Moldova':'Republic of Moldova','Republic of Serbia':'Serbia'}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def geod_area(geometry):return abs(GEOD.geometry_area_perimeter(geometry)[0])/10000


def run():
    checks=[]
    def check(name,condition):
        if not bool(condition):raise AssertionError(name)
        checks.append(name)
    manifest=json.loads((DATA/'download_manifest.json').read_text())
    archive_checks=[]
    for record in manifest:
        if record.get('checksum_verified'):
            path=DATA/record['file'];content=path.read_bytes()
            check(f"archive_size_{path.name}",len(content)==record['bytes'])
            check(f"archive_md5_{path.name}",hashlib.md5(content).hexdigest()==record['md5'])
            check(f"archive_sha256_{path.name}",hashlib.sha256(content).hexdigest()==record['sha256'])
            archive_checks.append(path.name)
    for kind in ['harvested','physical']:
        archive=DATA/f'spam2020V2r2_global_{kind}_area.geotiff.zip'
        with zipfile.ZipFile(archive) as z:
            for name in z.namelist():
                if '_WHEA_' in name and name.endswith('.tif'):
                    check(f'byte_identical_extracted_{Path(name).name}',hashlib.sha256(z.read(name)).hexdigest()==sha(DATA/'spam2020_wheat'/Path(name).name))
    registry=pd.read_parquet(DATA/'europe_wheat_cells_025.parquet')
    csv_registry=pd.read_csv(DATA/'europe_wheat_cells_025.csv')
    check('unique_target_cell_ids',not registry.cell_id.duplicated().any())
    check('csv_parquet_identity',np.allclose(csv_registry[list(RULES)],registry[list(RULES)],rtol=1e-14,atol=1e-8))
    check('positive_total_wheat',registry.harvested_total_ha.gt(0).all())
    check('target_row_col_bounds',registry.row.between(0,719).all() and registry.col.between(0,1439).all())
    check('grid_cell_id_identity',all(i==f'g025_r{r:03d}_c{c:04d}' for i,r,c in zip(registry.cell_id,registry.row,registry.col)))
    check('target_cell_centers',np.allclose(registry.longitude,-179.875+registry.col*.25) and np.allclose(registry.latitude,89.875-registry.row*.25))
    check('target_edges',np.allclose(registry.west,-180+registry.col*.25) and np.allclose(registry.north,90-registry.row*.25) and np.allclose(registry.east-registry.west,.25) and np.allclose(registry.north-registry.south,.25))
    metadata=pd.read_parquet(DATA/'spam2020_wheat/source_wheat_country_metadata.parquet')
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    ordinary={ALIASES.get(x,x) for x in countries.loc[countries.CONTINENT.eq('Europe'),'ADMIN']} - TRANS
    ordinary|={'Malta','Andorra','Liechtenstein','Monaco','San Marino','Holy See'}
    candidates=metadata.loc[metadata.ADM0_NAME.isin(ordinary|TRANS)&metadata.whea.gt(0)&metadata.x.between(-35,75)&metadata.y.between(30,90)].copy()
    candidates['rr']=(candidates.grid_code//4320).astype(int)
    candidates['cc']=(candidates.grid_code%4320).astype(int)
    check('native_grid_code_coordinates',np.max(abs(candidates.x-(-180+(candidates.cc+.5)/12)))<.00051 and np.max(abs(candidates.y-(90-(candidates.rr+.5)/12)))<.00051)
    physical=gpd.read_file(DATA/'ne_50m_geography_regions_polys.zip')
    europe=shapely.make_valid(physical.loc[physical.NAME.eq('EUROPE')].geometry.iloc[0])
    polygons=shapely.box(-180+candidates.cc.to_numpy()/12,90-(candidates.rr.to_numpy()+1)/12,
                         -180+(candidates.cc.to_numpy()+1)/12,90-candidates.rr.to_numpy()/12)
    fraction=np.ones(len(candidates));center_rule=np.ones(len(candidates));intersection_rule=np.ones(len(candidates))
    for i in np.flatnonzero(candidates.ADM0_NAME.isin(TRANS).to_numpy()):
        polygon=polygons[i]
        if shapely.covered_by(polygon,europe):fraction[i]=1.
        elif not shapely.intersects(polygon,europe):fraction[i]=0.
        else:fraction[i]=np.clip(geod_area(polygon.intersection(europe))/geod_area(polygon),0.,1.)
        center_rule[i]=shapely.covers(europe,shapely.Point(-180+(candidates.cc.iloc[i]+.5)/12,90-(candidates.rr.iloc[i]+.5)/12))
        intersection_rule[i]=shapely.intersects(polygon,europe)
    candidates['fraction']=fraction
    source=pd.read_parquet(DATA/'europe_source_wheat_pixels.parquet')
    expected=candidates.loc[candidates.fraction.gt(0)].sort_values('grid_code')
    source=source.sort_values('grid_code')
    check('all_and_only_declared_positive_source_pixels',np.array_equal(expected.grid_code.to_numpy(),source.grid_code.to_numpy()))
    fraction_difference=float(np.max(abs(expected.fraction.to_numpy()-source.europe_fraction.to_numpy())))
    check('independent_ellipsoidal_overlap_fraction',fraction_difference<1e-8)
    weights=source.europe_fraction.to_numpy();rr=(source.grid_code//4320).to_numpy(int);cc=(source.grid_code%4320).to_numpy(int)
    cells=(rr//3)*1440+(cc//3);target=(registry.row*1440+registry.col).to_numpy(int)
    check('exact_native_3_by_3_target_membership',np.array_equal(np.unique(cells),np.sort(target)))
    totals={};raw_values={};raw_valid={};source_difference={}
    for name,(kind,system) in RULES.items():
        path=DATA/f'spam2020_wheat/spam2020_V2r2_global_{kind}_WHEA_{system}.tif'
        with rasterio.open(path) as ds:
            check(f'geographic_native_grid_{name}',ds.crs.to_epsg()==4326 and ds.shape==(2160,4320) and np.allclose(tuple(ds.transform)[:6],[1/12,0,-180,0,-1/12,90],atol=1e-12))
            raw=ds.read(1);valid=ds.read_masks(1)>0
            values=raw[rr,cc].astype(float);ok=valid[rr,cc]
            check(f'preserved_source_nodata_mask_{name}',np.array_equal(ok,source[f'{name}_source_raster_valid']))
            if system=='A':check(f'total_pixels_available_{name}',ok.all())
            else:values=np.where(ok,values,0.)
            check(f'archived_source_values_{name}',np.array_equal(values,source[name].to_numpy()))
            raw_values[name]=values;raw_valid[name]=ok
            derived=np.bincount(cells,weights=values*weights,minlength=720*1440)[target]
            max_difference=float(np.max(abs(derived-registry[name].to_numpy())))
            check(f'independent_bincount_area_conservation_{name}',max_difference<1e-7)
            total=float(np.sum(values*weights));totals[name]=dict(included_native_ha=total,registry_ha=float(registry[name].sum()),max_cell_difference_ha=max_difference,total_difference_ha=float(abs(total-registry[name].sum())))
            if name=='harvested_total_ha':
                count=valid.astype(np.uint8).reshape(720,3,1440,3).sum(axis=(1,3))
                zero=(valid&(raw==0)).astype(np.uint8).reshape(720,3,1440,3).sum(axis=(1,3))
                check('valid_zero_nodata_distinction',np.array_equal(registry.source_valid_pixels,count[registry.row,registry.col]) and np.array_equal(registry.source_zero_wheat_pixels,zero[registry.row,registry.col]) and np.array_equal(registry.source_nodata_pixels,9-count[registry.row,registry.col]))
            if kind=='H':
                tech={'A':'TA','I':'TI','R':'TR'}[system]
                tab=pd.read_parquet(DATA/f'spam2020_wheat/spam2020V2r2_global_H_{tech}_wheat.parquet').set_index('grid_code').whea
                csv_values=tab.reindex(source.grid_code).to_numpy()
                mask_difference=(~np.isnan(csv_values))!=ok
                check(f'source_csv_raster_zero_encoding_{name}',((np.nan_to_num(csv_values[mask_difference])==0)&(values[mask_difference]==0)).all())
                check(f'source_csv_raster_values_{name}',np.allclose(np.nan_to_num(csv_values),values,rtol=0,atol=1e-9))
                source_difference[name]=dict(maximum_absolute_value_difference_ha=float(np.nanmax(abs(csv_values-values))),
                    CSV_raster_validity_disagreements_with_zero_value=int(mask_difference.sum()))
    technology={}
    for prefix in ['harvested','physical']:
        missing=~raw_valid[f'{prefix}_irrigated_ha']|~raw_valid[f'{prefix}_rainfed_ha']
        delta=raw_values[f'{prefix}_total_ha']-raw_values[f'{prefix}_irrigated_ha']-raw_values[f'{prefix}_rainfed_ha']
        check(f'structural_technology_zero_identity_{prefix}',np.max(abs(delta[missing]))==0)
        technology[prefix]=dict(max_native_residual_ha=float(abs(delta).max()),weighted_total_residual_ha=float(np.sum(delta*weights)),technology_nodata_structural_zero_count=int(missing.sum()))
    check('source_pixel_physical_area_within_geometric_area',(source.physical_total_ha<=source.source_pixel_geometric_area_ha+.01).all())
    check('unallocated_season_unique',registry.crop_season.eq('unallocated_wheat').all())
    check('harvested_area_weights',abs(registry.wheat_area_weight.sum()-1)<1e-12 and np.allclose(registry.wheat_area_weight,registry.harvested_total_ha/registry.harvested_total_ha.sum()))
    country_cells=pd.read_parquet(DATA/'europe_cell_country_wheat_areas.parquet')
    check('country_area_partition',np.allclose(country_cells.groupby('cell_id')[list(RULES)].sum().reindex(registry.cell_id).to_numpy(),registry[list(RULES)].to_numpy(),rtol=1e-14,atol=1e-8))
    scenarios=pd.read_parquet(DATA/'europe_wheat_calendar_scenarios.parquet');calendar_summary=[]
    check('one_each_of_four_calendar_scenarios',len(scenarios)==len(registry)*4 and not scenarios.duplicated(['cell_id','crop_season','water_system']).any())
    check('calendar_table_has_no_repeated_area_weights',not any(x.endswith('_ha') or 'area_weight' in x for x in scenarios.columns))
    for file,frame in scenarios.groupby('source_file'):
        frame=frame.set_index('cell_id').reindex(registry.cell_id)
        with xr.open_dataset(DATA/file) as ds:
            for old,new in [('planting_day','planting_doy'),('maturity_day','maturity_doy'),('growing_season_length','growing_season_length_days'),('data_source_used','provider_source_index')]:
                # Find containing 0.5-degree cell from verified target centers, independently of integer r//2.
                row=np.floor((90-registry.latitude.to_numpy())/.5).astype(int);col=np.floor((registry.longitude.to_numpy()+180)/.5).astype(int)
                expected=ds[old].to_numpy()[row,col]
                check(f'native_calendar_value_{file}_{new}',np.array_equal(expected,frame[new].to_numpy(),equal_nan=True))
            valid=frame.planting_doy.between(1,366)&frame.maturity_doy.between(1,366)
            check(f'calendar_missing_mask_{file}',np.array_equal(valid,frame.calendar_valid))
            missing_area=float(registry.loc[~valid.to_numpy(),'harvested_total_ha'].sum())
            length_residual=((frame.maturity_doy-frame.planting_doy)%365-frame.growing_season_length_days).loc[valid]
            calendar_summary.append(dict(file=file,valid_cells=int(valid.sum()),missing_cells=int((~valid).sum()),missing_wheat_area_ha=missing_area,
                planting_doy_range=[float(frame.loc[valid,'planting_doy'].min()),float(frame.loc[valid,'planting_doy'].max())],
                source_index_counts=frame.provider_source_index.value_counts(dropna=False).to_dict(),
                maximum_calendar_length_residual_days=float(length_residual.abs().max())))
    point_scenarios=pd.read_parquet(DATA/'trial_point_calendar_scenarios.parquet')
    point_csv=pd.read_csv(DATA/'trial_point_calendar_scenarios.csv')
    check('point_csv_parquet_values',np.allclose(point_csv[['planting_doy','maturity_doy']],point_scenarios[['planting_doy','maturity_doy']],equal_nan=True))
    check('point_unique_scenario_identity',not point_scenarios.duplicated(['dataset_id','point_id','crop_season','water_system']).any())
    check('point_calendar_is_scenario_not_observed_sowing',point_scenarios.sowing_date_basis.eq('scenario_planting_calendar_not_observed_trial_sowing').all())
    check('harvest_date_unavailable',point_scenarios.harvest_doy.isna().all() and (~point_scenarios.harvest_date_available).all())
    for file,frame in point_scenarios.groupby('source_file'):
        row=np.floor((90-frame.latitude.to_numpy())*2).astype(int);col=np.floor((frame.longitude.to_numpy()+180)*2).astype(int)
        check(f'point_containing_cell_{file}',np.array_equal(row,frame.native_calendar_row) and np.array_equal(col,frame.native_calendar_col))
        with xr.open_dataset(DATA/file) as ds:
            check(f'point_source_values_{file}',np.array_equal(ds.planting_day.to_numpy()[row,col],frame.planting_doy.to_numpy(),equal_nan=True) and np.array_equal(ds.maturity_day.to_numpy()[row,col],frame.maturity_doy.to_numpy(),equal_nan=True))
    check('point_missing_flags',np.array_equal(point_scenarios.native_missing_mask,~(point_scenarios.planting_doy.between(1,366)&point_scenarios.maturity_doy.between(1,366))))
    sensitivity=[]
    for country,group in candidates.assign(center_fraction=center_rule,intersection_fraction=intersection_rule).groupby('ADM0_NAME'):
        sensitivity.append(dict(country=country,positive_source_harvested_in_search_extent_ha=float(group.whea.sum()),
            primary_overlap_harvested_ha=float((group.whea*group.fraction).sum()),
            native_center_alternative_harvested_ha=float((group.whea*group.center_fraction).sum()),
            intersecting_pixel_upper_harvested_ha=float((group.whea*group.intersection_fraction).sum())))
    pd.DataFrame(sensitivity).to_csv(HERE/'boundary_sensitivity.csv',index=False)
    cyprus=metadata.loc[metadata.ADM0_NAME.eq('Cyprus')&metadata.whea.gt(0)]
    receipt=dict(status='all_checks_passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks_passed=len(checks),checks=checks,
        registry_cells=len(registry),native_positive_pixels=len(source),totals=totals,technology_source_identity=technology,
        independent_fraction_max_absolute_difference=fraction_difference,
        csv_raster_max_absolute_difference_ha=source_difference,calendar_scenarios=calendar_summary,
        trial_calendar_points=len(point_scenarios)//4,trial_calendar_missing_rows=int(point_scenarios.native_missing_mask.sum()),
        excluded_Cyprus_harvested_ha=float(cyprus.whea.sum()),archive_checks=archive_checks,
        registry_sha256=sha(DATA/'europe_wheat_cells_025.parquet'),verified_sources_sha256={p.name:sha(p) for p in (DATA/'spam2020_wheat').glob('*.tif')})
    (HERE/'independent_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],checks=len(checks),cells=len(registry),calendar_scenarios=calendar_summary),indent=2))


if __name__=='__main__':run()
