"""Integrity and native-grid checks on actually downloaded regional artifacts."""
from datetime import datetime,timezone
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT=Path(__file__).resolve().parents[3]
DATA=ROOT/'data'/'era5'/'europe'
OUT=ROOT/'analysis'/'era5'/'Europe'
ast.parse((OUT/'gather_europe_era5.py').read_text())
results=[]
for base,expected_count,expected_bands in [
    ('era5_fixed_intervals_2017_2019_native025',22140,6),
    ('era5_hourly_forcing_2019-05-01_2019-05-02',22198,8),
]:
    meta=json.loads((DATA/(base+'_retrieval.json')).read_text())
    manifest=json.loads((DATA/(base+'_manifest.json')).read_text())
    assert meta['download_bytes']<32_000_000
    assert hashlib.sha256((DATA/(base+'_gee_download.tif')).read_bytes()).hexdigest()==meta['download_sha256']
    assert hashlib.sha256((DATA/(base+'.tif')).read_bytes()).hexdigest()==meta['final_raster_sha256']
    grid=pd.read_parquet(DATA/(base+'_grid.parquet'))
    assert len(grid)==expected_count==grid.grid_cell_id.nunique()
    with rasterio.open(DATA/(base+'.tif')) as raster:
        assert raster.count==expected_bands and raster.crs.to_epsg()==4326
        assert raster.transform.a==0.25 and raster.transform.e==-0.25
        arrays=raster.read();valid=(arrays!=-9999).all(axis=0)
        assert valid.sum()==expected_count
        assert np.isfinite(arrays[:,valid]).all()
        rr,cc=rasterio.transform.rowcol(raster.transform,grid.longitude.to_numpy(),grid.latitude.to_numpy())
        for i,name in enumerate(raster.descriptions):assert np.array_equal(arrays[i,rr,cc],grid[name].to_numpy())
    with rasterio.open(DATA/(base+'_country_mask.tif')) as mask:
        assert np.array_equal(mask.read(1)>0,valid)
    assert grid.country.eq('United Kingdom').any() and grid.country.eq('Cyprus').any()
    assert not grid.country.isin(['Turkey','Morocco','Algeria','Tunisia','Egypt','Libya']).any()
    assert grid.loc[grid.country.eq('Russia'),'longitude'].le(60).all()
    if base.startswith('era5_fixed'):
        assert grid.grid_cell_id.str.startswith('era5monthlygrid_').all()
        assert np.allclose(np.mod(grid.longitude.to_numpy()*4,1),0.5)
        assert all(x['monthly_count']==13 and x['days']==395 for x in manifest['temporal_sources'])
        for year in (2017,2018,2019):assert grid[f'precipitation_{year}_mm'].ge(0).all()
    else:
        assert grid.grid_cell_id.str.startswith('era5grid_').all()
        assert np.allclose(np.mod(grid.longitude.to_numpy()*4,1),0)
        assert manifest['temporal_sources'][0]['hourly_count']==24
        for field in ('rh_hours_ge90','rain_hours_gt0_1mm'):
            assert grid[field].between(0,24).all() and np.array_equal(grid[field],np.round(grid[field]))
        assert grid.rh_mean_pct.between(0,100).all()
        assert grid.rain_mm.ge(0).all() and grid.precipitation_mm.ge(0).all()
        assert grid.rain_mm.le(grid.precipitation_mm+1e-5).all()
    results.append({'artifact':base,'result':'pass','valid_grid_cells':len(grid),'band_count':expected_bands,
        'raw_and_final_raster_hashes_verified':True,'all_table_values_match_raster':True,'native_grid_registration_preserved':True,
        'country_mask_matches_valid_cells':True,'country_count':grid.country.nunique(),'download_bytes':meta['download_bytes']})
comparison=json.loads((OUT/'hourly_provider_comparison.json').read_text())
assert comparison['matched_grid_cells']==comparison['BASF_grid_cells_on_date']==32
result={'checked_utc':datetime.now(timezone.utc).isoformat(),'result':'pass','artifacts':results,
    'provider_comparison_grid_cells':32,'provider_precision_differences_remain_material_for_threshold_counts':True,
    'model_output_role':'weather context / forcing only; no Septoria risk calculated',
    'wheat_specific_host_mask_created':False,'script_syntax':'pass'}
(OUT/'regional_archive_independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
