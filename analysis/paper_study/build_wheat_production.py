"""Conserve reference wheat-production tonnes on the frozen European domain."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import rasterio

ROOT=Path(__file__).resolve().parents[2]
AREA=ROOT/'data/paper_study/wheat_area'
OUT=ROOT/'data/paper_study/wheat_production'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipt=json.loads((OUT/'download_receipt.json').read_text())
    assert receipt['status']=='complete_and_hash_verified'
    pixels=pd.read_parquet(AREA/'europe_source_wheat_pixels.parquet')
    cells=pd.read_parquet(AREA/'europe_wheat_cells_025.parquet')
    assert sha(AREA/'europe_wheat_cells_025.parquet')=='380833fc139e7c76adc274384f7175438b202b217cdb4194c439f4ea69af6c92'
    assert not pixels.duplicated(['source_row','source_col']).any()
    hashes={};bands=[];global_support={}
    with rasterio.open(AREA/'spam2020_wheat/spam2020_V2r2_global_H_WHEA_A.tif') as raster:
        global_area=raster.read(1);global_area_valid=raster.read_masks(1).astype(bool)
    for technology,label in [('A','total'),('I','irrigated'),('R','rainfed')]:
        path=OUT/f'spam2020_V2r2_global_P_WHEA_{technology}.tif'
        with rasterio.open(path) as raster:
            assert raster.crs.to_epsg()==4326 and raster.shape==(2160,4320)
            np.testing.assert_allclose(tuple(raster.transform)[:6],[1/12,0,-180,0,-1/12,90],rtol=0,atol=1e-12)
            array=raster.read(1);mask=raster.read_masks(1).astype(bool)
            global_positive=mask&(array>0)
            outside=global_positive&~(global_area_valid&(global_area>0))
            global_support[label]=dict(positive_pixels=int(global_positive.sum()),
                positive_production_outside_positive_area_pixels=int(outside.sum()),
                tonnes_outside_positive_area=float(array[outside].sum()))
            # The fixed domain contains positive-area native pixels. Source
            # rounding can leave small positive production on zero-area pixels;
            # its global sum bounds any omitted European tonnes and is retained.
            values=array[pixels.source_row.to_numpy(int),pixels.source_col.to_numpy(int)].astype(float)
            valid=mask[pixels.source_row.to_numpy(int),pixels.source_col.to_numpy(int)]
            if technology=='A':assert valid.all()
            values=np.where(valid,values,0.)
            assert np.isfinite(values).all() and (values>=0).all()
            key=f'production_{label}_tonnes'
            pixels[key]=values;pixels[key+'_source_raster_valid']=valid
            bands.append(key);hashes[str(path.relative_to(ROOT))]=sha(path)
    residual=pixels.production_total_tonnes-pixels.production_irrigated_tonnes-pixels.production_rainfed_tonnes
    missing=~pixels.production_irrigated_tonnes_source_raster_valid|~pixels.production_rainfed_tonnes_source_raster_valid
    # Technology omission is a structural zero only when its complementary
    # source layer conserves the available total within source rounding.
    assert not ((~pixels.production_irrigated_tonnes_source_raster_valid)&
        (~pixels.production_rainfed_tonnes_source_raster_valid)&pixels.production_total_tonnes.gt(0)).any()
    assert residual[missing].abs().max()<.11
    weighted=pixels.copy()
    weighted[bands]=weighted[bands].mul(weighted.europe_fraction,axis=0)
    grouped=weighted.groupby(['row','col'])[bands].sum().reset_index()
    result=cells.merge(grouped,on=['row','col'],how='left',validate='one_to_one')
    assert result.cell_id.to_list()==cells.cell_id.to_list() and result[bands].notna().all().all()
    result['production_reference_year']=2020
    result['production_weight']=result.production_total_tonnes/result.production_total_tonnes.sum()
    result['reference_yield_tonnes_per_ha']=result.production_total_tonnes/result.harvested_total_ha
    country=weighted.groupby(['row','col','ADM0_NAME','FIPS0'])[bands].sum().reset_index()
    country['cell_id']=[f'g025_r{r:03d}_c{c:04d}' for r,c in zip(country.row,country.col)]
    summary=weighted.groupby(['ADM0_NAME','FIPS0'])[bands].sum().reset_index()
    conserved={}
    for key in bands:
        total=float(weighted[key].sum());difference=abs(total-result[key].sum())
        assert difference<1e-6
        np.testing.assert_allclose(country.groupby('cell_id')[key].sum().reindex(result.cell_id),result[key],rtol=0,atol=1e-8)
        conserved[key]=dict(included_native_tonnes=total,coarse_grid_tonnes=float(result[key].sum()),
            maximum_aggregation_residual_tonnes=float(difference))
    pixels.to_parquet(OUT/'europe_source_wheat_production.parquet',index=False)
    result.to_parquet(OUT/'europe_wheat_production_025.parquet',index=False)
    country.to_parquet(OUT/'europe_cell_country_wheat_production.parquet',index=False)
    summary.to_csv(OUT/'europe_country_wheat_production.csv',index=False)
    report=dict(status='passed',checked_utc=datetime.now(timezone.utc).isoformat(),registry_cells=len(result),
        source_pixels=len(pixels),totals=conserved,global_positive_production_support=global_support,
        technology_total_minus_irrigated_minus_rainfed_tonnes=float(np.dot(residual,pixels.europe_fraction)),
        maximum_technology_residual_tonnes=float(residual.abs().max()),
        maximum_residual_where_technology_omitted_tonnes=float(residual[missing].abs().max()),
        source_raster_sha256=hashes,frozen_area_registry_sha256=sha(AREA/'europe_wheat_cells_025.parquet'),
        source_boundary_pixels_sha256=sha(AREA/'europe_source_wheat_pixels.parquet'),
        production_source_receipt_sha256=sha(OUT/'download_receipt.json'),source_code_sha256=sha(Path(__file__)),
        production_is_reference_exposure=True,production_is_simulated_yield_loss=False,
        output_sha256={str(p.relative_to(ROOT)):sha(p) for p in OUT.glob('europe_*')})
    (OUT/'aggregation_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['output_sha256','source_raster_sha256']}),flush=True)


if __name__=='__main__':main()
