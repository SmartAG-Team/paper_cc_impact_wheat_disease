"""Export verified European positive-wheat support with explicit nodata."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT/'data/paper_study/europe_wheat_geotiff'
FIELDS = ['harvested_total_ha','physical_total_ha','harvested_rainfed_ha','harvested_irrigated_ha',
          'physical_rainfed_ha','physical_irrigated_ha']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(frame,row_name,col_name,resolution,label,source_path):
    first_row,last_row = int(frame[row_name].min()),int(frame[row_name].max())
    first_col,last_col = int(frame[col_name].min()),int(frame[col_name].max())
    rows,cols = last_row-first_row+1,last_col-first_col+1
    rr,cc = frame[row_name].to_numpy(int)-first_row,frame[col_name].to_numpy(int)-first_col
    if frame.duplicated([row_name,col_name]).any():
        raise ValueError('Duplicate raster cell keys.')
    arrays = np.full((6,rows,cols),np.nan,dtype='float64')
    for i,field in enumerate(FIELDS):
        arrays[i,rr,cc] = frame[field].to_numpy(float)
    transform = from_origin(-180+first_col*resolution,90-first_row*resolution,resolution,resolution)
    path = DEST/f'europe_wheat_area_{label}.tif'
    with rasterio.open(path,'w',driver='GTiff',width=cols,height=rows,count=6,dtype='float64',
        crs='EPSG:4326',transform=transform,nodata=np.nan,compress='DEFLATE',predictor=3,tiled=True) as raster:
        raster.write(arrays)
        raster.update_tags(source_dataset='IFPRI SPAM2020 Version2 Release2, WHEA',reference_year='2020',
            source_sha256=sha(source_path),units='hectares',geographic_scope='declared geographical Europe',
            missing_semantics='Nodata outside archived positive European wheat support; not a verified absence class',
            continental_boundary_allocation='source footprint European polygon overlap',
            winter_spring_allocation='not supplied by SPAM')
        for i,field in enumerate(FIELDS,1):
            raster.set_band_description(i,field)
            raster.update_tags(i,units='ha')
    mask_path = DEST/f'europe_wheat_positive_support_{label}.tif'
    mask = np.full((rows,cols),255,dtype='uint8');mask[rr,cc] = 1
    with rasterio.open(mask_path,'w',driver='GTiff',width=cols,height=rows,count=1,dtype='uint8',
        crs='EPSG:4326',transform=transform,nodata=255,compress='DEFLATE',tiled=True) as raster:
        raster.write(mask,1)
        raster.set_band_description(1,'positive_archived_European_wheat_support')
        raster.update_tags(value1='positive included wheat hectares',value255='outside archived positive support',
                           no_verified_zero_disease_or_wheat_absence_class='true',source_sha256=sha(source_path))
    with rasterio.open(path) as raster:
        actual = raster.read()
        np.testing.assert_array_equal(actual,arrays)
        if raster.crs.to_epsg()!=4326 or raster.transform!=transform:
            raise ValueError('GIS projection/transform drift.')
    with rasterio.open(mask_path) as raster:
        if int((raster.read(1)==1).sum())!=len(frame):
            raise ValueError('Wheat-support cell count drift.')
    totals = {field:float(np.nansum(arrays[i])) for i,field in enumerate(FIELDS)}
    for field,value in totals.items():
        if abs(value-float(frame[field].sum()))>1e-6:
            raise ValueError('Crop-area totals changed during raster export.')
    return dict(resolution_degrees=resolution,source=str(source_path.relative_to(ROOT)),source_sha256=sha(source_path),
                positive_cells=len(frame),totals_ha=totals,
                rasters=[dict(path=str(p.relative_to(ROOT)),sha256=sha(p),bytes=p.stat().st_size) for p in [path,mask_path]],
                bounds=dict(west=transform.c,north=transform.f,east=transform.c+cols*resolution,south=transform.f-rows*resolution),
                projection='EPSG:4326',nodata_is_absence=False)


def main():
    DEST.mkdir(parents=True,exist_ok=True)
    sources = ROOT/'data/paper_study/wheat_area'
    coarse = sources/'europe_wheat_cells_025.parquet'
    native = sources/'europe_source_wheat_pixels.parquet'
    coarse_frame = pd.read_parquet(coarse)
    native_frame = pd.read_parquet(native)
    # Source native hectares are full-footprint quantities. Match the study's
    # fractional European membership before exporting transcontinental pixels.
    for field in FIELDS:
        native_frame[field] *= native_frame.europe_fraction
        if abs(float(native_frame[field].sum())-float(coarse_frame[field].sum()))>1e-6:
            raise ValueError('Native European hectares do not reconcile with the common grid.')
    receipts = [export(coarse_frame,'row','col',.25,'025deg',coarse),
                export(native_frame,'source_row','source_col',1/12,'5arcmin',native)]
    (DEST/'receipt.json').write_text(json.dumps(dict(status='verified',source_modified=False,exports=receipts),indent=2)+'\n')
    (DEST/'README.txt').write_text('European wheat distribution, reference year2020\n\n'
        'Six-band GeoTIFFs retain SPAM wheat harvested total, physical total, harvested rainfed, harvested irrigated, physical rainfed and physical irrigated hectares. Band descriptions and units are embedded. Native5-arcminute and aligned0.25-degree products use EPSG:4326. European boundary footprints follow the verified study area registry.\n\n'
        'Positive-support masks contain1 for archived positive European wheat hectares and255 nodata elsewhere. Nodata does not establish absence of wheat. These maps identify the modelled wheat domain; they do not classify crops at individual-field resolution. The2020 distribution is a fixed land-use reference for climate comparisons, rather than a prediction of future wheat planting.\n\n'
        'Original data: IFPRI SPAM2020 Version2 Release2, WHEA. https://doi.org/10.7910/DVN/SWPENT\n')
    print(json.dumps(receipts),flush=True)


if __name__=='__main__':
    main()
