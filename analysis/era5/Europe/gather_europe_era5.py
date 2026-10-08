"""Native-grid Europe-region ERA5 extraction through authorized Earth Engine.

Default: genuine monthly-derived fixed-interval climate context for2017–2019.
Optional --hourly-start YYYY-MM-DD --hourly-end YYYY-MM-DD extracts an hourly
forcing window with exclusive end date. No disease model is applied.
Dependencies: earthengine-api,requests,pandas,numpy,rasterio,pyproj,pyshp,shapely.
OAuth credentials are reused by ee.Initialize and never read or printed here.
"""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import re
import zipfile

import ee
import numpy as np
import pandas as pd
from pyproj import Geod
import rasterio
from rasterio.features import rasterize
import requests
import shapefile
from shapely.geometry import shape,mapping,box

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis'/'era5'/'Europe'
DATA=ROOT/'data'/'era5'/'europe'
PROJECT='ee-gangzhaomodel'
BOUNDS=[-25,34,65,72]
YEARS=(2017,2018,2019)
MONTHLY='ECMWF/ERA5/MONTHLY'
HOURLY='ECMWF/ERA5/HOURLY'


def utc():return datetime.now(timezone.utc).isoformat()
def write_json(path,data):path.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
def safe_error(exc):return re.sub(r'https?://\S+','<redacted URL>',str(exc))[:1000]


def make_country_mask():
    archive=ROOT/'analysis'/'era5'/'ne_110m_admin_0_countries.zip'
    expected=json.loads((ROOT/'analysis'/'era5'/'natural_earth_provenance.json').read_text())['sha256']
    assert hashlib.sha256(archive.read_bytes()).hexdigest()==expected
    features=[]
    with zipfile.ZipFile(archive) as z:
        stem='ne_110m_admin_0_countries'
        reader=shapefile.Reader(shp=BytesIO(z.read(stem+'.shp')),shx=BytesIO(z.read(stem+'.shx')),dbf=BytesIO(z.read(stem+'.dbf')))
        selected=[r for r in reader.iterShapeRecords() if r.record.as_dict()['CONTINENT']=='Europe' or r.record.as_dict()['ADMIN']=='Cyprus']
        for index,record in enumerate(sorted(selected,key=lambda r:r.record.as_dict()['ADMIN']),1):
            properties=record.record.as_dict();geo=shape(record.shape.__geo_interface__)
            repaired=not geo.is_valid
            if repaired:geo=geo.buffer(0)
            clip=box(-25,34,60 if properties['ADMIN']=='Russia' else 65,72)
            geo=geo.intersection(clip)
            if geo.is_empty:continue
            features.append({'type':'Feature','properties':{'country':properties['ADMIN'],'country_code':properties['ADM0_A3'],
                'country_index':index,'natural_earth_continent':properties['CONTINENT'],'invalid_geometry_repaired':repaired},'geometry':mapping(geo)})
    geojson={'type':'FeatureCollection','features':features}
    write_json(DATA/'europe_region_countries.geojson',geojson)
    countries=pd.DataFrame([f['properties'] for f in features])
    countries.to_csv(DATA/'europe_region_countries.csv',index=False)
    write_json(DATA/'geographic_scope.json',{
        'label':'Europe-region country domain','bounding_box_wgs84':BOUNDS,
        'country_selection':'Natural Earth CONTINENT=Europe plus Cyprus; clipped to bounding box; Russia additionally clipped at60E',
        'included_non_EU_examples':['United Kingdom','Norway','Switzerland','Ukraine','Belarus','western Russia','Iceland','western Balkans'],
        'European_Turkey':'excluded; the domain is not the full geophysical continent',
        'Arctic_islands_north_of72N':'excluded','North_Africa':'excluded by country membership',
        'source_resolution':'Natural Earth 1:110m; very small states/islands may be absent',
        'country_count':len(features),'country_names':countries.country.tolist(),
        'boundary_source':'https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/',
        'boundary_license':'Public domain','boundary_zip_sha256':expected,
        'cell_inclusion_rule':'Raster pixel centre in a selected country polygon; not a wheat mask',
    })
    return features,countries


def probe_access():
    ee.Initialize(project=PROJECT)
    ee.data.setDeadline(180000)
    monthly=ee.ImageCollection(MONTHLY)
    hourly=ee.ImageCollection(HOURLY).filterDate('2018-09-01','2019-10-01')
    info={'access_checked_utc':utc(),'project':PROJECT,'credentials_reused_via_client_library':True,
        'monthly_collection':MONTHLY,'monthly_count':monthly.size().getInfo(),
        'monthly_last_date':ee.Date(monthly.aggregate_max('system:time_start')).format('YYYY-MM-dd').getInfo(),
        'monthly_bands':monthly.first().bandNames().getInfo(),
        'monthly_projection':monthly.first().select('mean_2m_air_temperature').projection().getInfo(),
        'hourly_collection':HOURLY,'hourly_2019_fixed_interval_count':hourly.size().getInfo(),
        'hourly_projection':hourly.first().select('temperature_2m').projection().getInfo(),
        'hourly_selected_bands':['temperature_2m','dewpoint_temperature_2m','total_precipitation','snowfall',
            'u_component_of_wind_10m','v_component_of_wind_10m','mean_surface_downward_short_wave_radiation_flux'],
        'monthly_limitation':'Collection ends June2020; monthly means cannot identify hourly humidity/rain conjunctions or daily infection timing',
    }
    assert info['hourly_2019_fixed_interval_count']==9480
    for name in ('monthly_projection','hourly_projection'):
        p=info[name];assert p['crs']=='EPSG:4326' and p['transform'][0]==0.25 and p['transform'][4]==-0.25
    write_json(OUT/'earthengine_access_validation.json',info)
    print('Verified monthly/hourly access and native0.25-degree grids',flush=True)
    return info


def monthly_climate():
    outputs=[];sources=[];names=[]
    for year in YEARS:
        start=f'{year-1}-09-01';end=f'{year}-10-01'
        collection=ee.ImageCollection(MONTHLY).filterDate(start,end)
        count=collection.size().getInfo();assert count==13
        dates=[pd.Timestamp(int(v),unit='ms',tz='UTC').date().isoformat() for v in collection.aggregate_array('system:time_start').getInfo()]
        expected=pd.date_range(start,pd.Timestamp(end)-pd.Timedelta(days=1),freq='MS').strftime('%Y-%m-%d').tolist()
        assert sorted(dates)==expected
        days=(pd.Timestamp(end)-pd.Timestamp(start)).days;assert days==395
        def weight(image):
            date=ee.Date(image.get('system:time_start'));n=date.advance(1,'month').difference(date,'day')
            return image.select('mean_2m_air_temperature').multiply(n)
        temperature=collection.map(weight).sum().divide(days).subtract(273.15).rename(f'tmean_{year}_c')
        precipitation=collection.select('total_precipitation').sum().multiply(1000).rename(f'precipitation_{year}_mm')
        outputs.extend([temperature,precipitation]);names.extend([f'tmean_{year}_c',f'precipitation_{year}_mm'])
        sources.append({'season_year':year,'window_start':start,'window_end_inclusive':f'{year}-09-30','days':days,
            'monthly_count':count,'source_month_starts':sorted(dates),'temperature_aggregation':'monthly mean Kelvin weighted by calendar month days, divided by395, then minus273.15',
            'precipitation_aggregation':'sum of monthly total precipitation in metres multiplied by1000; rain plus snow water equivalent'})
    image=ee.Image.cat(outputs).toFloat()
    return image,names,sources


def hourly_forcing(start,end):
    start_time=pd.Timestamp(start);end_time=pd.Timestamp(end)
    days=(end_time-start_time).days
    if days<=0 or days>31:raise ValueError('Hourly extraction must cover1–31completedUTCdays; larger jobs use repeated monthly/window chunks')
    collection=ee.ImageCollection(HOURLY).filterDate(start,end)
    count=collection.size().getInfo();assert count==days*24
    fields=['temperature_2m','dewpoint_temperature_2m','total_precipitation','snowfall',
            'u_component_of_wind_10m','v_component_of_wind_10m','mean_surface_downward_short_wave_radiation_flux']
    collection=collection.select(fields)
    def transform(image):
        t=image.select('temperature_2m').subtract(273.15)
        td=image.select('dewpoint_temperature_2m').subtract(273.15)
        # Magnus approximation over water; an atmospheric humidity variable, not leaf wetness.
        rh=td.multiply(17.625).divide(td.add(243.04)).subtract(t.multiply(17.625).divide(t.add(243.04))).exp().multiply(100).clamp(0,100)
        rain=image.select('total_precipitation').subtract(image.select('snowfall')).max(0).multiply(1000)
        wind=image.select('u_component_of_wind_10m').pow(2).add(image.select('v_component_of_wind_10m').pow(2)).sqrt()
        return ee.Image.cat([t.rename('tmean_c'),rh.rename('rh_mean_pct'),rh.gte(90).rename('rh_hours_ge90'),
            rain.rename('rain_mm'),rain.gt(0.1).rename('rain_hours_gt0_1mm'),image.select('total_precipitation').multiply(1000).rename('precipitation_mm'),
            wind.rename('wind_mean_m_s'),image.select('mean_surface_downward_short_wave_radiation_flux').multiply(0.0036).rename('shortwave_mj_m2')])
    converted=collection.map(transform)
    mean_names=['tmean_c','rh_mean_pct','wind_mean_m_s']
    sum_names=['rh_hours_ge90','rain_mm','rain_hours_gt0_1mm','precipitation_mm','shortwave_mj_m2']
    image=ee.Image.cat([converted.select(mean_names).mean(),converted.select(sum_names).sum()]).toFloat()
    sources=[{'window_start':start,'window_end_exclusive':end,'days':days,'hourly_count':count,
        'relative_humidity_method':'100exp(17.625Td/(243.04+Td)-17.625T/(243.04+T)), T/Td in°C; clipped0–100',
        'liquid_rain_method':'max(total_precipitation-snowfall,0)*1000; both native accumulations in metres water equivalent',
        'temporal_convention':'UTC timestamp; precipitation and radiation describe the preceding hour',
        'provider_precision_limitation':'GEE native temperature/dewpoint and derivedRH retain greater precision than rounded Open-Meteo fields; fitted-model transfer requires precision/feature compatibility checks',
        'model_role':'forcing summary only; no disease risk, wheat mask or sowing calendar applied'}]
    return image,mean_names+sum_names,sources


def fetch_raster(image,names,projection,features,basename):
    region=ee.Geometry.Rectangle(BOUNDS,proj='EPSG:4326',geodesic=False)
    countries=ee.FeatureCollection([ee.Feature(ee.Geometry(f['geometry'],proj='EPSG:4326',geodesic=False),f['properties']) for f in features])
    image=image.clipToCollection(countries).unmask(-9999).toFloat()
    payload={'format':'GEO_TIFF','filePerBand':False,'crs':projection['crs'],'crs_transform':projection['transform'],'region':region}
    estimated=(int((BOUNDS[2]-BOUNDS[0])/0.25)+2)*(int((BOUNDS[3]-BOUNDS[1])/0.25)+2)*len(names)*4
    assert estimated<32_000_000
    print(f'Requesting {basename}: {len(names)} bands, estimated {estimated} bytes; signed download URL is not logged',flush=True)
    url=image.getDownloadURL(payload)
    response=requests.get(url,timeout=(15,180))
    if not response.ok:raise RuntimeError(f'Earth Engine raster HTTP{response.status_code}')
    assert len(response.content)<32_000_000
    raw_path=DATA/(basename+'_gee_download.tif')
    raw_path.write_bytes(response.content)
    with rasterio.open(raw_path) as source:
        values=source.read();profile=source.profile;transform=source.transform
        assert source.count==len(names) and source.crs.to_epsg()==4326
        assert abs(transform.a-0.25)<1e-12 and abs(transform.e+0.25)<1e-12
        mask=rasterize([(f['geometry'],f['properties']['country_index']) for f in features],out_shape=(source.height,source.width),
                       transform=transform,fill=0,all_touched=False,dtype='uint16')
        valid=mask>0
        valid &= np.isfinite(values).all(axis=0) & (values!=-9999).all(axis=0)
        numeric_adjustments=[]
        for band,name in enumerate(names):
            if name=='precipitation_mm' or name.startswith('precipitation_'):
                negative=valid&(values[band]<0)
                if negative.any():
                    minimum=float(values[band,negative].min())
                    if minimum < -0.001:raise ValueError(f'Negative precipitation exceeds numerical-artifact tolerance: {name}, {minimum}mm')
                    numeric_adjustments.append({'band':name,'negative_aggregate_cells':int(negative.sum()),'raw_min_mm':minimum,
                        'prepared_summary_action':'negative totals set to zero; original downloaded raster unchanged',
                        'encoding_origin':'not determined; magnitudes are physically nonmeaningful numerical artifacts'})
                    values[band,negative]=0
        values[:,~valid]=-9999
        profile.update(dtype='float32',nodata=-9999,compress='deflate',predictor=3)
        final_path=DATA/(basename+'.tif')
        with rasterio.open(final_path,'w',**profile) as target:
            target.write(values)
            for number,name in enumerate(names,1):target.set_band_description(number,name)
            target.update_tags(source_collection=MONTHLY if basename.startswith('era5_fixed') else HOURLY,
                               domain='Europe-region; NaturalEarth Europe+Cyprus; Russia west60E; Turkey excluded',role='weather exposure; not disease risk',
                               attribution='Contains modified Copernicus Climate Change Service information; supplied via Google Earth Engine')
        mask_profile=profile|{'count':1,'dtype':'uint16','nodata':0,'predictor':2}
        mask_path=DATA/(basename+'_country_mask.tif')
        with rasterio.open(mask_path,'w',**mask_profile) as target:target.write(np.where(valid,mask,0).astype('uint16'),1)
        pixel_rows,pixel_cols=np.where(valid)
        longitudes,latitudes=rasterio.transform.xy(transform,pixel_rows,pixel_cols,offset='center')
        grid=pd.DataFrame({'latitude':latitudes,'longitude':longitudes,'country_index':mask[valid]})
        grid['grid_cell_id']=[f'era5monthlygrid_{a:.3f}_{b:.3f}' if basename.startswith('era5_fixed') else f'era5grid_{a:.2f}_{b:.2f}' for a,b in zip(grid.latitude,grid.longitude)]
        geod=Geod(ellps='WGS84')
        area_by_lat={lat:abs(geod.polygon_area_perimeter([0,0.25,0.25,0],[lat-0.125,lat-0.125,lat+0.125,lat+0.125])[0])/1e6 for lat in grid.latitude.unique()}
        grid['selected_grid_cell_area_km2']=grid.latitude.map(area_by_lat)
        countries=pd.DataFrame([f['properties'] for f in features])
        grid=grid.merge(countries[['country_index','country','country_code']],on='country_index',validate='many_to_one')
        for i,name in enumerate(names):grid[name]=values[i,valid]
        grid.to_csv(DATA/(basename+'_grid.csv.gz'),index=False,compression='gzip',float_format='%.8f')
        grid.to_parquet(DATA/(basename+'_grid.parquet'),index=False,compression='zstd')
        coverage=grid.groupby(['country','country_code'],as_index=False).agg(selected_grid_cells=('grid_cell_id','size'),
            selected_grid_cell_area_km2=('selected_grid_cell_area_km2','sum'))
        coverage.to_csv(DATA/(basename+'_country_coverage.csv'),index=False,float_format='%.4f')
    metadata={'retrieved_utc':utc(),'basename':basename,'native_projection':projection,'region_bounds':BOUNDS,
        'band_names':names,'width':profile['width'],'height':profile['height'],'valid_selected_grid_cells':len(grid),
        'country_mask_path':str(mask_path.relative_to(ROOT)),
        'selected_grid_cell_area_km2':float(grid.selected_grid_cell_area_km2.sum()),
        'area_limitation':'sum of full ERA5 cell areas selected by country-mask centres; not exact land area and not wheat area',
        'download_bytes':len(response.content),'download_sha256':hashlib.sha256(response.content).hexdigest(),
        'final_raster_sha256':hashlib.sha256(final_path.read_bytes()).hexdigest(),
        'download_URL_policy':'ephemeral signed URL excluded from archived provenance',
        'numeric_adjustments':numeric_adjustments,
        'ranges':{name:{'min':float(grid[name].min()),'max':float(grid[name].max())} for name in names},
        'country_count_with_grid_centres':coverage.country.nunique(),'country_names_with_grid_centres':coverage.country.tolist()}
    write_json(DATA/(basename+'_retrieval.json'),metadata)
    print(f'Downloaded {len(response.content)} bytes; {len(grid)} selected land-domain grid centres',flush=True)
    return metadata


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hourly-start');parser.add_argument('--hourly-end')
    args=parser.parse_args();OUT.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    features,countries=make_country_mask();access=probe_access()
    if args.hourly_start or args.hourly_end:
        if not(args.hourly_start and args.hourly_end):raise ValueError('Supply both hourly-start and exclusive hourly-end')
        image,names,sources=hourly_forcing(args.hourly_start,args.hourly_end)
        basename=f'era5_hourly_forcing_{args.hourly_start}_{args.hourly_end}'
        projection=access['hourly_projection'];collection=HOURLY
    else:
        image,names,sources=monthly_climate();basename='era5_fixed_intervals_2017_2019_native025'
        projection=access['monthly_projection'];collection=MONTHLY
    metadata=fetch_raster(image,names,projection,features,basename)
    write_json(DATA/(basename+'_manifest.json'),{'created_utc':utc(),'source_collection':collection,'source_catalog':
        'https://developers.google.com/earth-engine/datasets/catalog/'+collection.replace('/','_'),
        'temporal_sources':sources,'geographic_scope_file':'data/era5/europe/geographic_scope.json',
        'role':'climate exposure / forcing summary only; no Septoria model or wheat-specific host mask applied',
        'retrieval':metadata,'attribution':'Contains modified Copernicus Climate Change Service information; supplied via Google Earth Engine',
        'reproduction_script':'analysis/era5/Europe/gather_europe_era5.py'})
    print(json.dumps({'result':'success','artifact':f'data/era5/europe/{basename}.tif','valid_grid_cells':metadata['valid_selected_grid_cells']}),flush=True)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        OUT.mkdir(exist_ok=True)
        write_json(OUT/'earthengine_extraction_failure.json',{'checked_utc':utc(),'error_type':type(exc).__name__,'safe_message':safe_error(exc)})
        print(f'{type(exc).__name__}: {safe_error(exc)}',flush=True)
        raise SystemExit(1)
