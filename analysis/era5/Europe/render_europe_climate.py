"""Render genuine ERA5 regional exposure context and verify native raster geometry."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from pyproj import Transformer
import rasterio

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis'/'era5'/'Europe'
DATA=ROOT/'data'/'era5'/'europe'
BASE='era5_fixed_intervals_2017_2019_native025'

plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
    'text.color':'#26313a','axes.labelcolor':'#26313a','pdf.fonttype':42,'ps.fonttype':42,
    'figure.facecolor':'white','savefig.facecolor':'white'})

with rasterio.open(DATA/(BASE+'.tif')) as source:
    values=source.read(masked=True);transform=source.transform
    assert source.crs.to_epsg()==4326 and transform.a==0.25 and transform.e==-0.25
    shape=source.shape;descriptions=list(source.descriptions)
    lon_edges=transform.c+np.arange(source.width+1)*transform.a
    lat_edges=transform.f+np.arange(source.height+1)*transform.e
    latlon=np.meshgrid(lon_edges,lat_edges)
    projection=Transformer.from_crs('EPSG:4326','EPSG:3035',always_xy=True)
    xx,yy=projection.transform(latlon[0],latlon[1])
    masked_count=int((~values.mask[0]).sum());assert masked_count==22140
    assert np.array_equal(values.mask[0],values.mask[-1])
    profile_geometry={'width':source.width,'height':source.height,'crs':str(source.crs),'transform':list(transform)[:6]}
    coordinate_centre_fraction=np.mod((lon_edges[:-1]+0.125)*4,1)
    assert np.allclose(coordinate_centre_fraction,0.5)

countries=json.loads((DATA/'europe_region_countries.geojson').read_text())
paths=[]
for feature in countries['features']:
    geometry=feature['geometry']
    polygons=geometry['coordinates'] if geometry['type']=='MultiPolygon' else [geometry['coordinates']]
    for polygon in polygons:
        vertices,codes=[],[]
        for ring in polygon:
            ring=np.asarray(ring);x,y=projection.transform(ring[:,0],ring[:,1])
            vertices.extend(np.column_stack([x,y]).tolist())
            codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(ring)-2)+[MplPath.CLOSEPOLY])
        paths.append(MplPath(np.asarray(vertices),np.asarray(codes)))

valid=np.where(~values.mask[0])
valid_x=xx[valid];valid_y=yy[valid]
xlim=(float(valid_x.min()-180000),float(valid_x.max()+180000));ylim=(float(valid_y.min()-180000),float(valid_y.max()+180000))
fig,axes=plt.subplots(1,2,figsize=(10.7,6.1))
fig.subplots_adjust(left=0.03,right=0.97,bottom=0.21,top=0.79,wspace=0.20)
fields=[('tmean_2019_c','(a) Mean 2 m air temperature','°C','cividis',-6,23),
        ('precipitation_2019_mm','(b) Total precipitation','mm','Blues',0,4000)]
for axis,(name,title,unit,cmap,vmin,vmax) in zip(axes,fields):
    band=values[descriptions.index(name)]
    artist=axis.pcolormesh(xx,yy,band,cmap=cmap,vmin=vmin,vmax=vmax,shading='flat',rasterized=True)
    for path in paths:axis.add_patch(PathPatch(path,facecolor='none',edgecolor='#68747b',linewidth=0.35))
    axis.set_xlim(xlim);axis.set_ylim(ylim);axis.set_aspect('equal');axis.set_axis_off();axis.set_title(title,fontsize=11,pad=7)
    cbar=fig.colorbar(artist,ax=axis,orientation='horizontal',fraction=0.06,pad=0.03,shrink=0.9)
    cbar.set_label(unit,fontsize=10);cbar.outline.set_linewidth(0.5)
fig.text(0.03,0.957,'ERA5 weather context for the Europe-region domain, 2019',fontsize=14,ha='left',va='top')
fig.text(0.03,0.895,'1 September 2018–30 September 2019 · Fixed 395-day coverage',fontsize=10)
fig.text(0.03,0.120,'Country-polygon domain: UK/non-EU Europe and Cyprus included; Russia west 60°E; Turkey excluded.',fontsize=8.7)
fig.text(0.03,0.087,'Geographic cell selection covers all land types; wheat applicability requires a host map and phenology.',fontsize=8.5)
fig.text(0.03,0.054,'ERA5 via Google Earth Engine · Contains modified Copernicus Climate Change Service information (2019)',fontsize=8.2,url='https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY')
fig.text(0.03,0.021,'Boundaries: Natural Earth, public domain · Projection: EPSG:3035 · Weather exposure context',fontsize=8.2)
fig.savefig(OUT/'europe_region_era5_weather_context_2019.png',dpi=400)
fig.savefig(OUT/'europe_region_era5_weather_context_2019.pdf',metadata={'Title':'Europe-region ERA5 weather context,2019','Creator':'Reproducible Earth Engine/Matplotlib workflow'})
plt.close(fig)

grid=pd.read_parquet(DATA/(BASE+'_grid.parquet'))
assert len(grid)==masked_count and grid.grid_cell_id.nunique()==len(grid)
assert grid.grid_cell_id.str.startswith('era5monthlygrid_').all()
assert grid.country.ne('United Kingdom').any() and grid.country.eq('United Kingdom').any()
assert not grid.country.isin(['Turkey','Morocco','Algeria','Tunisia','Egypt','Libya']).any()
stats=[]
for year in [2017,2018,2019]:
    for name in [f'tmean_{year}_c',f'precipitation_{year}_mm']:
        x=grid[name]
        assert x.notna().all()
        if name.startswith('precipitation'):assert x.ge(0).all()
        stats.append({'season_year':year,'variable':name,'selected_grid_cells':len(x),'unweighted_mean':float(x.mean()),
            'median':float(x.median()),'min':float(x.min()),'max':float(x.max()),
            'grid_cell_area_weighted_mean':float(np.average(x,weights=grid.selected_grid_cell_area_km2))})
pd.DataFrame(stats).to_csv(OUT/'europe_regional_context_stats.csv',index=False,float_format='%.8f')
validation={'checked_utc':datetime.now(timezone.utc).isoformat(),'result':'pass','monthly_raster_geometry':profile_geometry,
    'valid_selected_grid_cells':len(grid),'country_count_with_grid_centres':grid.country.nunique(),
    'selected_grid_cell_area_km2':float(grid.selected_grid_cell_area_km2.sum()),
    'native_monthly_grid_centres_preserved':True,'monthly_identifiers_separate_from_hourly_grid':True,
    'no_missing_selected_weather_values':True,'North_Africa_and_Turkey_excluded':True,'UK_included':True,
    'layer_role':'regional weather context; no wheat-specific mask or disease risk model applied',
    'final_raster_sha256':hashlib.sha256((DATA/(BASE+'.tif')).read_bytes()).hexdigest()}
(OUT/'continental_weather_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
print(json.dumps(validation,indent=2))
