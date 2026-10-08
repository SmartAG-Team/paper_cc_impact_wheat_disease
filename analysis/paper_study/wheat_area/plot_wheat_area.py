"""Standalone source-backed map of conserved wheat harvest hectares."""
from pathlib import Path
import hashlib
import json

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/wheat_area'


def run():
    source=DATA/'europe_wheat_cells_025.parquet';cells=pd.read_parquet(source)
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    first_row,last_row=int(cells.row.min()),int(cells.row.max())
    first_col,last_col=int(cells.col.min()),int(cells.col.max())
    values=np.full((last_row-first_row+1,last_col-first_col+1),np.nan)
    values[cells.row-first_row,cells.col-first_col]=cells.harvested_total_ha
    lon=-180+np.arange(first_col,last_col+2)*.25
    lat=90-np.arange(first_row,last_row+2)*.25
    fig,ax=plt.subplots(figsize=(11,7))
    countries.plot(ax=ax,color='#f1f2f2',edgecolor='#969b9e',linewidth=.35)
    image=ax.pcolormesh(lon,lat,values,cmap='YlOrBr',norm=LogNorm(vmin=10,vmax=40000),rasterized=True)
    ax.set_xlim(-12,64);ax.set_ylim(33,71)
    ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°N)')
    ax.set_title('Wheat harvested area, SPAM2020 v2r2',loc='left',fontsize=15,pad=15)
    ax.text(0,1.01,f'0.25° cell totals · {cells.harvested_total_ha.sum()/1e6:.3f} million ha · all wheat',transform=ax.transAxes,fontsize=10,color='#444')
    ax.set_aspect(1/np.cos(np.deg2rad(52)))
    color=fig.colorbar(image,ax=ax,shrink=.8,pad=.025,extend='min')
    color.set_label('Harvested hectares per grid cell')
    color.set_ticks([10,100,1000,10000,40000]);color.set_ticklabels(['10','100','1,000','10,000','40,000'])
    ax.grid(alpha=.15,linewidth=.5)
    fig.text(.085,.055,'Source: IFPRI SPAM2020 v2r2 (reference year 2020). Positive wheat cells only.',fontsize=9,color='#444')
    fig.text(.085,.035,'Continental boundary: declared Natural Earth approximation. Winter/spring area shares unresolved.',fontsize=9,color='#444')
    fig.subplots_adjust(left=.085,right=.91,bottom=.15,top=.88)
    fig.savefig(HERE/'europe_wheat_area_025.png',dpi=250,facecolor='white')
    fig.savefig(HERE/'europe_wheat_area_025.pdf',facecolor='white')
    plt.close(fig)
    (HERE/'figure_provenance.json').write_text(json.dumps(dict(source_file=str(source.relative_to(ROOT)),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        units='harvested ha per 0.25-degree cell',color_scale='logarithmic; values below10ha use lowest colour',area_season='unallocated_all_wheat',
        caption='Wheat harvested area in 2020, from IFPRI SPAM2020 v2r2. Native wheat hectares are summed to aligned 0.25-degree cells; transcontinental boundary cells receive declared polygon-overlap fractions. Uncoloured land has no displayed positive wheat area and may contain zero or unavailable source cells. Winter/spring crop-area shares are unresolved.',
        source_doi='10.7910/DVN/SWPENT'),indent=2)+'\n')
    print('PNG and PDF saved')


if __name__=='__main__':run()
