"""Source-backed sowing-window allocation sensitivity figure."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/wheat_area'

def main():
    source=DATA/'europe_wheat_cells_025_seasonal_sensitivity.parquet';d=pd.read_parquet(source)
    geography=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    r0,r1=int(d.row.min()),int(d.row.max());c0,c1=int(d.col.min()),int(d.col.max())
    lon=-180+np.arange(c0,c1+2)*.25;lat=90-np.arange(r0,r1+2)*.25
    fig,axes=plt.subplots(1,2,figsize=(13,5.7))
    for ax,key,title in zip(axes,['autumn_sowing','spring_sowing'],['a  Autumn window (August–December)','b  Spring window (February–June)']):
        geography.plot(ax=ax,color='#f2f3f3',edgecolor='#92989a',linewidth=.25)
        v=np.full((r1-r0+1,c1-c0+1),np.nan);v[d.row-r0,d.col-c0]=d[key+'_fraction']*100
        im=ax.pcolormesh(lon,lat,v,cmap='YlGnBu',vmin=0,vmax=100,rasterized=True)
        ax.set_xlim(-12,64);ax.set_ylim(33,71);ax.set_aspect(1/np.cos(np.radians(52)))
        ax.set_title(title,loc='left',fontsize=11);ax.set_xlabel('Longitude (°E)')
        ax.text(.01,.02,f'{d[key+"_harvested_ha"].sum()/1e6:.3f} million allocated ha',transform=ax.transAxes,fontsize=9,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    axes[0].set_ylabel('Latitude (°N)')
    axes[1].set_ylabel('')
    fig.subplots_adjust(left=.065,right=.88,bottom=.22,top=.85,wspace=.12)
    cbax=fig.add_axes([.90,.27,.015,.50]);fig.colorbar(im,cax=cbax,label='Share of original SPAM harvested area (%)')
    fig.suptitle('Calendar-defined wheat-season allocation: MIRCA-to-SPAM reference sensitivity',fontsize=13,y=.97)
    fig.text(.065,.115,'2020 area; genetic winter/spring type unidentified. Russia seasonal-share applicability unresolved.',fontsize=9)
    fig.text(.065,.082,f'Unallocated: {d.unallocated_harvested_ha.sum()/1e6:.3f} million ha; ambiguous: {d.ambiguous_harvested_ha.sum():,.0f} ha. All wheat hectares retained.',fontsize=9)
    fig.text(.065,.049,'Source: MIRCA-OS v2 monthly subcrop areas and calendars; IFPRI SPAM2020 v2r2; declared European mask.',fontsize=8.5)
    for suffix in ['png','pdf']:fig.savefig(HERE/f'europe_wheat_sowing_window_sensitivity.{suffix}',dpi=250,facecolor='white')
    plt.close(fig)
    receipt=dict(source=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        units='percentage of original SPAM total harvested hectares within each cell',
        caption='Reference seasonal allocation of SPAM2020 wheat harvested hectares using native MIRCA-OS 2020 irrigated/rainfed subcrop support ratios and source planting months. Autumn and spring windows comprise August–December and February–June. Unallocated and ambiguous contributions remain separate. Sowing windows do not identify genetic wheat types; Russia seasonal shares have unresolved contemporary applicability. Colours represent valid wheat-cell fractions; uncoloured land contains no displayed positive SPAM wheat cell.')
    (HERE/'seasonal_figure_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print('Seasonal sensitivity PNG and PDF written')

if __name__=='__main__':main()
