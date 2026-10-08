"""Reproducible spatial figures from completed, version-labelled crop results."""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm,LinearSegmentedColormap,Normalize
from .prepare import HERE,ROOT,ARCHIVE

BLUE='#235e83';GOLD='#a17937';GREEN='#52796f';GREY='#6e7880'
DIVERGING=LinearSegmentedColormap.from_list('signed_crop_change',[BLUE,'#f8f8f5',GOLD])


def geography(ax,countries):
    countries.plot(ax=ax,color='#f0f2f3',edgecolor='#9fa8ae',linewidth=.30,zorder=0)
    ax.set(xlim=(-12,64),ylim=(33,70),xticks=[0,20,40,60],yticks=[40,50,60,70])
    ax.set_aspect(1/np.cos(np.deg2rad(52)));ax.tick_params(labelsize=8.2,length=2)
    ax.set_xlabel('Longitude (°E)',fontsize=8.5);ax.set_ylabel('Latitude (°N)',fontsize=8.5)
    ax.spines[['top','right']].set_visible(False)


def raster(ax,cells,value,**kwargs):
    lo_r,hi_r=int(cells.row.min()),int(cells.row.max());lo_c,hi_c=int(cells.col.min()),int(cells.col.max())
    image=np.full((hi_r-lo_r+1,hi_c-lo_c+1),np.nan)
    image[cells.row-lo_r,cells.col-lo_c]=cells[value]
    return ax.pcolormesh(-180+np.arange(lo_c,hi_c+2)*.25,90-np.arange(lo_r,hi_r+2)*.25,
        image,rasterized=True,zorder=1,**kwargs)


def selected(kind,period,metric,scenario='ssp585'):
    frame=pd.read_csv(HERE/'map_ensemble_values.csv')
    return frame.loc[frame.kind.eq(kind)&frame.period.eq(period)&frame.metric.eq(metric)&frame.scenario.eq(scenario)]


def dots(ax,rows,norm,cmap):
    good=rows.value.notna()
    image=ax.scatter(rows.loc[good,'longitude'],rows.loc[good,'latitude'],c=rows.loc[good,'value'],
        s=31,cmap=cmap,norm=norm,edgecolors='#45515a',linewidths=.35,zorder=3)
    if (~good).any():ax.scatter(rows.loc[~good,'longitude'],rows.loc[~good,'latitude'],marker='x',s=20,color=GREY,zorder=3)
    return image


def symmetric_norm(values,step):
    bound=step*np.ceil(np.nanmax(np.abs(values))/step)
    bound=max(step,bound);return TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound)


def main_maps(destination,export):
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    cells=pd.read_csv(HERE/'map_wheat_domain.csv');fields=pd.read_csv(HERE/'map_field_locations.csv')
    baseline=selected('level','1991-2020','F1_symptom_relative_anthesis_days')
    timing=selected('change','2071-2100','F1_symptom_relative_anthesis_days')
    transfer=selected('change','2071-2100','conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai')
    fig,axes=plt.subplots(2,2,figsize=(9.3,7.3));fig.subplots_adjust(left=.075,right=.97,bottom=.07,top=.92,wspace=.22,hspace=.34)
    fig.suptitle('European wheat area and climate-linked conditional production responses',fontsize=11.2,weight='bold',x=.075,ha='left')
    for ax in axes.flat:
        geography(ax,countries);ax.set_xlabel('');ax.set_xticklabels(['0°','20°E','40°E','60°E'])
    im=raster(axes[0,0],cells,'harvested_total_ha',cmap='YlOrBr',norm=LogNorm(vmin=10,vmax=40000))
    for partition,color,marker,label in [('calibration',BLUE,'^','BASF 2017–18'),
            ('reused_development_2019',GREEN,'s','BASF 2019'),('reused_external_strict',GREY,'o','Corteva')]:
        points=fields.loc[fields.partition.eq(partition)].drop_duplicates(['latitude','longitude'])
        axes[0,0].scatter(points.longitude,points.latitude,s=20,marker=marker,facecolors='none',
            edgecolors=color,linewidths=.9,label=label,zorder=4)
    axes[0,0].legend(frameon=False,fontsize=7.8,loc='lower right',handlelength=1.,labelspacing=.3)
    axes[0,0].set_title('a   Reference harvested area and field evidence',loc='left',fontsize=9.6,weight='bold')
    bar=fig.colorbar(im,ax=axes[0,0],orientation='horizontal',fraction=.042,pad=.13,shrink=.85)
    bar.set_label('Harvested hectares per 0.25° cell',fontsize=8.2);bar.set_ticks([10,100,1000,10000,40000]);bar.ax.tick_params(labelsize=7.7)
    base_norm=Normalize(vmin=np.floor(baseline.value.min()/5)*5,vmax=np.ceil(baseline.value.max()/5)*5)
    im=dots(axes[0,1],baseline,base_norm,'Blues')
    axes[0,1].set_title('b   Historical symptoms at sampled cells',loc='left',fontsize=9.6,weight='bold')
    bar=fig.colorbar(im,ax=axes[0,1],orientation='horizontal',fraction=.042,pad=.13,shrink=.85)
    bar.set_label('F1 symptoms relative to flowering (days)',fontsize=8.2);bar.ax.tick_params(labelsize=7.7)
    im=dots(axes[1,0],timing,symmetric_norm(timing.value.to_numpy(),2),DIVERGING)
    axes[1,0].set_title('c   Late-century symptom timing change',loc='left',fontsize=9.6,weight='bold')
    bar=fig.colorbar(im,ax=axes[1,0],orientation='horizontal',fraction=.042,pad=.13,shrink=.85)
    bar.set_label('Future − reference F1 timing (days)',fontsize=8.2);bar.ax.tick_params(labelsize=7.7)
    im=dots(axes[1,1],transfer,symmetric_norm(transfer.value.to_numpy(),.025),DIVERGING)
    axes[1,1].set_title('d   Change in conditional yield-loss proxy',loc='left',fontsize=9.6,weight='bold')
    bar=fig.colorbar(im,ax=axes[1,1],orientation='horizontal',fraction=.042,pad=.13,shrink=.85)
    bar.set_label('Change in t ha⁻¹ per nominal upper-three LAI',fontsize=8.2);bar.ax.tick_params(labelsize=7.7)
    export(fig,Path(destination),'fig6_european_spatial_results')
    return ('Figure 6 | European wheat production context and spatially explicit climate–infection diagnostics. '
        '(a) Fixed SPAM2020 harvested area on all14,941 wheat cells, with source locations for28 calibration,45 BASF2019 '
        'and143 location-disjoint Corteva field seasons. Coincident locations retain their true rounded coordinates. '
        'The logarithmic colour scale shows10–40,000 ha; smaller positive values use its lower endpoint. '
        '(b) Arithmetic three-climate-model mean F1 symptom timing relative to the flowering proxy in1991–2020, using the SSP5–8.5 reference continuation. '
        '(c,d) Paired2071–2100 minus1991–2020 SSP5–8.5 changes in symptom timing and the0.018-slope conditional yield transfer. '
        'Circles in b–d represent62 selected cells from64 registered area-proportional draws; duplicate draw locations are collapsed only for display. '
        'Colours retain the full displayed ranges. The point maps do not interpolate uncomputed cells or represent a completed current-model European census. '
        'Fixed reference area, predicted symptom timing and nominal-LAI yield-transfer units have separate meanings. '
        'The fixed harvested-area layer places sampled climate responses in the European wheat-production domain, while disease and yield panels remain conditional diagnostics because field disease and absolute-yield transfer are not validated. '
        'All SSPs and both future periods are retained in Figures S13–S14; earlier full-grid maps retain their archived identities in Figures S15–S18.')


def scenario_maps(destination,export,metric,stem,number,title,unit,step):
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    data=pd.read_csv(HERE/'map_ensemble_values.csv');data=data.loc[data.kind.eq('change')&data.metric.eq(metric)]
    norm=symmetric_norm(data.value.to_numpy(),step)
    fig,axes=plt.subplots(2,3,figsize=(10.4,6.7));fig.subplots_adjust(left=.065,right=.91,bottom=.075,top=.91,wspace=.14,hspace=.30)
    fig.suptitle(title,fontsize=11.2,weight='bold',x=.065,ha='left')
    for row,period in enumerate(['2031-2060','2071-2100']):
        for col,(scenario,label) in enumerate([('ssp126','SSP1–2.6'),('ssp245','SSP2–4.5'),('ssp585','SSP5–8.5')]):
            ax=axes[row,col];geography(ax,countries)
            part=data.loc[data.period.eq(period)&data.scenario.eq(scenario)]
            im=dots(ax,part,norm,DIVERGING)
            ax.set_title(f'{chr(97+row*3+col)}   {label}; {period}',loc='left',fontsize=9.3,weight='bold')
            if col>0:ax.set_ylabel('');ax.set_yticklabels([])
    cax=fig.add_axes([.935,.20,.016,.52]);bar=fig.colorbar(im,cax=cax);bar.set_label(unit,fontsize=9.);bar.ax.tick_params(labelsize=8.2)
    export(fig,Path(destination),stem)
    return (f'Figure S{number} | {title}. Each circle is a selected grid cell, with arithmetic means of three model-specific '
        'paired changes on valid relative-year coverage. The same62unique spatial cells and64draw identities underlie every panel; '
        'duplicates collapse for the display while area-weighted estimates retain all draw identities. Colours share one symmetric '
        f'zero-centred scale in {unit}; no spatial interpolation or missing-value zero filling is used. '
        'Reference periods follow their own SSP continuation. This is the frozen current-model sample, not an observed disease distribution '
        'or a full-grid climate census. Cell-level coverage, individual-model values and ranges remain in Source Data.')


def full_grid_example(destination,export):
    source=ROOT/'analysis/paper_study/nature_food_revision_20261007/continental_replay/annual_outputs/nasa/ACCESS-CM2/ssp585/2001.parquet'
    if not source.exists():return None
    data=pd.read_parquet(source);cells=pd.read_csv(HERE/'map_wheat_domain.csv')
    data=cells.merge(data.drop(columns=['latitude','longitude','harvested_total_ha']),on='cell_id',validate='one_to_one')
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    fig,axes=plt.subplots(1,2,figsize=(9.,4.4));fig.subplots_adjust(left=.065,right=.97,bottom=.20,top=.88,wspace=.18)
    fig.suptitle('Completed full-grid example: 2001 harvest',fontsize=11.2,weight='bold',x=.065,ha='left')
    for ax,column,label,cmap in zip(axes,['F1_symptom_relative_anthesis_days','GS65_85_lost_had3'],
            ['F1 symptoms relative to flowering (days)','Lost GLAI-days per nominal upper-three LAI'],['Blues','YlOrBr']):
        geography(ax,countries);data[column]=data[column].where(data.valid_complete_season)
        im=raster(ax,data,column,cmap=cmap,vmin=np.nanmin(data[column]),vmax=np.nanmax(data[column]))
        bar=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.15,shrink=.9);bar.set_label(label,fontsize=8.3);bar.ax.tick_params(labelsize=8.)
    axes[0].set_title('a   Current frozen-model symptom timing',loc='left',fontsize=9.8,weight='bold')
    axes[1].set_title('b   Model functional-area deficit',loc='left',fontsize=9.8,weight='bold')
    export(fig,Path(destination),'figS19_full_grid_example_2001')
    return ('Figure S19 | Completed full-European example for the 2001 harvest under historically aligned ACCESS-CM2 forcing. '
        'All 14,941 reference wheat cells remain in the output; 14,932 have an eligible winter/rainfed calendar, 14,772 complete the modeled season, '
        '160 do not reach BBCH 85 and 9 lack the selected crop calendar. Invalid outcomes are not set to zero. '
        'This single-year replay reproduces 58 numeric and 10 date metrics of the previous 64-draw run at matching cells, with unchanged biology. '
        'The maps demonstrate actual full-grid coverage for that year; they do not constitute a 30-year reference or future-period projection, '
        'independent epidemic validation or actual field-yield loss.')


def supplementary(destination,export):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    captions={}
    captions['figS13_spatial_symptom_changes']=scenario_maps(destination,export,'F1_symptom_relative_anthesis_days',
        'figS13_spatial_symptom_changes',13,'Spatial changes in flag-leaf symptom timing','Future − reference timing (days)',2)
    captions['figS14_spatial_yield_transfers']=scenario_maps(destination,export,'conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai',
        'figS14_spatial_yield_transfers',14,'Spatial changes in conditional yield transfer','Change in t ha⁻¹ per nominal upper-three LAI',.025)
    for record in json.loads((HERE/'archived_map_registry.json').read_text()):
        stem=record['figure'];number=int(stem.split('_')[0].replace('figS',''))
        for suffix,key in [('png','source_png'),('pdf','source_pdf')]:shutil.copy2(ROOT/record[key],destination/f'{stem}.{suffix}')
        captions[stem]=(f'Figure S{number} | Archived spatial evidence from the earlier seasonal-v1 study. '
            'The original figure is reproduced unchanged and retains its source definitions. It is separate from the current seasonal-source '
            'model and empirical yield-response evaluations. '+record['original_caption'].split(' | ',1)[1])
    caption=full_grid_example(destination,export)
    if caption:captions['figS19_full_grid_example_2001']=caption
    return captions
