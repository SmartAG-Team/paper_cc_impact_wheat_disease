"""Publication maps and environmental comparisons from completed simulations."""
import shutil
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from analysis.paper_study.map_restoration_20261007.figures import geography, raster, DIVERGING, symmetric_norm
from .spatial import ROOT, HERE, DEST, PERIODS, SCENARIOS, METRICS, MAJOR_REGIONS

BLUE='#235e83'; GREEN='#52796f'; GOLD='#a17937'; GREY='#737d85'
SCENARIO_COLORS={'ssp126':BLUE,'ssp245':GREEN,'ssp585':GOLD}
LABELS={'ssp126':'SSP1–2.6','ssp245':'SSP2–4.5','ssp585':'SSP5–8.5'}


def main_map(destination, export):
    data=pd.read_csv(DEST/'sampled_cell_scenario_changes.csv')
    data=data[data.metric.eq(METRICS[0])]
    wheat=pd.read_csv(HERE/'spatial_sources/wheat_cells_eea_regions.csv')
    wheat['mask']=1.
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    norm=symmetric_norm(data.mean_change.to_numpy(),2)
    fig,axes=plt.subplots(2,3,figsize=(10.5,6.6))
    fig.subplots_adjust(left=.055,right=.985,top=.925,bottom=.21,wspace=.12,hspace=.26)
    for row,period in enumerate(PERIODS):
        for col,scenario in enumerate(SCENARIOS):
            ax=axes[row,col]; geography(ax,countries)
            raster(ax,wheat,'mask',cmap=ListedColormap(['#dce1e4']),vmin=0,vmax=1)
            part=data[data.period.eq(period)&data.scenario.eq(scenario)]
            agree=part.positive_gcm_count.eq(3)|part.negative_gcm_count.eq(3)
            for mask,marker in [(agree,'s'),(~agree,'D')]:
                good=part[mask&part.complete_gcm_count.eq(3)]
                image=ax.scatter(good.longitude,good.latitude,c=good.mean_change,cmap=DIVERGING,norm=norm,
                    s=20 if marker=='s' else 17,marker=marker,edgecolors='#43515b',linewidths=.4,zorder=3)
            ax.set_title(f'{chr(97+row*3+col)}   {LABELS[scenario]}; {period.replace("-","–")}',loc='left',fontsize=9.2,fontweight='bold',pad=6)
            ax.set_xlabel('');ax.set_ylabel('Latitude (°N)' if col==0 else '')
            if col:ax.set_yticklabels([])
            if row==0:ax.set_xticklabels([])
            else:ax.set_xticklabels(['0°','20°E','40°E','60°E'])
    barax=fig.add_axes([.24,.12,.53,.025])
    bar=fig.colorbar(image,cax=barax,orientation='horizontal')
    bar.ax.tick_params(labelsize=8.5)
    bar.set_label('Change in HAD deficit (days per reference upper-three-leaf LAI)',fontsize=9)
    handles=[Line2D([],[],marker='s',color='#43515b',markerfacecolor='white',linestyle='none',markersize=5,label='Same sign in all three GCMs'),
             Line2D([],[],marker='D',color='#43515b',markerfacecolor='white',linestyle='none',markersize=4.5,label='GCM signs not unanimous'),
             Patch(facecolor='#dce1e4',edgecolor='none',label='Reference wheat cells')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.51,.002),ncol=3,frameon=False,fontsize=8.3,handletextpad=.5,columnspacing=1.8)
    export(fig,destination,'fig3_spatial_scenario_response')
    data.to_csv(destination/'fig3_sampled_cell_canopy_changes.csv',index=False)
    shutil.copy2(DEST/'canopy_change_sign_area_shares.csv',destination/'fig3_canopy_sign_area_shares.csv')
    return ('Figure 3 | Spatial variation in modeled wheat canopy damage under three emissions pathways. '
        '(a–f) Three-climate-model mean paired changes in upper-three-leaf healthy-area-duration (HAD) deficit for 2031–2060 and 2071–2100 relative to each SSP-specific 1991–2020 reference. '
        'Colors share one zero-centered scale and retain the full range. Squares identify cells with the same change sign in all three GCMs; diamonds identify non-unanimous signs. '
        'Markers locate 62 actually simulated 0.25° wheat cells and are enlarged for legibility. Gray pixels show the reference all-wheat area mask; they have no assigned future response. '
        'Duplicate draw locations collapse for display, while all 64 draw identities remain in area-weighted estimates. '
        'Positive values indicate greater modeled canopy deficit. These sampled-cell responses are conditional on the imposed crop calendar and fixed management. '
        'GCM sign agreement is deterministic scenario consistency, not a probability of future damage. Environmental-domain estimates and coverage are reported in Supplementary Fig. S19 and Tables S14a–e.')


def environmental_comparison(destination, export):
    context=pd.read_csv(DEST/'region_sampling_context.csv').set_index('environment_region')
    changes=pd.read_csv(DEST/'region_paired_changes.csv')
    decomposition=pd.read_csv(DEST/'region_supported_decomposition.csv').set_index('environment_region')
    cells=pd.read_csv(HERE/'spatial_sources/wheat_cells_eea_regions.csv')
    draws=pd.read_csv(DEST/'sample_region_membership.csv').drop_duplicates('cell_id')
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    fig,axes=plt.subplots(2,2,figsize=(10.7,8.0))
    fig.subplots_adjust(left=.11,right=.98,bottom=.13,top=.94,wspace=.45,hspace=.40)
    region_colors=[BLUE,GOLD,GREEN,'#ad7188','#7d8050']
    colors=ListedColormap(region_colors+['#d6dbdf'])
    cells['region_index']=cells.environment_region.map({name:i for i,name in enumerate(MAJOR_REGIONS)}).fillna(5)
    ax=axes[0,0];geography(ax,countries)
    raster(ax,cells,'region_index',cmap=colors,vmin=-.5,vmax=5.5)
    ax.scatter(draws.longitude,draws.latitude,s=10,facecolors='none',edgecolors='#26343e',linewidths=.45,zorder=3)
    ax.set_xlabel('');ax.set_ylabel('');ax.set_title('a   EEA regions in the wheat domain',loc='left',fontsize=10,fontweight='bold')
    handles=[Patch(facecolor=color,label=name) for color,name in zip(region_colors,MAJOR_REGIONS)]
    handles.append(Patch(facecolor='#d6dbdf',label='Other / unassigned'))
    ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,-.11),ncol=3,frameon=False,fontsize=7.8,
              columnspacing=.8,handlelength=1.1,handletextpad=.4)
    for ax,metric,panel,label in [(axes[0,1],METRICS[0],'b','HAD deficit change (days / reference LAI)'),
                                 (axes[1,0],METRICS[1],'c','Symptom timing change relative to flowering (days)')]:
        y=np.arange(len(MAJOR_REGIONS))
        for index,scenario in enumerate(SCENARIOS):
            part=changes[changes.scenario.eq(scenario)&changes.period.eq('2071-2100')&changes.metric.eq(metric)].set_index('environment_region').loc[MAJOR_REGIONS]
            ax.errorbar(part.mean_change,y+(index-1)*.20,xerr=part.spatial_mcse,color=SCENARIO_COLORS[scenario],
                        marker=['o','s','^'][index],linestyle='none',markersize=4.5,capsize=2,label=LABELS[scenario])
        ax.axvline(0,color=GREY,lw=.8,ls='--');ax.grid(axis='x',color='#e8ecee',lw=.65)
        ax.set_yticks(y,[f'{region} (n={int(context.loc[region,"sample_draws"])})' for region in MAJOR_REGIONS],fontsize=8.2)
        ax.invert_yaxis();ax.tick_params(axis='x',labelsize=8.5);ax.set_xlabel(label,fontsize=8.6)
        ax.spines[['top','right']].set_visible(False)
        ax.set_title(panel+'   Late-century regional response',loc='left',fontsize=10,fontweight='bold')
    axes[0,1].legend(frameon=False,fontsize=8,loc='lower right')
    ax=axes[1,1];y=np.arange(len(MAJOR_REGIONS))
    for i,(component,label,color) in enumerate([('weather','Disease weather',BLUE),('host','Host development',GREEN),('net','Net change',GOLD)]):
        rows=decomposition.loc[MAJOR_REGIONS]
        ax.errorbar(rows[component],y+(i-1)*.19,xerr=rows[component+'_spatial_mcse'],color=color,marker=['o','s','D'][i],
                    markersize=4.5,linestyle='none',capsize=2,label=label)
    ax.set_yticks(y,MAJOR_REGIONS,fontsize=8.2);ax.invert_yaxis()
    ax.axvline(0,color=GREY,lw=.8,ls='--');ax.grid(axis='x',color='#e8ecee',lw=.65)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(axis='x',labelsize=8.5)
    ax.set_xlabel('HAD contribution (days / reference LAI)',fontsize=8.6)
    ax.set_title('d   SSP5–8.5 weather–host contributions',loc='left',fontsize=10,fontweight='bold')
    ax.legend(frameon=False,fontsize=7.8,loc='lower right')
    fig.text(.11,.025,'b–d: markers show three-model means; whiskers show ±1 spatial Monte Carlo SE.\nGCM ranges are retained separately in Tables S14b–e; these intervals exclude model-fitting and biological uncertainty.',fontsize=8.1,color=GREY)
    export(fig,destination,'figS19_environmental_region_responses')
    for name in ['region_sampling_context','region_paired_changes','region_supported_decomposition']:
        shutil.copy2(DEST/f'{name}.csv',destination/f'figS19_{name}.csv')
    return ('Figure S19 | Environmental-region differences in sampled European wheat responses. '
        '(a) Official EEA 2016 region assignments at reference wheat-cell centroids, with sampled cells outlined. '
        '(b,c) Late-century paired changes in modeled canopy HAD deficit and flag-leaf symptom timing under three SSPs. '
        '(d) Supported-forcing weather and host Shapley contributions and their net under late-century SSP5–8.5. '
        'The five named regions have at least four registered draws; n counts draws, including duplicate locations. '
        'Point estimates retain original design weights and are conditional domain ratios, rather than equally weighted cell means. '
        'Whiskers are ±1 spatial Monte Carlo standard error and retain covariance from shared draws across GCMs; they do not measure comprehensive projection uncertainty. '
        'Sparse and unsampled domains retain their coverage and available estimates in Tables S14a–e and source data. '
        'All-wheat reference area with an imposed winter/rainfed calendar defines the population; response maps and regional summaries do not establish actual yield loss or adaptation efficacy.')
