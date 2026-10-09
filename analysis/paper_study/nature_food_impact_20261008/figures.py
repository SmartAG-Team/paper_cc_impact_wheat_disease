"""Publication figures of completed grid impacts and pooled evaluation evidence."""
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, Normalize, LogNorm, BoundaryNorm
from matplotlib.lines import Line2D
import geopandas as gpd
from .data import (ROOT,HERE,GRID,DERIVED,CANOPY,SEVERITY,ONSET,FREQUENCY,YIELD,
                   SCENARIOS,REGIONS,pooled,grid_data)
from analysis.paper_study.map_restoration_20261007.figures import geography,raster,symmetric_norm,DIVERGING

BLUE='#235e83';GREEN='#52796f';GOLD='#a17937';GREY='#737d85'
COLORS=[BLUE,GREEN,GOLD]
LABELS=['SSP1–2.6','SSP2–4.5','SSP5–8.5']
STAGES={10:'Emergence',31:'First node',32:'Second node',33:'Third node',37:'Flag-leaf appearance',
        39:'Flag-leaf unfolding',51:'Heading',65:'Flowering',85:'Soft dough'}


def export(fig,destination,stem):
    destination.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','pdf','svg']:
        fig.savefig(destination/f'{stem}.{suffix}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def style():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
        'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})


def clean(ax):
    ax.grid(axis='x',color='#e8ecee',lw=.65);ax.set_axisbelow(True);ax.tick_params(labelsize=8.5)


def map_axes(ax,countries,cells,mask_color='#dce1e4'):
    geography(ax,countries)
    mask=cells.copy();mask['available_wheat']=1.
    raster(ax,mask,'available_wheat',cmap=ListedColormap([mask_color]),vmin=0,vmax=1)
    ax.set_xlabel('');ax.set_ylabel('');ax.set_xticklabels(['0°','20°E','40°E','60°E'])


def discrete_scale(values,breaks,step):
    bound=max(breaks[0],step*np.ceil(np.nanmax(np.abs(values))/step))
    positive=sorted(set([v for v in breaks if v<bound]+[bound]))
    boundaries=np.r_[-np.array(positive)[::-1],positive]
    colors=np.array([.5]) if len(boundaries)==2 else np.linspace(0,1,len(boundaries)-1)
    cmap=ListedColormap(DIVERGING(colors))
    ticks=sorted(set([positive[len(positive)//2],positive[-1]]))
    return BoundaryNorm(boundaries,cmap.N),cmap,[-v for v in ticks[::-1]]+[0]+ticks


def onset_change_scale(values):
    finite=np.asarray(values,float)
    finite=finite[np.isfinite(finite)]
    bound=max(10.,10.*np.ceil(np.max(np.abs(finite))/10.)) if len(finite) else 10.
    return Normalize(vmin=-bound,vmax=bound),DIVERGING,np.linspace(-bound,bound,5)


def scenario_map(destination,data,countries,cells,metric,stem,unit,scale=1.,periods=None):
    selected=data[data.metric.eq(metric)].copy();selected['display']=scale*selected.mean_change
    step=2 if scale==1 else (1 if scale==100 else 25)
    breaks=[10,25,50,100,200] if abs(scale)==1000 else [.5,1,2,5,10]
    norm,cmap,ticks=discrete_scale(selected.display.to_numpy(),breaks,step)
    periods=sorted(selected.period.unique()) if periods is None else periods
    fig,axes=plt.subplots(len(periods),3,figsize=(10.2,6.2 if len(periods)==2 else 3.8),squeeze=False)
    fig.subplots_adjust(left=.05,right=.985,top=.95,bottom=.15,wspace=.12,hspace=.26)
    for r,period in enumerate(periods):
        for c,scenario in enumerate(SCENARIOS):
            ax=axes[r,c];map_axes(ax,countries,cells)
            part=selected[selected.scenario.eq(scenario)&selected.period.eq(period)]
            assert len(part)==14941
            im=raster(ax,part,'display',cmap=cmap,norm=norm)
            ax.set_title(f'{chr(97+r*3+c)}   {LABELS[c]} · {period.replace("-","–")}',
                         loc='left',fontsize=9.4,fontweight='bold',pad=7)
            if c:ax.set_yticklabels([])
            if r<len(periods)-1:ax.set_xticklabels([])
    cb=fig.colorbar(im,cax=fig.add_axes([.25,.065,.53,.024]),orientation='horizontal')
    cb.set_ticks(ticks)
    cb.set_label(unit,fontsize=9);cb.ax.tick_params(labelsize=8.5)
    export(fig,destination,stem)
    selected.to_csv(destination/f'{stem}.csv.gz',index=False)


def disease_map(destination,data,countries,cells,period='2071-2100'):
    specs=[(FREQUENCY,100,'a   Symptom frequency','Change in symptom frequency (percentage points)'),
           (ONSET,1,'b   Symptom timing','Change relative to flowering (days)'),
           (SEVERITY,100,'c   Relative HAD loss','Change in relative HAD loss (percentage points)'),
           ('F1_symptom_day_after_sowing',1,'d   Symptom onset after sowing','Change in days from sowing to symptoms')]
    fig,axes=plt.subplots(2,2,figsize=(8.6,7.0))
    fig.subplots_adjust(left=.055,right=.975,top=.94,bottom=.08,wspace=.26,hspace=.36)
    sources=[]
    for ax,(metric,scale,title,unit) in zip(axes.flat,specs):
        part=data[data.metric.eq(metric)&data.scenario.eq('ssp585')&data.period.eq(period)].copy()
        part['display']=scale*part.mean_change;sources.append(part)
        map_axes(ax,countries,cells)
        step=.1 if metric==FREQUENCY else (2 if scale==1 else 5)
        breaks=[.1,1,5,10] if metric==FREQUENCY else [.5,1,2,5,10,20]
        if metric=='F1_symptom_day_after_sowing':
            norm,cmap,ticks=onset_change_scale(part.display.to_numpy())
        else:
            norm,cmap,ticks=discrete_scale(part.display.to_numpy(),breaks,step)
        im=raster(ax,part,'display',cmap=cmap,norm=norm)
        ax.set_title(title,loc='left',fontsize=10,fontweight='bold',pad=9)
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.14,shrink=.93)
        cb.set_ticks(ticks)
        cb.set_label(unit,fontsize=8.1);cb.ax.tick_params(labelsize=8)
    export(fig,destination,'fig2_disease_frequency_timing_severity')
    pd.concat(sources).to_csv(destination/'fig2_disease_frequency_timing_severity.csv.gz',index=False)


def domain_points(ax,data,domain,names,metric,scale=1.,period='2071-2100'):
    y=np.arange(len(names))
    for i,(scenario,color,label) in enumerate(zip(SCENARIOS,COLORS,LABELS)):
        part=data[data.scenario.eq(scenario)&data.period.eq(period)&data.metric.eq(metric)].set_index(domain).loc[names]
        mid=scale*part.mean_change;lo=np.minimum(scale*part.gcm_min,scale*part.gcm_max)
        hi=np.maximum(scale*part.gcm_min,scale*part.gcm_max)
        ax.errorbar(mid,y+(i-1)*.19,xerr=[mid-lo,hi-mid],color=color,marker=['o','s','^'][i],
                    linestyle='none',markersize=4.5,capsize=2,label=label)
    ax.set_yticks(y,[n.replace('United Kingdom of Great Britain and Northern Ireland','United Kingdom') for n in names],fontsize=8.5)
    ax.invert_yaxis();ax.axvline(0,color=GREY,lw=.8,ls='--');clean(ax)


def domain_figure(destination,regions,countries,period='2071-2100'):
    refs=countries.drop_duplicates('country').sort_values('reference_area_ha',ascending=False)
    names=refs.head(12).country.tolist()
    fig,axes=plt.subplots(1,2,figsize=(10.3,6.2),gridspec_kw={'width_ratios':[1,1.13]})
    fig.subplots_adjust(left=.11,right=.98,bottom=.15,top=.87,wspace=.56)
    domain_points(axes[0],regions,'environment_region',REGIONS,CANOPY,period=period)
    domain_points(axes[1],countries,'country',names,YIELD,scale=-1000,period=period)
    axes[0].set_title('a   Canopy damage across environmental regions',loc='left',fontsize=10,fontweight='bold',pad=12)
    axes[1].set_title('b   Estimated yield change by country',loc='left',fontsize=10,fontweight='bold',pad=12)
    axes[0].set_xlabel('Change in normalized HAD loss (days)',fontsize=9)
    axes[1].set_xlabel('Estimated yield change (kg ha⁻¹ per unit reference LAI)',fontsize=8.9)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.52,.995),ncol=3,frameon=False,fontsize=9)
    export(fig,destination,'fig3_regional_and_country_impacts')
    pd.concat([regions.assign(domain_name=regions.environment_region),
               countries.assign(domain_name=countries.country)]).to_csv(destination/'fig3_regional_and_country_impacts.csv.gz',index=False)


def contribution_figure(destination,stem='fig4_weather_and_crop_contributions',full=False):
    source=ROOT/'analysis/paper_study/nature_food_fix_20261007/climate/decomposition/supported_forcing_ensemble_decomposition.csv'
    data=pd.read_csv(source).set_index('component')
    offsets=pd.read_csv(ROOT/'analysis/paper_study/nature_food_submission_20261008/derived/weather_host_offset.csv')
    components=['weather_shapley','host_shapley','total_change']+(['interaction'] if full else [])
    labels=['Weather effect','Wheat phenology','Net change']+(['Weather × phenology'] if full else [])
    fig,axes=plt.subplots(1,2,figsize=(9.3,4.5),gridspec_kw={'width_ratios':[1,1]})
    fig.subplots_adjust(left=.15,right=.96,bottom=.21,top=.88,wspace=.59)
    for y,key,color in zip(range(len(components)),components,[BLUE,GREEN,GOLD,GREY]):
        row=data.loc[key]
        axes[0].plot([row.gcm_min,row.gcm_max],[y,y],color='#b1b6ba',lw=2.4)
        axes[0].errorbar(row.gcm_mean,y,xerr=row.spatial_mcse_gcm_mean,fmt='o',color=color,capsize=3,ms=6)
        axes[0].annotate(f'{row.gcm_mean:+.2f}',(row.gcm_mean,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9,color=color)
    axes[0].set_yticks(range(len(labels)),labels,fontsize=9);axes[0].invert_yaxis()
    axes[0].axvline(0,color=GREY,ls='--',lw=.8);clean(axes[0])
    axes[0].set_xlabel('Contribution to normalized HAD loss (days)',fontsize=8.8)
    axes[0].set_title('a   Weather and wheat-phenology effects',loc='left',fontsize=10,fontweight='bold',pad=14)
    part=offsets[offsets.climate_model.ne('Three-model ensemble')]
    axes[1].barh(range(3),part.host_offset_percent,color=GREEN,height=.52)
    axes[1].set_yticks(range(3),part.climate_model,fontsize=8.4);axes[1].invert_yaxis()
    for y,value in enumerate(part.host_offset_percent):axes[1].text(value+1.5,y,f'{value:.1f}%',va='center',fontsize=9)
    ensemble=float(offsets[offsets.climate_model.eq('Three-model ensemble')].host_offset_percent.iloc[0])
    axes[1].axvline(ensemble,color=GREY,ls='--',lw=.9);axes[1].set_xlim(0,103)
    axes[1].set_xlabel('Weather contribution offset by wheat phenology (%)',fontsize=8.7)
    axes[1].set_title('b   Offset from wheat phenology',loc='left',fontsize=10,fontweight='bold',pad=14);clean(axes[1])
    axes[1].text(.02,-.22,f'Ensemble: {ensemble:.1f}%',transform=axes[1].transAxes,fontsize=9,color=GREEN)
    export(fig,destination,stem)
    data.reset_index().to_csv(destination/f'{stem}_components.csv',index=False)
    offsets.to_csv(destination/f'{stem}_offsets.csv',index=False)


def main(destination):
    style();data,regions,country=grid_data()
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    cells=data.drop_duplicates('cell_id')
    scenario_map(destination,data,countries,cells,CANOPY,'fig1_grid_climate_canopy_impacts',
                 'Change in normalized HAD loss (days)')
    disease_map(destination,data,countries,cells)
    domain_figure(destination,regions,country)
    contribution_figure(destination)
    captions={
      'fig1_grid_climate_canopy_impacts': 'Figure 1 | Projected changes in Septoria-related canopy damage across European wheat-growing areas. (a–f) Three-climate-model mean changes in normalized loss of healthy-area duration (HAD) during flowering to soft dough for two future periods relative to 1991–2020. HAD loss is divided by maximum reference upper-three-leaf LAI and expressed in days. Each 0.25° pixel represents a separately simulated wheat-growing cell. Positive values indicate increasing accumulated canopy damage; negative values indicate decreasing damage. The six maps share a symmetric colour scale. The fixed SPAM2020 all-wheat mask contains 14,941 cells, of which 14,932 have a winter-wheat rainfed calendar. Grey wheat pixels have unavailable estimates. Country and regional summaries use fixed harvested-area weights. Supplementary Fig. S19 shows agreement among climate models.',
      'fig2_disease_frequency_timing_severity': 'Figure 2 | Projected changes in Septoria symptoms and canopy damage. Panels show three-model mean changes under SSP5–8.5 in 2071–2100 relative to 1991–2020. (a) Frequency of seasons with flag-leaf symptoms before soft dough. (b) First symptoms relative to flowering; negative changes indicate earlier symptoms. (c) Relative HAD loss, the fraction of reference healthy-area duration lost during grain filling, expressed as a percentage-point change. This modelled loss of canopy function differs from measured lesion percentage. (d) Days from the fixed sowing date to first symptoms; negative changes indicate a shorter interval. Each panel has its own symmetric colour scale; panel d uses a continuous linear scale spanning all finite changes. Timing requires symptoms in both paired seasons, whereas frequency includes complete seasons without symptoms. Grey wheat pixels have unavailable estimates.',
      'fig3_regional_and_country_impacts': 'Figure 3 | Regional differences in canopy damage and inferred yield response. (a) Harvested-area-weighted changes in normalized HAD loss within eight EEA biogeographical regions. (b) Estimated disease-related yield changes for the twelve countries with the largest reference wheat area within the study domain. Both panels compare 2071–2100 with 1991–2020 under three emissions pathways. Points show three-model means; whiskers span model-specific estimates and are not confidence intervals. Country assignment follows the dominant SPAM source-country label of each cell. The yield estimate uses the change in HAD loss and the fixed coefficient 0.018 t ha⁻¹ per GLAI-day, normalized to unit reference upper-canopy LAI. Negative values indicate increased estimated disease-related yield loss. These estimates require validation of the canopy–yield relationship and do not describe total national production. Source Data includes all country groups.',
      'fig4_weather_and_crop_contributions': 'Figure 4 | Wheat phenology offsets part of the weather contribution to Septoria damage. (a) Weather and wheat-phenology contributions to normalized HAD loss during grain filling, and their sum, under late-century SSP5–8.5. The phenology contribution includes changes in upper-leaf appearance and unfolding and the flowering-to-soft-dough interval. The analysis uses 64 harvested-area-proportional draws at 62 cells and 5,731 valid season pairs. Grey segments span climate-model means; coloured whiskers show one spatial Monte Carlo standard error. Each Shapley contribution includes half of the weather–phenology interaction, shown separately in Supplementary Fig. S16. (b) Percentage of the weather contribution offset by wheat phenology in each climate model. The dashed line shows the ratio of ensemble contributions, 70.6%. These comparisons separate effects within the model; they do not estimate the effectiveness of changing sowing dates or cultivars.',
    }
    (destination/'captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    return captions


def field_panels(destination,exporter,stages,stem,cohort,number):
    d=pd.read_csv(DERIVED/'pooled_field_stage_records.csv')
    d=d[d.cohort.eq(cohort)&d.genuinely_bracketed].copy()
    for c in ['lower_exclusive','upper_inclusive','predicted_date']:d[c]=pd.to_datetime(d[c])
    anchor=pd.to_datetime(d.season_year.astype(str)+'-01-01')
    d['lo']=(d.lower_exclusive-anchor).dt.days+2;d['hi']=(d.upper_inclusive-anchor).dt.days+1
    d['predicted']=(d.predicted_date-anchor).dt.days+1
    fig,axes=plt.subplots(2,2,figsize=(8.1,7.8))
    fig.subplots_adjust(left=.10,right=.98,bottom=.10,top=.93,wspace=.29,hspace=.45)
    for ax,stage,letter in zip(axes.flat,stages,'abcd'):
        part=d[d.event.eq(stage)]
        lo=5*np.floor((min(part.lo.min(),part.predicted.min())-3)/5) if len(part) else 100
        hi=5*np.ceil((max(part.hi.max(),part.predicted.max())+3)/5) if len(part) else 180
        ax.plot([lo,hi],[lo,hi],ls='--',lw=.8,color=GREY)
        if len(part):
            ax.hlines(part.predicted,part.lo,part.hi,color=BLUE,lw=.8,alpha=.32)
            ax.scatter((part.lo+part.hi)/2,part.predicted,s=18,color=BLUE,edgecolors='white',lw=.45,zorder=3)
            ax.text(.03,.96,f'{int(part.compatible.sum())}/{len(part)} within interval',transform=ax.transAxes,va='top',fontsize=8)
        else:ax.text(.5,.5,'No two-sided observations',transform=ax.transAxes,ha='center',fontsize=9)
        ax.set(xlim=(lo,hi),ylim=(lo,hi),xlabel='Observed interval (day of year)',ylabel='Predicted date (day of year)')
        ax.set_aspect('equal');ax.set_title(f'{letter}   {STAGES[stage]} (BBCH {stage})',loc='left',fontsize=9.7,fontweight='bold')
        ax.tick_params(labelsize=8.5);ax.xaxis.label.set_size(8.6);ax.yaxis.label.set_size(8.6);ax.grid(color='#ededed',lw=.6)
    exporter(fig,destination,stem)
    return (f'Figure S{number} | {cohort} of wheat phenological stage dates. '
            'Horizontal segments show observed two-sided stage intervals; their centers locate markers and are not treated as exact event dates. '
            'The dashed diagonal denotes equal dates, and an interval intersecting it contains the prediction. '
            'Counts give compatible intervals among all informative records. Observations from both field archives are pooled; calibration and evaluation remain separate. '
            'Intervals express observation resolution rather than statistical confidence.')


def validation_supplementary(destination,exporter):
    return {stem:field_panels(destination,exporter,stages,stem,cohort,number) for stages,stem,cohort,number in [
      ([31,32,33,37],'figS1_field_early_stages','Field evaluation',1),
      ([39,51,65,85],'figS2_field_late_stages','Field evaluation',2),
      ([31,32,33,37],'figS3_calibration_early_stages','Calibration',3),
      ([39,51,65,85],'figS4_calibration_late_stages','Calibration',4)]}


def yield_supplementary(destination,exporter):
    from analysis.paper_study.yield_evidence_20261007 import figures as previous
    def polish(fig,folder,stem):
        replacements={'Held-background prediction':'Predicted yield loss','Relative transfer':'Relative yield loss',
          'Absolute transfer':'Absolute yield loss','Transferred centre: 0.018':'Reference coefficient: 0.018',
          'Published protected–unprotected yield contrasts':'Yield differences between protected and unprotected wheat'}
        for ax in fig.axes:
            legend=ax.get_legend()
            artists=[*ax.texts,ax.xaxis.label,ax.yaxis.label]
            if legend:artists.extend(legend.get_texts())
            for artist in artists:
                value=artist.get_text()
                for old,new in replacements.items():value=value.replace(old,new)
                artist.set_text(value)
            for loc in ['left','center','right']:
                value=ax.get_title(loc=loc)
                for old,new in replacements.items():value=value.replace(old,new)
                if stem in ['figS5_german_event_dates','figS7_stage_identification_profiles']:
                    import re
                    value=re.sub(r'BBCH(\d+)',lambda m:STAGES[int(m.group(1))]+f' (BBCH {m.group(1)})',value)
                ax.set_title(value,loc=loc)
        if stem=='figS11_published_yield_group_holdout':
            for ax,kind,unit in zip(fig.axes,['relative','absolute'],['%','t ha⁻¹']):
                ax.set_xlabel(f'Published {kind} yield loss ({unit})',fontsize=9)
                ax.set_ylabel(f'Predicted {kind} yield loss ({unit})',fontsize=9)
        exporter(fig,folder,stem)
    functions=[('figS5_german_event_dates',previous.german_dates),
      ('figS7_stage_identification_profiles',previous.stage_profiles),
      ('figS9_published_STB_yield_contrasts',previous.parker_yields),
      ('figS10_top3_yield_response_coefficients',previous.parker_slopes),
      ('figS11_published_yield_group_holdout',previous.group_benchmark)]
    captions={stem:f(destination,polish) for stem,f in functions}
    onset=pd.read_csv(DERIVED/'pooled_symptom_onset_membership.csv')
    fig,ax=plt.subplots(figsize=(7.3,4.3));fig.subplots_adjust(left=.10,right=.98,bottom=.20,top=.81)
    for leaf in range(3):
        p=onset[onset.leaf_index.eq(leaf)].sort_values('endpoint_series')
        shifts=np.linspace(-.09,.09,len(p))
        for col,offset,marker,color,label in [('model',-.18,'s',BLUE,'Seasonal model'),('benchmark',.18,'o',GREY,'Phenology benchmark')]:
            ax.scatter(leaf+offset+shifts,p[col],s=20,color=color,marker=marker,alpha=.55,label=label if leaf==0 else None)
    counts=onset.groupby('leaf_index').size()
    ax.set_xticks(range(3),[f'Source leaf {i+1}\nn = {counts.get(i,0)}' for i in range(3)])
    ax.set_ylabel('Distance outside symptom-onset interval (days)',fontsize=9)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.07),ncol=2,frameon=False);ax.grid(axis='y',color='#e8ecee',lw=.65)
    exporter(fig,destination,'figS6_leaf_onset_distances')
    onset.to_csv(destination/'figS6_leaf_onset_distances.csv',index=False)
    captions['figS6_leaf_onset_distances']=('Figure S6 | Pooled symptom-onset errors for source-numbered leaves 1–3. '
      'All 136 genuinely two-sided histories remain represented. Horizontal displacement separates coincident records. '
      'Zero indicates a predicted date within the observation interval; missing positive predictions retain their predefined penalty. '
      'Source leaf numbers do not establish a common final-rank identity across archives. Scores use equal coordinate-year, field and leaf weighting.')
    replay=pd.read_csv(ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/manuscript/replay/full_season_top3_timing_and_conditional_yield.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.7,4.4));fig.subplots_adjust(left=.09,right=.98,bottom=.18,top=.88,wspace=.33)
    axes[0].scatter(replay.F1_symptom_relative_anthesis_days,100*replay.model_proxy_had_loss_fraction,color=BLUE,s=19,alpha=.55)
    axes[1].scatter(replay.grain_fill_days,replay.model_proxy_lost_had3,color=BLUE,s=19,alpha=.55)
    axes[0].set(xlabel='Flag-leaf symptoms relative to flowering (days)',ylabel='Relative HAD loss (%)')
    axes[1].set(xlabel='Flowering to soft dough (days)',ylabel='Normalized HAD loss (days)')
    for ax,t in zip(axes,['a   Disease timing and canopy damage','b   Grain-fill duration and canopy damage']):
        ax.set_title(t,loc='left',fontsize=9.7,fontweight='bold');clean(ax)
        ax.xaxis.label.set_size(8.6);ax.yaxis.label.set_size(8.6)
    exporter(fig,destination,'figS8_field_grainfill_exposure')
    captions['figS8_field_grainfill_exposure']=('Figure S8 | Simulated disease timing and grain-fill canopy damage in field weather replays. '
      'Each point represents one of 218 full-season simulations. (a) Flag-leaf symptoms relative to flowering and relative HAD loss. '
      '(b) Flowering-to-soft-dough duration and HAD deficit. Reference upper-three-leaf LAI is nominally one. '
      'Both axes describe simulated quantities; these relationships do not validate observed disease severity or grain-yield loss.')
    return captions


def map_supplementary(destination,exporter):
    data,regions,country=grid_data();cells=data.drop_duplicates('cell_id')
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    captions={}
    for metric,stem,num,unit,scale in [(ONSET,'figS13_spatial_symptom_changes',13,'Symptom timing change relative to flowering (days)',1),
          (YIELD,'figS14_spatial_yield_transfers',14,'Estimated yield change (kg ha⁻¹ per unit reference LAI)',-1000)]:
        scenario_map(destination,data,countries,cells,metric,stem,unit,scale)
        captions[stem]=(f'Figure S{num} | '+('Changes in flag-leaf symptom timing.' if metric==ONSET else 'Disease-related yield changes inferred from canopy damage.')+
          ' Panels show three-climate-model means at every eligible wheat land-use grid for two future periods relative to 1991–2020. '
          'Each figure shares symmetric discrete color intervals retaining the full data range. Gray wheat pixels have unavailable estimates. '
          'Timing requires detected symptoms in both paired years. Estimated disease-related yield change is −0.018 times the HAD-loss change, converted to kg ha⁻¹ per reference LAI. '
          'The yield product depends on a fixed canopy–yield coefficient and nominal leaf area rather than a validated grain-yield model.')
    # This historical comparison replaces the superseded single-year example.
    base=data[data.scenario.eq('ssp585')&data.period.eq('2071-2100')]
    fig,axes=plt.subplots(1,2,figsize=(8.9,4.3));fig.subplots_adjust(left=.05,right=.98,bottom=.13,top=.90,wspace=.22)
    for ax,metric,scale,title,unit in [(axes[0],ONSET,1,'a   Historical symptom timing','Symptoms relative to flowering (days)'),
        (axes[1],SEVERITY,100,'b   Historical canopy damage','Relative HAD loss (%)')]:
        p=base[base.metric.eq(metric)].copy();p['value']=scale*p.reference_on_common
        map_axes(ax,countries,cells)
        lo=np.floor(p.value.min()/5)*5;hi=np.ceil(p.value.max()/5)*5
        im=raster(ax,p,'value',cmap='Blues' if scale==1 else 'YlOrBr',norm=Normalize(lo,hi))
        ax.set_title(title,loc='left',fontsize=10,fontweight='bold')
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.14,shrink=.93);cb.set_label(unit,fontsize=8.6)
    exporter(fig,destination,'figS19_full_grid_example_2001')
    captions['figS19_full_grid_example_2001']=('Figure S19 | Historical simulated disease timing and canopy damage across the full wheat domain. '
       'Panels show the 1991–2020 reference means on the corresponding seasons included in the late-century SSP5–8.5 comparison, averaged across three climate models. '
       '(a) Flag-leaf symptoms relative to flowering. (b) Relative HAD loss during grain filling. '
       'Gray wheat pixels have unavailable estimates. These modeled reference distributions differ from observed historical disease prevalence.')
    return captions


def pooled_validation_figure(destination):
    fig,axes=plt.subplots(1,3,figsize=(10.2,4.4));fig.subplots_adjust(left=.07,right=.98,bottom=.19,top=.86,wspace=.44)
    for ax,endpoint,model,bench,title,unit in [(axes[0],'symptom_onset','model','benchmark','a   Symptom-onset error','Interval-excess distance (days)'),
        (axes[1],'severity','model_rmse','benchmark_rmse','b   Severity prediction error','RMSE (percentage points)')]:
        for i,(metric,label,color) in enumerate([(model,'Seasonal model',BLUE),(bench,'Benchmark',GREY)]):
            r=pooled(endpoint,metric)
            ax.bar(i,r.value,color=color,width=.56)
            ax.errorbar(i,r.value,yerr=[[r.value-r.lower95],[r.upper95-r.value]],fmt='none',color='#35434d',capsize=4,lw=1)
            ax.text(i,r.upper95+.10 if endpoint=='symptom_onset' else r.upper95+.6,f'{r.value:.2f}',ha='center',fontsize=9)
        ax.set_xticks([0,1],['Seasonal\nmodel','Benchmark']);ax.set_ylabel(unit,fontsize=8.8)
        ax.set_title(title,loc='left',fontsize=10,fontweight='bold',pad=12);ax.grid(axis='y',color='#e8ecee',lw=.65);ax.set_axisbelow(True)
    x=np.arange(3)
    for prefix,offset,color in [('model',-.18,BLUE),('benchmark',.18,GREY)]:
        rows=[pooled('symptom_detection',prefix+'_'+m) for m in ['sensitivity','specificity','balanced_accuracy']]
        values=100*np.array([r.value for r in rows]);lo=100*np.array([r.lower95 for r in rows]);hi=100*np.array([r.upper95 for r in rows])
        axes[2].bar(x+offset,values,width=.32,color=color)
        axes[2].errorbar(x+offset,values,yerr=[values-lo,hi-values],fmt='none',color='#35434d',capsize=3,lw=.8)
    axes[2].set_xticks(x,['Sensitivity','Specificity','Balanced\naccuracy'],fontsize=8)
    axes[2].set(ylim=(0,108),ylabel='Detection performance (%)')
    axes[2].set_title('c   Symptom detection',loc='left',fontsize=10,fontweight='bold',pad=12)
    axes[2].grid(axis='y',color='#e8ecee',lw=.65);axes[2].set_axisbelow(True)
    export(fig,destination,'figS20_pooled_field_evaluation')
    return ('Figure S20 | Pooled evaluation of disease predictions. '
        '(a) Mean distance outside genuine two-sided symptom-onset intervals for 136 histories from 78 field seasons and 62 coordinate-years. '
        '(b) Numerical severity RMSE for 2,033 assessments from 188 field seasons and 140 coordinate-years. '
        '(c) Assessment-level symptom-detection sensitivity, specificity and balanced accuracy on the same 2,033 records. '
        'Blue bars represent the seasonal model and gray bars the corresponding calibration-selected benchmark. '
        'Whiskers are 95% percentile intervals from 20,000 paired coordinate-year bootstrap resamples. '
        'Coordinate-years, fields, source leaves and their assessments receive equal weight within each hierarchy level. '
        'Calibration records are excluded. Source scoring conventions and unresolved final leaf-rank mappings remain in the pooled evidence.')


def supplementary_phenology(destination):
    from analysis.paper_study.full_validation_20261007.figures import render_main
    # The station panel retains its observed distributions; the field panel pools
    # both archives and shows one marker population per developmental stage.
    def combine(fig,folder,stem):
        ax=fig.axes[1];ax.clear()
        d=pd.read_csv(DERIVED/'pooled_field_stage_records.csv')
        d=d[d.cohort.eq('Field evaluation')&d.genuinely_bracketed]
        stages=[31,32,33,37,39,51,65,85];labels=[]
        for i,stage in enumerate(stages):
            p=d[d.event.eq(stage)].sort_values('signed_distance_days')
            ax.scatter(i+np.linspace(-.12,.12,len(p)),p.signed_distance_days,s=16,color=BLUE,alpha=.6)
            labels.append(f'{stage}\nn={len(p)}')
        ax.set_xticks(range(8),labels,fontsize=8)
        ax.set(xlim=(-.5,7.5),ylim=(-26,21),ylabel='Distance outside observed interval (days)',xlabel='Development stage (BBCH)')
        ax.set_title('b   Pooled field evaluation',loc='left',fontsize=10.4,fontweight='bold',pad=10)
        ax.axhline(0,color=GREY,lw=.8,ls='--');ax.grid(axis='y',color='#e8ecee',lw=.65)
        ax.text(0,-.40,'Negative: early; positive: late; zero: within observed interval',transform=ax.transAxes,fontsize=8,va='top')
        export(fig,destination,'figS18_flag_stage_validation')
    render_main(destination,combine)
    return 'figS18_flag_stage_validation',('Figure S18 | Evaluation of retained crop-development stages. '
      '(a) German station-held-out date errors at BBCH 10, 31, 51 and 85. Boxes span the interquartile range; whiskers span the fifth to ninety-fifth percentiles. '
      '(b) Pooled field errors outside genuine two-sided developmental-stage intervals, with all informative records represented. '
      'The station and field endpoints have different observation resolutions. Counts beneath stages give sample sizes; observation intervals do not define statistical confidence intervals.')


def agreement_figure(destination,exporter):
    data,_,_=grid_data();cells=data.drop_duplicates('cell_id');countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    p=data[data.metric.eq(CANOPY)].copy()
    p['agreement']=np.where(p.positive_gcm_count.eq(3),1,np.where(p.negative_gcm_count.eq(3),-1,0)).astype(float)
    p.loc[p.available_gcms.ne(3),'agreement']=np.nan
    cmap=ListedColormap([BLUE,'#d9d9d9',GOLD]);fig,axes=plt.subplots(2,3,figsize=(10.2,6.3))
    fig.subplots_adjust(left=.05,right=.98,bottom=.13,top=.95,wspace=.12,hspace=.26)
    for r,period in enumerate(['2031-2060','2071-2100']):
        for c,scenario in enumerate(SCENARIOS):
            ax=axes[r,c];map_axes(ax,countries,cells,mask_color='#697780');part=p[p.period.eq(period)&p.scenario.eq(scenario)]
            im=raster(ax,part,'agreement',cmap=cmap,vmin=-1.5,vmax=1.5)
            ax.set_title(f'{chr(97+r*3+c)}   {LABELS[c]} · {period.replace("-","–")}',loc='left',fontsize=9.4,fontweight='bold')
            if c:ax.set_yticklabels([])
            if not r:ax.set_xticklabels([])
    cb=fig.colorbar(im,cax=fig.add_axes([.23,.065,.55,.024]),orientation='horizontal',ticks=[-1,0,1])
    cb.ax.set_xticklabels(['All models: less damage','Different signs / zero','All models: more damage'],fontsize=8.4)
    fig.text(.88,.078,'■ Unavailable',fontsize=8,color='#697780',ha='left')
    exporter(fig,destination,'figS19_environmental_region_responses')
    return ('Figure S19 | Agreement in projected canopy-damage direction across climate models. '
      'Colors indicate a negative change in all three models, a positive change in all three models, or other sign combinations. '
      'Each panel shows independently simulated wheat grids for one emissions pathway and future period relative to 1991–2020. '
      'Model agreement is deterministic scenario consistency rather than a probability estimate. Unavailable wheat pixels remain distinguishable from complete mixed-sign estimates in Source Data.')


def sensitivity_figure(destination):
    source=ROOT/'analysis/paper_study/nature_food_fix_20261007/climate/headline_robustness.csv'
    data=pd.read_csv(source);data=data[data.population.eq('all_settings_common_paired')]
    order=data.setting.drop_duplicates().tolist()
    labels={'baseline':'Reference settings','primary_objective_tied_runnerup':'Alternative tied calibration',
      'local_source_off':'No initial local source','rank_spacing80':'Shorter leaf-rank spacing',
      'rank_spacing160':'Longer leaf-rank spacing','stage_profile_early37':'Earlier flag-leaf appearance',
      'stage_profile_late37':'Later flag-leaf appearance','detection0p0001':'Lower detection threshold',
      'detection0p01':'Higher detection threshold','reference_decline_end0p5':'Reference canopy declines to 50%',
      'reference_decline_end0':'Reference canopy declines to zero','functional_conversion0p5':'Functional conversion: 50%',
      'functional_conversion0p75':'Functional conversion: 75%','decline_end0_conversion0p5':'Declining canopy; 50% conversion'}
    fig,axes=plt.subplots(1,2,figsize=(10.3,6.7),sharey=True)
    fig.subplots_adjust(left=.30,right=.98,bottom=.11,top=.92,wspace=.29)
    for ax,metric,letter,title,unit,color in [(axes[0],ONSET,'a','Symptom timing','Change relative to flowering (days)',BLUE),
       (axes[1],CANOPY,'b','Canopy damage','Change in normalized HAD loss (days)',GOLD)]:
        p=data[data.metric.eq(metric)].set_index('setting').loc[order]
        for y,row in enumerate(p.itertuples()):
            ax.plot([row.gcm_min_change,row.gcm_max_change],[y,y],lw=2.3,color='#b1b6ba')
            ax.errorbar(row.gcm_mean_change,y,xerr=row.spatial_mcse_gcm_mean_change,fmt='o',ms=5,color=color,capsize=3)
        ax.axvline(0,color=GREY,lw=.8,ls='--');clean(ax)
        ax.set_xlabel(unit,fontsize=8.4);ax.set_title(letter+'   '+title,loc='left',fontsize=10,fontweight='bold',pad=12)
    axes[0].set_yticks(range(len(order)),[labels.get(s,s.replace('_',' ')) for s in order],fontsize=8.5)
    axes[0].invert_yaxis();export(fig,destination,'figS15_climate_robustness')
    data.to_csv(destination/'figS15_climate_robustness.csv',index=False)
    return ('Figure S15 | Sensitivity of simulated climate responses to structural and reporting assumptions. '
      '(a) Flag-leaf symptom timing relative to flowering. (b) Normalized HAD loss during grain filling. '
      'All fourteen settings retain their common paired late-century SSP5–8.5 diagnostic population. '
      'Gray segments span three climate-model means; colored whiskers show one spatial Monte Carlo standard error for 64 area-proportional draws. '
      'Settings describe selected model assumptions rather than a probability distribution of future disease outcomes; precise settings and coverage remain in Supplementary Table S10.')


def study_context(destination):
    cells=pd.read_csv(ROOT/'analysis/paper_study/nature_food_submission_20261008/spatial_sources/wheat_cells_eea_regions.csv')
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    fields=pd.read_csv(ROOT/'analysis/paper_study/map_restoration_20261007/map_field_locations.csv')
    fields['cohort']=np.where(fields.partition.eq('calibration'),'Calibration','Field evaluation')
    fig,axes=plt.subplots(1,2,figsize=(9.9,4.8));fig.subplots_adjust(left=.05,right=.98,bottom=.24,top=.92,wspace=.22)
    geography(axes[0],countries);axes[0].set_xlabel('');axes[0].set_ylabel('')
    im=raster(axes[0],cells,'harvested_total_ha',cmap='YlOrBr',norm=LogNorm(vmin=1,vmax=float(cells.harvested_total_ha.max())))
    for cohort,color,marker in [('Calibration',BLUE,'^'),('Field evaluation',GREY,'o')]:
        p=fields[fields.cohort.eq(cohort)].drop_duplicates(['latitude','longitude'])
        axes[0].scatter(p.longitude,p.latitude,s=17,facecolors='none',edgecolors=color,marker=marker,lw=.7,zorder=3,label=cohort)
    axes[0].legend(frameon=False,fontsize=8,loc='lower right')
    cb=fig.colorbar(im,ax=axes[0],orientation='horizontal',fraction=.04,pad=.14,shrink=.93)
    cb.set_label('Wheat harvested area (ha per 0.25° cell)',fontsize=8.6)
    axes[0].set_title('a   Wheat area and pooled field observations',loc='left',fontsize=10,fontweight='bold',pad=10)
    names=REGIONS+['Outside EEA regions','Unassigned'];cells['region_code']=cells.environment_region.map(dict(zip(names,range(len(names)))))
    colors=plt.get_cmap('tab10').colors
    geography(axes[1],countries);axes[1].set_xlabel('');axes[1].set_ylabel('')
    raster(axes[1],cells,'region_code',cmap=ListedColormap(colors),vmin=-.5,vmax=len(names)-.5)
    from matplotlib.patches import Patch
    axes[1].legend(handles=[Patch(facecolor=colors[i],label=n) for i,n in enumerate(names)],
        loc='upper left',bbox_to_anchor=(-.05,-.16),ncol=3,frameon=False,fontsize=7.6,columnspacing=.7,handlelength=.9)
    axes[1].set_title('b   Environmental regions',loc='left',fontsize=10,fontweight='bold',pad=10)
    export(fig,destination,'figS21_study_domain')
    return ('Figure S21 | Wheat-land-use domain, pooled observation locations and environmental regions. '
      '(a) SPAM2020 all-wheat harvested area with calibration and pooled evaluation locations; coincident sites retain their true reported coordinates. '
      '(b) EEA biogeographical regions assigned by wheat-cell centroid intersection. Outside-region and unassigned cells remain separate. '
      'The 14,941-cell fixed mask and imposed winter-wheat rainfed calendar define the climate-analysis scenario. '
      'These maps specify the study design and do not display projected disease outcomes.')


def supplementary_framework(destination):
    from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
    fig,ax=plt.subplots(figsize=(10.5,5.2));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,label,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01',fc=color,ec='#9ba8b0',lw=1))
        ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=9.5,linespacing=1.5,color='#27343d')
    def arrow(start,end,color=GREY):
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,color=color,lw=1.4))
    box(.025,.43,.18,.18,'Climate and weather\nTemperature, humidity\nand rainfall','#e6eef4')
    box(.275,.71,.25,.20,'Wheat phenology\nLeaf appearance and unfolding\nFlowering and soft dough','#e8f0eb')
    box(.275,.15,.25,.20,'Septoria disease\nInfection and progression\nSymptoms and leaf damage','#e6eef4')
    box(.635,.60,.32,.22,'Loss of healthy canopy\nDamage-to-function assumption\nLoss of healthy-area duration','#e8f0eb')
    box(.635,.13,.32,.22,'Disease-related yield estimate\nPublished HAD–yield coefficient\nConditional on canopy assumptions','#f2ebde')
    arrow((.21,.57),(.27,.79),BLUE);arrow((.21,.46),(.27,.26),BLUE)
    arrow((.40,.70),(.40,.36),GREEN)
    ax.text(.41,.52,'Leaf availability\nand seasonal exposure',ha='left',va='center',fontsize=8.5,color=GREEN)
    arrow((.535,.81),(.63,.76),GREEN);arrow((.535,.25),(.63,.64),BLUE)
    arrow((.80,.59),(.80,.36),GOLD)
    ax.text(.03,.035,'Wheat phenology drives disease exposure; disease damage does not feed back into developmental rates.',fontsize=8.5,color=GREY)
    export(fig,destination,'figS17_framework')
    return 'figS17_framework',('Figure S17 | Relationships among climate, wheat phenology, Septoria disease and the estimated yield response. '
      'Wheat phenology comprises leaf appearance, unfolding, flowering and soft dough. Weather drives these stage dates and disease. Leaf availability and seasonal timing connect wheat phenology to infection and damage. '
      'Disease damage is mapped to an assumed loss of canopy function and integrated as HAD loss. A published coefficient provides a conditional disease-related yield estimate. '
      'The model has no disease feedback on crop-development rates and no complete crop carbon-balance calculation.')
