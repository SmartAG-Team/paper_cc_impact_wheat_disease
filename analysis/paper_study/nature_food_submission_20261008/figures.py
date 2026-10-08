"""Six source-backed crop infection, canopy damage and conditional yield figures."""
from pathlib import Path
import hashlib,json,shutil,re
from tempfile import TemporaryDirectory
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from analysis.paper_study.full_validation_20261007.figures import render_main
from analysis.paper_study.map_restoration_20261007.figures import main_maps

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
EVIDENCE=ROOT/'analysis/paper_study/nature_food_fix_20261007'
STUDY=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006'
YIELD=ROOT/'analysis/paper_study/yield_evidence_20261007'
RESPONSE=ROOT/'analysis/paper_study/nature_food_revision_20261007'
BLUE='#235e83';GREEN='#52796f';GOLD='#a17937';GREY='#737d85';DARK='#27343d'


def export(fig,destination,stem):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','pdf','svg'):
        fig.savefig(destination/f'{stem}.{suffix}',dpi=300,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def clean(ax):
    ax.grid(axis='y',color='#e8ecee',lw=.65);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=8.5)


def title(ax,text):ax.set_title(text,loc='left',fontsize=10.3,fontweight='bold',pad=10)


def scientific_framework(destination):
    fig,ax=plt.subplots(figsize=(11.5,4.4));ax.set(xlim=(0,1),ylim=(0,1));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=.016',fc=color,ec='#9ba8b0',lw=1.0))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=10.1,color=DARK,linespacing=1.45)
    def arrow(a,b,color=GREY):ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,color=color,lw=1.3))
    box(.025,.385,.17,.235,'Daily climate / weather\nMean and maximum T\nHumidity and rainfall','#e6eef4')
    box(.265,.66,.22,.245,'Crop development\nTemperature–photoperiod–\nvernalization; BBCH stages\nLeaf appearance / unfolding','#e8f0eb')
    box(.265,.11,.22,.245,'Crop infection / symptoms\nExposure and source pressure\nEstablishment and progression','#e6eef4')
    box(.565,.385,.195,.235,'Upper-leaf damage\nF1–F3 availability and area\nSymptom-gated damage','#e8f0eb')
    box(.81,.69,.17,.215,'HAD deficit proxy\nGS31–85 or GS65–85\nNominal upper-three LAI','#f2ebde')
    box(.81,.11,.17,.24,'Conditional yield effect\nFixed HAD transfer\n0.0141–0.0207\nt ha⁻¹ per GLAI-day','#f2ebde')
    arrow((.2,.56),(.25,.78),BLUE);arrow((.2,.445),(.25,.23),BLUE)
    arrow((.375,.645),(.375,.37),GREEN)
    ax.text(.382,.50,'Leaf availability\nand renewal',fontsize=8.7,ha='left',va='center',color=GREEN)
    arrow((.493,.77),(.555,.575),GREEN);arrow((.493,.235),(.555,.445),BLUE)
    arrow((.772,.56),(.80,.775),GOLD);arrow((.895,.675),(.895,.365),GOLD)
    ax.text(.91,.52,'Fixed canopy–yield\ntransfer coefficients',ha='left',va='center',fontsize=8.5,color=GOLD)
    ax.text(.035,.975,'Climate and weather effects on crop infection, canopy function and yield',ha='left',va='top',fontsize=12,fontweight='bold')
    ax.text(.025,.015,'Crop-stage observations  •  Source-numbered disease records  •  Independent canopy–yield treatment contrasts',fontsize=9,color=GREY,ha='left')
    fig.subplots_adjust(left=.01,right=.97,bottom=.02,top=.98);export(fig,destination,'fig1_framework')
    return ('Figure 1 | Climate–crop–infection pathways linking canopy damage to conditional yield effects. '
        'Daily temperature, humidity and rainfall drive the retained crop-development and disease-weather operators. '
        'Development determines modeled leaf availability, unfolding and renewal, while weather and fixed source assumptions drive infection and symptom progression. '
        'Symptom-gated upper-three-leaf damage is integrated against a nominal reference canopy over GS31–85 or GS65–85 to give the healthy-area-duration (HAD) deficit proxy. '
        'Fixed coefficients of 0.0141–0.0207 t ha⁻¹ per GLAI-day provide conditional yield-loss diagnostics. '
        'Crop-stage dates, source-numbered disease measurements and independent canopy–yield treatment contrasts constrain different links in the chain. '
        'The canopy reference, damage-to-functional-area conversion and current simulated yield effects have no local quantitative validation.')


def validation_figure(destination):
    onset=pd.read_csv(EVIDENCE/'validation/paired_onset_primary.csv')
    onset=onset[onset.leaf_scope.eq('top3_source_numbered')&onset.comparison_population.eq('original_eligible_records')&onset.metric.eq('difference')&onset.comparator.eq('phenology_only')]
    severity=pd.read_csv(EVIDENCE/'validation/paired_severity_primary.csv')
    severity=severity[severity.endpoint.eq('all_assessments_severity_pp')&severity.comparison_population.eq('original_eligible_records')&severity.comparator.eq('calibration_leaf_mean')]
    parts=['reused_BASF2019','reused_strict_Corteva'];labels=['BASF 2019','Corteva\nsource-numbered leaves']
    onset=onset.set_index('partition').loc[parts].reset_index()
    severity.to_csv(destination/'fig3_severity_bootstrap_source.csv',index=False);onset.to_csv(destination/'fig3_paired_onset_source.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(8.9,4.65));fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.86,wspace=.45)
    for ax,data,letter,ylabel in [(axes[0],onset,'a','Model − phenology benchmark\nonset excess (days)'),
        (axes[2],severity[severity.metric.eq('rmse_difference')].set_index('partition').loc[parts].reset_index(),'c','Model − calibration-leaf mean\nRMSE (percentage points)')]:
        x=np.arange(2);mean=data.point.to_numpy();low=data.lower95.to_numpy();high=data.upper95.to_numpy()
        ax.errorbar(x,mean,yerr=[mean-low,high-mean],fmt='o',color=BLUE,capsize=5,ms=6,lw=1.4)
        ax.axhline(0,color=GREY,ls='--',lw=.9);ax.set_xticks(x,labels);ax.set_xlim(-.5,1.5);ax.set_ylabel(ylabel,fontsize=9.1)
        for i,row in enumerate(data.itertuples()):
            ax.annotate(f'{row.point:+.2f}',(i,row.point),xytext=(8,4),textcoords='offset points',fontsize=8.7,color=BLUE)
        title(ax,letter+'   '+('Paired symptom-onset error' if letter=='a' else 'Paired severity-error difference'));clean(ax)
    for metric,offset,color,label in [('model_rmse',-.18,BLUE,'Seasonal model'),('benchmark_rmse',.18,GREY,'Calibration-leaf mean')]:
        data=severity[severity.metric.eq(metric)].set_index('partition').loc[parts]
        axes[1].bar(np.arange(2)+offset,data.point,width=.32,color=color,label=label)
        for i,row in enumerate(data.itertuples()):axes[1].text(i+offset,row.point+.6,f'{row.point:.2f}',ha='center',fontsize=8.2)
    axes[1].set_xticks(range(2),labels);axes[1].set_ylim(0,39);axes[1].set_ylabel('Severity RMSE (percentage points)',fontsize=9.1)
    title(axes[1],'b   Quantitative disease transfer');axes[1].legend(frameon=False,fontsize=8,loc='upper left',bbox_to_anchor=(0,1.08));clean(axes[1])
    axes[0].text(0,-.27,'13 / 123 histories; 8 / 54 coordinate-years',transform=axes[0].transAxes,fontsize=8,va='top')
    axes[1].text(0,-.27,'330 / 1,703 assessments; all source leaves',transform=axes[1].transAxes,fontsize=8,va='top')
    axes[2].text(0,-.27,'36 / 104 coordinate-year groups',transform=axes[2].transAxes,fontsize=8,va='top')
    export(fig,destination,'fig3_onset_and_detection')
    return ('Figure 3 | Paired validation of crop-infection timing and disease magnitude. '
        '(a) Seasonal-model minus phenology-benchmark first-symptom interval-excess distance for source-numbered leaves 1–3. '
        'BASF 2019 contributes 13 histories in eight fields/eight coordinate-years; location-disjoint Corteva contributes 123 histories in 70 fields/54 coordinate-years. '
        'Positive differences indicate greater seasonal-model error. BASF source leaf 1 is explicitly the flag leaf; Corteva final-rank identity remains unresolved. '
        '(b) Severity RMSE against the calibration-leaf-mean comparator on all numeric source-leaf assessments: 330 BASF records in 45 fields/36 coordinate-years and 1,703 Corteva records in 143 fields/104 coordinate-years. '
        '(c) Paired RMSE differences on the same assessment populations. Whiskers in a and c are 95% grouped-bootstrap intervals from 20,000 coordinate-year resamples, preserving field, leaf and assessment histories; b shows point scores. '
        'Severity percentages retain the source scoring conventions and are compared with the model damage proxy. '
        'These reused validation records assess symptom timing and disease magnitude, not observed infection dates or measured functional green-area loss.')


def crop_response_figure(destination):
    points=pd.read_csv(YIELD/'foulkes2006_group_separated_predictions.csv')
    scores=pd.read_csv(YIELD/'published_yield_response_metrics.csv').set_index('response')
    nordic=pd.read_csv(RESPONSE/'yield_response/model_comparison_metrics.csv');basf=pd.read_csv(RESPONSE/'basf_crop_response/year_transfer_metrics.csv')
    records=[]
    for domain,label in [('endpoint67','Nordic endpoint\n67 contrasts\n5 trials'),('integral50','Nordic window\n50 contrasts\n5 trials')]:
        for family,suffix in [('Linear','_loto'),('Exponential','_damage_loto')]:
            row=nordic[nordic.domain.eq(domain+suffix)].iloc[0]
            records.append(dict(domain=domain,label=label,family=family,model_RMSE_pp=row.model_rmse_pp,training_mean_RMSE_pp=row.baseline_rmse_pp,
                normalized_RMSE=row.model_rmse_pp/row.baseline_rmse_pp,contrasts=int(row.rows),groups=int(row.independent_trials)))
    for family,function in [('Linear','linear'),('Exponential','exponential')]:
        row=basf[basf.domain.eq('all_three_separate_windows')&basf.function.eq(function)].iloc[0]
        records.append(dict(domain='BASF_top3',label='BASF all-three\n12 contrasts\n11 fields',family=family,model_RMSE_pp=row.RMSE_pp,
            training_mean_RMSE_pp=row.baseline_RMSE_pp,normalized_RMSE=row.RMSE_pp/row.baseline_RMSE_pp,
            contrasts=int(row.evaluation_contrasts),groups=int(row.evaluation_fields)))
    ratios=pd.DataFrame(records);ratios.to_csv(destination/'fig4_holdout_response_errors.csv',index=False);points.to_csv(destination/'fig4_Foulkes_held_background_points.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.9,5.15));fig.subplots_adjust(left=.08,right=.985,bottom=.27,top=.85,wspace=.34)
    for background,marker in zip(points.genetic_background.unique(),['o','s','D','^','v']):
        p=points[points.genetic_background.eq(background)]
        exposure=p.genotype_observed_target_diseases.iloc[0]
        color={'STB':BLUE,'stripe rust':GREY,'STB;stripe rust':GOLD}[exposure]
        disease_label={'STB':'STB','stripe rust':'yellow rust','STB;stripe rust':'mixed'}[exposure]
        axes[0].scatter(100*p.derived_yield_loss_fraction,100*p.relative_held_out_prediction,s=43,marker=marker,color=color,label=f'{background} ({disease_label})',edgecolors='white',lw=.6,zorder=3)
    axes[0].plot([0,45],[0,45],ls='--',color=GREY,lw=.9);axes[0].set(xlim=(0,45),ylim=(0,45),xlabel='Published relative yield loss (%)',ylabel='Held-background prediction (%)')
    title(axes[0],'a   Published canopy–yield holdout');clean(axes[0]);axes[0].set_aspect('equal',adjustable='box')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,frameon=False,ncol=3,fontsize=8,loc='lower left',bbox_to_anchor=(.075,.01),columnspacing=1.2,handlelength=1.)
    score=scores.loc['relative'];axes[0].text(.04,.96,f'8 means; 5 backgrounds\nRMSE {100*score.group_balanced_RMSE:.2f} pp; R² {score.group_balanced_R2:.3f}',transform=axes[0].transAxes,fontsize=8.5,va='top',linespacing=1.5)
    absolute=scores.loc['absolute']
    axes[0].text(.04,.06,f'Absolute transfer:\nRMSE {absolute.group_balanced_RMSE:.3f} t ha⁻¹; R² {absolute.group_balanced_R2:.3f}',
                 transform=axes[0].transAxes,fontsize=8.0,va='bottom',linespacing=1.4,color=DARK)
    domains=['endpoint67','integral50','BASF_top3'];domain_labels=ratios.drop_duplicates('domain').set_index('domain').loc[domains,'label'].tolist()
    for family,offset,color in [('Linear',-.18,BLUE),('Exponential',.18,GOLD)]:
        r=ratios[ratios.family.eq(family)].set_index('domain').loc[domains]
        axes[1].bar(np.arange(3)+offset,r.normalized_RMSE,width=.32,color=color,label=family)
        for i,row in enumerate(r.itertuples()):axes[1].text(i+offset,row.normalized_RMSE+.035,f'{row.normalized_RMSE:.2f}',ha='center',fontsize=8.5)
    axes[1].axhline(1,color=GREY,ls='--',lw=.9);axes[1].set(ylim=(0,1.75),ylabel='RMSE / own training-mean RMSE')
    axes[1].set_xticks(range(3),domain_labels);axes[1].legend(frameon=False,ncol=2,fontsize=8.3,loc='upper left',bbox_to_anchor=(0,1.08));title(axes[1],'b   Independent crop-response transfer');clean(axes[1])
    axes[1].text(0,-.30,'Dashed line: training-mean benchmark (= 1)\nLower values indicate lower prediction error',transform=axes[1].transAxes,fontsize=8.3,va='top',linespacing=1.5)
    export(fig,destination,'fig4_timing_and_yield_relevance')
    return ('Figure 4 | Held-out evidence for infection-related crop-yield response. '
        '(a) Published observed and leave-one-genetic-background-out relative-yield-loss means from Foulkes et al. (2006): eight contrasts across five backgrounds, with two STB-only, two yellow-rust-only and four mixed-exposure means retained. '
        'One proportional HAD–yield coefficient is trained on the other four backgrounds; backgrounds receive equal evaluation weight. '
        'Symbols identify backgrounds, and the dashed line is equality. The points are aggregate treatment means from top-five postanthesis canopy measurements, not independent raw plots or calibration of the current top-three simulator. '
        '(b) Linear and exponential response RMSE divided by each dataset’s own training-mean RMSE. Nordic endpoint and common-date window datasets contain 67 and 50 treatment contrasts across five held-out trial environments. '
        'BASF all-three separate severity-window models train on 12 contrasts in eight 2017–2018 fields and evaluate 12 contrasts in 11 2019 fields. '
        'Ratios above one indicate poorer transfer than the corresponding training-mean benchmark; bars are point estimates, not confidence intervals. '
        'Nordic and BASF predictors are disease severity or percentage-days and their outcomes are management-associated relative yield responses; they are not measured HAD or identified causal STB losses.')


def climate_figure(destination):
    climate=pd.read_csv(STUDY/'climate/ensemble_paired_period_changes.csv')
    selected=climate[climate.metric.isin(['F1_symptom_relative_anthesis_days','GS65_85_lost_had3'])]
    selected.to_csv(destination/'fig5_all_SSP_paired_climate_changes.csv',index=False)
    decomp=pd.read_csv(EVIDENCE/'climate/decomposition/supported_forcing_ensemble_decomposition.csv');decomp.to_csv(destination/'fig5_supported_weather_host_decomposition.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(8.6,6.8),gridspec_kw={'height_ratios':[1,1.4]})
    fig.subplots_adjust(left=.12,right=.985,bottom=.12,top=.94,wspace=.52,hspace=.51)
    scenarios=['ssp126','ssp245','ssp585'];scenario_labels=['SSP1–2.6','SSP2–4.5','SSP5–8.5']
    for ax,metric,letter,ylabel in [(axes[0,0],'F1_symptom_relative_anthesis_days','a','F1 symptom shift relative to anthesis (days)'),
        (axes[0,1],'GS65_85_lost_had3','b','HAD proxy change (days / nominal LAI)')]:
        for period,offset,marker,color in [('2031-2060',-.12,'o',BLUE),('2071-2100',.12,'s',GOLD)]:
            rows=selected[selected.future_period.eq(period)&selected.metric.eq(metric)].set_index('scenario').loc[scenarios]
            x=np.arange(3)+offset
            for i,row in zip(x,rows.itertuples()):ax.plot([i,i],[row.gcm_min_change,row.gcm_max_change],color='#b1b6ba',lw=2.1,zorder=1)
            ax.errorbar(x,rows.gcm_mean_change,yerr=rows.spatial_mcse_gcm_mean_change,fmt=marker,color=color,ms=5,capsize=3,label=period,zorder=3)
        ax.axhline(0,color=GREY,lw=.8);ax.set_xticks(range(3),scenario_labels);ax.set_ylabel(ylabel,fontsize=8.9);clean(ax)
        title(ax,letter+'   '+('Climate-driven symptom timing' if letter=='a' else 'Climate-driven canopy deficit'))
    axes[0,0].legend(frameon=False,fontsize=8,ncol=2,loc='lower left',bbox_to_anchor=(0,1.075))
    order=['weather_shapley','host_shapley','total_change','interaction'];labels=['Disease weather\n(Shapley)','Host development\n(Shapley)','Supported net','Interaction\n(full term)']
    ax=axes[1,0];data=decomp.set_index('component').loc[order]
    for y,row,color in zip(range(4),data.itertuples(),[BLUE,GREEN,GOLD,GREY]):
        ax.plot([row.gcm_min,row.gcm_max],[y,y],color='#b1b6ba',lw=2.7)
        ax.errorbar(row.gcm_mean,y,xerr=row.spatial_mcse_gcm_mean,fmt='o',color=color,ms=5.8,capsize=3)
        ax.annotate(f'{row.gcm_mean:+.2f}',(row.gcm_mean,y),xytext=(0,10),textcoords='offset points',ha='center',fontsize=8.5,color=color)
    ax.set_yticks(range(4),labels,fontsize=8.5);ax.invert_yaxis();ax.axvline(0,color=GREY,ls='--',lw=.8);ax.axhline(2.5,color='#dddddd',lw=.7)
    ax.set_xlabel('HAD proxy change (days / nominal upper-three LAI)',fontsize=8.8);ax.grid(axis='x',color='#e8ecee',lw=.65);ax.tick_params(axis='x',labelsize=8.3)
    title(ax,'c   Weather–host model decomposition')
    offset=pd.read_csv(HERE/'derived/weather_host_offset.csv')
    offset.to_csv(destination/'fig5_weather_host_offset.csv',index=False)
    models=offset[offset.climate_model.ne('Three-model ensemble')]
    ax=axes[1,1]
    ax.barh(np.arange(3),models.host_offset_percent,color=GREEN,height=.55)
    for y,value in enumerate(models.host_offset_percent):
        ax.text(value+2,y,f'{value:.1f}%',va='center',fontsize=8.7,color=GREEN)
    ensemble=float(offset[offset.climate_model.eq('Three-model ensemble')].host_offset_percent.iloc[0])
    ax.axvline(ensemble,color=GREY,ls='--',lw=1)
    ax.set_yticks(range(3),models.climate_model.tolist(),fontsize=8.4)
    ax.invert_yaxis();ax.set_xlim(0,106)
    ax.set_xlabel('Host offset of weather contribution (%)',fontsize=8.8)
    ax.grid(axis='x',color='#e8ecee',lw=.65);ax.set_axisbelow(True)
    ax.tick_params(axis='x',labelsize=8.3)
    title(ax,'d   Host offset in each climate model')
    ax.text(.01,-.23,f'Ensemble ratio: {ensemble:.1f}%\nDescriptive model values; no intervention tested',transform=ax.transAxes,fontsize=8.1,va='top',color=GREY)
    fig.text(.12,.016,'a–c: gray segments span three climate models; colored bars show ±1 spatial Monte Carlo SE.\nc,d: supported-forcing population. Model ranges are not confidence intervals.',fontsize=8,color=GREY)
    export(fig,destination,'fig5_conditional_climate_changes')
    return ('Figure 5 | Weather and host development contribute opposing changes in modeled wheat canopy damage. '
        '(a,b) Paired changes in flag-leaf symptom timing relative to flowering and upper-three-leaf HAD deficit under three SSPs, relative to each 1991–2020 reference. '
        '(c) Late-century SSP5–8.5 weather and host Shapley contributions and their net on 5,731 common supported-forcing pairs; 20 finite pairs exceeding the supported hybrid temperature range are excluded. '
        'The full −2.78-day interaction is displayed separately and is already shared equally between the contributions. '
        '(d) Model-specific host-offset ratios, −100 × host/weather. The dashed line is the ratio of ensemble contributions, 70.6%, not the mean of model-specific percentages. '
        'Gray ranges in a–c span climate-model means; colored bars show ±1 spatial Monte Carlo standard error for shared draws. Panel d has no probability intervals. '
        'Timing requires detection in both periods. Baseline and supported-forcing populations differ, giving canopy net changes of 2.62 and 2.64 days. '
        'Reference LAI denotes a nominal upper-three-leaf area index of one. These within-model responses are not measured yield effects or adaptation benefits.')


def maps_figure(destination):
    import geopandas as gpd
    from matplotlib.colors import LogNorm,Normalize
    from analysis.paper_study.map_restoration_20261007.figures import geography,raster,dots,symmetric_norm,DIVERGING
    m=ROOT/'analysis/paper_study/map_restoration_20261007'
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    cells=pd.read_csv(m/'map_wheat_domain.csv');fields=pd.read_csv(m/'map_field_locations.csv')
    values=pd.read_csv(m/'map_ensemble_values.csv')
    baseline=values[values.kind.eq('level')&values.period.eq('1991-2020')&values.scenario.eq('ssp585')&values.metric.eq('F1_symptom_relative_anthesis_days')]
    timing=values[values.kind.eq('change')&values.period.eq('2071-2100')&values.scenario.eq('ssp585')&values.metric.eq('F1_symptom_relative_anthesis_days')]
    canopy=pd.read_csv(HERE/'derived/map_canopy_changes.csv')
    canopy=canopy[canopy.period.eq('2071-2100')&canopy.scenario.eq('ssp585')]
    fig,axes=plt.subplots(2,2,figsize=(8.1,7.0));fig.subplots_adjust(left=.08,right=.975,bottom=.07,top=.94,wspace=.24,hspace=.34)
    for ax in axes.flat:geography(ax,countries);ax.set_xlabel('')
    im=raster(axes[0,0],cells,'harvested_total_ha',cmap='YlOrBr',norm=LogNorm(vmin=10,vmax=40000))
    for part,color,marker,label in [('calibration',BLUE,'^','BASF 2017–18'),('reused_development_2019',GREEN,'s','BASF 2019'),('reused_external_strict',GREY,'o','Corteva')]:
        q=fields[fields.partition.eq(part)].drop_duplicates(['latitude','longitude'])
        axes[0,0].scatter(q.longitude,q.latitude,s=17,marker=marker,facecolors='none',edgecolors=color,linewidths=.8,label=label,zorder=4)
    axes[0,0].legend(frameon=False,fontsize=7.5,loc='lower right',handlelength=1)
    title(axes[0,0],'a   Wheat area and field evidence')
    bar=fig.colorbar(im,ax=axes[0,0],orientation='horizontal',fraction=.042,pad=.14,shrink=.88)
    bar.set_label('Harvested hectares per 0.25° cell',fontsize=8);bar.set_ticks([10,100,1000,10000]);bar.ax.tick_params(labelsize=7.5)
    im=dots(axes[0,1],baseline,Normalize(vmin=np.floor(baseline.value.min()/5)*5,vmax=np.ceil(baseline.value.max()/5)*5),'Blues')
    title(axes[0,1],'b   Historical symptom timing')
    bar=fig.colorbar(im,ax=axes[0,1],orientation='horizontal',fraction=.042,pad=.14,shrink=.88)
    bar.set_label('Symptoms relative to flowering (days)',fontsize=8);bar.ax.tick_params(labelsize=7.5)
    im=dots(axes[1,0],timing,symmetric_norm(timing.value.to_numpy(),2),DIVERGING)
    title(axes[1,0],'c   Change in symptom timing')
    bar=fig.colorbar(im,ax=axes[1,0],orientation='horizontal',fraction=.042,pad=.14,shrink=.88)
    bar.set_label('Future − reference timing (days)',fontsize=8);bar.ax.tick_params(labelsize=7.5)
    im=dots(axes[1,1],canopy,symmetric_norm(canopy.value.to_numpy(),2),DIVERGING)
    title(axes[1,1],'d   Change in canopy deficit')
    bar=fig.colorbar(im,ax=axes[1,1],orientation='horizontal',fraction=.042,pad=.14,shrink=.88)
    bar.set_label('HAD change (days per reference LAI)',fontsize=8);bar.ax.tick_params(labelsize=7.5)
    export(fig,destination,'fig6_european_spatial_results')
    for frame,name in [(cells,'wheat_area'),(fields,'field_locations'),(baseline,'historical_symptoms'),(timing,'symptom_changes'),(canopy,'canopy_changes')]:
        frame.to_csv(destination/f'fig6_{name}.csv',index=False)
    return ('Figure 6 | European wheat area and sampled climate responses of the current model. '
        '(a) Fixed SPAM2020 harvested area in 14,941 wheat cells with the locations of 28 calibration, 45 BASF 2019 and 143 location-disjoint Corteva field seasons. '
        '(b) Three-climate-model mean flag-leaf symptom timing relative to flowering in 1991–2020. '
        '(c,d) Paired SSP5–8.5 changes in symptom timing and flowering-to-soft-dough upper-three-leaf HAD deficit in 2071–2100 relative to 1991–2020. '
        'Circles represent 62 sampled cells from 64 registered draws; duplicate locations are collapsed for display only. '
        'Cell values are arithmetic three-model means, not interpolated continental predictions or observed disease prevalence. '
        'Positive canopy values indicate greater modeled deficit. Reference LAI denotes a nominal upper-three-leaf area index of one. '
        'The fixed all-wheat area and imposed winter/rainfed calendar define a management scenario. '
        'Country-level production losses and adaptation benefits are not estimated.')


def _legacy_main(destination:Path)->dict[str,str]:
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.5,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    captions={}
    captions['fig1_framework']=scientific_framework(destination)
    render_main(destination,export)
    captions['fig2_flag_stage_validation']=('Figure 2 | Complete crop-stage validation in station and field evidence. '
        '(a) German station-held-out temperature–photoperiod–vernalization date errors at BBCH10,31,51 and85 on the shared matched events of the archived model comparison. '
        'Boxes show interquartile ranges, median lines and fifth–ninety-fifth percentile whiskers; counts give validation followed by test events. '
        '(b) Individual signed interval-excess distances at field BBCH31,32,33,37,39,51,65 and85. Negative values indicate premature prediction, positive values delayed prediction and zero an admissible date inside a genuine two-sided observation window. '
        'Counts give BASF2019/location-disjoint Corteva records; zero counts indicate unavailable interval evidence. Horizontal displacement separates coincident records without changing their distance. '
        'The German holdout assesses retained crop phenology, not added flag stages or disease outcomes. Field predictions use frozen parameters and full-season forcing. '
        'Exact-date errors and interval-excess distances have different observational resolutions. Whiskers are distribution percentiles, not confidence intervals; tail records and missing-prediction coverage remain in the source data.')
    captions['fig3_onset_and_detection']=validation_figure(destination)
    captions['fig4_timing_and_yield_relevance']=crop_response_figure(destination)
    captions['fig5_conditional_climate_changes']=climate_figure(destination)
    captions['fig6_european_spatial_results']=maps_figure(destination)
    for name in ['fig2_field_stage_intervals.csv','fig2_german_stage_errors.csv']:
        source=STUDY/'manuscript/figures'/name
        if source.exists() and source.resolve()!=(destination/name).resolve():shutil.copy2(source,destination/name)
        elif not (destination/name).exists():raise FileNotFoundError(source)
    assert len(captions)==6 and all(len(c.split())<=220 for c in captions.values())
    assert not any('risk' in c.lower() for c in captions.values())
    (destination/'captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    (destination/'figure_source_manifest.json').write_text(json.dumps(dict(module_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        figure_ready_csv_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.csv')},
        caption_words={stem:len(text.split()) for stem,text in captions.items()}),indent=2)+'\n')
    return captions


def main(destination:Path)->dict[str,str]:
    from .spatial_figures import main_map
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    mapping=[('fig6_european_spatial_results','fig1_european_wheat_context',6,1),
             ('fig5_conditional_climate_changes','fig2_weather_host_climate_response',5,2),
             ('fig3_onset_and_detection','fig4_onset_and_detection',3,4),
             ('fig4_timing_and_yield_relevance','fig5_canopy_yield_evidence',4,5)]
    with TemporaryDirectory(dir=destination) as tmp:
        raw=_legacy_main(Path(tmp));captions={}
        prefixes={f'fig{oldnum}_':f'fig{newnum}_' for _,_,oldnum,newnum in mapping}
        names={old:new for old,new,_,_ in mapping}
        for file in Path(tmp).iterdir():
            if not file.is_file() or file.name.startswith(('fig1_','fig2_')) or not file.name.startswith('fig'):continue
            name=file.name
            if file.stem in names:name=names[file.stem]+file.suffix
            else:
                for old,new in prefixes.items():
                    if name.startswith(old):name=name.replace(old,new,1);break
            shutil.copy2(file,destination/name)
        for old,new,_,num in mapping:
            captions[new]=re.sub(r'^Figure \d+',f'Figure {num}',raw[old])
    captions['fig3_spatial_scenario_response']=main_map(destination,export)
    captions=dict(sorted(captions.items(),key=lambda item:int(re.match(r'fig(\d+)',item[0]).group(1))))
    (destination/'captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    (destination/'figure_source_manifest.json').write_text(json.dumps({
        'module_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'figure_ready_csv_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.glob('*.csv')},
        'caption_words':{k:len(v.split()) for k,v in captions.items()}},indent=2)+'\n')
    return captions


def supplementary_phenology(destination):
    destination=Path(destination)
    with TemporaryDirectory(dir=destination) as temporary:
        render_main(Path(temporary),export)
        for file in Path(temporary).iterdir():
            if file.is_file() and file.name.startswith('fig2_'):
                shutil.copy2(file,destination/file.name.replace('fig2_','figS18_',1))
    caption=('Figure S18 | Crop-stage evaluation in station and field observations. '
        '(a) German station-held-out date errors at BBCH10,31,51 and85. Boxes show interquartile ranges, median lines and fifth–ninety-fifth percentile whiskers; counts give validation followed by test events. '
        '(b) Signed distances outside genuine two-sided field observation intervals at BBCH31,32,33,37,39,51,65 and85. Negative values indicate early prediction, positive values late prediction and zero an admissible date. '
        'Counts give BASF2019/location-disjoint Corteva records; zero counts identify unavailable interval evidence. Horizontal displacement separates coincident records. '
        'Station holdouts evaluate retained crop phenology, whereas added leaf stages have sparse field evidence. Field predictions use frozen parameters and full-season forcing. '
        'Exact-date errors and interval-excess distances have different observational resolutions. Whiskers are distribution percentiles, not confidence intervals.')
    return 'figS18_flag_stage_validation',caption


def supplementary_framework(destination):
    destination=Path(destination)
    with TemporaryDirectory(dir=destination) as tmp:
        caption=scientific_framework(Path(tmp))
        for file in Path(tmp).iterdir():
            if file.is_file():shutil.copy2(file,destination/file.name.replace('fig1_', 'figS17_',1))
    return 'figS17_framework',caption.replace('Figure 1 |','Figure S17 |',1)


def current_map_figures(destination,save):
    from analysis.paper_study.map_restoration_20261007.figures import scenario_maps,full_grid_example
    captions={}
    captions['figS13_spatial_symptom_changes']=scenario_maps(destination,save,'F1_symptom_relative_anthesis_days',
        'figS13_spatial_symptom_changes',13,'Spatial changes in flag-leaf symptom timing','Future − reference timing (days)',2)
    captions['figS14_spatial_yield_transfers']=scenario_maps(destination,save,'conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai',
        'figS14_spatial_yield_transfers',14,'Spatial changes in conditional yield transfer','Change in t ha⁻¹ per nominal upper-three LAI',.025)
    caption=full_grid_example(destination,save)
    if caption is None:raise FileNotFoundError('The retained current-model 2001 grid replay is required.')
    captions['figS19_full_grid_example_2001']=caption
    return captions


if __name__=='__main__':main(HERE/'figure_preview')
