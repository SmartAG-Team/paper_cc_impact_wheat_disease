"""Six source-backed crop infection, canopy damage and conditional yield figures."""
from pathlib import Path
import hashlib,json,shutil
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
    onset=pd.read_csv(HERE/'validation/paired_onset_primary.csv')
    onset=onset[onset.leaf_scope.eq('top3_source_numbered')&onset.comparison_population.eq('original_eligible_records')&onset.metric.eq('difference')&onset.comparator.eq('phenology_only')]
    severity=pd.read_csv(HERE/'validation/paired_severity_primary.csv')
    severity=severity[severity.endpoint.eq('all_assessments_severity_pp')&severity.comparison_population.eq('original_eligible_records')&severity.comparator.eq('calibration_leaf_mean')]
    parts=['reused_BASF2019','reused_strict_Corteva'];labels=['BASF 2019','Corteva\nsource-numbered leaves']
    onset=onset.set_index('partition').loc[parts].reset_index()
    severity.to_csv(destination/'fig3_severity_bootstrap_source.csv',index=False);onset.to_csv(destination/'fig3_paired_onset_source.csv',index=False)
    fig,axes=plt.subplots(1,3,figsize=(11.7,4.65));fig.subplots_adjust(left=.075,right=.985,bottom=.24,top=.86,wspace=.45)
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
    for domain,label in [('endpoint67','Nordic endpoint\n67 contrasts; 5 trials'),('integral50','Nordic window\n50 contrasts; 5 trials')]:
        for family,suffix in [('Linear','_loto'),('Exponential','_damage_loto')]:
            row=nordic[nordic.domain.eq(domain+suffix)].iloc[0]
            records.append(dict(domain=domain,label=label,family=family,model_RMSE_pp=row.model_rmse_pp,training_mean_RMSE_pp=row.baseline_rmse_pp,
                normalized_RMSE=row.model_rmse_pp/row.baseline_rmse_pp,contrasts=int(row.rows),groups=int(row.independent_trials)))
    for family,function in [('Linear','linear'),('Exponential','exponential')]:
        row=basf[basf.domain.eq('all_three_separate_windows')&basf.function.eq(function)].iloc[0]
        records.append(dict(domain='BASF_top3',label='BASF all-three\n12 contrasts; 11 fields',family=family,model_RMSE_pp=row.RMSE_pp,
            training_mean_RMSE_pp=row.baseline_RMSE_pp,normalized_RMSE=row.RMSE_pp/row.baseline_RMSE_pp,
            contrasts=int(row.evaluation_contrasts),groups=int(row.evaluation_fields)))
    ratios=pd.DataFrame(records);ratios.to_csv(destination/'fig4_holdout_response_errors.csv',index=False);points.to_csv(destination/'fig4_Foulkes_held_background_points.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(10.5,4.8));fig.subplots_adjust(left=.08,right=.985,bottom=.27,top=.85,wspace=.34)
    for background,marker,color in zip(points.genetic_background.unique(),['o','s','D','^','v'],[BLUE,GREEN,GOLD,GREY,DARK]):
        p=points[points.genetic_background.eq(background)]
        axes[0].scatter(100*p.derived_yield_loss_fraction,100*p.relative_held_out_prediction,s=43,marker=marker,color=color,label=background,edgecolors='white',lw=.6,zorder=3)
    axes[0].plot([0,45],[0,45],ls='--',color=GREY,lw=.9);axes[0].set(xlim=(0,45),ylim=(0,45),xlabel='Published relative yield loss (%)',ylabel='Held-background prediction (%)')
    title(axes[0],'a   Published canopy–yield holdout');clean(axes[0]);axes[0].set_aspect('equal',adjustable='box')
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,frameon=False,ncol=5,fontsize=8,loc='lower left',bbox_to_anchor=(.075,.045),columnspacing=1.2,handlelength=1.)
    score=scores.loc['relative'];axes[0].text(.04,.96,f'8 means; 5 backgrounds\nRMSE {100*score.group_balanced_RMSE:.2f} pp; R² {score.group_balanced_R2:.3f}',transform=axes[0].transAxes,fontsize=8.5,va='top',linespacing=1.5)
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
        '(a) Published observed and leave-one-genetic-background-out relative-yield-loss means from Foulkes et al. (2006): eight contrasts across five backgrounds. '
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
    decomp=pd.read_csv(HERE/'climate/decomposition/supported_forcing_ensemble_decomposition.csv');decomp.to_csv(destination/'fig5_supported_weather_host_decomposition.csv',index=False)
    robust=pd.read_csv(HERE/'climate/ensemble_paired_changes.csv');robust=robust[robust.metric.eq('GS65_85_lost_had3')&robust.population.eq('all_settings_common_paired')]
    robust.to_csv(destination/'fig5_14_setting_HAD_robustness.csv',index=False)
    fig,axes=plt.subplots(2,2,figsize=(11.8,8.25),gridspec_kw={'height_ratios':[1,1.55]})
    fig.subplots_adjust(left=.11,right=.985,bottom=.08,top=.94,wspace=.61,hspace=.51)
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
    setting_order=['baseline','primary_objective_tied_runnerup','local_source_off','rank_spacing80','rank_spacing160','stage_profile_early37','stage_profile_late37',
        'detection0p0001','detection0p01','reference_decline_end0p5','reference_decline_end0','functional_conversion0p5','functional_conversion0p75','decline_end0_conversion0p5']
    names=['Baseline','Primary-objective runner-up','Initial local source = 0','Rank spacing = 80','Rank spacing = 160','Earlier37 profile','Later37 profile',
        'Detection = 0.0001','Detection = 0.01','Reference → 0.5','Reference → 0','Conversion c = 0.5','Conversion c = 0.75','Reference → 0; c = 0.5']
    ax=axes[1,1];data=robust.set_index('setting').loc[setting_order]
    for y,row in zip(range(14),data.itertuples()):
        ax.plot([row.gcm_min_change,row.gcm_max_change],[y,y],color='#b1b6ba',lw=2.5)
        ax.errorbar(row.gcm_mean_change,y,xerr=row.spatial_mcse_gcm_mean_change,fmt='o',color=BLUE,ms=4,capsize=2)
    ax.set_yticks(range(14),names,fontsize=7.6);ax.invert_yaxis();ax.axvline(0,color=GREY,ls='--',lw=.8);ax.grid(axis='x',color='#e8ecee',lw=.65)
    ax.set_xlabel('HAD proxy change (days / nominal upper-three LAI)',fontsize=8.8);ax.tick_params(axis='x',labelsize=8.3)
    title(ax,'d   SSP5–8.5 structural sensitivities')
    fig.text(.11,.012,'Gray segments: three-GCM ranges. Colored bars: ±1 spatial Monte Carlo SE, retaining shared draws. Scenario ranges are not confidence intervals.',fontsize=8,color=GREY)
    export(fig,destination,'fig5_conditional_climate_changes')
    return ('Figure 5 | Climate, disease-weather and host-development effects on canopy damage. '
        '(a,b) Paired changes in modeled F1 symptom timing relative to anthesis and GS65–85 HAD proxy loss under all three SSPs, for 2031–2060 and 2071–2100 relative to 1991–2020. '
        '(c) Late-century SSP5–8.5 disease-weather and host-development Shapley contributions on the supported-forcing common population. '
        'Twenty pairs with mean temperature above 40 °C in either active hybrid are excluded, leaving 5,731 pairs and 99.205–99.864% weighted coverage. '
        'The two Shapley terms sum to the supported net change; the full interaction is displayed separately and is already allocated equally between them. '
        '(d) Fourteen registered structural, detection, reference-canopy and conversion scenarios on common finite HAD pairs (1,913–1,920 per GCM; 99.9719–100% coverage). '
        'Points average three GCMs using the same 64 spatial draws in 16 strata; gray segments span GCM means, and colored bars show ±1 spatial Monte Carlo standard error with shared-draw covariance retained. '
        'Timing populations additionally require detected symptoms. Ranges are not confidence intervals. '
        'All outputs are within-model quantities per nominal upper-three LAI; disease-weather and host combinations have no physical causal or actual-yield attribution.')


def maps_figure(destination):
    def map_export(fig,path,stem):
        if fig._suptitle is not None:fig._suptitle.set_text('European wheat area and climate-linked conditional production responses')
        for ax in fig.axes:
            if 'conditional yield transfer' in ax.get_title(loc='left'):ax.set_title('d   Change in conditional yield-loss proxy',loc='left',fontsize=9.6,fontweight='bold')
            if 'Change in t ha' in ax.get_xlabel():ax.set_xlabel('Conditional yield effect: Δloss\n(t ha⁻¹ per nominal upper-three LAI)',fontsize=8.2)
        export(fig,path,stem)
    main_maps(destination,map_export)
    m=ROOT/'analysis/paper_study/map_restoration_20261007'
    for name,target in [('map_ensemble_values.csv','fig6_spatial_ensemble_source.csv'),('map_wheat_domain.csv','fig6_wheat_domain_source.csv'),('map_field_locations.csv','fig6_field_location_source.csv')]:shutil.copy2(m/name,destination/target)
    return ('Figure 6 | European wheat production context and spatially explicit climate–infection diagnostics. '
        '(a) Fixed SPAM2020 harvested area in 14,941 wheat cells with the locations of 28 calibration, 45 BASF2019 and 143 location-disjoint Corteva field seasons. '
        'The logarithmic colour scale denotes harvested hectares per 0.25° cell. '
        '(b) Three-GCM mean modeled F1 symptom timing relative to anthesis in 1991–2020 under the SSP5–8.5 reference continuation. '
        '(c,d) Paired 2071–2100 minus 1991–2020 SSP5–8.5 changes in symptom timing and conditional yield effect at coefficient 0.018 t ha⁻¹ per GLAI-day; positive yield-effect values denote greater modeled loss. '
        'Circles in b–d show 62 selected cells from 64 registered spatial draws; duplicate locations are collapsed only for display. '
        'Actual point coordinates and computed values are retained without interpolation to uncomputed cells. '
        'Colours show conditional point estimates, not confidence intervals or a completed current-model European census. '
        'The fixed harvested-area layer places sampled climate responses in the European wheat-production domain. Disease and yield panels remain conditional diagnostics because field disease and absolute-yield transfer are not validated; the mapped yields are not total production losses.')


def main(destination:Path)->dict[str,str]:
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
    for name in ['fig2_field_stage_intervals.csv','fig2_german_stage_errors.csv','fig4_specified_yield_scenarios.csv']:
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


if __name__=='__main__':main(HERE/'figure_preview')
