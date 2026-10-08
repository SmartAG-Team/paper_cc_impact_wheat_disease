"""Published yield data and source-backed supplementary diagnostic figures."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,TwoSlopeNorm
from matplotlib.lines import Line2D
from .published_response import HERE,ROOT

STUDY=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006'
VALIDATION=ROOT/'analysis/paper_study/full_validation_20261007'
BLUE='#235e83';GREEN='#52796f';GREY='#6e7880';GOLD='#a17937'


def plot_style(ax):
    ax.grid(color='#e9edef',linewidth=.6);ax.set_axisbelow(True);ax.tick_params(labelsize=8.6)


def published_pairs():
    return pd.read_csv(HERE/'foulkes2006_group_separated_predictions.csv')


def main_yield(destination,export):
    replay=pd.read_csv(STUDY/'manuscript/replay/full_season_top3_timing_and_conditional_yield.csv')
    pairs=published_pairs()
    fig,axes=plt.subplots(1,2,figsize=(8.8,4.5),gridspec_kw={'width_ratios':[1,1.12]})
    fig.subplots_adjust(left=.075,right=.98,bottom=.18,top=.79,wspace=.27)
    ranks=np.arange(1,4)
    for suffix,offset,color,label in [('infection',-.17,BLUE,'Effective infection'),('symptom',.17,GREEN,'Symptoms')]:
        values=[replay[f'F{leaf}_{suffix}_relative_anthesis_days'].dropna().to_numpy() for leaf in ranks]
        bp=axes[0].boxplot(values,positions=ranks+offset,widths=.28,patch_artist=True,manage_ticks=False,
            medianprops={'color':'white','linewidth':1.2},whiskerprops={'color':color},capprops={'color':color},
            flierprops={'marker':'.','markersize':2,'markeredgecolor':color})
        for box in bp['boxes']:box.set(facecolor=color,edgecolor=color)
    axes[0].set_xticks(ranks,['F1','F2','F3'])
    axes[0].set(xlim=(.5,3.5),ylabel='Days relative to flowering proxy',xlabel='Final leaf rank')
    axes[0].axhline(0,color=GREY,linestyle='--',linewidth=.8)
    axes[0].set_title('a   Seasonal model timing',loc='left',fontsize=10.3,weight='bold')
    axes[0].legend(handles=[Line2D([],[],color=BLUE,linewidth=4,label='Effective infection'),
        Line2D([],[],color=GREEN,linewidth=4,label='Symptoms')],frameon=False,fontsize=8.4,ncol=2,
        loc='lower left',bbox_to_anchor=(0,1.08),handlelength=1.2)
    domains={'STB':(GREY,'o','STB site'), 'stripe rust':(BLUE,'s','Yellow-rust site'),
        'STB;stripe rust':(GREEN,'D','Both disease sites')}
    for domain,(color,marker,label) in domains.items():
        rows=pairs.loc[pairs.genotype_observed_target_diseases.eq(domain)]
        axes[1].scatter(100*rows.derived_HAD_loss_fraction,100*rows.derived_yield_loss_fraction,
            color=color,marker=marker,s=38,edgecolors='white',linewidth=.5,label=label,zorder=3)
    coefficient=pd.read_csv(HERE/'published_yield_response_metrics.csv').set_index('response').loc['relative','full_data_slope']
    x=np.linspace(0,70,200);axes[1].plot(x,coefficient*x,color=GREY,linewidth=1.,linestyle='--')
    axes[1].set(xlim=(0,70),ylim=(0,45),xlabel='Top-five HAD loss (% of protected mean)',
        ylabel='Grain yield loss (% of protected mean)')
    axes[1].set_title('b   Published paired yield evidence',loc='left',fontsize=10.3,weight='bold')
    axes[1].legend(frameon=False,fontsize=8.1,ncol=2,loc='lower left',bbox_to_anchor=(0,1.065),
        handlelength=1.,columnspacing=.8)
    for ax in axes:plot_style(ax)
    export(fig,Path(destination),'fig4_timing_and_yield_relevance')
    return ('Figure 4 | Upper-leaf timing and published healthy-area/yield response evidence. '
        '(a) Model effective-infection and symptom dates for F1–F3 across218 complete field replays; zero denotes the flowering proxy. '
        'Boxes show interquartile ranges and medians, with Tukey whiskers and individual outliers. These infection dates are model states. '
        '(b) Eight published genotype-by-treatment mean contrasts from Foulkes et al. (2006), using top-five postanthesis HAD and '
        'grain yield at85% dry matter. Points distinguish genotypes observed at STB, yellow-rust or both sites. '
        'The dashed line is a descriptive group-balanced proportional fit within that top-five domain; group-separated predictions '
        'and both absolute and relative response metrics are retained separately. The values are mixed-model predicted treatment means, '
        'not independent raw plots. The current top-three transfer remains anchored to Parker et al. (2004), whose117 reported yield '
        'contrasts and25 cultivar slopes are retained in Figures S9–S10 and Table S7. Different leaf scopes are not pooled. '
        'Published yield-response evidence supports the functional-area connection; it does not validate actual grain loss in the current STB field simulations.')


def german_dates(destination,export):
    data=pd.read_parquet(VALIDATION/'german_testing_tpv_events.parquet')
    data=data.loc[data.common_matched].copy()
    for column in ['true_date','predicted_date','SOWING_DATE']:data[column]=pd.to_datetime(data[column])
    data['observed_days_after_sowing']=(data.true_date-data.SOWING_DATE).dt.days
    data['predicted_days_after_sowing']=(data.predicted_date-data.SOWING_DATE).dt.days
    fig,axes=plt.subplots(2,2,figsize=(7.8,7.2));hexagons=[]
    fig.subplots_adjust(left=.095,right=.88,bottom=.075,top=.925,wspace=.28,hspace=.31)
    for ax,stage,letter in zip(axes.flat,[10,31,51,85],'abcd'):
        rows=data.loc[data.BBCH.eq(stage)]
        maximum=10*np.ceil(max(rows.observed_days_after_sowing.max(),rows.predicted_days_after_sowing.max())/10)
        minimum=10*np.floor(min(rows.observed_days_after_sowing.min(),rows.predicted_days_after_sowing.min())/10)
        hexagons.append(ax.hexbin(rows.observed_days_after_sowing,rows.predicted_days_after_sowing,
            gridsize=35,mincnt=1,cmap='Blues',linewidths=0.,norm=LogNorm(vmin=1)))
        ax.plot([minimum,maximum],[minimum,maximum],color=GREY,linestyle='--',linewidth=.8)
        ax.set(xlim=(minimum,maximum),ylim=(minimum,maximum),xlabel='Observed days after sowing',ylabel='Predicted days after sowing')
        ax.set_aspect('equal');ax.set_title(f'{letter}   BBCH{stage}; n={len(rows):,}',loc='left',fontsize=10.,weight='bold')
        ax.xaxis.label.set_size(9.);ax.yaxis.label.set_size(9.);ax.tick_params(labelsize=8.5)
    maximum=max(float(h.get_array().max()) for h in hexagons)
    for h in hexagons:h.set_clim(1,maximum)
    cax=fig.add_axes([.91,.25,.018,.47]);bar=fig.colorbar(hexagons[-1],cax=cax);bar.set_label('Events per hexagon',fontsize=9.)
    export(fig,Path(destination),'figS5_german_event_dates')
    return ('Figure S5 | Retained T–P–V predictions on the German station test cohort at BBCH10,31,51 and85. '
        'Every common matched test event is represented, including date-error tails. Both axes use days from the recorded sowing anchor '
        'to avoid calendar-year ambiguity. Hexagon colour shows count on a shared logarithmic scale; axis ranges vary by stage. '
        'Missing predictions remain in the coverage tables. The station holdout concerns realized-weather phenology, rather than disease or field-yield forecasts.')


def onset_distances(destination,export):
    all_rows=pd.read_parquet(STUDY/'disease/all_first_onset_bracket_predictions.parquet')
    rows=all_rows.loc[all_rows.scenario.eq('baseline')&all_rows.censoring.eq('two_sided')&all_rows.leaf_index.lt(3)
        &all_rows.partition.isin(['reused_BASF2019','reused_strict_Corteva'])].copy()
    rows.to_csv(Path(destination)/'figS6_leaf_onset_distances.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(8.,4.6),sharey=True)
    fig.subplots_adjust(left=.085,right=.985,bottom=.19,top=.79,wspace=.19)
    for ax,partition,title,letter in zip(axes,['reused_BASF2019','reused_strict_Corteva'],['BASF 2019','Corteva location-disjoint'],'ab'):
        for offset,model,color,marker in [(-.17,'overwinter_source_model',BLUE,'s'),(.17,'phenology_only',GREY,'o')]:
            for leaf in range(3):
                values=rows.loc[rows.partition.eq(partition)&rows.model.eq(model)&rows.leaf_index.eq(leaf)].sort_values('endpoint_series')
                shifts=np.linspace(-.085,.085,len(values)) if len(values)>1 else np.zeros(len(values))
                present=values.predicted_symptom_day.ge(0)
                ax.scatter(leaf+offset+shifts[present],values.loc[present,'onset_distance_days'],s=20,
                    color=color,marker=marker,alpha=.65,linewidths=.6,facecolors='none' if marker=='o' else color)
                if (~present).any():ax.scatter(leaf+offset+shifts[~present],values.loc[~present,'onset_distance_days'],s=35,marker='x',color=color)
        counts=[int((rows.partition.eq(partition)&rows.model.eq('overwinter_source_model')&rows.leaf_index.eq(leaf)).sum()) for leaf in range(3)]
        ax.set_xticks(range(3),[f'F{rank+1}\nn={n}' for rank,n in enumerate(counts)])
        ax.set(xlim=(-.5,2.5),xlabel='Leaf-specific symptom histories')
        ax.set_title(f'{letter}   {title}',loc='left',fontsize=10.,weight='bold');plot_style(ax)
    axes[0].set_ylabel('Interval-excess loss (days)')
    fig.legend(handles=[Line2D([],[],marker='s',color=BLUE,linestyle='none',label='Seasonal model'),
        Line2D([],[],marker='o',markerfacecolor='none',color=GREY,linestyle='none',label='Phenology benchmark'),
        Line2D([],[],marker='x',color=GREY,linestyle='none',label='Missing predicted symptom')],
        loc='upper center',ncol=3,frameon=False,fontsize=9.)
    export(fig,Path(destination),'figS6_leaf_onset_distances')
    return ('Figure S6 | Individual F1–F3 symptom-onset losses under the frozen seasonal model and independently selected phenology benchmark. '
        'Only genuine two-sided histories are included. Horizontal displacement separates coincident records without changing loss values. '
        'Crosses identify missing predicted symptoms, whose registered loss includes the missing-positive penalty. These are interval-excess '
        'losses rather than exact-date errors. The figure retains individual histories; the main metrics use coordinate-year, field and leaf hierarchy.')


def stage_profiles(destination,export):
    frame=pd.read_csv(STUDY/'phenology/ordered_threshold_joint_profiles.csv')
    selected=json.loads((STUDY/'phenology/calibrated_stage_thresholds.json').read_text())
    minimum=selected['minimum_loss_days']
    fig,axes=plt.subplots(2,2,figsize=(7.8,6.3));fig.subplots_adjust(left=.09,right=.98,bottom=.09,top=.94,wspace=.24,hspace=.38)
    for ax,event,letter in zip(axes.flat,[32,33,37,39],'abcd'):
        part=frame.loc[frame.event.eq(event)]
        ax.plot(part.threshold,part.joint_profile_loss_days-minimum,color=BLUE,linewidth=1.2)
        ax.axvline(selected['thresholds'][str(event)],color=GOLD,linestyle='--',linewidth=1.)
        for value in [1,2]:ax.axhline(value,color=GREY,linestyle=':',linewidth=.8)
        ax.set(ylim=(-.06,3.05),xlabel='Development threshold (TPV units)',ylabel='Joint loss above minimum (days)')
        ax.set_title(f'{letter}   BBCH{event}',loc='left',fontsize=10.,weight='bold');plot_style(ax)
        ax.xaxis.label.set_size(9.);ax.yaxis.label.set_size(9.)
    export(fig,Path(destination),'figS7_stage_identification_profiles')
    return ('Figure S7 | Ordered stage-threshold identification from calibration-only observations. '
        'Each blue curve is the minimum joint loss conditional on that threshold, after optimizing the other ordered stage thresholds. '
        'Dashed vertical lines mark the selected values; horizontal lines show declared one- and two-day loss tolerances. '
        'The displayed range is zero to three days above the joint minimum; all399 grid values for each event remain in Source Data. '
        'Loss-tolerance regions describe identification and are not statistical confidence intervals.')


def replay_exposure(destination,export):
    data=pd.read_csv(STUDY/'manuscript/replay/full_season_top3_timing_and_conditional_yield.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.,4.2));fig.subplots_adjust(left=.085,right=.985,bottom=.17,top=.91,wspace=.27)
    for source,color,marker,label in [('BASF',GREY,'o','BASF'),('Corteva',BLUE,'s','Corteva')]:
        rows=data.loc[data.source.eq(source)]
        axes[0].scatter(rows.F1_symptom_relative_anthesis_days,rows.model_proxy_had_loss_fraction,
            s=20,color=color,marker=marker,alpha=.6,label=label)
        axes[1].scatter(rows.grain_fill_days,rows.model_proxy_lost_had3,s=20,color=color,marker=marker,alpha=.6)
    axes[0].set(xlabel='F1 symptoms relative to flowering (days)',ylabel='Model functional HAD-loss fraction')
    axes[0].axvline(0,color=GREY,linestyle='--',linewidth=.8)
    axes[1].set(xlabel='Flowering–BBCH85 window (days)',ylabel='Lost GLAI-days per nominal upper-three LAI')
    for ax,letter in zip(axes,'ab'):ax.set_title(letter,loc='left',fontsize=11.,weight='bold');plot_style(ax);ax.xaxis.label.set_size(9.);ax.yaxis.label.set_size(9.)
    axes[0].legend(frameon=False,fontsize=8.8)
    export(fig,Path(destination),'figS8_field_grainfill_exposure')
    return ('Figure S8 | Conditional full-season field replay. Each point is one of218 simulated field seasons. '
        '(a) F1 symptom timing and the model functional healthy-area-duration loss fraction; (b) flowering–BBCH85 window duration '
        'and lost healthy-area duration. Reference upper-three-leaf LAI is nominally one. These are covariations within the specified '
        'simulation and do not establish observed functional loss or measured field-yield effects.')


def parker_yields(destination,export):
    matrix=pd.read_csv(HERE/'parker2004_table4_yield_loss_matrix.csv').set_index('cultivar')
    fig,ax=plt.subplots(figsize=(6.9,7.7));fig.subplots_adjust(left=.22,right=.88,bottom=.12,top=.95)
    cmap=plt.get_cmap('BrBG_r').copy();cmap.set_bad('#eeeeee')
    im=ax.imshow(matrix.to_numpy(),cmap=cmap,norm=TwoSlopeNorm(vmin=-.4,vcenter=0,vmax=4.5),aspect='auto')
    ax.set_xticks(range(6),['Starcross\n1995','Starcross\n1996','Starcross\n1997','Rosemaund\n1995','Rosemaund\n1996','Rosemaund\n1997'],fontsize=8.)
    ax.set_yticks(range(25),matrix.index,fontsize=8.7)
    ax.tick_params(length=0)
    ax.set_title('Published protected–unprotected yield contrasts',loc='left',fontsize=10.2,weight='bold',pad=12)
    cax=fig.add_axes([.91,.27,.018,.46]);bar=fig.colorbar(im,cax=cax);bar.set_label('Grain yield contrast (t ha⁻¹)',fontsize=9.)
    ax.set_xticks(np.arange(-.5,6,1),minor=True);ax.set_yticks(np.arange(-.5,25,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=.7);ax.tick_params(which='minor',length=0)
    export(fig,Path(destination),'figS9_published_STB_yield_contrasts')
    return ('Figure S9 | All117 reported protected-minus-unprotected grain-yield contrasts in Parker et al. (2004), Table4. '
        'The25 cultivars and six site-years are retained, including five negative contrasts; grey cells indicate unreported combinations. '
        'Values are published treatment contrasts from a three-replicate field design, not independent plot records. '
        'They supply empirical yield outcomes for the study whose crop-impact relationship integrates absolute green area over the top three leaves.')


def parker_slopes(destination,export):
    data=pd.read_csv(HERE/'parker2004_table5_yield_slopes.csv').sort_values('random_effect_slope_t_ha_per_GLAI_day')
    fig,ax=plt.subplots(figsize=(6.9,7.6));fig.subplots_adjust(left=.24,right=.97,bottom=.09,top=.94)
    positions=np.arange(len(data))
    valid=data.fixed_effect_slope_t_ha_per_GLAI_day.notna()
    ax.errorbar(data.loc[valid,'fixed_effect_slope_t_ha_per_GLAI_day'],positions[valid]-.12,
        xerr=data.loc[valid,'fixed_effect_slope_SE'],fmt='o',color=GREY,markersize=3.,linewidth=.8,capsize=2,
        label='Fixed estimate ± 1 SE')
    ax.scatter(data.random_effect_slope_t_ha_per_GLAI_day,positions+.12,color=BLUE,marker='s',s=17,
        label='Random-effects prediction',zorder=3)
    ax.axvline(.018,color=GOLD,linestyle='--',linewidth=1.,label='Transferred centre: 0.018')
    ax.set_yticks(positions,data.cultivar,fontsize=8.8);ax.invert_yaxis()
    ax.set_xlabel('Yield sensitivity (t ha⁻¹ per top-three GLAI-day)',fontsize=9.3)
    ax.set_title('Published cultivar response coefficients',loc='left',fontsize=10.2,weight='bold',pad=12)
    ax.legend(frameon=False,fontsize=8.3,loc='upper left',bbox_to_anchor=(-.26,-.105),ncol=1)
    ax.grid(axis='x',color='#e9edef',linewidth=.6);ax.set_axisbelow(True)
    export(fig,Path(destination),'figS10_top3_yield_response_coefficients')
    return ('Figure S10 | All25 cultivar response predictions from Parker et al. (2004), Table5, with available fixed-effect estimates '
        'and their published standard errors. The dashed line identifies the retained0.018 t ha⁻¹ per GLAI-day transfer centre. '
        'Random-effect predictions span0.0141–0.0207; that cultivar range is not a confidence interval. Error bars represent one '
        'standard error of fixed-effect coefficients. BLUP prediction-error variances concern deviations from the population mean and '
        'are not substituted for uncertainty of complete slopes. Different cultivars share environments and are not independent field replications.')


def group_benchmark(destination,export):
    data=published_pairs();metrics=pd.read_csv(HERE/'published_yield_response_metrics.csv').set_index('response')
    fig,axes=plt.subplots(1,2,figsize=(8.,4.5));fig.subplots_adjust(left=.08,right=.985,bottom=.25,top=.92,wspace=.28)
    for ax,mode,actual,scale,label,letter in zip(axes,['relative','absolute'],
            ['derived_yield_loss_fraction','derived_yield_loss_t_ha'],[100,1],['Yield loss (% of protected mean)','Yield loss (t ha⁻¹)'],'ab'):
        x=scale*data[actual];y=scale*data[f'{mode}_held_out_prediction']
        for background,marker in zip(data.genetic_background.unique(),['o','s','D','^','v']):
            selected=data.genetic_background.eq(background)
            ax.scatter(x[selected],y[selected],s=38,marker=marker,color=BLUE,edgecolors='white',linewidth=.5,label=background)
        limit=45 if mode=='relative' else 3.
        ax.plot([0,limit],[0,limit],color=GREY,linestyle='--',linewidth=.9)
        ax.set(xlim=(0,limit),ylim=(0,limit),xlabel='Published '+label.lower(),ylabel='Held-background prediction')
        ax.set_title(f'{letter}   {mode.capitalize()} transfer',loc='left',fontsize=10.1,weight='bold');plot_style(ax)
        score=metrics.loc[mode]
        unit=' percentage points' if mode=='relative' else ' t ha⁻¹'
        ax.text(0,-.29,f'RMSE {scale*score.group_balanced_RMSE:.2f}{unit}\nGroup-balanced R² {score.group_balanced_R2:.3f}',
            transform=ax.transAxes,fontsize=8.5,va='top',linespacing=1.5)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,frameon=False,fontsize=7.8,ncol=5,loc='upper center',
        bbox_to_anchor=(.52,1.045),handlelength=1.,columnspacing=1.3)
    export(fig,Path(destination),'figS11_published_yield_group_holdout')
    return ('Figure S11 | Leave-one-genetic-background-out reproduction of the published top-five paired means. '
        'One nonnegative origin-constrained coefficient is fitted to the other four genetic backgrounds in each fold. '
        '(a) Relative HAD-loss to relative yield-loss transfer; (b) absolute HAD-loss to absolute grain-loss transfer. '
        'Metrics give each background equal weight, divided among its lines. This is a small aggregate-data benchmark with shared '
        'mixed-model means and unbalanced disease/site coverage. It is not raw-plot validation, does not remove environmental confounding '
        'and does not calibrate the current top-three transfer. Negative absolute-response R² identifies failure of a universal absolute slope.')


def cross_disease(destination,export):
    yellow=pd.read_csv(HERE/'bryson1997_reported_yield_fits.csv')
    brown=pd.read_csv(HERE/'subbarao1989_reported_yield_fits.csv')
    fig,axes=plt.subplots(1,2,figsize=(8.,4.4),sharey=True)
    fig.subplots_adjust(left=.08,right=.985,bottom=.18,top=.84,wspace=.22)
    for offset,predictor,color in [(-.18,'Healthy area duration',GREY),(.18,'Healthy area absorption',BLUE)]:
        frame=yellow.loc[yellow.predictor.eq(predictor)].sort_values('year')
        bars=axes[0].bar(np.arange(2)+offset,frame.value,width=.32,color=color,label=predictor)
        for bar,value in zip(bars,frame.value):axes[0].text(bar.get_x()+bar.get_width()/2,value+.02,f'{value:.2f}',ha='center',fontsize=8.4)
    axes[0].set_xticks([0,1],['1994\nn=60 combinations','1995\nn=52 combinations'],fontsize=8.6)
    axes[0].set_title('a   Yellow rust: canopy indicators',loc='left',fontsize=10.,weight='bold')
    axes[0].legend(frameon=False,fontsize=8.,loc='upper left',bbox_to_anchor=(0,-.30))
    for offset,status,color in [(-.18,'leaf rust affected',BLUE),(.18,'control',GREY)]:
        frame=brown.loc[brown.crop_status.eq(status)].sort_values('crop_season')
        bars=axes[1].bar(np.arange(2)+offset,frame.value,width=.32,color=color,label=status)
        for bar,value in zip(bars,frame.value):axes[1].text(bar.get_x()+bar.get_width()/2,value+.02,f'{value:.2f}',ha='center',fontsize=8.4)
    axes[1].set_xticks([0,1],['1986–87','1987–88'])
    axes[1].set_title('b   Leaf rust: leaf-specific duration',loc='left',fontsize=10.,weight='bold')
    axes[1].legend(frameon=False,fontsize=8.,loc='upper left',bbox_to_anchor=(0,-.17))
    axes[0].set_ylabel('Published fit coefficient of determination')
    for ax in axes:ax.set_ylim(0,1.);plot_style(ax)
    export(fig,Path(destination),'figS12_cross_disease_yield_evidence')
    return ('Figure S12 | Published wheat-rust crop-response evidence. (a) Bryson et al. (1997) report within-year grain-yield R² '
        'for healthy-area duration and radiation-weighted healthy-area absorption under yellow rust. (b) Subba Rao et al. (1989) '
        'report adjusted R² for leaf-specific relative healthy-area duration and tiller grain weight under leaf rust and control conditions. '
        'These are original fitted-study summaries rather than independent prediction scores. Canopy radiation, leaf rank, outcome scale '
        'and experimental design differ between the studies; bars are not pooled or used to rank diseases or calibrate a common disease coefficient.')


def supplementary(destination,export):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    plots=[german_dates,onset_distances,stage_profiles,replay_exposure,parker_yields,parker_slopes,group_benchmark,cross_disease]
    stems=['figS5_german_event_dates','figS6_leaf_onset_distances','figS7_stage_identification_profiles','figS8_field_grainfill_exposure',
        'figS9_published_STB_yield_contrasts','figS10_top3_yield_response_coefficients','figS11_published_yield_group_holdout','figS12_cross_disease_yield_evidence']
    return {stem:plot(destination,export) for stem,plot in zip(stems,plots)}
