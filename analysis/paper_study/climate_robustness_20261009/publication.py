"""Publication displays for pairing sensitivity and annual canopy damage."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .run import HERE,read


def supplementary_figures(destination):
    from analysis.paper_study.nature_food_impact_20261008.figures import export,clean,BLUE,GOLD,GREEN,GREY
    from analysis.paper_study.nature_food_impact_20261008.data import REGIONS
    names=['Europe',*REGIONS]
    pair=read('pairing_sensitivity').query("scenario == 'ssp585' and period == '2071-2100' and weighting == 'harvested_area'").set_index('environment_region').loc[names]
    support=read('support_ensemble').query("scenario == 'ssp585' and period == '2071-2100' and weighting == 'harvested_area'")
    fig,axes=plt.subplots(1,2,figsize=(10.2,5.8),gridspec_kw={'width_ratios':[1,1.25]})
    fig.subplots_adjust(left=.12,right=.98,bottom=.18,top=.89,wspace=.40)
    y=np.arange(len(names))
    ax=axes[0]
    ax.barh(y,1000*pair.max_absolute_difference,color=BLUE,height=.65)
    ax.set_yticks(y,names);ax.invert_yaxis();ax.set_xlabel('Largest change across year pairings\n(thousandths of a HAD day)',fontsize=9)
    ax.set_title('a   Sensitivity to incomplete year pairs',loc='left',fontsize=10,weight='bold',pad=12);clean(ax)
    ax=axes[1]
    for offset,method,label,color,marker in [(-.18,'positional_pairs','Original pairs',BLUE,'o'),(0,'separate_available_periods','Separate period means',GOLD,'s'),(.18,'complete_common_cells','Complete common cells',GREEN,'^')]:
        p=support[support.method.eq(method)].set_index('environment_region').loc[names]
        ax.scatter(p.change,y+offset,s=22,color=color,marker=marker,label=label,zorder=3)
    ax.set_yticks(y,[]);ax.set_ylim(len(names)-.5,-.5);ax.axvline(0,color=GREY,lw=.8,ls='--');clean(ax)
    ax.set_xlabel('Change in normalized HAD loss (days)',fontsize=9)
    ax.set_title('b   Sensitivity to population support',loc='left',fontsize=10,weight='bold',pad=12)
    ax.legend(frameon=False,fontsize=8,loc='upper center',bbox_to_anchor=(.46,-.12),ncol=1)
    stem='figS28_year_pairing_and_population'
    export(fig,destination,stem)
    pair.reset_index().to_csv(destination/f'{stem}_pairing.csv',index=False)
    support.to_csv(destination/f'{stem}_support.csv',index=False)
    captions={stem:('Figure S28 | Sensitivity to year pairing and population support under late-century SSP5–8.5. '
        '(a) Largest absolute difference from the original mean across all 30 circular shifts of future-year positions. '
        'The horizontal scale uses thousandths of a day. Shifts change the overlap of incomplete seasons; they are not independent replicates. '
        '(b) Harvested-area-weighted changes using original valid pairs, separately available period means, or cells complete in every year and all three climate models. '
        'The common complete population retains 98.1% of European baseline harvested area. Points average three model-specific means. '
        'Pairing and support sensitivity describe aggregation; they do not assess biological prediction accuracy.')}
    tails=read('annual_distribution_ensemble').query("scenario == 'ssp585' and period == '2071-2100' and weighting == 'baseline_production'").set_index('environment_region').loc[names]
    fig,axes=plt.subplots(1,2,figsize=(10.2,5.8))
    fig.subplots_adjust(left=.12,right=.98,bottom=.20,top=.89,wspace=.38)
    ax=axes[0]
    for offset,key,label,color in [(-.12,'historical','1991–2020',BLUE),(.12,'future','2071–2100',GOLD)]:
        ax.hlines(y+offset,tails[key+'_q10'],tails[key+'_q90'],color=color,lw=2,label=label)
        ax.scatter(tails[key+'_mean'],y+offset,color=color,s=22,zorder=3)
    ax.set_yticks(y,names);ax.invert_yaxis();clean(ax)
    ax.set_xlabel('Annual normalized HAD loss (days)',fontsize=9)
    ax.set_title('a   Annual mean and 10th–90th quantiles',loc='left',weight='bold',fontsize=10,pad=12)
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.13),ncol=2,fontsize=8.5)
    ax=axes[1]
    v=100*tails.future_exceedance_fraction
    ax.errorbar(v,y,xerr=[v-100*tails.future_exceedance_fraction_gcm_min,100*tails.future_exceedance_fraction_gcm_max-v],color=GOLD,marker='o',ls='none',capsize=3,ms=5)
    ax.set_yticks(y,[f'{100*c:.1f}%' for c in tails.complete_support_fraction]);ax.yaxis.tick_right()
    ax.set_ylim(len(names)-.5,-.5);ax.set_xlim(-3,103);clean(ax)
    ax.axvline(10,color=GREY,ls='--',lw=.8)
    ax.set_xlabel('Years exceeding historical 90th percentile (%)',fontsize=8.7)
    ax.set_title('b   High-damage years · retained production',loc='left',weight='bold',fontsize=10,pad=12)
    stem='figS29_annual_canopy_damage_distribution'
    export(fig,destination,stem)
    tails.reset_index().to_csv(destination/f'{stem}.csv',index=False)
    captions[stem]=('Figure S29 | Interannual regional canopy damage on an identical complete population. '
        'SPAM2020 production weights aggregate annual canopy damage under SSP5–8.5; the same cells are complete in both 30-year periods and all three climate models. '
        '(a) Points average model-specific annual means; segments connect model-specific 10th and 90th quantiles averaged across models. '
        'Segments describe temporal variability, not confidence intervals. (b) Future-year frequencies above each model’s historical 90th percentile, with model min–max ranges. '
        'Right-hand labels give the share of baseline production retained by the common-cell requirement. '
        'Annual canopy damage is not harvested-yield variability or STB-attributable production loss.')
    return captions


def supplementary_tables():
    tables=[]
    support=read('support_ensemble').query("scenario == 'ssp585' and period == '2071-2100' and environment_region == 'Europe'")
    rows=[['Spatial weights','Season support','HAD change (days)','Baseline support (%)']]
    labels={'positional_pairs':'Original valid pairs','separate_available_periods':'Separate period means','complete_common_cells':'Common complete cells'}
    for r in support.itertuples():
        rows.append([r.weighting.replace('_',' '),labels[r.method],f'{r.change:+.4f}',f'{100*r.coverage_fraction:.3f}'])
    tables.append((rows,'Table S26 | European aggregation sensitivity under late-century SSP5–8.5. '
        'Common complete cells are observed in all 60 seasons and all three climate models. Original-pair coverage is a weighted area–time or production–time fraction; '
        'separate-period coverage reports the smaller period fraction within each climate model, averaged across models. Common-cell coverage is the fixed spatial share. '
        'Rainfed weights change aggregation only. The largest European harvested-area mean difference across 30 circular pairings is less than 0.001 day.'))
    q=read('annual_distribution_ensemble').query("period == '2071-2100' and environment_region == 'Europe' and weighting == 'baseline_production'")
    rows=[['SSP','Historical 90th quantile (days)','Future 90th quantile (days)','Future exceedance (%)','Model range (%)','Retained production (%)']]
    for r in q.itertuples():
        rows.append([r.scenario.upper(),f'{r.historical_q90:.3f}',f'{r.future_q90:.3f}',f'{100*r.future_exceedance_fraction:.1f}',
            f'{100*r.future_exceedance_fraction_gcm_min:.1f}–{100*r.future_exceedance_fraction_gcm_max:.1f}',f'{100*r.complete_support_fraction:.2f}'])
    tables.append((rows,'Table S27 | European annual canopy-damage distribution on common complete cells. '
        'Each model supplies 30 annual production-weighted values per period; quantiles use linear interpolation. Future exceedance uses strict comparison to the same model’s historical 90th quantile. '
        'Three-model averages and ranges retain separate climate-model thresholds. Empirical frequencies are conditional model outcomes, not probabilities calibrated against harvested yields.'))
    rows=[['Environmental region','Total production (Mt)','Rainfed production (Mt)','Irrigated production (%)','Source residual (t)']]
    for r in read('crop_population_coverage').itertuples():
        rows.append([r.environment_region,f'{r.baseline_production_tonnes/1e6:.3f}',f'{r.rainfed_production_tonnes/1e6:.3f}',f'{r.irrigated_share_pct:.3f}',f'{r.source_component_residual_tonnes:+.3f}'])
    tables.append((rows,'Table S28 | Observed irrigation-component coverage of the all-wheat production baseline. '
        'SPAM2020 total and management-component rasters retain their published values and small non-additive residual. '
        'The total-production raster remains the baseline. No winter–spring wheat partition is inferred from these data, and rainfed reweighting does not validate the imposed crop calendar.'))
    return tables


def source_paths():
    return [HERE/(name+'.csv') for name in ['pairing_sensitivity','support_ensemble','annual_domain_values',
        'annual_distribution_by_gcm','annual_distribution_ensemble','complete_population_coverage','crop_population_coverage']]
