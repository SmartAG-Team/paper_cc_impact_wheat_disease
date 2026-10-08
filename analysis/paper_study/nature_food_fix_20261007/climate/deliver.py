"""Exact manuscript-facing robustness numbers and a standalone scientific figure."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    independent=json.loads((HERE/'independent_verification.json').read_text());assert independent['status']=='verified'
    ensemble=pd.read_parquet(HERE/'ensemble_paired_changes.parquet')
    reg=json.loads((HERE/'registration_before_robustness_results.json').read_text())
    metrics=['F1_infection_relative_anthesis_days','F1_symptom_relative_anthesis_days','F2_infection_relative_anthesis_days','F2_symptom_relative_anthesis_days',
        'F3_infection_relative_anthesis_days','F3_symptom_relative_anthesis_days','GS65_85_lost_had3','GS31_85_lost_had3','GS65_85_functional_lost_fraction',
        'conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai','conditional_yield_loss_GS31_85_b0180_t_ha_per_unit_lai','any_top3_infection_before85','any_top3_symptom_before85']
    headline=ensemble.loc[ensemble.metric.isin(metrics)].copy()
    moments=pd.read_parquet(HERE/'paired_draw_moments.parquet')
    by_gcm=pd.read_parquet(HERE/'paired_changes_by_setting_gcm.parquet')
    draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv')
    common='all_settings_common_paired'
    vectors={}
    for (setting,metric),group in moments.loc[moments.population.eq(common)&moments.metric.isin(metrics)].groupby(['setting','metric']):
        parts=[]
        for model,g in group.groupby('model',sort=True):
            g=g.sort_values('spatial_draw_id')
            row=by_gcm.loc[by_gcm.setting.eq(setting)&by_gcm.model.eq(model)&by_gcm.metric.eq(metric)&by_gcm.population.eq(common)].iloc[0]
            parts.append((g.numerator.to_numpy()-row.change*g.denominator.to_numpy())/row.common_paired_area_time_coverage)
        vectors[(setting,metric)]=np.mean(parts,axis=0)
    differences=[]
    for metric in metrics:
        part=headline.loc[headline.population.eq(common)&headline.metric.eq(metric)]
        baseline=part.loc[part.setting.eq('baseline')].iloc[0]
        for row in part.itertuples():
            influence=vectors[(row.setting,metric)]-vectors[('baseline',metric)]
            variance=0.
            for _,g in draws.groupby('stratum',sort=False):
                idx=g.index.to_numpy(int);n=len(g);weight=float(g.area_mean_weight.sum())
                variance+=weight**2*np.var(influence[idx],ddof=1)/n
            differences.append(dict(setting=row.setting,metric=metric,population=common,
                difference_in_climate_change_from_baseline=row.gcm_mean_change-baseline.gcm_mean_change,
                shared_draw_spatial_mcse=float(np.sqrt(variance))))
    pd.DataFrame(differences).to_csv(HERE/'paired_setting_difference_from_baseline.csv',index=False)
    records={}
    for population in headline.population.unique():
        records[population]={}
        for metric in metrics:
            part=headline.loc[headline.population.eq(population)&headline.metric.eq(metric)]
            baseline=part.loc[part.setting.eq('baseline')].iloc[0]
            records[population][metric]=dict(baseline=baseline.to_dict(),
                settings=part.to_dict(orient='records'),setting_mean_min=float(part.gcm_mean_change.min()),setting_mean_max=float(part.gcm_mean_change.max()),
                all_setting_gcm_min=float(part.gcm_min_change.min()),all_setting_gcm_max=float(part.gcm_max_change.max()),
                all_setting_means_positive=bool((part.gcm_mean_change>0).all()),all_setting_means_negative=bool((part.gcm_mean_change<0).all()),
                all_setting_all_GCM_positive=bool((part.gcm_min_change>0).all()),all_setting_all_GCM_negative=bool((part.gcm_max_change<0).all()),
                minimum_paired_coverage=float(part.minimum_gcm_paired_coverage.min()),maximum_paired_coverage=float(part.maximum_gcm_paired_coverage.max()),
                minimum_spatial_mcse=float(part.spatial_mcse_gcm_mean_change.min()),maximum_spatial_mcse=float(part.spatial_mcse_gcm_mean_change.max()))
    old=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/climate/ensemble_paired_period_changes.csv'
    cached=pd.read_csv(old);cached=cached.loc[cached.future_period.eq('2071-2100')&cached.metric.isin(metrics)]
    cached.to_csv(HERE/'cached_baseline_all_ssps.csv',index=False)
    v=records['within_setting_paired']['conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai']['baseline']['gcm_mean_change']
    wider=records['within_setting_paired']['conditional_yield_loss_GS31_85_b0180_t_ha_per_unit_lai']['baseline']['gcm_mean_change']
    coefficient_ranges={}
    for population in headline.population.unique():
        coefficient_ranges[population]={}
        for window in ['GS65_85','GS31_85']:
            names=[f'conditional_yield_loss_{window}_{label}_t_ha_per_unit_lai' for label in ['b0141','b0180','b0207']]
            p=ensemble.loc[ensemble.population.eq(population)&ensemble.metric.isin(names)]
            coefficient_ranges[population][window]=dict(setting_coefficient_mean_min=float(p.gcm_mean_change.min()),
                setting_coefficient_mean_max=float(p.gcm_mean_change.max()),all_GCM_setting_coefficient_min=float(p.gcm_min_change.min()),
                all_GCM_setting_coefficient_max=float(p.gcm_max_change.max()),coefficient_range_is_confidence_interval=False)
    result=dict(scenario='ssp585',historical_period='1991-2020',future_period='2071-2100',registered_settings=14,
        manuscript_values=records,wider_window_diagnostic=dict(GS65_85=v,GS31_85=wider,relative_increase_pct=(wider/v-1)*100),
        diagnostic_ranges_across_all_three_coefficients=coefficient_ranges,
        supplemental_cached_baseline_all_ssps_source=str(old.relative_to(ROOT)),supplemental_cached_baseline_all_ssps_source_sha256=sha(old),
        supplemental_scope='Cached original baseline only; robustness reruns restricted to SSP585.',
        daily_baseline_storage='Float32 daily state archive; integration and annual outputs float64.',
        no_European_yield_tonnage=True,no_policy_benefit_estimate=True,settings_are_not_plausibility_distribution=True,
        registration_sha256=sha(HERE/'registration_before_robustness_results.json'),independent_verification_sha256=sha(HERE/'independent_verification.json'))
    with (HERE/'results_for_manuscript.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    labels={'baseline':'Baseline','primary_objective_tied_runnerup':'Primary-objective tie (runner-up)','local_source_off':'Initial local source = 0',
        'rank_spacing80':'Effective rank spacing = 80','rank_spacing160':'Effective rank spacing = 160',
        'stage_profile_early37':'Early C37 stage profile','stage_profile_late37':'Late C37 stage profile',
        'detection0p0001':'Detection threshold = 0.0001','detection0p01':'Detection threshold = 0.01',
        'reference_decline_end0p5':'Reference declines to 0.5','reference_decline_end0':'Reference declines to 0',
        'functional_conversion0p5':'Functional conversion = 0.5','functional_conversion0p75':'Functional conversion = 0.75',
        'decline_end0_conversion0p5':'Reference to 0; conversion = 0.5'}
    order=list(reg['settings']);y=np.arange(len(order))
    fig,axes=plt.subplots(1,2,figsize=(11.2,6.4),sharey=True)
    for ax,metric,title,xlabel,color in zip(axes,['F1_symptom_relative_anthesis_days','GS65_85_lost_had3'],
            ['Modeled flag-leaf symptom timing','Upper-three-leaf HAD proxy loss'],
            ['Future − historical timing relative to anthesis (days)','Future − historical loss (days per nominal upper-three LAI)'],['#35668a','#ae5734']):
        part=ensemble.loc[ensemble.population.eq('all_settings_common_paired')&ensemble.metric.eq(metric)].set_index('setting').loc[order]
        for position,row in zip(y,part.itertuples()):
            ax.plot([row.gcm_min_change,row.gcm_max_change],[position,position],color='#a9aaad',lw=3,solid_capstyle='round',zorder=1)
            ax.errorbar(row.gcm_mean_change,position,xerr=row.spatial_mcse_gcm_mean_change,fmt='o',color=color,ms=5,capsize=3,elinewidth=1.2,zorder=3)
        ax.axvline(0,color='#555555',ls='--',lw=.8)
        ax.set_title(title,fontsize=11,pad=12);ax.set_xlabel(xlabel,fontsize=9);ax.grid(axis='x',alpha=.14)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=9)
    axes[0].set_yticks(y,[labels[s] for s in order],fontsize=9);axes[0].invert_yaxis()
    fig.suptitle('SSP585, 2071–2100 versus 1991–2020: registered structural sensitivities',fontsize=12,y=.98)
    fig.text(.02,.015,'Gray segments: three-GCM ranges. Colored points: GCM means; bars: ±1 spatial Monte Carlo SE.\nAll-setting common paired coverage; 64 registered draws. Scenario ranges are not confidence intervals.',fontsize=8,color='#454545')
    fig.tight_layout(rect=(0,.065,1,.955))
    fig.savefig(HERE/'climate_robustness.pdf',bbox_inches='tight');fig.savefig(HERE/'climate_robustness.png',dpi=240,bbox_inches='tight');plt.close(fig)
    with (HERE/'delivery_receipt.json').open('x') as f:
        json.dump(dict(status='complete',delivery_code_sha256=sha(Path(__file__)),
            independent_verification_sha256=sha(HERE/'independent_verification.json'),
            output_sha256={name:sha(HERE/name) for name in ['results_for_manuscript.json','cached_baseline_all_ssps.csv',
                'paired_setting_difference_from_baseline.csv','climate_robustness.pdf','climate_robustness.png','methods_and_limits.md']}),f,indent=2)
        f.write('\n')
    core=headline.loc[headline.population.eq('all_settings_common_paired')&headline.metric.isin(['F1_symptom_relative_anthesis_days','GS65_85_lost_had3','conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai'])]
    print(core[['setting','metric','gcm_mean_change','gcm_min_change','gcm_max_change','spatial_mcse_gcm_mean_change','minimum_gcm_paired_coverage']].to_string(index=False))


if __name__=='__main__':main()
