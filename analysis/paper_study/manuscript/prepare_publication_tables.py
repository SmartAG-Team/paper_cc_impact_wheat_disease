"""Inspectable publication tables and supplementary sensitivity figures."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional import sha,MODELS,SCENARIOS
from analysis.paper_study.aggregate_projection_uncertainty import PERIODS

DEST=ROOT/'publication/european_wheat_stb';DATA=ROOT/'data/paper_study/publication'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})


def write_supplementary_provenance():
    inputs={
        'figS1_parameter_identification':['analysis/paper_study/seasonal_statistics/calibration_parameter_bootstrap.csv',
            'analysis/paper_study/seasonal_statistics/secondary_beta_training_profile.csv'],
        'figS2_declared_sensitivities':['data/paper_study/calendar_sensitivity_reporting/sampled_sensitivity_periods.csv'],
        'figS3_paired_scenario_contrasts':['data/paper_study/publication/paired_scenario_contrasts.csv']}
    receipt=dict(source_code_sha256=sha(Path(__file__)),
        figures={name:{path:sha(ROOT/path) for path in paths} for name,paths in inputs.items()})
    (DEST/'figures/supplementary_provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')


def main():
    metrics=pd.read_csv(ROOT/'analysis/paper_study/empirical_benchmarks/severity_metrics.csv')
    keep=metrics.endpoint.eq('final_leaf_assessment')&metrics.model.isin(['seasonal_seir','weighted_ridge','constrained_forest','calibration_only_leaf_rank_mean'])
    keep&=(metrics.partition.eq('validation')|metrics.partition.eq('external')&metrics.geography.eq('strict_location_disjoint'))
    table=metrics[keep].copy();table['evaluation']=table.partition.map({'validation':'2019 development validation','external':'Strict external evaluation'})
    table['leaf_scope']=table.leaf_scope.map({'all_ordinal':'All ordinal ranks','upper_three':'Upper three'})
    table['model']=table.model.map({'seasonal_seir':'Seasonal model','weighted_ridge':'Weather ridge','constrained_forest':'Weather forest','calibration_only_leaf_rank_mean':'Leaf-rank mean'})
    table=table[['evaluation','leaf_scope','model','n','rmse','mae','bias']]
    table.to_csv(DATA/'table1_field_severity.csv',index=False)
    bands=pd.read_csv(ROOT/'analysis/paper_study/projection_uncertainty_reporting/complete_period_summary.csv')
    production=pd.read_csv(ROOT/'data/paper_study/production_exposure_reporting/conditional_production_projection_bands.csv')
    rows=[]
    for period in list(PERIODS)[1:]:
        for scenario in SCENARIOS:
            q=bands[bands.model.eq('three_model_mean')&bands.period.eq(period)&bands.scenario.eq(scenario)]
            main=q[q.quantity.eq('future_minus_model_baseline')].iloc[0]
            p=production[production.model.eq('three_model_mean')&production.period.eq(period)&production.scenario.eq(scenario)&production.quantity.eq('production_future_minus_model_baseline')].iloc[0]
            weather=q[q.quantity.eq('sampled_counterfactual_weather_effect')].iloc[0]
            host=q[q.quantity.eq('sampled_counterfactual_host_effect')].iloc[0]
            rows.append(dict(scenario=scenario,period=period,area_change=main.point_estimate_percent,
                area_low=main.conditional_parameter_p025,area_high=main.conditional_parameter_p975,
                production_change=p.point_estimate_percent,production_low=p.conditional_parameter_p025,
                production_high=p.conditional_parameter_p975,weather_contribution=weather.point_estimate_percent,
                host_contribution=host.point_estimate_percent,weather_point_mcse=weather.point_spatial_mcse,
                host_point_mcse=host.point_spatial_mcse))
    pd.DataFrame(rows).to_csv(DATA/'table2_climate_contrasts.csv',index=False)
    # S1: identifiability, using calibration outcomes only.
    bootstrap=pd.read_csv(ROOT/'analysis/paper_study/seasonal_statistics/calibration_parameter_bootstrap.csv')
    profile=pd.read_csv(ROOT/'analysis/paper_study/seasonal_statistics/secondary_beta_training_profile.csv')
    print('Bootstrap columns',bootstrap.columns.to_list(),flush=True)
    fig,axes=plt.subplots(1,2,figsize=(8.8,3.8),layout='constrained')
    axes[0].plot(profile.beta_fixed,profile.training_rmse_pp,color='#0072B2');axes[0].set(xlabel='Fixed secondary coefficient β (day⁻¹)',ylabel='Calibration RMSE (percentage points)',title='a  Calibration-only coefficient profile')
    axes[1].scatter(bootstrap.alpha,bootstrap.beta,s=15,alpha=.5,color='#D55E00');axes[1].set(xlabel='External coefficient α (day⁻¹)',ylabel='Secondary coefficient β (day⁻¹)',title='b  Joint calibration bootstrap estimates')
    fig.savefig(DEST/'figures/figS1_parameter_identification.png',dpi=300,bbox_inches='tight');fig.savefig(DEST/'figures/figS1_parameter_identification.pdf',bbox_inches='tight');plt.close(fig)
    # S2: identical future–reference contrasts under declared alternatives.
    sensitivity=pd.read_csv(ROOT/'data/paper_study/calendar_sensitivity_reporting/sampled_sensitivity_periods.csv')
    sensitivity=sensitivity[sensitivity.model.eq('three_model_mean')&sensitivity.quantity.eq('sampled_future_minus_case_baseline')]
    cases=['main','sowing_minus14','sowing_plus14','endpoint75','unaligned_climate']
    labels=['Main','Sowing −14 days','Sowing +14 days','Stage 75 interpolation','Unaligned climate']
    fig,axes=plt.subplots(2,3,figsize=(10.0,6.0),sharey=True,sharex=True,layout='constrained')
    for i,period in enumerate(list(PERIODS)[1:]):
        for j,scenario in enumerate(SCENARIOS):
            ax=axes[i,j]
            d=sensitivity[sensitivity.period.eq(period)&sensitivity.scenario.eq(scenario)].set_index('case').reindex(cases)
            assert len(d)==5 and d.estimate_percent.notna().all()
            ax.errorbar(d.estimate_percent,range(5),xerr=1.96*d.spatial_mcse,fmt='o',color=['#0072B2','#009E73','#D55E00'][j],capsize=3)
            ax.axvline(0,color='#777',linewidth=.7);ax.set_yticks(range(5),labels)
            ax.set_title(chr(97+i*3+j)+'  '+{'ssp126':'SSP1-2.6','ssp245':'SSP2-4.5','ssp585':'SSP5-8.5'}[scenario]+' | '+('2031–60' if i==0 else '2071–2100'),loc='left',fontsize=8)
            ax.grid(axis='x',alpha=.15)
    fig.supxlabel('Future minus same-case reference severity (percentage points)',fontsize=9)
    axes[0,0].invert_yaxis();fig.savefig(DEST/'figures/figS2_declared_sensitivities.png',dpi=300,bbox_inches='tight');fig.savefig(DEST/'figures/figS2_declared_sensitivities.pdf',bbox_inches='tight');plt.close(fig)
    # S3: direct paired comparisons, preserving covariance across scenarios.
    contrasts=pd.read_csv(DATA/'paired_scenario_contrasts.csv')
    contrasts=contrasts[contrasts.model.eq('three_model_mean')]
    pairs=[('ssp245','ssp126'),('ssp585','ssp126'),('ssp585','ssp245')]
    pair_labels=['SSP2-4.5 − SSP1-2.6','SSP5-8.5 − SSP1-2.6','SSP5-8.5 − SSP2-4.5']
    quantities=['future_severity_difference','difference_in_baseline_relative_changes']
    fig,axes=plt.subplots(2,2,figsize=(9.4,5.2),sharex=True,sharey=True,layout='constrained')
    for i,period in enumerate(list(PERIODS)[1:]):
        for j,quantity in enumerate(quantities):
            ax=axes[i,j]
            for k,(high,low) in enumerate(pairs):
                row=contrasts[contrasts.period.eq(period)&contrasts.quantity.eq(quantity)&contrasts.scenario_high.eq(high)&contrasts.scenario_low.eq(low)].iloc[0]
                ax.plot([row.conditional_parameter_p025_pp,row.conditional_parameter_p975_pp],[k,k],color='#0072B2',linewidth=3,alpha=.65)
                ax.scatter(row.point_estimate_pp,k,color='#0072B2',s=18,zorder=3)
                ax.plot([row.climate_model_point_min_pp,row.climate_model_point_max_pp],[k+.14,k+.14],color='#777',linewidth=1)
            ax.set_yticks(range(3),pair_labels,fontsize=8);ax.axvline(0,color='#777',linewidth=.7);ax.grid(axis='x',alpha=.15)
            ax.set_title(chr(97+i*2+j)+'  '+('2031–2060' if i==0 else '2071–2100')+' | '+('Future severity' if j==0 else 'Reference-relative changes'),loc='left',fontsize=8)
    axes[0,0].invert_yaxis();fig.supxlabel('Higher minus lower SSP (percentage points)',fontsize=9)
    axes[0,0].plot([],[],color='#0072B2',linewidth=3,label='Conditional parameter percentiles')
    axes[0,0].plot([],[],color='#777',linewidth=1,label='Climate-model point range')
    axes[0,0].legend(frameon=False,fontsize=7,loc='upper left',bbox_to_anchor=(0,1.34))
    fig.savefig(DEST/'figures/figS3_paired_scenario_contrasts.png',dpi=300,bbox_inches='tight');fig.savefig(DEST/'figures/figS3_paired_scenario_contrasts.pdf',bbox_inches='tight');plt.close(fig)
    captions={
        'figS1_parameter_identification':'Supplementary Figure 1 | Weak identification of infection routes. a, Calibration-only profile with β fixed and α reoptimized using the declared three starts. This objective profile is not a likelihood-ratio interval. b, One hundred joint calibration-only coordinate-year bootstrap estimates with the selected latent delay and all response and host assumptions fixed. Boundary fits remain included; no external outcome selects a parameter draw.',
        'figS2_declared_sensitivities':'Supplementary Figure 2 | Sowing, endpoint and forcing sensitivity across three climate scenarios. The six panels cover SSP1-2.6, SSP2-4.5 and SSP5-8.5 in 2031–2060 and 2071–2100, with common axes. Three-model mean future-minus-same-case reference contrasts use the same 64 registered spatial draws and frozen point disease parameters. Whiskers are ±1.96 spatial Monte Carlo standard errors, retaining shared draws for contrasts; they do not represent parameter or field uncertainty. Stage 75 is a threshold interpolation rather than an independently validated stage model. These spatial-sample values differ from exact full-grid point contrasts.',
        'figS3_paired_scenario_contrasts':'Supplementary Figure 3 | Paired differences between climate scenarios. a,c, Higher minus lower SSP final-severity means in the same future period. b,d, Differences between each SSP\'s change from its own 1991–2020 reference. Reference continuations differ over 2015–2020. Blue points have exact whole-grid anchors; blue segments are 2.5th–97.5th percentiles from the same 100 joint calibration-parameter draws across scenarios and models. Grey segments are the three climate-model point ranges. Spatial covariance is retained through the same registered draws; perturbation Monte Carlo errors accompany the source data. Climate-model ranges are not probability intervals. Separate marginal intervals are not subtracted to construct these comparisons.'}
    (DEST/'figures/supplementary_captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    write_supplementary_provenance()
    print(json.dumps(dict(status='complete',table1_rows=len(table),table2_rows=len(rows),supplementary_figures=len(captions))),flush=True)


if __name__=='__main__':main()
