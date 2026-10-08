"""Formal crop infection/canopy decomposition artifacts and standalone figure."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    verification=json.loads((HERE/'independent_verification.json').read_text());assert verification['status']=='verified'
    ensemble=pd.read_csv(HERE/'ensemble_decomposition.csv');gcm=pd.read_csv(HERE/'decomposition_by_gcm.csv')
    corners=pd.read_parquet(HERE/'four_corner_draw_year_outputs.parquet');draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv');weight=draws.area_mean_weight.to_numpy()
    reg=json.loads((HERE/'registration_before_decomposition_results.json').read_text());corner_records=[]
    for model in reg['models']:
        arrays={name:group.sort_values(['spatial_draw_id','reference_year'])[reg['metrics']].to_numpy().reshape(64,30,len(reg['metrics'])) for name,group in corners[corners.model.eq(model)].groupby('corner')}
        common=np.logical_and.reduce([np.isfinite(a) for a in arrays.values()])
        for j,metric in enumerate(reg['metrics']):
            z=common[:,:,j].mean(axis=1);coverage=weight@z
            for name,values in arrays.items():
                x=np.where(common[:,:,j],values[:,:,j],0.).mean(axis=1)
                corner_records.append(dict(model=model,metric=metric,corner=name,mean=float(weight@x/coverage),common_area_time_coverage=float(coverage),pairs=int(common[:,:,j].sum())))
    corner_summary=pd.DataFrame(corner_records);corner_summary.to_csv(HERE/'corner_means_by_gcm.csv',index=False)
    corner_mean=corner_summary.groupby(['metric','corner'])['mean'].agg(['mean','min','max']).reset_index();corner_mean.to_csv(HERE/'ensemble_corner_means.csv',index=False)
    # Descriptive stage-coordinate diagnostics use the same verified current-kernel baseline.
    baseline_path=HERE.parent/'draw_season_outputs.parquet';baseline=pd.read_parquet(baseline_path);baseline=baseline[baseline.setting.eq('baseline')];stage_records=[]
    for model in reg['models']:
        reference=baseline[baseline.model.eq(model)&baseline.harvest_year.le(2020)].sort_values(['spatial_draw_id','harvest_year'])
        future=baseline[baseline.model.eq(model)&baseline.harvest_year.ge(2071)].sort_values(['spatial_draw_id','harvest_year'])
        common=(reference.valid_complete_season.to_numpy()&future.valid_complete_season.to_numpy()).reshape(64,30);z=common.mean(axis=1);coverage=weight@z
        for stage in (31,65,85):
            a=(pd.to_datetime(reference[f'BBCH{stage}_date'])-pd.to_datetime(reference.calendar_sowing_date)).dt.days.to_numpy().reshape(64,30)
            b=(pd.to_datetime(future[f'BBCH{stage}_date'])-pd.to_datetime(future.calendar_sowing_date)).dt.days.to_numpy().reshape(64,30)
            x=np.where(common,b-a,0).mean(axis=1)
            stage_records.append(dict(model=model,metric=f'BBCH{stage}_days_after_sowing',change=float(weight@x/coverage),paired_coverage=float(coverage),pairs=int(common.sum())))
    stages=pd.DataFrame(stage_records);stages.to_csv(HERE/'host_stage_calendar_diagnostics.csv',index=False)
    stage_summary=stages.groupby('metric').change.agg(['mean','min','max']).reset_index()
    result=dict(scenario='ssp585',reference='1991-2020',future='2071-2100',factor_order_invariant=True,
        ensemble=ensemble.to_dict(orient='records'),GCMs=gcm.to_dict(orient='records'),corner_means=corner_mean.to_dict(orient='records'),
        current_kernel_host_stage_shifts=stage_summary.to_dict(orient='records'),independent_verification=verification,
        caveat='Within-model decomposition of disease-weather forcing and the complete host trajectory; no physical causal identification or actual-yield attribution.',
        interaction_allocation='Interaction reported separately and divided equally between weather and host Shapley contributions; Shapley terms sum to total.',
        calendar='One-to-one elapsed days since each actual calendar sowing; no interpolation, repeated source values, leap-day duplication or active forcing padding.',
        event_timing_population='Common finite detection across all four corners, separate for each metric.',
        registration_sha256=sha(HERE/'registration_before_decomposition_results.json'))
    (HERE/'results_for_manuscript.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    order=['weather_shapley','host_shapley','total_change','interaction'];labels=['Disease weather (Shapley)','Host development (Shapley)','Total change','Interaction (full term)']
    colors=['#35668a','#4a7a61','#ae5734','#757575'];y=np.arange(4)
    fig,axes=plt.subplots(1,2,figsize=(10.8,4.25),sharey=True)
    for ax,metric,title,xlabel in zip(axes,['F1_symptom_relative_anthesis_days','GS65_85_lost_had3'],
            ['Modeled flag-leaf symptom timing','Upper-three-leaf HAD proxy loss'],
            ['Timing shift relative to anthesis (days)','Change (days per nominal upper-three LAI)']):
        table=ensemble[ensemble.metric.eq(metric)].set_index('component').loc[order]
        for position,row,color in zip(y,table.itertuples(),colors):
            ax.plot([row.gcm_min,row.gcm_max],[position,position],color='#b0b1b3',lw=3,solid_capstyle='round')
            ax.errorbar(row.gcm_mean,position,xerr=row.spatial_mcse_gcm_mean,fmt='o',color=color,ms=6,capsize=3,elinewidth=1.2)
        ax.axvline(0,color='#555555',ls='--',lw=.8);ax.axhline(2.5,color='#dddddd',lw=.8)
        ax.set_title(title,fontsize=11,pad=12);ax.set_xlabel(xlabel,fontsize=9);ax.grid(axis='x',alpha=.14)
        ax.spines[['top','right','left']].set_visible(False);ax.tick_params(axis='y',length=0);ax.tick_params(axis='x',labelsize=9)
    axes[0].set_yticks(y,labels,fontsize=9);axes[0].invert_yaxis()
    fig.suptitle('SSP585 weather–host decomposition: 2071–2100 versus 1991–2020',fontsize=12,y=.98)
    fig.text(.02,.025,'Gray segments: three-GCM ranges; colored bars: ±1 spatial Monte Carlo SE.\nWeather and host Shapley terms sum to total. Interaction is split equally between them and shown separately.',fontsize=8,color='#454545')
    fig.tight_layout(rect=(0,.11,1,.94));fig.savefig(HERE/'weather_host_decomposition.pdf',bbox_inches='tight');fig.savefig(HERE/'weather_host_decomposition.png',dpi=240,bbox_inches='tight');plt.close(fig)
    (HERE/'delivery_receipt.json').write_text(json.dumps(dict(status='complete',code_sha256=sha(Path(__file__)),
        source_baseline_sha256=sha(baseline_path),independent_verification_sha256=sha(HERE/'independent_verification.json'),
        output_sha256={p.name:sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ('.csv','.json','.pdf','.png') and p.name!='delivery_receipt.json'}),indent=2)+'\n')
    print('DELIVERED current-kernel decomposition',flush=True)


if __name__=='__main__':main()
