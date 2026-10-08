"""Standalone publication figures from frozen field and full ERA5 results."""

from pathlib import Path
import hashlib,json,sys

import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import FancyBboxPatch,FancyArrowPatch
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_empirical_benchmarks import endpoint

DEST=ROOT/'publication/european_wheat_stb/figures'
DATA=ROOT/'data/paper_study/publication';DATA.mkdir(parents=True,exist_ok=True);DEST.mkdir(parents=True,exist_ok=True)
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';GRAY='#50565c'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,
    'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
COUNTRIES=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
CELLS=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet')
PRODUCTION=pd.read_parquet(ROOT/'data/paper_study/wheat_production/europe_wheat_production_025.parquet')
PRED=pd.read_parquet(ROOT/'data/paper_study/empirical_benchmarks/frozen_all_predictions.parquet')
CAPTIONS={};SOURCES={}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def save(fig,name,caption,source_paths):
    fig.savefig(DEST/f'{name}.png',dpi=300,facecolor='white',bbox_inches='tight',pad_inches=.12)
    fig.savefig(DEST/f'{name}.pdf',facecolor='white',bbox_inches='tight',pad_inches=.12);plt.close(fig)
    CAPTIONS[name]=caption;SOURCES[name]={str(p.relative_to(ROOT)):sha(p) for p in source_paths}


def letter(ax,label):ax.text(-.10,1.04,label,transform=ax.transAxes,fontsize=11,fontweight='bold',va='bottom')


def grid_map(ax,frame,value,cmap,vmin,vmax,log=False):
    registry=CELLS[['cell_id','row','col']].merge(frame[['cell_id',value]],on='cell_id',how='left',validate='one_to_one')
    r0,r1=int(registry.row.min()),int(registry.row.max());c0,c1=int(registry.col.min()),int(registry.col.max())
    array=np.full((r1-r0+1,c1-c0+1),np.nan);array[registry.row-r0,registry.col-c0]=registry[value]
    lon=-180+np.arange(c0,c1+2)*.25;lat=90-np.arange(r0,r1+2)*.25
    COUNTRIES.plot(ax=ax,color='#eff0f0',edgecolor='#acb0b3',linewidth=.28)
    kwargs={'norm':LogNorm(vmin=vmin,vmax=vmax)} if log else {'vmin':vmin,'vmax':vmax}
    image=ax.pcolormesh(lon,lat,array,cmap=cmap,rasterized=True,**kwargs)
    ax.set_xlim(-12,63);ax.set_ylim(33,70);ax.set_aspect(1/np.cos(np.deg2rad(52)))
    ax.set_xticks([0,20,40,60]);ax.set_yticks([40,50,60,70]);ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°N)')
    return image


def box(ax,x,y,w,h,label,color='#eff4f7'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.006',facecolor=color,edgecolor='#707a81',linewidth=.7))
    ax.text(x+w/2,y+h/2,label,ha='center',va='center',fontsize=8)


def arrow(ax,start,end,**kwargs):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9,linewidth=.8,color=GRAY,**kwargs))


def figure1():
    fig=plt.figure(figsize=(8.6,9.1));gs=fig.add_gridspec(2,1,height_ratios=[2.2,1],hspace=.35)
    ax=fig.add_subplot(gs[0]);letter(ax,'a')
    image=grid_map(ax,PRODUCTION,'production_total_tonnes','YlOrBr',100,130000,True)
    points=[]
    for name,part,color,marker,label in [('BASF','calibration',BLUE,'^','Calibration, 2017–2018'),
            ('BASF','validation',GREEN,'s','Development validation, 2019'),
            ('Corteva','external',ORANGE,'o','External archive, 2014–2018')]:
        p=PRED[PRED.source.eq(name)&PRED.partition.eq(part)&PRED.model.eq('seasonal_seir')].drop_duplicates('site_id').copy()
        ax.scatter(p.longitude,p.latitude,s=18,marker=marker,facecolors='none',edgecolors=color,linewidths=.8,label=label,zorder=5)
        p['display_scope']=label;points.append(p[['site_id','latitude','longitude','display_scope']])
    ax.legend(loc='lower right',frameon=True,framealpha=.95,fontsize=7)
    ax.set_title('Wheat production exposure and European field evaluation',loc='left')
    colorbar=fig.colorbar(image,ax=ax,fraction=.025,pad=.02);colorbar.set_label('2020 reference wheat production (tonnes per cell)')
    colorbar.set_ticks([100,1000,10000,100000]);colorbar.set_ticklabels(['100','1,000','10,000','100,000'])
    ax=fig.add_subplot(gs[1]);letter(ax,'b');ax.set_axis_off();ax.set_xlim(0,1);ax.set_ylim(0,1)
    box(ax,.01,.79,.30,.15,'Daily temperature, humidity, rain')
    box(ax,.36,.79,.29,.15,'T-P-V crop development')
    box(ax,.70,.79,.28,.15,'Visibility and untreated severity')
    ax.text(.02,.64,'Effective external + canopy pressure',fontsize=8)
    ax.text(.50,.69,'Cohort availability',ha='center',fontsize=8)
    ax.add_patch(FancyBboxPatch((.01,.32),.97,.24,boxstyle='round,pad=.005',
        facecolor='none',edgecolor='#9da6ad',linewidth=.7,linestyle='--'))
    labels=['S','E₁','E₂','E₃','V','I','R'];positions=[.03,.21,.34,.47,.62,.75,.88]
    for x,label in zip(positions,labels):box(ax,x,.35,.08,.16,label,'#eef2f4' if label=='S' else '#e3eff5')
    for x,next_x in zip(positions[:-1],positions[1:]):arrow(ax,(x+.083,.43),(next_x-.008,.43))
    arrow(ax,(.16,.79),(.16,.53));arrow(ax,(.50,.79),(.50,.73));arrow(ax,(.50,.65),(.50,.57))
    ax.text(.80,.60,'Visible damage: V + I + R',ha='center',fontsize=7)
    arrow(ax,(.84,.64),(.84,.78))
    ax.plot([.79,.79,.075],[.34,.25,.25],color=ORANGE,linewidth=.8)
    arrow(ax,(.075,.25),(.075,.34))
    ax.text(.5,.14,'Rain-mediated transmission from infectious cohorts',color=ORANGE,fontsize=8,ha='center')
    ax.text(.5,.03,'Two origins retained; seven final leaves + juvenile reservoir',fontsize=8,ha='center')
    fig.subplots_adjust(left=.10,right=.93,top=.96,bottom=.07)
    pd.concat(points,ignore_index=True).to_csv(DATA/'fig1_evaluation_locations.csv',index=False)
    CELLS[['cell_id','row','col','latitude','longitude','harvested_total_ha']].to_csv(DATA/'fig1_wheat_area.csv',index=False)
    PRODUCTION[['cell_id','row','col','latitude','longitude','production_total_tonnes']].to_csv(DATA/'fig1_wheat_production.csv',index=False)
    save(fig,'fig1_domain_and_model',
        'Figure 1 | European wheat production exposure, field evidence and seasonal model. a, SPAM2020 reference wheat production, summed to 0.25° cells under the declared geographical boundary. Tonnes describe fixed production exposure rather than simulated disease loss. Symbols identify rounded field-coordinate locations, including the full external source scope; overlapping locations are not displaced. The strict external benchmark excludes locations shared with the original development archive. Uncoloured land is outside displayed positive wheat support and does not establish verified crop absence. b, Susceptible cohorts acquire infection through effective external pressure and infectious-cohort transmission. Origin labels remain distinct through latent, visible nonsporulating, infectious and retained-damage states. Crop development controls cohort availability. The diagram shows model structure; it does not validate spore identities or estimated transmission contributions.',
        [ROOT/'data/paper_study/wheat_production/europe_wheat_production_025.parquet',ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet',ROOT/'data/paper_study/empirical_benchmarks/frozen_all_predictions.parquet',ROOT/'data/geography/ne_110m_admin_0_countries.zip'])


def figure2():
    fig,axes=plt.subplots(2,2,figsize=(8.6,7.2));fig.subplots_adjust(wspace=.39,hspace=.42,left=.10,right=.97,bottom=.09,top=.94)
    source_rows=[]
    for ax,part,label in [(axes[0,0],'validation','2019 development validation'),(axes[0,1],'external','Location-disjoint external evaluation')]:
        frame=PRED[PRED.model.eq('seasonal_seir')&PRED.partition.eq(part)].copy()
        if part=='external':frame=frame[frame.strict_location_disjoint]
        frame=endpoint(frame,True,False);source_rows.append(frame)
        ax.scatter(frame.value,frame.predicted_percent,s=12,alpha=.4,color=BLUE,edgecolors='none',rasterized=True)
        ax.plot([0,100],[0,100],color='#777',linewidth=.8,linestyle='--');ax.set(xlim=(-2,102),ylim=(-2,102),xlabel='Observed final leaf score (%)',ylabel='Predicted score (%)',title=label)
        error=frame.predicted_percent-frame.value;rmse=np.sqrt(np.average(error**2,weights=frame.weight))
        bias=np.average(error,weights=frame.weight)
        observed_mean=np.average(frame.value,weights=frame.weight)
        variance=np.average((frame.value-observed_mean)**2,weights=frame.weight)
        r_squared=1-rmse**2/variance
        ax.text(.05,.94,f'n = {len(frame)} leaf endpoints\nRMSE = {rmse:.1f}; bias = {bias:+.1f} pp\nWeighted R² = {r_squared:.3f}',transform=ax.transAxes,fontsize=7.5,va='top')
    letter(axes[0,0],'a');letter(axes[0,1],'b')
    comparator=pd.read_csv(ROOT/'analysis/paper_study/empirical_benchmarks/paired_model_comparator_intervals.csv')
    basf=pd.read_csv(ROOT/'analysis/paper_study/seasonal_statistics/paired_final_severity_intervals.csv')
    external=pd.read_csv(ROOT/'analysis/paper_study/external_statistics/severity_bootstrap_intervals.csv')
    ax=axes[1,0];letter(ax,'c');rows=[]
    for part,color,offset in [('validation',GREEN,-.12),('external',ORANGE,.12)]:
        if part=='validation':
            row=basf[basf.partition.eq(part)&basf.subset.eq('upper_three')&basf.metric.eq('rmse')&basf.comparison.eq('paired_model_minus_baseline')].iloc[0]
        else:
            row=external[external.geography.eq('strict_location_disjoint')&external.leaf_scope.eq('upper_three')&external.sensitivity.eq('all_dates')&external.endpoint.eq('final_numeric_assessment')&external.weighting.eq('literal_equal_coordinate_year')&external.metric.eq('rmse')&external.comparison.eq('paired_model_minus_baseline')].iloc[0]
        entries=[('Leaf-rank mean',row.estimate_pp,row.ci95_lower_pp,row.ci95_upper_pp)]
        for family,label in [('weighted_ridge','Weather ridge'),('constrained_forest','Weather forest')]:
            candidates=comparator[comparator.partition.eq(part)&comparator.leaf_scope.eq('upper_three')&comparator.endpoint.eq('final_leaf_assessment')&comparator.comparator.eq(family)]
            if part=='external':candidates=candidates[candidates.geography.eq('strict_location_disjoint')]
            r=candidates.iloc[0];entries.append((label,r.rmse_difference,r.rmse_difference_lower,r.rmse_difference_upper))
        for j,(label,value,lower,upper) in enumerate(entries):
            ax.errorbar(value,j+offset,xerr=[[value-lower],[upper-value]],fmt='o',markersize=4,
                capsize=2,color=color,label='2019' if part=='validation' and j==0 else 'External' if j==0 else None)
            rows.append(dict(partition=part,comparator=label,rmse_difference=value,lower=lower,upper=upper))
    ax.axvline(0,color='#555',linewidth=.7);ax.set_yticks(range(3),['Leaf-rank mean','Weather ridge','Weather forest']);ax.invert_yaxis()
    ax.set_xlabel('Model minus comparator RMSE (percentage points)');ax.set_title('Final upper-three severity');ax.legend(fontsize=7,loc='lower left')
    onset_external=pd.read_csv(ROOT/'analysis/paper_study/external_statistics/onset_censoring_statistics.csv')
    onset_basf=pd.read_csv(ROOT/'analysis/paper_study/seasonal_statistics/onset_compatibility_by_censoring.csv')
    ax=axes[1,1];letter(ax,'d');onset_rows=[]
    for part,color,offset in [('validation',GREEN,-.16),('external',ORANGE,.16)]:
        for j,cutoff in enumerate([.1,1.,5.]):
            if part=='validation':
                r=onset_basf[onset_basf.model.eq('seasonal_seir')&onset_basf.partition.eq(part)&onset_basf.cutoff_percent.eq(cutoff)&onset_basf.censoring.eq('interval')&onset_basf.subset.eq('all_ordinal_leaves')].iloc[0]
                mean=r.equal_cluster_compatible_fraction
            else:
                r=onset_external[onset_external.model.eq('seasonal_seir')&onset_external.geography.eq('strict_location_disjoint')&onset_external.cutoff_percent.eq(cutoff)&onset_external.censoring.eq('interval')&onset_external.leaf_scope.eq('all_ordinal_leaves')&onset_external.sensitivity.eq('all_dates')].iloc[0]
                mean=r.equal_coordinate_year_compatible_fraction
            ax.errorbar(j+offset,mean*100,yerr=[[(mean-r.ci95_lower)*100],[(r.ci95_upper-mean)*100]],fmt='o',color=color,capsize=2,markersize=4)
            onset_rows.append(dict(partition=part,cutoff_percent=cutoff,compatible_fraction=mean,lower=r.ci95_lower,upper=r.ci95_upper))
    ax.set_xticks(range(3),['0.1','1','5']);ax.set(xlabel='Visible-disease cutoff (%)',ylabel='Interval-compatible onset (%)',ylim=(-3,65),title='Visibility timing under two-sided bounds')
    pd.concat(source_rows,ignore_index=True)[['field_id','endpoint_series','partition','coordinate_year','date','value','predicted_percent','weight']].to_csv(DATA/'fig2_final_leaf_scores.csv',index=False)
    pd.DataFrame(rows).to_csv(DATA/'fig2_paired_rmse_intervals.csv',index=False);pd.DataFrame(onset_rows).to_csv(DATA/'fig2_visibility_intervals.csv',index=False)
    save(fig,'fig2_field_validation',
        'Figure 2 | Frozen severity and visibility predictions. a,b, Final numeric scores for each source leaf series in 2019 development validation and the location-disjoint external subset. Different leaves can end on different dates. Scatter points are individual source endpoints; displayed errors use equal coordinate-year, field and leaf weighting. Bias is predicted minus observed score; weighted R² is one minus weighted squared error divided by weighted observed variance. Its reference is the evaluation-sample mean, distinct from the calibration-only comparator. c, Model-minus-comparator RMSE differences for final upper-three scores, with 95% paired coordinate-year-bootstrap confidence intervals. Negative differences indicate smaller model error. Empirical comparator families were introduced after initial external results; their fitting and selection used calibration data only. d, Visibility compatibility for interval-censored leaf series at three fixed thresholds. Whiskers are 95% coordinate-year-bootstrap intervals. One-sided bounds are excluded; visible onset is distinct from unobserved primary infection.',
        [ROOT/'data/paper_study/empirical_benchmarks/frozen_all_predictions.parquet',ROOT/'analysis/paper_study/empirical_benchmarks/paired_model_comparator_intervals.csv',ROOT/'analysis/paper_study/external_statistics/onset_censoring_statistics.csv'])


def figure3():
    base=ROOT/'data/paper_study/regional_reporting/era5/ERA5/baseline/original'
    folder=base/'winter_wheat_rainfed_sow+0_stage85/baseline1991_2020'
    summary=pd.read_parquet(folder/'cell_period_summary.parquet')
    fig,axes=plt.subplots(2,2,figsize=(8.6,7.6));fig.subplots_adjust(left=.08,right=.98,bottom=.09,top=.95,wspace=.28,hspace=.35)
    specs=[('mean_upper3_final_damage_percent','YlOrRd',0,100,'Final upper-three damage (%)'),
        ('mean_exceeds_50pct','YlGnBu',0,1,'Annual final-damage exceedance ≥50%'),
        ('mean_flag_onset5_from_sowing_days','YlGnBu',120,330,'Days from sowing to 5% flag visibility')]
    for ax,label,(value,cmap,low,high,title) in zip(axes.flat,['a','b','c'],specs):
        image=grid_map(ax,summary,value,cmap,low,high);letter(ax,label);ax.set_title(title,loc='left')
        colorbar=fig.colorbar(image,ax=ax,fraction=.026,pad=.015);colorbar.ax.tick_params(labelsize=7)
    ax=axes[1,1];letter(ax,'d');annuals=[]
    for season,color,label in [('winter_wheat',BLUE,'Winter calendar'),('spring_wheat',ORANGE,'Spring-sowing sensitivity')]:
        p=base/f'{season}_rainfed_sow+0_stage85/baseline1991_2020/europe_annual_metrics.csv'
        annual=pd.read_csv(p);annual=annual[annual.area_basis.eq('harvested')].copy();annual['calendar']=season;annuals.append(annual)
        ax.plot(annual.harvest_year,annual.mean_final_damage_percent,color=color,linewidth=1.2,label=label)
    ax.set(xlabel='Harvest year',ylabel='Wheat-area mean final damage (%)',title='ERA5 annual variation');ax.legend(frameon=False,fontsize=7);ax.grid(axis='y',alpha=.15)
    summary[['cell_id',*list(s[0] for s in specs),'complete_years','expected_years','all_30_years_valid']].to_csv(DATA/'fig3_era5_cell_summary.csv',index=False)
    pd.concat(annuals,ignore_index=True).to_csv(DATA/'fig3_era5_europe_annual.csv',index=False)
    save(fig,'fig3_era5_baseline',
        'Figure 3 | Conditional European baseline under 1991–2020 ERA5 weather. a, Mean final upper-three damage at predicted soft dough. b, Fraction of completed years with final upper-three damage≥50%; this is a modelled diagnostic frequency, not an observed occurrence probability or treatment threshold. c, First 5% visibility on the individual flag cohort, in days since the scenario sowing date. It is not onset of the upper-three mean. Maps apply the winter-rainfed calendar to fixed all-wheat 2020 exposure and average completed years, with valid-year counts in source data. d, Annual harvested-area-weighted means under two separately applied full-area calendar scenarios. Spring development disables vernalization while transferring winter stage thresholds. The curves are nonadditive calendar sensitivities. No missing crop outcome is set to zero.',
        [folder/'cell_period_summary.parquet',folder/'europe_annual_metrics.csv',base/'spring_wheat_rainfed_sow+0_stage85/baseline1991_2020/europe_annual_metrics.csv'])


def main():
    figure1();figure2();figure3()
    (DEST/'field_and_baseline_captions.json').write_text(json.dumps(CAPTIONS,indent=2)+'\n')
    (DEST/'field_and_baseline_provenance.json').write_text(json.dumps(SOURCES,indent=2)+'\n')
    print(json.dumps({'figures':list(CAPTIONS),'formats':['PNG300dpi','PDF'],'source_data':str(DATA.relative_to(ROOT))}))


if __name__=='__main__':main()
