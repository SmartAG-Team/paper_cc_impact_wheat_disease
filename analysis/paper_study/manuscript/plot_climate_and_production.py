"""Publication maps, pathway comparisons and reference-production exposure."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional import MODELS,SCENARIOS
from analysis.paper_study.aggregate_projection_uncertainty import PERIODS

DEST=ROOT/'publication/european_wheat_stb/figures'
DATA=ROOT/'data/paper_study/publication'
ANNUAL=ROOT/'analysis/paper_study/regional_v1/annual/nasa'
CELLS=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet')
COUNTRIES=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
PRODUCTION=pd.read_parquet(ROOT/'data/paper_study/wheat_production/europe_wheat_production_025.parquet')
COUNTRY_PRODUCTION=pd.read_parquet(ROOT/'data/paper_study/wheat_production/europe_cell_country_wheat_production.parquet')
CONTRACT=ROOT/'analysis/paper_study/manuscript/climate_figure_contract.json'
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':9,'axes.labelsize':8,
    'xtick.labelsize':7,'ytick.labelsize':7,'pdf.fonttype':42,'ps.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False})
CAPTIONS={};HASHES={}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def letter(ax,label):ax.text(-.08,1.04,label,transform=ax.transAxes,fontsize=11,fontweight='bold',va='bottom')


def save(fig,name,caption):
    fig.savefig(DEST/f'{name}.png',dpi=300,bbox_inches='tight',pad_inches=.12,facecolor='white')
    fig.savefig(DEST/f'{name}.pdf',bbox_inches='tight',pad_inches=.12,facecolor='white')
    plt.close(fig);CAPTIONS[name]=caption


def grid_map(ax,values):
    r0,r1=CELLS.row.min(),CELLS.row.max();c0,c1=CELLS.col.min(),CELLS.col.max()
    grid=np.full((r1-r0+1,c1-c0+1),np.nan)
    grid[CELLS.row-r0,CELLS.col-c0]=values
    COUNTRIES.plot(ax=ax,color='#eff0f0',edgecolor='#acb0b3',linewidth=.25)
    image=ax.pcolormesh(-180+np.arange(c0,c1+2)*.25,90-np.arange(r0,r1+2)*.25,
        grid,cmap='RdBu_r',vmin=-20,vmax=20,rasterized=True)
    ax.set_xlim(-12,63);ax.set_ylim(33,70);ax.set_aspect(1/np.cos(np.deg2rad(52)))
    ax.set_xticks([0,20,40,60]);ax.set_yticks([40,50,60,70])
    ax.set_xlabel('Longitude (°E)');ax.set_ylabel('Latitude (°N)')
    return image


def period_arrays():
    ids={key:index for index,key in enumerate(CELLS.cell_id)}
    names=sorted(COUNTRY_PRODUCTION.ADM0_NAME.unique())
    name_indices={name:index for index,name in enumerate(names)}
    matrix=coo_matrix((COUNTRY_PRODUCTION.production_total_tonnes,
        (COUNTRY_PRODUCTION.ADM0_NAME.map(name_indices),COUNTRY_PRODUCTION.cell_id.map(ids))),
        shape=(len(names),len(CELLS))).tocsr()
    full=matrix.sum(axis=1).A1
    np.testing.assert_allclose(matrix.sum(axis=0).A1,PRODUCTION.production_total_tonnes,rtol=0,atol=1e-8)
    output={};country_rows=[];common=np.ones(len(CELLS),bool)
    for model in MODELS:
        for scenario in SCENARIOS:
            for period,years in PERIODS.items():
                scores=[];valids=[];country_num=np.zeros(len(names));country_den=country_num.copy()
                for year in years:
                    path=ANNUAL/model/scenario/'adjusted/winter_wheat_rainfed_sow+0_stage85'/f'{year}.parquet'
                    receipt=json.loads(path.with_suffix('.json').read_text());assert sha(path)==receipt['parquet_sha256']
                    frame=pd.read_parquet(path,columns=['cell_id','status','upper3_final_damage_percent'])
                    assert frame.cell_id.to_list()==CELLS.cell_id.to_list()
                    valid=frame.status.eq('complete').to_numpy();score=frame.upper3_final_damage_percent.to_numpy()
                    assert np.array_equal(np.isfinite(score),valid)
                    scores.append(score);valids.append(valid)
                    country_num+=matrix@np.where(valid,score,0.);country_den+=matrix@valid.astype(float)
                    HASHES[str(path.relative_to(ROOT))]=receipt['parquet_sha256']
                values=np.stack(scores);valid=np.stack(valids)
                all_valid=valid.all(axis=0);common&=all_valid
                numerator=np.where(valid,values,0.).sum(axis=0);den=valid.sum(axis=0)
                mean=np.divide(numerator,den,out=np.full(len(CELLS),np.nan),where=den>0)
                frequency=np.divide((valid&(values>=50)).sum(axis=0),den,out=np.full(len(CELLS),np.nan),where=den>0)
                output[model,scenario,period]=dict(mean=mean,exceed50=frequency,all_valid=all_valid)
                for j,name in enumerate(names):
                    country_rows.append(dict(model=model,scenario=scenario,period=period,country=name,
                        reference_production_tonnes=full[j],valid_production_year_fraction=country_den[j]/30/full[j] if full[j] else np.nan,
                        production_weighted_mean_severity_percent=country_num[j]/country_den[j] if country_den[j] else np.nan))
                print(json.dumps(dict(model=model,scenario=scenario,period=period)),flush=True)
    pd.DataFrame(country_rows).to_csv(DATA/'country_production_weighted_periods.csv',index=False)
    return output,common,pd.DataFrame(country_rows)


def figure4(periods,common):
    fig,axes=plt.subplots(2,3,figsize=(10.2,6.2),layout='constrained')
    source=[];coverage=[]
    for i,period in enumerate(list(PERIODS)[1:]):
        for j,scenario in enumerate(SCENARIOS):
            differences=np.stack([periods[m,scenario,period]['mean']-periods[m,scenario,'baseline1991_2020']['mean'] for m in MODELS])
            values=differences.mean(axis=0);values[~common]=np.nan
            image=grid_map(axes[i,j],values);letter(axes[i,j],chr(ord('a')+i*3+j))
            axes[i,j].set_title({'ssp126':'SSP1-2.6','ssp245':'SSP2-4.5','ssp585':'SSP5-8.5'}[scenario]+' | '+('2031–2060' if i==0 else '2071–2100'),loc='left')
            agreement=(differences.min(axis=0)>0)|(differences.max(axis=0)<0)
            frame=CELLS[['cell_id','latitude','longitude','harvested_total_ha']].copy()
            frame['scenario']=scenario;frame['period']=period;frame['all_comparisons_common_complete']=common
            frame['three_model_mean_change_pp']=values
            frame['three_model_point_min_change_pp']=np.where(common,differences.min(axis=0),np.nan)
            frame['three_model_point_max_change_pp']=np.where(common,differences.max(axis=0),np.nan)
            frame['three_models_same_direction']=common&agreement
            source.append(frame)
            coverage.append(dict(scenario=scenario,period=period,common_cells=int(common.sum()),
                common_harvested_ha=float(CELLS.loc[common,'harvested_total_ha'].sum()),
                common_area_fraction=float(CELLS.loc[common,'harvested_total_ha'].sum()/CELLS.harvested_total_ha.sum()),
                same_direction_fraction_common_area=float(CELLS.loc[common&agreement,'harvested_total_ha'].sum()/CELLS.loc[common,'harvested_total_ha'].sum())))
    bar=fig.colorbar(image,ax=axes.ravel().tolist(),orientation='horizontal',fraction=.045,pad=.03,extend='both')
    bar.set_label('Future minus model-specific 1991–2020 severity (percentage points)')
    pd.concat(source,ignore_index=True).to_csv(DATA/'fig4_climate_change_cells.csv',index=False)
    pd.DataFrame(coverage).to_csv(DATA/'fig4_common_map_coverage.csv',index=False)
    save(fig,'fig4_climate_change_maps',
        'Figure 4 | Conditional climate-change effects on final upper-leaf disease. Panels show the arithmetic mean of three climate-model changes from each model–SSP-specific 1991–2020 reference. The fixed all-wheat winter-calendar scenario uses predicted soft dough as its endpoint. Mapped cells complete all 30 annual seasons in every displayed model, scenario and period; common coverage and individual-model ranges accompany the source data. Values beyond ±20 percentage points retain their source values and use endpoint colours. The map is a disease-severity contrast and does not estimate yield loss. Model direction agreement is supplied as a diagnostic rather than statistical significance. Regional means elsewhere use all valid hectare-years, separately from this common-cell display mask.')


def figure5():
    path=ROOT/'analysis/paper_study/projection_uncertainty_reporting/complete_period_summary.csv'
    bands=pd.read_csv(path);bands=bands[bands.model.eq('three_model_mean')]
    keys=[(s,p) for p in list(PERIODS)[1:] for s in SCENARIOS]
    labels=[{'ssp126':'1-2.6','ssp245':'2-4.5','ssp585':'5-8.5'}[s]+'\n'+('2031–60' if 'mid' in p else '2071–2100') for s,p in keys]
    fig,axes=plt.subplots(2,1,figsize=(8.6,7.1),sharex=True);fig.subplots_adjust(hspace=.28,left=.12,right=.97,bottom=.12,top=.95)
    source=[]
    for i,(scenario,period) in enumerate(keys):
        row=bands[bands.scenario.eq(scenario)&bands.period.eq(period)&bands.quantity.eq('future_minus_model_baseline')].iloc[0]
        axes[0].plot([i,i],[row.conditional_parameter_p025,row.conditional_parameter_p975],color=BLUE,linewidth=3,alpha=.5)
        axes[0].scatter(i,row.point_estimate_percent,s=22,color=BLUE,zorder=4)
        axes[0].plot([i-.15,i-.15],[row.climate_model_point_min,row.climate_model_point_max],color='#777',linewidth=.8)
        for end in [row.climate_model_point_min,row.climate_model_point_max]:
            axes[0].plot([i-.19,i-.11],[end,end],color='#777',linewidth=.8)
        for offset,quantity,color in [(-.15,'sampled_counterfactual_weather_effect',ORANGE),(.15,'sampled_counterfactual_host_effect',GREEN)]:
            r=bands[bands.scenario.eq(scenario)&bands.period.eq(period)&bands.quantity.eq(quantity)].iloc[0]
            x=i+offset;axes[1].bar(x,r.point_estimate_percent,width=.28,color=color,alpha=.75)
            axes[1].plot([x,x],[r.conditional_parameter_p025,r.conditional_parameter_p975],color=color,linewidth=1.4)
            axes[1].errorbar(x,r.point_estimate_percent,yerr=1.96*r.point_spatial_mcse,fmt='.',color='black',capsize=2,markersize=3)
            source.append(r.to_dict())
        source.append(row.to_dict())
    for ax,lab in zip(axes,['a','b']):letter(ax,lab);ax.axhline(0,color='#666',linewidth=.7);ax.grid(axis='y',alpha=.15)
    axes[0].set_ylabel('Full-grid change (percentage points)')
    axes[0].set_title('Net severity change and conditional parameter uncertainty',loc='left')
    axes[0].plot([],[],color=BLUE,linewidth=3,alpha=.5,label='2.5–97.5% parameter percentiles');axes[0].plot([],[],color='#777',linewidth=.8,label='Climate-model point range');axes[0].legend(frameon=False,fontsize=7,ncol=2)
    axes[1].set_ylabel('Within-model contribution (percentage points)');axes[1].set_title('Disease weather and host-development pathways',loc='left')
    axes[1].legend(handles=[Patch(facecolor=ORANGE,label='Disease weather'),Patch(facecolor=GREEN,label='Host development')],frameon=False,fontsize=7,ncol=2)
    axes[1].set_xticks(range(6),labels);axes[1].set_xlabel('SSP and future harvest period')
    pd.DataFrame(source).to_csv(DATA/'fig5_uncertainty_and_pathways.csv',index=False);HASHES[str(path.relative_to(ROOT))]=sha(path)
    save(fig,'fig5_uncertainty_and_pathways',
        'Figure 5 | Parameter uncertainty and opposing seasonal pathways. a, Three-model mean final-severity change from model-specific baselines. Blue points are exact full-grid point-parameter estimates; thick segments show 2.5th–97.5th percentiles from 100 joint calibration-parameter draws. Grey brackets show the range of the three point-parameter climate models, a separate quantity. Parameter perturbations use the registered spatial sample and an exact full-grid anchor; sampling errors are reported in source data. b, Symmetric crossed-weather/host contributions, averaged over three models. Coloured bars are spatial-sample point estimates and coloured whiskers are conditional parameter percentiles. Black whiskers show ±1.96 point spatial Monte Carlo standard errors. Host development includes cohort emergence, renewal and the crop endpoint. The two contributions sum to the sampled total change, which can differ from the exact full-grid change in panel a. The decomposition identifies pathways within the declared model, without empirical causal attribution.')


def figure6(country):
    totals=COUNTRY_PRODUCTION.groupby('ADM0_NAME').production_total_tonnes.sum().sort_values(ascending=False)
    names=totals.head(10).index.tolist();source=[]
    fig,axes=plt.subplots(1,2,figsize=(8.8,5.0),sharey=True);fig.subplots_adjust(left=.19,right=.96,bottom=.13,top=.93,wspace=.3)
    label_map={'Russian Federation':'European Russia','United Kingdom of Great Britain and Northern Ireland':'United Kingdom'}
    for j,name in enumerate(names):
        axes[0].barh(j,totals[name]/1e6,color=BLUE,alpha=.8)
        for scenario,offset,color,marker in zip(SCENARIOS,[-.23,0,.23],[BLUE,GREEN,ORANGE],['o','s','^']):
            changes=[]
            for model in MODELS:
                q=country[country.country.eq(name)&country.model.eq(model)&country.scenario.eq(scenario)]
                base=q[q.period.eq('baseline1991_2020')].iloc[0].production_weighted_mean_severity_percent
                future=q[q.period.eq('latecentury2071_2100')].iloc[0].production_weighted_mean_severity_percent
                changes.append(future-base)
                source.append(dict(country=name,model=model,scenario=scenario,period='latecentury2071_2100',reference_production_tonnes=totals[name],
                    baseline_production_weighted_severity_percent=base,future_production_weighted_severity_percent=future,
                    change_percentage_points=future-base))
            axes[1].plot([min(changes),max(changes)],[j+offset,j+offset],color=color,linewidth=.8,alpha=.7)
            axes[1].scatter(np.mean(changes),j+offset,color=color,marker=marker,s=20,zorder=3)
    axes[0].set_yticks(range(10),[label_map.get(name,name) for name in names]);axes[0].invert_yaxis()
    axes[0].set_xlabel('2020 reference wheat production (million tonnes)');axes[0].set_title('Production exposure',loc='left');letter(axes[0],'a')
    axes[1].axvline(0,color='#666',linewidth=.7);axes[1].set_xlabel('Severity change (percentage points)');axes[1].set_title('Scenario comparison, 2071–2100',loc='left');letter(axes[1],'b')
    for scenario,color,marker in zip(SCENARIOS,[BLUE,GREEN,ORANGE],['o','s','^']):
        axes[1].scatter([],[],color=color,marker=marker,s=20,label={'ssp126':'SSP1-2.6','ssp245':'SSP2-4.5','ssp585':'SSP5-8.5'}[scenario])
    axes[1].legend(loc='lower left',bbox_to_anchor=(-.08,1.04),ncol=3,fontsize=6.5,frameon=False,columnspacing=.9,handletextpad=.35)
    for ax in axes:ax.grid(axis='x',alpha=.15)
    pd.DataFrame(source).to_csv(DATA/'fig6_country_production_exposure.csv',index=False)
    save(fig,'fig6_production_exposure',
        'Figure 6 | Reference wheat production and disease response across three climate scenarios. a, Ten countries with the largest fixed SPAM2020 production inside the declared geographical European domain, selected independently of disease-response direction. The European portion of Russia is identified by the domain convention. b, Production-weighted final-disease changes under SSP1-2.6 (circles), SSP2-4.5 (squares) and SSP5-8.5 (triangles) in 2071–2100 relative to each model–SSP-specific 1991–2020 baseline. Markers are three-model means and horizontal lines are climate-model point ranges. Within-cell source-country tonnes supply weights, including shared boundary cells. Country comparisons use frozen point disease parameters; their range is not a confidence interval. Production totals describe reference exposure and cannot be interpreted as projected production or tonnes lost to STB.')


def main():
    DEST.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    assert json.loads(CONTRACT.read_text())['map_color_limits_percentage_points']==[-20,20]
    periods,common,country=period_arrays()
    figure4(periods,common);figure5();figure6(country)
    HASHES[str(CONTRACT.relative_to(ROOT))]=sha(CONTRACT)
    HASHES['data/paper_study/wheat_production/europe_wheat_production_025.parquet']=sha(ROOT/'data/paper_study/wheat_production/europe_wheat_production_025.parquet')
    (DEST/'climate_and_production_captions.json').write_text(json.dumps(CAPTIONS,indent=2)+'\n')
    (DEST/'climate_and_production_provenance.json').write_text(json.dumps(dict(source_hashes=HASHES,source_code_sha256=sha(Path(__file__))),indent=2)+'\n')
    print(json.dumps(dict(figures=list(CAPTIONS),common_map_cells=int(common.sum()),source_data=str(DATA.relative_to(ROOT)))),flush=True)


if __name__=='__main__':main()
