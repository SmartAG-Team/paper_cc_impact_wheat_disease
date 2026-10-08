"""Independent reconstruction of parameter bands and counterfactual arithmetic."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
PARAM=ROOT/'data/paper_study/regional_parameter_uncertainty'
COUNTER=ROOT/'data/paper_study/climate_counterfactuals'
REPORT=ROOT/'data/paper_study/projection_uncertainty_reporting'
DEST=ROOT/'analysis/paper_study/projection_uncertainty_reporting'
MODELS=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0'];SCENARIOS=['ssp126','ssp245','ssp585']
PERIODS={'baseline1991_2020':range(1991,2021),'midcentury2031_2060':range(2031,2061),'latecentury2071_2100':range(2071,2101)}


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def sampling_error(influence,draws):
    result=np.zeros(influence.shape[:-1])
    for name in sorted(draws.stratum.unique()):
        mask=draws.stratum.eq(name).to_numpy();values=influence[...,mask]
        mean=values.sum(axis=-1)/4
        variance=((values-mean[...,None])**2).sum(axis=-1)/3
        weight=float(draws.loc[mask,'area_mean_weight'].sum())
        result+=weight*weight*variance/4
    return np.sqrt(result)


def main():
    draws=pd.read_csv(PARAM/'spatial_draws.csv');weights=draws.area_mean_weight.to_numpy()
    prod=pd.read_parquet(ROOT/'data/paper_study/wheat_production/europe_wheat_production_025.parquet')
    calendar=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_calendar_scenarios.parquet')
    ids=calendar.loc[calendar.crop_season.eq('winter_wheat')&calendar.water_system.eq('rainfed')&calendar.calendar_valid,'cell_id']
    eligible=prod[prod.cell_id.isin(ids)]
    eligible_tonnes=float(eligible.production_total_tonnes.sum());eligible_ha=float(eligible.harvested_total_ha.sum())
    p=prod.set_index('cell_id').reindex(draws.cell_id)
    importance=(p.production_total_tonnes/p.harvested_total_ha).to_numpy()/(eligible_tonnes/eligible_ha)
    production_bands=pd.read_csv(ROOT/'data/paper_study/production_exposure_reporting/conditional_production_projection_bands.csv')
    parameter_results={};production_results={};counter_results={};production_counter_results={}
    checks=0;maximum=0.
    def compare(a,b):
        nonlocal checks,maximum
        a,b=np.asarray(a),np.asarray(b)
        np.testing.assert_allclose(a,b,rtol=0,atol=1e-10,equal_nan=True)
        maximum=max(maximum,float(np.nanmax(np.abs(a-b))))
        checks+=a.size
    for model in MODELS:
        for scenario in SCENARIOS:
            for period,years in PERIODS.items():
                annual=[];full={key:[0.,0.] for key in ['area','production']}
                for year in years:
                    path=PARAM/'annual'/model/scenario/f'{year}.npz';r=json.loads(path.with_suffix('.json').read_text());assert sha(path)==r['output_sha256']
                    annual.append(np.load(path)['cell_damage_percent'])
                    path=ROOT/'analysis/paper_study/regional_v1/annual/nasa'/model/scenario/'adjusted/winter_wheat_rainfed_sow+0_stage85'/f'{year}.parquet'
                    f=pd.read_parquet(path,columns=['cell_id','status','upper3_final_damage_percent']);assert f.cell_id.to_list()==prod.cell_id.to_list()
                    valid=f.status.eq('complete').to_numpy();values=f.upper3_final_damage_percent.to_numpy()
                    for key,w in [('area',prod.harvested_total_ha.to_numpy()),('production',prod.production_total_tonnes.to_numpy())]:
                        full[key][0]+=float(np.sum(w[valid]*values[valid]));full[key][1]+=float(w[valid].sum())
                values=np.stack(annual,axis=1);valid=np.isfinite(values[0]);delta=np.where(valid[None,:,:],values-values[0:1],0.).sum(axis=1)/30
                for key,population,ratio in [('area',eligible_ha,1.),('production',eligible_tonnes,importance[None,:])]:
                    point=full[key][0]/full[key][1];coverage=full[key][1]/30/population
                    influence=delta*ratio/coverage;means=point+np.sum(influence*weights,axis=-1)
                    mcse=sampling_error(influence,draws)
                    if key=='area':
                        path=REPORT/'parameter_periods'/model/scenario/f'{period}.npz';saved=dict(np.load(path))
                        compare(means,saved['means']);compare(mcse,saved['mcse']);compare(influence,saved['contributions'])
                        parameter_results[model,scenario,period]=dict(means=means,influence=influence)
                    else:
                        selected=production_bands[production_bands.model.eq(model)&production_bands.scenario.eq(scenario)&production_bands.period.eq(period)&production_bands.quantity.eq('production_weighted_final_severity')].iloc[0]
                        compare([means[0],*np.quantile(means[1:],[.025,.975])],[selected.point_estimate_percent,selected.conditional_parameter_p025,selected.conditional_parameter_p975])
                        compare(mcse[1:].max(),selected.bootstrap_spatial_mcse_max)
                        production_results[model,scenario,period]=dict(means=means,influence=influence)
                if period=='baseline1991_2020':continue
                cases=[]
                for year in years:
                    path=COUNTER/'annual'/model/scenario/f'{year}.npz';r=json.loads(path.with_suffix('.json').read_text());assert sha(path)==r['output_sha256']
                    case=dict(np.load(path));assert case['y00'].shape==(101,64)
                    compare(case['weather_effect'],.5*((case['y10']-case['y00'])+(case['y11']-case['y01'])))
                    compare(case['host_effect'],.5*((case['y01']-case['y00'])+(case['y11']-case['y10'])))
                    compare(case['weather_effect']+case['host_effect'],case['y11']-case['y00']);cases.append(case)
                valid=np.stack([case['common_valid'] for case in cases]).sum(axis=0)/30
                saved=dict(np.load(REPORT/'counter_periods'/model/scenario/f'{period}.npz'))
                for quantity in ['weather_effect','host_effect','total_effect']:
                    numerator=np.nan_to_num(np.stack([c[quantity] for c in cases],axis=1)).sum(axis=1)/30
                    for key,ratio in [('area',1.),('production',importance)]:
                        den=float(np.sum(valid*ratio*weights));mean=np.sum(numerator*ratio*weights,axis=-1)/den
                        influence=(numerator*ratio-mean[:,None]*valid*ratio)/den;mcse=sampling_error(influence,draws)
                        if key=='area':
                            compare(mean,saved[quantity+'_means']);compare(mcse,saved[quantity+'_mcse'])
                            counter_results[model,scenario,period,quantity]=dict(means=mean,influence=influence)
                        else:
                            selected=production_bands[production_bands.model.eq(model)&production_bands.scenario.eq(scenario)&production_bands.period.eq(period)&production_bands.quantity.eq('production_sampled_'+quantity)].iloc[0]
                            compare([mean[0],*np.quantile(mean[1:],[.025,.975])],[selected.point_estimate_percent,selected.conditional_parameter_p025,selected.conditional_parameter_p975])
                            compare(mcse[0],selected.point_spatial_mcse)
                            production_counter_results[model,scenario,period,quantity]=dict(means=mean,influence=influence)
    # Ensemble contrasts preserve shared parameter and spatial draws.
    area_bands=pd.read_csv(DEST/'complete_period_summary.csv')
    for key,bands,results in [('area',area_bands,parameter_results),('production',production_bands,production_results)]:
        label='future_minus_model_baseline' if key=='area' else 'production_future_minus_model_baseline'
        for scenario in SCENARIOS:
            for period in list(PERIODS)[1:]:
                means=np.mean([results[m,scenario,period]['means']-results[m,scenario,'baseline1991_2020']['means'] for m in MODELS],axis=0)
                influence=np.mean([results[m,scenario,period]['influence']-results[m,scenario,'baseline1991_2020']['influence'] for m in MODELS],axis=0)
                row=bands[bands.model.eq('three_model_mean')&bands.scenario.eq(scenario)&bands.period.eq(period)&bands.quantity.eq(label)].iloc[0]
                compare([means[0],*np.quantile(means[1:],[.025,.975])],[row.point_estimate_percent,row.conditional_parameter_p025,row.conditional_parameter_p975])
                compare(sampling_error(influence,draws)[1:].max(),row.bootstrap_spatial_mcse_max)
    report=dict(status='passed',checked_parameter_periods=27,checked_counterfactual_periods=18,
        counterfactual_year_pairs=540,independent_scalar_arithmetic_checks=checks,
        maximum_absolute_difference_percentage_points=maximum,
        production_importance_ratio_and_shared_draw_covariance_verified=True,
        source_code_sha256=sha(Path(__file__)))
    (DEST/'independent_reporting_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)


if __name__=='__main__':main()
