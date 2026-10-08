"""Complete-period conditional parameter bands and spatial sampling error."""

import argparse,json,sys,time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional import MODELS,SCENARIOS,sha
from calibration.seasonal_septoria.spatial_estimation import anchored_parameter_means, stratified_mcse, sampled_ratio_mean

DEST=ROOT/'analysis/paper_study/projection_uncertainty_reporting'
DATA=ROOT/'data/paper_study/projection_uncertainty_reporting'
PARAM=ROOT/'data/paper_study/regional_parameter_uncertainty'
COUNTER=ROOT/'data/paper_study/climate_counterfactuals'
ANNUAL=ROOT/'analysis/paper_study/regional_v1/annual/nasa'
PERIODS={'baseline1991_2020':list(range(1991,2021)),
    'midcentury2031_2060':list(range(2031,2061)),'latecentury2071_2100':list(range(2071,2101))}


def read_verified(path,key):
    receipt=path.with_suffix('.json')
    if not path.exists() or not receipt.exists():raise FileNotFoundError(path)
    info=json.loads(receipt.read_text())
    if sha(path)!=info[key]:raise ValueError('Input byte digest differs: '+str(path))
    return info


def full_point(model,scenario,years,eligible_area):
    numerator,denominator=0.,0.;hashes={}
    for year in years:
        path=ANNUAL/model/scenario/'adjusted/winter_wheat_rainfed_sow+0_stage85'/f'{year}.parquet'
        receipt=read_verified(path,'parquet_sha256');hashes[str(path.relative_to(ROOT))]=receipt['parquet_sha256']
        frame=pd.read_parquet(path,columns=['cell_id','status','harvested_total_ha','upper3_final_damage_percent'])
        if len(frame)!=14941 or not frame.cell_id.is_unique:raise ValueError('Full-grid point membership differs.')
        valid=frame.status.eq('complete');y=frame.loc[valid,'upper3_final_damage_percent'];w=frame.loc[valid,'harvested_total_ha']
        numerator+=float(np.dot(y,w));denominator+=float(w.sum())
    return numerator/denominator,denominator/eligible_area/len(years),hashes


def percentile_summary(model,scenario,period,quantity,means,mcse,**extra):
    low,high=np.quantile(means[1:],[.025,.975])
    return dict(model=model,scenario=scenario,period=period,quantity=quantity,
        point_estimate_percent=float(means[0]),conditional_parameter_p025=float(low),
        conditional_parameter_p975=float(high),point_spatial_mcse=float(mcse[0]),
        bootstrap_spatial_mcse_median=float(np.median(mcse[1:])),
        bootstrap_spatial_mcse_max=float(np.max(mcse[1:])),bootstrap_draws=100,spatial_draws=64,
        observation_residual_error_included=False,phenology_parameter_uncertainty_included=False,**extra)


def parameter_period(model,scenario,period,draws,eligible_area):
    years=PERIODS[period];records=[];hashes={}
    for year in years:
        path=PARAM/'annual'/model/scenario/f'{year}.npz'
        receipt=read_verified(path,'output_sha256');hashes[str(path.relative_to(ROOT))]=receipt['output_sha256']
        if receipt['spatial_sample_sha256']!=sha(PARAM/'spatial_draws.csv') or receipt['bootstrap_draws']!=100:
            raise ValueError('Frozen bootstrap/spatial membership differs.')
        array=np.load(path)['cell_damage_percent']
        if array.shape!=(101,64):raise ValueError('Parameter ensemble shape differs.')
        records.append(array)
    values=np.stack(records,axis=1)
    point,valid,point_hashes=full_point(model,scenario,years,eligible_area)
    result=anchored_parameter_means(values,point,valid,draws.area_mean_weight,draws.stratum)
    sample_valid=np.isfinite(values[0]).mean(axis=0)
    sample=sampled_ratio_mean(np.nan_to_num(values[0]).mean(axis=0),sample_valid,
        draws.area_mean_weight,draws.stratum)
    result.update(full_point=point,full_valid_fraction=valid,sample_point=float(sample['means']),
        sample_point_mcse=float(sample['mcse']))
    path=DATA/'parameter_periods'/model/scenario/f'{period}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,means=result['means'],mcse=result['mcse'],contributions=result['contributions'])
    path.with_suffix('.json').write_text(json.dumps(dict(source_hashes={**hashes,**point_hashes},
        output_sha256=sha(path),full_point_mean=point,full_valid_fraction=valid,
        sample_point_mean=result['sample_point'],sample_point_mcse=result['sample_point_mcse'],
        source_code_sha256=sha(Path(__file__))),indent=2)+'\n')
    return result


def counter_period(model,scenario,period,draws):
    arrays=[];hashes={}
    for year in PERIODS[period]:
        path=COUNTER/'annual'/model/scenario/f'{year}.npz';r=read_verified(path,'output_sha256')
        hashes[str(path.relative_to(ROOT))]=r['output_sha256'];arrays.append(dict(np.load(path)))
    valid=np.stack([a['common_valid'] for a in arrays]).mean(axis=0)
    output={}
    for key in ['y00','y10','y01','y11','weather_effect','host_effect','total_effect']:
        values=np.stack([a[key] for a in arrays],axis=1)
        if values.shape!=(101,30,64):raise ValueError('Counterfactual draw membership differs.')
        numerator=np.nan_to_num(values).mean(axis=1)
        result=sampled_ratio_mean(numerator,valid,draws.area_mean_weight,draws.stratum)
        result['influence']=(numerator-result['means'][:,None]*valid)/result['valid_fraction']
        output[key]=result
    np.testing.assert_allclose(output['weather_effect']['means']+output['host_effect']['means'],
        output['total_effect']['means'],rtol=0,atol=1e-11)
    path=DATA/'counter_periods'/model/scenario/f'{period}.npz';path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,**{key+'_'+metric:value[metric] for key,value in output.items() for metric in ['means','mcse','influence']})
    path.with_suffix('.json').write_text(json.dumps(dict(source_hashes=hashes,output_sha256=sha(path),
        valid_sample_area_year_fraction=float(output['total_effect']['valid_fraction']),
        source_code_sha256=sha(Path(__file__))),indent=2)+'\n')
    return output


def aggregate():
    draws=pd.read_csv(PARAM/'spatial_draws.csv');weights=draws.area_mean_weight.to_numpy();strata=draws.stratum.to_numpy()
    eligible=json.loads((PARAM/'sampling_receipt.json').read_text())['eligible_harvested_ha']
    parameter,counter={},{};rows=[];inventory=[]
    for model in MODELS:
        for scenario in SCENARIOS:
            for period in PERIODS:
                try:p=parameter_period(model,scenario,period,draws,eligible)
                except FileNotFoundError:
                    inventory.append(dict(type='parameter',model=model,scenario=scenario,period=period,status='incomplete'));continue
                parameter[model,scenario,period]=p
                rows.append(percentile_summary(model,scenario,period,'final_upper3_severity',p['means'],p['mcse'],
                    exact_full_grid_point_anchor=True,full_valid_area_year_fraction=p['full_valid_fraction'],
                    raw_sample_point_mean=p['sample_point'],raw_sample_point_mcse=p['sample_point_mcse']))
                if period!='baseline1991_2020':
                    try:c=counter_period(model,scenario,period,draws)
                    except FileNotFoundError:
                        inventory.append(dict(type='counterfactual',model=model,scenario=scenario,period=period,status='incomplete'));continue
                    counter[model,scenario,period]=c
                    for quantity in ['weather_effect','host_effect','total_effect']:
                        q=c[quantity];rows.append(percentile_summary(model,scenario,period,'sampled_counterfactual_'+quantity,q['means'],q['mcse'],exact_full_grid_point_anchor=False))
    for scenario in SCENARIOS:
        for period in list(PERIODS)[1:]:
            contrast=[];contributions=[]
            for model in MODELS:
                key=model,scenario,period;basekey=model,scenario,'baseline1991_2020'
                if key not in parameter or basekey not in parameter:continue
                future,base=parameter[key],parameter[basekey]
                mean=future['means']-base['means'];influence=future['contributions']-base['contributions']
                mcse=stratified_mcse(influence,weights,strata)
                contrast.append(mean);contributions.append(influence)
                rows.append(percentile_summary(model,scenario,period,'future_minus_model_baseline',mean,mcse,exact_full_grid_point_anchor=True))
            if len(contrast)==3:
                mean=np.mean(contrast,axis=0);mcse=stratified_mcse(np.mean(contributions,axis=0),weights,strata)
                rows.append(percentile_summary('three_model_mean',scenario,period,'future_minus_model_baseline',mean,mcse,
                    exact_full_grid_point_anchor=True,climate_model_point_min=float(min(x[0] for x in contrast)),
                    climate_model_point_max=float(max(x[0] for x in contrast))))
            c=[counter.get((model,scenario,period)) for model in MODELS]
            if all(q is not None for q in c):
                for quantity in ['weather_effect','host_effect','total_effect']:
                    mean=np.mean([q[quantity]['means'] for q in c],axis=0)
                    mcse=stratified_mcse(np.mean([q[quantity]['influence'] for q in c],axis=0),weights,strata)
                    rows.append(percentile_summary('three_model_mean',scenario,period,'sampled_counterfactual_'+quantity,mean,mcse,exact_full_grid_point_anchor=False))
    pd.DataFrame(rows).to_csv(DEST/'complete_period_summary.csv',index=False)
    (DEST/'period_inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    report=dict(complete_parameter_periods=len(parameter),expected_parameter_periods=27,
        complete_counterfactual_periods=len(counter),expected_counterfactual_periods=18,
        source_code_sha256=sha(Path(__file__)),field_parameters_modified=False,
        parameter_percentiles_are_observation_prediction_intervals=False,
        spatial_mcse_reported_separately=True,exact_full_point_anchor=True)
    (DEST/'receipt.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
    return len(parameter)==27 and len(counter)==18


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    while True:
        complete=aggregate()
        if complete or not args.watch:return
        time.sleep(30)


if __name__=='__main__':main()
