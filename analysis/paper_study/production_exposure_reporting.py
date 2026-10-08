"""Reference-production exposure with full-grid points and sampled uncertainty."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional import sha,MODELS,SCENARIOS
from calibration.seasonal_septoria.spatial_estimation import anchored_parameter_means, stratified_mcse
from analysis.paper_study.aggregate_projection_uncertainty import PERIODS,percentile_summary

DEST=ROOT/'analysis/paper_study/production_exposure_reporting'
DATA=ROOT/'data/paper_study/production_exposure_reporting'
ANNUAL=ROOT/'analysis/paper_study/regional_v1/annual'
PARAM=ROOT/'data/paper_study/regional_parameter_uncertainty'
COUNTER=ROOT/'data/paper_study/climate_counterfactuals'


def point_period(provider,model,scenario,period,calendar,production):
    rows=[];hashes={};numerator=denominator=0.
    alignment='original' if provider=='era5' else 'adjusted'
    for year in PERIODS[period]:
        path=ANNUAL/provider/model/scenario/alignment/f'{calendar}_rainfed_sow+0_stage85'/f'{year}.parquet'
        r=json.loads(path.with_suffix('.json').read_text());assert sha(path)==r['parquet_sha256']
        frame=pd.read_parquet(path,columns=['cell_id','status','upper3_final_damage_percent'])
        assert frame.cell_id.to_list()==production.cell_id.to_list()
        valid=frame.status.eq('complete').to_numpy();p=production.production_total_tonnes.to_numpy()
        y=frame.upper3_final_damage_percent.to_numpy();den=float(p[valid].sum());num=float(np.dot(p[valid],y[valid]))
        numerator+=num;denominator+=den;row=dict(provider=provider,model=model,scenario=scenario,period=period,
            calendar=calendar,year=year,reference_production_tonnes=float(p.sum()),valid_reference_production_tonnes=den,
            mean_severity_percent=num/den,valid_production_fraction=den/p.sum())
        for cutoff in [5,25,50]:
            exposed=float(p[valid&(y>=cutoff)].sum())
            row[f'reference_production_exposure_ge{cutoff}_tonnes']=exposed
            row[f'production_weighted_exceedance_ge{cutoff}_fraction']=exposed/den
        row['missing_outcome_lower_severity_percent']=num/p.sum()
        row['missing_outcome_upper_severity_percent']=(num+100*(p.sum()-den))/p.sum()
        rows.append(row);hashes[str(path.relative_to(ROOT))]=r['parquet_sha256']
    frame=pd.DataFrame(rows)
    pooled=dict(provider=provider,model=model,scenario=scenario,period=period,calendar=calendar,
        mean_severity_percent=numerator/denominator,valid_production_fraction=denominator/len(rows)/production.production_total_tonnes.sum(),
        reference_production_tonnes=float(production.production_total_tonnes.sum()))
    for cutoff in [5,25,50]:
        pooled[f'annual_mean_reference_production_exposure_ge{cutoff}_tonnes']=float(frame[f'reference_production_exposure_ge{cutoff}_tonnes'].mean())
        pooled[f'production_weighted_exceedance_ge{cutoff}_fraction']=float(frame[f'reference_production_exposure_ge{cutoff}_tonnes'].sum()/denominator)
    return frame,pooled,hashes


def sampled_production_ratio(numerator_by_draw,valid_by_draw,importance,weights,strata):
    numerator=np.asarray(numerator_by_draw)*importance
    denominator_by_draw=np.asarray(valid_by_draw)*importance
    denominator=float(denominator_by_draw@weights)
    assert denominator>0
    means=(numerator@weights)/denominator
    influence=(numerator-means[...,None]*denominator_by_draw)/denominator
    return dict(means=means,influence=influence,mcse=stratified_mcse(influence,weights,strata),
        estimated_population_weight_coverage=denominator)


def main():
    DEST.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    production_path=ROOT/'data/paper_study/wheat_production/europe_wheat_production_025.parquet'
    production=pd.read_parquet(production_path)
    cal=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_calendar_scenarios.parquet')
    ids=cal.loc[cal.crop_season.eq('winter_wheat')&cal.water_system.eq('rainfed')&cal.calendar_valid,'cell_id']
    pop=production[production.cell_id.isin(ids)]
    eligible_tonnes=float(pop.production_total_tonnes.sum())
    eligible_ha=float(pop.harvested_total_ha.sum())
    draws=pd.read_csv(PARAM/'spatial_draws.csv');weights=draws.area_mean_weight.to_numpy();strata=draws.stratum.to_numpy()
    sample=production.set_index('cell_id').reindex(draws.cell_id)
    importance=sample.production_total_tonnes.to_numpy()/sample.harvested_total_ha.to_numpy()/(eligible_tonnes/eligible_ha)
    annuals=[];periods=[];hashes={};parameter={};counter={};bands=[]
    jobs=[('era5','ERA5','baseline','baseline1991_2020')]+[
        ('nasa',m,s,p) for m in MODELS for s in SCENARIOS for p in PERIODS]
    for provider,model,scenario,period in jobs:
        for calendar in ['winter_wheat','spring_wheat']:
            annual,point,sources=point_period(provider,model,scenario,period,calendar,production)
            annuals.append(annual);periods.append(point);hashes.update(sources)
            if provider=='era5' or calendar!='winter_wheat':continue
            sample_arrays=[]
            for year in PERIODS[period]:
                path=PARAM/'annual'/model/scenario/f'{year}.npz';r=json.loads(path.with_suffix('.json').read_text())
                assert sha(path)==r['output_sha256'];hashes[str(path.relative_to(ROOT))]=r['output_sha256']
                sample_arrays.append(np.load(path)['cell_damage_percent'])
            values=np.stack(sample_arrays,axis=1)
            full_valid=point['valid_production_fraction']*point['reference_production_tonnes']/eligible_tonnes
            result=anchored_parameter_means(values*importance[None,None,:],point['mean_severity_percent'],full_valid,weights,strata)
            parameter[model,scenario,period]=result
            bands.append(percentile_summary(model,scenario,period,'production_weighted_final_severity',result['means'],result['mcse']))
            if period=='baseline1991_2020':continue
            cases=[]
            for year in PERIODS[period]:
                path=COUNTER/'annual'/model/scenario/f'{year}.npz';r=json.loads(path.with_suffix('.json').read_text())
                assert sha(path)==r['output_sha256'];hashes[str(path.relative_to(ROOT))]=r['output_sha256'];cases.append(dict(np.load(path)))
            valid=np.stack([case['common_valid'] for case in cases]).mean(axis=0)
            output={}
            for key in ['weather_effect','host_effect','total_effect']:
                values=np.stack([case[key] for case in cases],axis=1)
                result=sampled_production_ratio(np.nan_to_num(values).mean(axis=1),valid,importance,weights,strata)
                output[key]=result
                bands.append(percentile_summary(model,scenario,period,'production_sampled_'+key,result['means'],result['mcse']))
            np.testing.assert_allclose(output['weather_effect']['means']+output['host_effect']['means'],output['total_effect']['means'],rtol=0,atol=1e-11)
            counter[model,scenario,period]=output
        print(json.dumps(dict(provider=provider,model=model,scenario=scenario,period=period)),flush=True)
    for scenario in SCENARIOS:
        for period in list(PERIODS)[1:]:
            contrasts=[];influences=[]
            for model in MODELS:
                base,future=parameter[model,scenario,'baseline1991_2020'],parameter[model,scenario,period]
                means=future['means']-base['means'];influence=future['contributions']-base['contributions']
                contrasts.append(means);influences.append(influence)
                bands.append(percentile_summary(model,scenario,period,'production_future_minus_model_baseline',means,stratified_mcse(influence,weights,strata)))
            bands.append(percentile_summary('three_model_mean',scenario,period,'production_future_minus_model_baseline',
                np.mean(contrasts,axis=0),stratified_mcse(np.mean(influences,axis=0),weights,strata),
                climate_model_point_min=float(min(x[0] for x in contrasts)),climate_model_point_max=float(max(x[0] for x in contrasts))))
            for key in ['weather_effect','host_effect','total_effect']:
                selected=[counter[m,scenario,period][key] for m in MODELS]
                means=np.mean([x['means'] for x in selected],axis=0)
                mcse=stratified_mcse(np.mean([x['influence'] for x in selected],axis=0),weights,strata)
                bands.append(percentile_summary('three_model_mean',scenario,period,'production_sampled_'+key,means,mcse))
    pd.concat(annuals,ignore_index=True).to_csv(DATA/'europe_annual_production_exposure.csv',index=False)
    pd.DataFrame(periods).to_csv(DATA/'europe_period_production_exposure.csv',index=False)
    pd.DataFrame(bands).to_csv(DATA/'conditional_production_projection_bands.csv',index=False)
    report=dict(status='complete',full_point_periods=len(periods),parameter_periods=len(parameter),counterfactual_periods=len(counter),
        source_hashes=hashes,source_production_sha256=sha(production_path),source_code_sha256=sha(Path(__file__)),
        reference_production_tonnes=float(production.production_total_tonnes.sum()),winter_calendar_eligible_tonnes=eligible_tonnes,
        parameter_band_sampling='area-PPS draws with explicit production/area importance ratio and exact full-production point anchor',
        production_is_untreated_potential_yield=False,exposure_tonnes_are_yield_loss=False,
        future_production_change_simulated=False)
    (DEST/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'}),flush=True)


if __name__=='__main__':main()
