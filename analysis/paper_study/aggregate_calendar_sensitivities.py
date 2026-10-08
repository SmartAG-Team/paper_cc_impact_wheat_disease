"""Calendar-mixture and declared point-parameter sensitivity summaries."""
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional import sha,MODELS,SCENARIOS
from analysis.paper_study.aggregate_projection_uncertainty import PERIODS
from calibration.seasonal_septoria.spatial_estimation import sampled_ratio_mean, stratified_mcse
from analysis.paper_study.run_regional_sensitivities import CASES

DEST=ROOT/'analysis/paper_study/calendar_sensitivity_reporting'
DATA=ROOT/'data/paper_study/calendar_sensitivity_reporting'
SAMPLE=ROOT/'data/paper_study/regional_sensitivities/annual'
ANNUAL=ROOT/'analysis/paper_study/regional_v1/annual'


def main():
    DEST.mkdir(parents=True,exist_ok=True);DATA.mkdir(parents=True,exist_ok=True)
    draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv')
    weights=draws.area_mean_weight.to_numpy();strata=draws.stratum.to_numpy()
    area=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025_seasonal_sensitivity.parquet')
    results={};rows=[];hashes={};mixed=[]
    jobs=[('era5','ERA5','baseline','baseline1991_2020')]+[('nasa',m,s,p) for m in MODELS for s in SCENARIOS for p in PERIODS]
    for provider,model,scenario,period in jobs:
        records=[];numerator=denominator=0.;total_area_years=area.harvested_total_ha.sum()*30
        for year in PERIODS[period]:
            path=SAMPLE/provider/model/scenario/f'{year}.npz';r=json.loads(path.with_suffix('.json').read_text());assert sha(path)==r['output_sha256']
            a=dict(np.load(path));assert a['damage'].shape==(5,64);records.append(a['damage']);hashes[str(path.relative_to(ROOT))]=r['output_sha256']
            alignment='original' if provider=='era5' else 'adjusted'
            for calendar,key in [('winter_wheat','autumn_sowing_harvested_ha'),('spring_wheat','spring_sowing_harvested_ha')]:
                path=ANNUAL/provider/model/scenario/alignment/f'{calendar}_rainfed_sow+0_stage85'/f'{year}.parquet'
                r=json.loads(path.with_suffix('.json').read_text());assert sha(path)==r['parquet_sha256']
                f=pd.read_parquet(path,columns=['cell_id','status','upper3_final_damage_percent'])
                assert f.cell_id.to_list()==area.cell_id.to_list();valid=f.status.eq('complete')
                w=area.loc[valid,key].to_numpy();y=f.loc[valid,'upper3_final_damage_percent'].to_numpy()
                numerator+=float(np.dot(w,y));denominator+=float(w.sum());hashes[str(path.relative_to(ROOT))]=r['parquet_sha256']
        values=np.stack(records,axis=1)
        outputs=[]
        for j,case in enumerate(CASES):
            valid=np.isfinite(values[j]);den=valid.mean(axis=0);num=np.nan_to_num(values[j]).mean(axis=0)
            q=sampled_ratio_mean(num,den,weights,strata)
            q['influence']=(num-q['means']*den)/q['valid_fraction'];outputs.append(q)
            rows.append(dict(provider=provider,model=model,scenario=scenario,period=period,case=case,
                quantity='sampled_mean_final_severity',estimate_percent=float(q['means']),spatial_mcse=float(q['mcse']),valid_sample_area_year_fraction=q['valid_fraction']))
        results[provider,model,scenario,period]=outputs
        mixed.append(dict(provider=provider,model=model,scenario=scenario,period=period,
            conditional_mean_final_severity_percent=numerator/denominator,
            assigned_complete_fraction_all_area_years=denominator/total_area_years,
            all_area_missing_lower_severity_percent=numerator/total_area_years,
            all_area_missing_upper_severity_percent=(numerator+100*(total_area_years-denominator))/total_area_years,
            area_allocation='MIRCA autumn and spring area weights with corresponding GGCMI calendar trajectories; unknown seasons excluded from conditional mean'))
    for scenario in SCENARIOS:
        for period in list(PERIODS)[1:]:
            bycase={case:[] for case in CASES};influences={case:[] for case in CASES}
            for model in MODELS:
                base=results['nasa',model,scenario,'baseline1991_2020'];future=results['nasa',model,scenario,period]
                for j,case in enumerate(CASES):
                    mean=future[j]['means']-base[j]['means'];influence=future[j]['influence']-base[j]['influence']
                    rows.append(dict(provider='nasa',model=model,scenario=scenario,period=period,case=case,
                        quantity='sampled_future_minus_case_baseline',estimate_percent=float(mean),spatial_mcse=float(stratified_mcse(influence,weights,strata))))
                    bycase[case].append(mean);influences[case].append(influence)
            for case in CASES:
                mean=np.mean(bycase[case]);influence=np.mean(influences[case],axis=0)
                rows.append(dict(provider='nasa',model='three_model_mean',scenario=scenario,period=period,case=case,
                    quantity='sampled_future_minus_case_baseline',estimate_percent=float(mean),spatial_mcse=float(stratified_mcse(influence,weights,strata))))
    pd.DataFrame(rows).to_csv(DATA/'sampled_sensitivity_periods.csv',index=False)
    pd.DataFrame(mixed).to_csv(DATA/'mirca_mixed_calendar_periods.csv',index=False)
    report=dict(status='complete',periods=len(jobs),spatial_draws=64,parameters_refitted=False,
        source_hashes=hashes,source_code_sha256=sha(Path(__file__)),
        calendar_mixture_unknown_area_retained_in_bounds=True,mixture_is_verified_genotype_partition=False)
    (DEST/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_hashes'}),flush=True)


if __name__=='__main__':main()
