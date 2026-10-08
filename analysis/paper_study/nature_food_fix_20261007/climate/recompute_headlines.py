"""Independent headline recomputation from the complete registered season table."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
METRICS=['F1_infection_relative_anthesis_days','F1_symptom_relative_anthesis_days','F2_infection_relative_anthesis_days','F2_symptom_relative_anthesis_days',
    'F3_infection_relative_anthesis_days','F3_symptom_relative_anthesis_days','GS65_85_lost_had3','GS31_85_lost_had3','GS65_85_functional_lost_fraction',
    'conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai','conditional_yield_loss_GS31_85_b0180_t_ha_per_unit_lai','any_top3_infection_before85','any_top3_symptom_before85']


def main():
    reg=json.loads((HERE/'registration_before_robustness_results.json').read_text())
    seasons=pd.read_parquet(HERE/'draw_season_outputs.parquet');assert len(seasons)==161280
    draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv');weights=draws.area_mean_weight.to_numpy()
    strata=[g.index.to_numpy() for _,g in draws.groupby('stratum',sort=False)]
    influences={};results={}
    for model in reg['models']:
        arrays={}
        for (setting,period),g in seasons.loc[seasons.model.eq(model)].groupby(['setting','period']):
            arrays[(setting,period)]=g.sort_values(['spatial_draw_id','harvest_year'])[METRICS].to_numpy().reshape(64,30,len(METRICS))
        valid={s:np.isfinite(arrays[(s,'1991-2020')])&np.isfinite(arrays[(s,'2071-2100')]) for s in reg['settings']}
        all_common=np.logical_and.reduce(list(valid.values()))
        for setting in reg['settings']:
            a=arrays[(setting,'1991-2020')];b=arrays[(setting,'2071-2100')]
            for population,mask in [('within_setting_paired',valid[setting]),('all_settings_common_paired',all_common)]:
                denominator=mask.mean(axis=1);numerator=np.where(mask,b-a,0.).mean(axis=1)
                coverage=weights@denominator;delta=(weights@numerator)/coverage
                ref=(weights@np.where(mask,a,0.).mean(axis=1))/coverage;future=(weights@np.where(mask,b,0.).mean(axis=1))/coverage
                influence=(numerator-delta[None,:]*denominator)/coverage[None,:]
                for j,metric in enumerate(METRICS):
                    key=(setting,metric,population);influences.setdefault(key,[]).append(influence[:,j])
                    results.setdefault(key,[]).append(dict(model=model,change=float(delta[j]),reference=float(ref[j]),future=float(future[j]),coverage=float(coverage[j]),pairs=int(mask[:,:,j].sum())))
    output=[]
    for key,parts in results.items():
        setting,metric,population=key;inf=np.mean(influences[key],axis=0)
        variance=sum(float(weights[idx].sum())**2*np.var(inf[idx],ddof=1)/len(idx) for idx in strata)
        output.append(dict(setting=setting,metric=metric,population=population,gcm_mean_change=float(np.mean([p['change'] for p in parts])),
            gcm_min_change=min(p['change'] for p in parts),gcm_max_change=max(p['change'] for p in parts),spatial_mcse_gcm_mean_change=float(np.sqrt(variance)),
            gcm_mean_reference=float(np.mean([p['reference'] for p in parts])),gcm_mean_future=float(np.mean([p['future'] for p in parts])),
            minimum_gcm_paired_coverage=min(p['coverage'] for p in parts),maximum_gcm_paired_coverage=max(p['coverage'] for p in parts),
            minimum_gcm_draw_year_pairs=min(p['pairs'] for p in parts),maximum_gcm_draw_year_pairs=max(p['pairs'] for p in parts)))
    frame=pd.DataFrame(output);frame.to_csv(HERE/'independently_recomputed_headlines.csv',index=False)
    for metric in ['F1_infection_relative_anthesis_days','F1_symptom_relative_anthesis_days','GS65_85_lost_had3','conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai']:
        p=frame.loc[frame.population.eq('all_settings_common_paired')&frame.metric.eq(metric)]
        print(metric,'mean range',p.gcm_mean_change.min(),p.gcm_mean_change.max(),'GCM range',p.gcm_min_change.min(),p.gcm_max_change.max(),'MCSE range',p.spatial_mcse_gcm_mean_change.min(),p.spatial_mcse_gcm_mean_change.max())
    print(frame.loc[frame.population.eq('all_settings_common_paired')&frame.metric.eq('GS65_85_lost_had3')].to_string(index=False))


if __name__=='__main__':main()
