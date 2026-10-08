"""Independent checks against individual frozen seasons and corner trajectories."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DEST=HERE/'derived/spatial'
METRICS=['GS65_85_lost_had3','F1_symptom_relative_anthesis_days','grain_fill_days']


def verify():
    source=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/climate/draw_season_outputs.csv'
    seasons=pd.read_csv(source,usecols=['model','scenario','harvest_year','spatial_draw_id',*METRICS])
    draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv').sort_values('spatial_draw_id').reset_index(drop=True)
    membership=pd.read_csv(DEST/'sample_region_membership.csv')
    registry=pd.read_csv(HERE/'spatial_sources/wheat_cells_eea_regions.csv')
    expected=draws.merge(registry[['cell_id','environment_region']],on='cell_id',validate='many_to_one')
    assert expected[['spatial_draw_id','cell_id','environment_region']].equals(membership[['spatial_draw_id','cell_id','environment_region']])
    regions=expected.environment_region.to_numpy()
    weights=draws.area_mean_weight.to_numpy()
    domains=['Europe']+sorted(set(registry.environment_region))
    observed=pd.read_csv(DEST/'region_paired_changes_by_gcm.csv').set_index(['environment_region','scenario','period','metric','climate_model'])
    ensemble=pd.read_csv(DEST/'region_paired_changes.csv').set_index(['environment_region','scenario','period','metric'])
    maps=pd.read_csv(DEST/'sampled_cell_scenario_changes.csv')
    signs=pd.read_csv(DEST/'canopy_change_sign_area_shares.csv').set_index(['scenario','period','classification'])
    model_checks=0; ensemble_checks=0; map_checks=0; sign_checks=0
    for scenario in ['ssp126','ssp245','ssp585']:
        for first,end in [(2031,2060),(2071,2100)]:
            period=f'{first}-{end}'; shift=first-1991
            estimates={}; residuals={}; cell_values={}
            for model in ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']:
                part=seasons[seasons.model.eq(model)&seasons.scenario.eq(scenario)]
                history=part[part.harvest_year.between(1991,2020)]
                future=part[part.harvest_year.between(first,end)].copy()
                future['harvest_year']=future.harvest_year-shift
                paired=history.merge(future,on=['spatial_draw_id','harvest_year'],suffixes=('_reference','_future'),validate='one_to_one')
                assert len(paired)==1920
                paired=paired.sort_values(['spatial_draw_id','harvest_year'])
                for metric in METRICS:
                    baseline=paired[metric+'_reference'].to_numpy().reshape(64,30)
                    future_values=paired[metric+'_future'].to_numpy().reshape(64,30)
                    valid=np.isfinite(baseline)&np.isfinite(future_values)
                    difference=future_values-baseline
                    cell_values[model,metric]=np.array([difference[i,valid[i]].mean() if valid[i].any() else np.nan for i in range(64)])
                    for region in domains:
                        eligible=np.ones(64,bool) if region=='Europe' else regions==region
                        mask=valid&eligible[:,None]
                        long_weights=np.repeat(weights[:,None]/30,30,axis=1)
                        denominator=long_weights[mask].sum()
                        mean=(long_weights[mask]*difference[mask]).sum()/denominator if denominator else np.nan
                        recorded=observed.loc[region,scenario,period,metric,model]
                        assert np.isclose(mean,recorded.mean_change,atol=1e-11,equal_nan=True)
                        model_checks+=1
                        estimates[region,model,metric]=mean
                        if denominator:
                            residuals[region,model,metric]=np.array([
                                ((difference[i,mask[i]]-mean).sum()/30/denominator) if mask[i].any() else 0. for i in range(64)])
            for region in domains:
                for metric in METRICS:
                    values=[estimates[region,model,metric] for model in ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']]
                    mean=float(np.mean(values))
                    recorded=ensemble.loc[region,scenario,period,metric]
                    assert np.isclose(mean,recorded.mean_change,atol=1e-11,equal_nan=True)
                    if np.isfinite(mean) and recorded.sample_draws>=4:
                        vector=np.mean([residuals[region,model,metric] for model in ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']],axis=0)
                        variance=0.
                        for stratum in draws.stratum.unique():
                            ix=np.flatnonzero(draws.stratum.eq(stratum))
                            stratum_weight=weights[ix].sum()
                            centered=vector[ix]-vector[ix].mean()
                            variance+=stratum_weight**2*np.dot(centered,centered)/(len(ix)*(len(ix)-1))
                        assert np.isclose(np.sqrt(variance),recorded.spatial_mcse,atol=1e-11)
                    elif recorded.sample_draws<4:
                        assert pd.isna(recorded.spatial_mcse)
                    ensemble_checks+=1
            for metric in METRICS:
                matrix=np.stack([cell_values[model,metric] for model in ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']],axis=1)
                for cell, indices in draws.groupby('cell_id').groups.items():
                    first_draw=list(indices)[0]
                    row=maps[maps.scenario.eq(scenario)&maps.period.eq(period)&maps.metric.eq(metric)&maps.cell_id.eq(cell)].iloc[0]
                    assert np.isclose(matrix[first_draw].mean(),row.mean_change,atol=1e-11,equal_nan=True)
                    assert row.positive_gcm_count==np.sum(matrix[first_draw]>0)
                    assert row.negative_gcm_count==np.sum(matrix[first_draw]<0)
                    map_checks+=1
                if metric==METRICS[0]:
                    assert np.isfinite(matrix).all()
                    classifications={'positive_ensemble':matrix.mean(axis=1)>0,
                                     'positive_all_three_gcm':(matrix>0).all(axis=1),
                                     'negative_all_three_gcm':(matrix<0).all(axis=1),
                                     'mixed_gcm_sign':(matrix>0).any(axis=1)&(matrix<0).any(axis=1)}
                    for label,indicator in classifications.items():
                        fraction=100*weights[indicator].sum()/weights.sum()
                        assert np.isclose(fraction,signs.loc[scenario,period,label].estimated_reference_area_share_percent,atol=1e-11)
                        sign_checks+=1
                    assert np.isclose(sum(signs.loc[scenario,period,label].estimated_reference_area_share_percent for label in
                        ['positive_all_three_gcm','negative_all_three_gcm','mixed_gcm_sign']),100.)
    decomp=ROOT/'analysis/paper_study/nature_food_fix_20261007/climate/decomposition'
    corners=pd.read_parquet(decomp/'four_corner_draw_year_outputs.parquet')
    wide=corners.pivot(index=['model','spatial_draw_id','reference_year'],columns='corner',values=METRICS[0]).reset_index()
    flags=pd.read_csv(decomp/'supported_forcing_draw_pair_flags.csv')
    wide=wide.merge(flags[['model','spatial_draw_id','reference_year','excluded_unsupported_hybrid_forcing']],
                    on=['model','spatial_draw_id','reference_year'],validate='one_to_one')
    wide=wide.merge(expected[['spatial_draw_id','environment_region','area_mean_weight']],on='spatial_draw_id',validate='many_to_one')
    cols=['Wreference_Hreference','Wfuture_Hreference','Wreference_Hfuture','Wfuture_Hfuture']
    wide=wide[wide[cols].notna().all(axis=1)&~wide.excluded_unsupported_hybrid_forcing].copy()
    assert len(wide)==5731
    wide['weather']=.5*((wide.Wfuture_Hreference-wide.Wreference_Hreference)+(wide.Wfuture_Hfuture-wide.Wreference_Hfuture))
    wide['host']=.5*((wide.Wreference_Hfuture-wide.Wreference_Hreference)+(wide.Wfuture_Hfuture-wide.Wfuture_Hreference))
    wide['net']=wide.Wfuture_Hfuture-wide.Wreference_Hreference
    recorded=pd.read_csv(DEST/'region_supported_decomposition_by_gcm.csv').set_index(['environment_region','climate_model'])
    decomposition_checks=0
    for (region,model), row in recorded.iterrows():
        part=wide[wide.model.eq(model)]
        if region!='Europe':part=part[part.environment_region.eq(region)]
        for component in ['weather','host','net']:
            value=np.average(part[component],weights=part.area_mean_weight) if len(part) else np.nan
            assert np.isclose(value,row[component],atol=1e-11,equal_nan=True)
            decomposition_checks+=1
    context=pd.read_csv(DEST/'region_sampling_context.csv')
    assert context[context.environment_region.eq('Europe')].iloc[0].sample_draws==64
    assert context[context.environment_region.eq('Europe')].iloc[0].unique_sampled_cells==62
    assert all(ensemble.reset_index().query('sample_draws == 0').mean_change.isna())
    report=dict(status='passed',source='Individual frozen seasons and original supported four-corner trajectories',
        GCM_domain_ratios_checked=model_checks,ensemble_domain_means_and_MCSE_checked=ensemble_checks,
        sampled_cell_map_values_and_signs_checked=map_checks,design_weighted_area_sign_shares_checked=sign_checks,
        supported_regional_decomposition_components_checked=decomposition_checks,
        supported_draw_year_pairs=5731,unsampled_regions_unestimated=True,sparse_region_MCSE_withheld=True)
    (DEST/'independent_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    print(json.dumps(verify(),indent=2))
