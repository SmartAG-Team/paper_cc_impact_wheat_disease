"""Independent arithmetic verification of registered paired climate robustness."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def variance(influence,draws):
    value=0.
    for stratum in draws.stratum.unique():
        idx=np.flatnonzero(draws.stratum.eq(stratum).to_numpy())
        weight=float(draws.area_mean_weight.iloc[idx].sum())
        assert len(idx)==4
        value+=weight*weight*np.var(influence[idx],ddof=1)/len(idx)
    return float(value)


def main():
    reg=json.loads((HERE/'registration_before_robustness_results.json').read_text())
    receipt=json.loads((HERE/'completion_receipt.json').read_text())
    for name,h in reg['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in receipt['output_sha256'].items():assert sha(HERE/name)==h,name
    original_climate=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/climate'
    cache_receipt=json.loads((original_climate/'weather_cache_receipt.json').read_text())
    forcing_archives=set()
    for model in reg['models']:
        for year in reg['periods']['1991-2020']+reg['periods']['2071-2100']:
            for forcing_year in (year-1,year):
                branch='historical' if forcing_year<=2014 else 'ssp585'
                relative=Path('annual_weather_cache')/model/branch/f'{forcing_year}.npz'
                assert sha(original_climate/relative)==cache_receipt['output_sha256'][str(relative)]
                assert sha((original_climate/relative).with_suffix('.json'))==cache_receipt['output_sha256'][str(relative.with_suffix('.json'))]
                forcing_archives.add(str(relative))
    seasons=pd.read_parquet(HERE/'draw_season_outputs.parquet')
    by_gcm=pd.read_parquet(HERE/'paired_changes_by_setting_gcm.parquet')
    ensemble=pd.read_parquet(HERE/'ensemble_paired_changes.parquet')
    draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv').sort_values('spatial_draw_id').reset_index(drop=True)
    assert len(seasons)==161280 and len(reg['settings'])==14
    assert not seasons.duplicated(['setting','model','harvest_year','spatial_draw_id']).any()
    assert seasons.groupby(['setting','model','period']).size().eq(1920).all()
    assert seasons.groupby(['setting','model','harvest_year']).size().eq(64).all()
    assert draws.cell_id.nunique()==62 and len(draws)==64
    weights=draws.area_mean_weight.to_numpy(float)
    metrics=['F1_infection_relative_anthesis_days','F1_symptom_relative_anthesis_days','F2_infection_relative_anthesis_days',
        'F2_symptom_relative_anthesis_days','F3_infection_relative_anthesis_days','F3_symptom_relative_anthesis_days',
        'GS65_85_lost_had3','GS31_85_lost_had3','GS65_85_reference_had3','GS65_85_functional_lost_fraction',
        'conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai','conditional_yield_loss_GS31_85_b0180_t_ha_per_unit_lai',
        'any_top3_infection_before85','any_top3_symptom_before85']
    influences={};summary_errors=[];mcse_errors=[];pair_count_checks=0
    for model in reg['models']:
        arrays={}
        for setting in reg['settings']:
            a=seasons.loc[seasons.model.eq(model)&seasons.setting.eq(setting)&seasons.period.eq('1991-2020')].sort_values(['spatial_draw_id','harvest_year'])
            b=seasons.loc[seasons.model.eq(model)&seasons.setting.eq(setting)&seasons.period.eq('2071-2100')].sort_values(['spatial_draw_id','harvest_year'])
            assert np.array_equal(a.harvest_year.to_numpy()+80,b.harvest_year.to_numpy())
            assert np.array_equal(a.spatial_draw_id,b.spatial_draw_id)
            arrays[setting]={m:(a[m].to_numpy(float).reshape(64,30),b[m].to_numpy(float).reshape(64,30)) for m in metrics}
        for metric in metrics:
            intersection=np.logical_and.reduce([np.isfinite(arrays[s][metric][0])&np.isfinite(arrays[s][metric][1]) for s in reg['settings']])
            for setting in reg['settings']:
                a,b=arrays[setting][metric]
                for population,mask in [('within_setting_paired',np.isfinite(a)&np.isfinite(b)),('all_settings_common_paired',intersection)]:
                    denominator_by_draw=mask.sum(axis=1)/30
                    delta_by_draw=np.where(mask,b-a,0).sum(axis=1)/30
                    historical_by_draw=np.where(mask,a,0).sum(axis=1)/30
                    future_by_draw=np.where(mask,b,0).sum(axis=1)/30
                    coverage=float(weights@denominator_by_draw)
                    delta=float(weights@delta_by_draw)/coverage
                    influence=(delta_by_draw-delta*denominator_by_draw)/coverage
                    mcse=np.sqrt(variance(influence,draws))
                    row=by_gcm.loc[by_gcm.setting.eq(setting)&by_gcm.model.eq(model)&by_gcm.metric.eq(metric)&by_gcm.population.eq(population)].iloc[0]
                    summary_errors.extend([abs(delta-row.change),abs(coverage-row.common_paired_area_time_coverage),
                        abs(float(weights@historical_by_draw)/coverage-row.reference_on_common),abs(float(weights@future_by_draw)/coverage-row.future_on_common)])
                    mcse_errors.append(abs(mcse-row.spatial_mcse_change))
                    assert int(mask.sum())==int(row.common_draw_year_pairs)
                    assert int(mask.all(axis=1).sum())==int(row.draws_complete_all30_pairs)
                    pair_count_checks+=1
                    influences.setdefault((setting,metric,population),[]).append(influence)
    for (setting,metric,population),vectors in influences.items():
        row=ensemble.loc[ensemble.setting.eq(setting)&ensemble.metric.eq(metric)&ensemble.population.eq(population)].iloc[0]
        direct=np.sqrt(variance(np.mean(vectors,axis=0),draws))
        mcse_errors.append(abs(direct-row.spatial_mcse_gcm_mean_change))
    assert max(summary_errors)<1e-10 and max(mcse_errors)<1e-10
    # Verify all reference-decline/functional-conversion transformations from archived daily histories.
    daily_errors=[];daily_error_ratios=[];daily_bounds=[];daily_checks=0;daily_archives=0;event_checks=0
    for model in reg['models']:
        for year in reg['periods']['1991-2020']+reg['periods']['2071-2100']:
            r=json.loads((HERE/'annual_outputs'/model/f'{year}.json').read_text())
            archive=HERE/'daily_baseline'/model/f'{year}.npz'
            assert sha(archive)==r['baseline_daily_sha256']
            assert sha(HERE/'annual_outputs'/model/f'{year}.parquet')==r['output_sha256']
            with np.load(archive) as daily:
                daily_archives+=1
                ids=daily['spatial_draw_id'];assert np.array_equal(ids,draws.spatial_draw_id)
                area=daily['reference_area'].astype(float);functional=daily['functional_loss'].astype(float)
                area_rounding=np.spacing(daily['reference_area']).astype(float)/2
                functional_rounding=np.spacing(daily['functional_loss']).astype(float)/2
                s31,s65,s85=[daily[f'BBCH{stage}_index'] for stage in (31,65,85)]
                days=np.arange(area.shape[1])[None,:]
                elapsed=np.clip((days-s65[:,None])/np.maximum(s85-s65,1)[:,None],0,1)
                for setting,s in reg['settings'].items():
                    if s['kind']!='daily_had_transfer':continue
                    rows=seasons.loc[seasons.setting.eq(setting)&seasons.model.eq(model)&seasons.harvest_year.eq(year)].sort_values('spatial_draw_id')
                    decline=1-(1-s['reference_decline_end_factor'])*elapsed
                    ref_area=area*decline[:,:,None]
                    for window,start in [('GS31_85',s31),('GS65_85',s65)]:
                        mask=(days>=start[:,None])&(days<=s85[:,None])
                        reference=(ref_area*mask[:,:,None]).sum(axis=(1,2))
                        loss=(ref_area*functional*s['functional_conversion']*mask[:,:,None]).sum(axis=(1,2))
                        valid=rows.valid_complete_season.to_numpy(bool)
                        reference_error=abs(reference-rows[f'{window}_reference_had3'].to_numpy(float))
                        loss_error=abs(loss-rows[f'{window}_lost_had3'].to_numpy(float))
                        reference_bound=(area_rounding*decline[:,:,None]*mask[:,:,None]).sum(axis=(1,2))+1e-10
                        product_rounding=area*functional_rounding+functional*area_rounding+area_rounding*functional_rounding
                        loss_bound=(product_rounding*decline[:,:,None]*s['functional_conversion']*mask[:,:,None]).sum(axis=(1,2))+1e-10
                        assert np.all(reference_error[valid]<=reference_bound[valid])
                        assert np.all(loss_error[valid]<=loss_bound[valid])
                        daily_errors.extend(reference_error[valid]);daily_errors.extend(loss_error[valid])
                        daily_error_ratios.extend(reference_error[valid]/reference_bound[valid]);daily_error_ratios.extend(loss_error[valid]/loss_bound[valid])
                        daily_bounds.extend(reference_bound[valid]);daily_bounds.extend(loss_bound[valid])
                        daily_checks+=int(valid.sum())*2
                # Detection sensitivity has identical trajectories; verify actual first crossings from cached daily state.
                if year in (1991,2071):
                    for setting,cutoff in [('baseline',.001),('detection0p0001',.0001),('detection0p01',.01)]:
                        rows=seasons.loc[seasons.setting.eq(setting)&seasons.model.eq(model)&seasons.harvest_year.eq(year)].sort_values('spatial_draw_id')
                        dates=pd.DatetimeIndex(daily['dates'])
                        for event,array_name in [('infection','affected'),('symptom','damage')]:
                            crossing=(daily[array_name]>=cutoff)&daily['crop_mask'][:,:,None]
                            first=np.where(crossing.any(axis=1),crossing.argmax(axis=1),-1)
                            for leaf in range(3):
                                found=first[:,leaf]>=0
                                predicted=pd.to_datetime(np.where(found,dates.to_numpy()[np.maximum(first[:,leaf],0)],np.datetime64('NaT','ns')))
                                expected=pd.to_datetime(rows[f'F{leaf+1}_{event}_date']).to_numpy()
                                assert np.array_equal(pd.isna(predicted),pd.isna(expected))
                                assert np.array_equal(predicted.to_numpy()[found],expected[found])
                                event_checks+=len(rows)
    assert max(daily_error_ratios)<=1
    result=dict(status='verified',verified_utc=datetime.now(timezone.utc).isoformat(),source_and_output_sha256_checks=True,
        settings=14,GCMs=3,draw_season_rows=161280,paired_year_offset=80,all_spatial_draws=64,unique_cells=62,
        independent_statistics_metrics=len(metrics),paired_population_gcm_checks=pair_count_checks,
        maximum_independent_summary_difference=float(max(summary_errors)),maximum_independent_mcse_difference=float(max(mcse_errors)),
        daily_archives_checked=daily_archives,daily_HAD_transform_checks=daily_checks,maximum_daily_float32_reconstruction_difference=float(max(daily_errors)),
        daily_float32_precision_check='Per-row bound from half-ULP area and damage rounding; product error includes cross term; additional1e-10 floating arithmetic allowance.',
        maximum_daily_error_to_precision_bound_ratio=float(max(daily_error_ratios)),maximum_daily_precision_bound=float(max(daily_bounds)),
        first_crossing_date_checks=event_checks,first_crossing_sample_years=[1991,2071],shared_draw_covariance_preserved=True,
        original_frozen_raw_forcing_archives_verified=len(forcing_archives),verification_code_sha256=sha(Path(__file__)),
        registration_sha256=sha(HERE/'registration_before_robustness_results.json'),completion_receipt_sha256=sha(HERE/'completion_receipt.json'))
    with (HERE/'independent_verification.json').open('x') as f:json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
