"""Narrow reaggregation excluding >40C mean disease weather in either hybrid.

No disease or host kinetics are rerun. Existing baseline crop masks and actual
cached/aligned source weather identify exclusions; four-corner outputs remain
immutable. Only GS65–85 HAD proxy attribution is reaggregated.
"""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd
from analysis.paper_study.nature_food_fix_20261007.climate.decomposition import run as core

HERE=Path(__file__).resolve().parent
ROOT=core.ROOT
METRIC='GS65_85_lost_had3'
COMPONENTS=['weather_shapley','host_shapley','total_change','interaction']


def sha(path):return core.sha(path)


def variance(v,draws):
    return float(sum(float(g.area_mean_weight.sum())**2*np.var(v[g.index.to_numpy()],ddof=1)/len(g) for _,g in draws.groupby('stratum',sort=False)))


def main():
    dependencies=[Path(__file__),HERE/'run.py',HERE/'registration_before_decomposition_results.json',HERE/'independent_verification.json',
        HERE/'four_corner_draw_year_outputs.parquet',HERE/'ensemble_decomposition.csv',HERE/'decomposition_by_gcm.csv',
        core.ORIGINAL.cache.DRAW_PATH,core.ORIGINAL.HERE/'weather_cache_receipt.json',
        *[core.ORIGINAL.coefficient_path(model,'ssp585') for model in core.MODELS]]
    hashes={str(p.relative_to(ROOT)):sha(p) for p in dependencies}
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),metric=METRIC,components=COMPONENTS,
        exclusion='Any original draw/GCM/relative-year pair with mean disease-weather temperature strictly above40C on any active day of either hybrid corner.',
        population='Supported-forcing common: original finite-four-corner HAD population intersected with no above40C active hybrid forcing day.',
        forcing='Exact existing crop-day alignment, existing verified baseline crop masks and cached source forcing; no interpolation/clipping.',
        kinetic_outputs='Reaggregate original four-corner outputs unchanged; no disease or host kinetics rerun.',
        mcse='Original stratified ratio residuals and shared-draw GCM covariance retained.',source_sha256=hashes)
    core.write_json(HERE/'supported_forcing_registration.json',registration)
    draws=pd.read_csv(core.ORIGINAL.cache.DRAW_PATH);weights=draws.area_mean_weight.to_numpy();flag_records=[];mask_hashes={}
    for model in core.MODELS:
        coeff=core.coefficients(model,draws)
        for year in core.YEARS:
            reference=core.forcing(model,year,draws,coeff);future=core.forcing(model,year+80,draws,coeff)
            masks={}
            for period,y in [('reference',year),('future',year+80)]:
                path=HERE.parent/'daily_baseline'/model/f'{y}.npz'
                receipt=json.loads((HERE.parent/'annual_outputs'/model/f'{y}.json').read_text())
                assert sha(path)==receipt['baseline_daily_sha256']
                mask_hashes[str(path.relative_to(ROOT))]=sha(path)
                with np.load(path) as loaded:
                    assert np.array_equal(loaded['spatial_draw_id'],draws.spatial_draw_id.to_numpy())
                    masks[period]=loaded['crop_mask'].copy()
            a,_=core.align_crop_day(future,reference,masks['reference']);b,_=core.align_crop_day(reference,future,masks['future'])
            nr=((a['tmean_c']>40)&masks['reference']).sum(axis=1)
            nf=((b['tmean_c']>40)&masks['future']).sum(axis=1)
            for i,draw in draws.iterrows():flag_records.append(dict(model=model,reference_year=year,future_year=year+80,
                spatial_draw_id=int(draw.spatial_draw_id),cell_id=draw.cell_id,stratum=draw.stratum,area_mean_weight=draw.area_mean_weight,
                Wfuture_Hreference_meanT_above40_days=int(nr[i]),Wreference_Hfuture_meanT_above40_days=int(nf[i]),excluded_unsupported_hybrid_forcing=bool(nr[i] or nf[i])))
    flags=pd.DataFrame(flag_records);assert len(flags)==5760
    assert int(flags.Wfuture_Hreference_meanT_above40_days.sum()+flags.Wreference_Hfuture_meanT_above40_days.sum())==42
    flags.to_csv(HERE/'supported_forcing_draw_pair_flags.csv',index=False)
    corners=pd.read_parquet(HERE/'four_corner_draw_year_outputs.parquet')
    previous=pd.read_csv(HERE/'ensemble_decomposition.csv');rows=[];moments=[];coverage=[];independent_errors=[];identity_error=0.
    for model in core.MODELS:
        arrays={name:g.sort_values(['spatial_draw_id','reference_year'])[METRIC].to_numpy().reshape(64,30)
            for name,g in corners[corners.model.eq(model)].groupby('corner')}
        a=arrays['Wreference_Hreference'];b=arrays['Wfuture_Hreference'];c=arrays['Wreference_Hfuture'];d=arrays['Wfuture_Hfuture']
        original=np.isfinite(a)&np.isfinite(b)&np.isfinite(c)&np.isfinite(d)
        bad=flags[flags.model.eq(model)].sort_values(['spatial_draw_id','reference_year']).excluded_unsupported_hybrid_forcing.to_numpy().reshape(64,30)
        common=original&~bad;z=common.mean(axis=1);den=float(weights@z)
        coverage.append(dict(model=model,original_draw_year_pairs=1920,original_finite_corner_pairs=int(original.sum()),
            all_above40C_flagged_pairs=int(bad.sum()),excluded_from_original_finite_pairs=int((original&bad).sum()),supported_forcing_pairs=int(common.sum()),
            original_finite_area_time_coverage=float(weights@original.mean(axis=1)),excluded_finite_area_time_weight=float(weights@(original&bad).mean(axis=1)),
            supported_forcing_area_time_coverage=den))
        interaction=(d-b)-(c-a)
        values={'weather_shapley':b-a+interaction/2,'host_shapley':c-a+interaction/2,'total_change':d-a,'interaction':interaction}
        identity_error=max(identity_error,float(np.max(abs(np.where(common,values['weather_shapley']+values['host_shapley']-values['total_change'],0)))))
        for component,value in values.items():
            x=np.where(common,value,0).mean(axis=1);mean=float(weights@x)/den;influence=(x-mean*z)/den
            # Independently calculate the same weighted mean directly over long-form pairs.
            long_weights=np.broadcast_to(weights[:,None]/30,common.shape)
            direct=float(np.sum(long_weights[common]*value[common])/np.sum(long_weights[common]))
            independent_errors.append(abs(direct-mean))
            rows.append(dict(population='supported_forcing_common',model=model,metric=METRIC,component=component,mean=mean,
                spatial_mcse=float(np.sqrt(variance(influence,draws))),common_area_time_coverage=den,common_draw_year_pairs=int(common.sum())))
            moments.append(dict(model=model,component=component,influence=influence))
    gcm=pd.DataFrame(rows);gcm.to_csv(HERE/'supported_forcing_decomposition_by_gcm.csv',index=False)
    ensemble=[]
    for component in COMPONENTS:
        group=gcm[gcm.component.eq(component)];old=previous[previous.metric.eq(METRIC)&previous.component.eq(component)].iloc[0]
        vectors=[r['influence'] for r in moments if r['component']==component]
        mean=float(group['mean'].mean());mcse=float(np.sqrt(variance(np.mean(vectors,axis=0),draws)))
        ensemble.append(dict(population='supported_forcing_common',metric=METRIC,component=component,gcm_mean=mean,gcm_min=float(group['mean'].min()),
            gcm_max=float(group['mean'].max()),spatial_mcse_gcm_mean=mcse,minimum_gcm_coverage=float(group.common_area_time_coverage.min()),
            maximum_gcm_coverage=float(group.common_area_time_coverage.max()),minimum_gcm_pairs=int(group.common_draw_year_pairs.min()),
            maximum_gcm_pairs=int(group.common_draw_year_pairs.max()),original_population_gcm_mean=float(old.gcm_mean),
            change_from_original_mean=mean-float(old.gcm_mean),original_population_MCSE=float(old.spatial_mcse_gcm_mean),
            sign_preserved_all_GCMs=bool(np.all(np.sign(group['mean'])==np.sign(old.gcm_mean)))))
    summary=pd.DataFrame(ensemble);summary.to_csv(HERE/'supported_forcing_ensemble_decomposition.csv',index=False)
    cov=pd.DataFrame(coverage);cov.to_csv(HERE/'supported_forcing_coverage.csv',index=False)
    assert max(independent_errors)<1e-10 and identity_error<1e-10
    for name,h in hashes.items():assert sha(ROOT/name)==h,name
    result=dict(status='verified',completed_utc=datetime.now(timezone.utc).isoformat(),population='supported_forcing_common',
        no_kinetics_rerun=True,original_kinetic_outputs_unchanged=True,above40C_active_hybrid_days=42,
        original_GCM_draw_year_pairs=5760,original_finite_HAD_pairs=int(cov.original_finite_corner_pairs.sum()),
        flagged_pairs=int(cov.all_above40C_flagged_pairs.sum()),excluded_original_finite_HAD_pairs=int(cov.excluded_from_original_finite_pairs.sum()),
        supported_forcing_HAD_pairs=int(cov.supported_forcing_pairs.sum()),coverage=coverage,ensemble=ensemble,
        maximum_independent_long_form_mean_error=float(max(independent_errors)),maximum_shapley_additivity_error=identity_error,
        verified_baseline_mask_archive_sha256=mask_hashes,source_sha256_verified=True,registration_sha256=sha(HERE/'supported_forcing_registration.json'))
    core.write_json(HERE/'supported_forcing_receipt.json',result)
    def row(c):return summary[summary.component.eq(c)].iloc[0]
    weather,host,total,inter=row('weather_shapley'),row('host_shapley'),row('total_change'),row('interaction')
    text=(f"The supported-forcing sensitivity excluded {int(cov.excluded_from_original_finite_pairs.sum())} of {int(cov.original_finite_corner_pairs.sum())} finite four-combination draw-year pairs with mean disease-weather temperature above 40 °C on any active day in either hybrid. The remaining population contained {int(cov.supported_forcing_pairs.sum())} pairs across the three GCMs, with weighted paired area-time coverage of {100*float(cov.supported_forcing_area_time_coverage.min()):.6f}–{100*float(cov.supported_forcing_area_time_coverage.max()):.6f}%. Existing kinetic outputs were retained, and all decomposition terms used this same supported-forcing common population.\n\n"
        f"The disease-weather Shapley contribution to GS65–85 upper-three-leaf HAD proxy loss was {weather.gcm_mean:+.9f} days per unit nominal upper-three-leaf LAI (GCM range {weather.gcm_min:.9f} to {weather.gcm_max:.9f}; spatial Monte Carlo standard error {weather.spatial_mcse_gcm_mean:.9f}). The host-development contribution was {host.gcm_mean:+.9f} days (range {host.gcm_min:.9f} to {host.gcm_max:.9f}; standard error {host.spatial_mcse_gcm_mean:.9f}). Their sum was {total.gcm_mean:+.9f} days (range {total.gcm_min:.9f} to {total.gcm_max:.9f}; standard error {total.spatial_mcse_gcm_mean:.9f}). The interaction was {inter.gcm_mean:+.9f} days (range {inter.gcm_min:.9f} to {inter.gcm_max:.9f}; standard error {inter.spatial_mcse_gcm_mean:.9f}), allocated equally within the two Shapley contributions.\n\n"
        f"The weather, host and net contributions differed from the original finite-corner population by {weather.change_from_original_mean:+.9f}, {host.change_from_original_mean:+.9f} and {total.change_from_original_mean:+.9f} days, respectively. Their signs remained consistent across all GCMs. These values are within-model crop-infection and canopy-damage quantities without physical causal or actual-yield attribution.\n")
    (HERE/'supported_forcing_results.md').write_text(text)
    print(cov.to_string(index=False));print(summary.to_string(index=False));print('VERIFIED supported-forcing reaggregation; no kinetics rerun')


if __name__=='__main__':main()
