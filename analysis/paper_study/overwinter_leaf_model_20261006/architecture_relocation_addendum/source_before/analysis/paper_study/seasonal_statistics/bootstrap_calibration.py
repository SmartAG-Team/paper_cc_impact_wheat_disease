#!/usr/bin/env python3
"""Calibration-only uncertainty with frozen latency and unchanged model code."""
from pathlib import Path
from datetime import datetime,timezone
from dataclasses import replace
import hashlib,json,os,sys,time
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from frozen_source import activate,resolve,drift,manifest

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/paper_study/seasonal_statistics'
ARCHIVE=ROOT/'analysis/paper_study/seasonal_calibration_v1'
SEED=20261005
N=100


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    HERE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    sys.dont_write_bytecode=True
    os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
    activate()
    from model.seasonal_septoria.field_data import prepare_fields
    from model.seasonal_septoria.calibrate import fit,predict
    from model.seasonal_septoria.core import Parameters,simulate_season
    model_paths={name:str(Path(sys.modules['model.seasonal_septoria.'+name].__file__).resolve().relative_to(ROOT))
                 for name in ['field_data','calibrate','core','host']}
    assert all(str(resolve('model/seasonal_septoria/'+name+'.py').relative_to(ROOT))==path for name,path in model_paths.items())
    files=[ARCHIVE/'basf_source_assessments_snapshot.csv',ARCHIVE/'target_membership.csv',ARCHIVE/'frozen_main_fit.json',
           ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv',ROOT/'data/paper_study/field_weather/daily_weather.parquet',
           ROOT/'process_model/parameters/calibration.json']+[ROOT/row['original_path'] for row in manifest()['rows']
                                                            if row['original_path'].startswith('model/seasonal_septoria/')]
    source_hashes={str(p.relative_to(ROOT)):sha(resolve(p)) for p in files}
    # The source is filtered to calibration seasons before the numerical model receives targets.
    source=pd.read_csv(files[0]);source=source.loc[source.season_year.isin([2017,2018])].copy()
    calendars=pd.read_csv(files[3]);weather=pd.read_parquet(files[4])
    data=prepare_fields(source,calendars,weather)
    assert data.metadata.season_year.isin([2017,2018]).all() and len(data.metadata)==28 and len(data.targets)==254
    frozen=json.loads(files[2].read_text());latent=float(frozen['parameters']['latent_days']);assert latent==30
    archived=pd.read_csv(files[1]);archived=archived.loc[archived.partition.eq('calibration')]
    identity=['sourcekeys','field_id','date','leaf_index']
    check=data.targets[identity+['value','weight']].copy();check.date=check.date.dt.strftime('%Y-%m-%d')
    joined=check.merge(archived[identity+['value','weight']],on=identity,validate='one_to_one',suffixes=('_new','_archive'))
    assert len(joined)==254 and np.allclose(joined.value_new,joined.value_archive) and np.allclose(joined.weight_new,joined.weight_archive)
    forecast,_=predict(data,Parameters(**frozen['parameters']))
    reference=pd.read_parquet(ARCHIVE/'basf_frozen_predictions.parquet',columns=['sourcekeys','field_id','partition','model','predicted_percent'])
    reference=reference.loc[reference.partition.eq('calibration')&reference.model.eq('seasonal_seir')]
    forecasts=data.targets[['sourcekeys','field_id']].copy();forecasts['reconstructed_prediction']=forecast
    fc=forecasts.merge(reference,on=['sourcekeys','field_id'],validate='one_to_one')
    reconstruction=float(abs(fc.reconstructed_prediction-fc.predicted_percent).max());assert reconstruction<1e-9
    clusters=sorted(data.targets.coordinate_year.unique());rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(clusters),size=(N,len(clusters)));np.save(OUT/'parameter_cluster_draw_indices.npy',draws)
    memberships=[];results=[]
    folder=OUT/'bootstrap_fits';folder.mkdir(exist_ok=True)
    for number,draw in enumerate(draws):
        counts=np.bincount(draw,minlength=len(clusters));multiplicity=dict(zip(clusters,counts))
        target=data.targets.copy();factor=target.coordinate_year.map(multiplicity).to_numpy(int)
        target['weight']=target.weight*factor
        indices=np.flatnonzero(factor>0)
        boot=replace(data,targets=target)
        for c,count in zip(clusters,counts):memberships.append(dict(bootstrap_id=number,coordinate_year=c,multiplicity=int(count),
            source_field_ids=';'.join(sorted(target.loc[target.coordinate_year.eq(c),'field_id'].unique())),complete_field_seasons_preserved=True))
        path=folder/f'fit_{number:03d}.json';start=time.monotonic()
        if path.exists():
            result=json.loads(path.read_text())
            assert result['draw_indices']==draw.tolist() and result['source_sha256']==source_hashes
        else:
            fitted=fit(boot,indices,latent_days=latent)
            result={'bootstrap_id':number,'draw_indices':draw.tolist(),'cluster_multiplicities':{c:int(n) for c,n in multiplicity.items()},
                'source_sha256':source_hashes,'fit':fitted,'elapsed_seconds':time.monotonic()-start,
                'calibration_only':True,'latent_days_fixed':latent,'validation_or_external_targets_used':False}
            path.write_text(json.dumps(result,indent=2)+'\n')
        fitted=result['fit'];winner=fitted['multistart_records'][fitted['winning_start']]
        params=fitted['parameters']
        results.append(dict(bootstrap_id=number,alpha=params['alpha'],beta=params['beta'],latent_days_fixed=latent,
            training_rmse_pp=fitted['training_rmse'],winning_success=winner['success'],winning_nfev=winner['evaluations'],
            weighted_sse=winner['weighted_sse'],unique_coordinate_years=int((counts>0).sum()),unique_field_seasons=fitted['field_count'],
            resampled_coordinate_year_draws=len(draw),resampled_field_seasons=int(sum(int(counts[i])*target.loc[target.coordinate_year.eq(c),'field_id'].nunique() for i,c in enumerate(clusters))),
            beta_at_lower_boundary=params['beta']<=1e-6,beta_at_upper_boundary=params['beta']>=10-1e-6,
            alpha_at_upper_boundary=params['alpha']>=.1-1e-6,
            elapsed_seconds=result['elapsed_seconds'],calibration_only=True))
        pd.DataFrame(results).to_csv(HERE/'calibration_parameter_bootstrap.csv',index=False)
        if number==0 or (number+1)%5==0:print(f'completed{number+1}/{N}; alpha={params["alpha"]:.6g} beta={params["beta"]:.6g}; elapsed={time.monotonic()-start:.2f}s',flush=True)
    pd.DataFrame(memberships).to_csv(OUT/'parameter_cluster_membership.csv',index=False)
    result_table=pd.DataFrame(results);intervals=[]
    for parameter in ['alpha','beta']:
        low,high=result_table[parameter].quantile([.025,.975]);intervals.append(dict(parameter=parameter,frozen_estimate=frozen['parameters'][parameter],
            bootstrap_median=result_table[parameter].median(),ci95_lower=low,ci95_upper=high,resamples=N,seed=SEED,
            latent_days_fixed=latent,lower_boundary_fraction=float(result_table.beta_at_lower_boundary.mean()) if parameter=='beta' else None,
            upper_boundary_fraction=float(result_table.alpha_at_upper_boundary.mean()) if parameter=='alpha' else float(result_table.beta_at_upper_boundary.mean()),
            selection_and_structural_uncertainty_included=False))
    pd.DataFrame(intervals).to_csv(HERE/'calibration_parameter_intervals.csv',index=False)
    # Training-only profile: beta is fixed, alpha is reoptimized under the original alpha bounds.
    profile=[];target=data.targets
    field,day,leaf=[target[c].to_numpy(int) for c in ['field_index','day_index','leaf_index']]
    observed=target.value.to_numpy(float);weight=target.weight.to_numpy(float)
    grid=[0.,.0001,.001,.01,.03,.1,.3,1.,3.,5.,10.]
    for beta in grid:
        records=[]
        def residual(x):
            parameters=Parameters(alpha=float(10**x[0]),beta=beta,latent_days=latent)
            t=simulate_season(data.temperature,data.humidity,data.rain,data.host_active,data.host_renewal,parameters)
            return np.sqrt(weight)*(t.damage[field,day,leaf]*100-observed)
        for initial in [-4.,-3.,-2.3]:
            s=least_squares(residual,[initial],bounds=([-8.],[-1.]),max_nfev=150,ftol=1e-8,xtol=1e-8,gtol=1e-8)
            records.append({'alpha':float(10**s.x[0]),'sse':float(np.square(s.fun).sum()),'success':bool(s.success),'nfev':int(s.nfev)})
        best=min(records,key=lambda q:q['sse']);profile.append(dict(beta_fixed=beta,alpha_optimized=best['alpha'],weighted_sse=best['sse'],
            training_rmse_pp=float(np.sqrt(best['sse']/weight.sum())),optimization_success=best['success'],all_starts=json.dumps(records),
            calibration_only=True,latent_days_fixed=latent))
        pd.DataFrame(profile).to_csv(HERE/'secondary_beta_training_profile.csv',index=False)
        print(f'profile beta={beta:g} alpha={best["alpha"]:.6g}',flush=True)
    profile_table=pd.DataFrame(profile);profile_table['delta_weighted_sse_from_grid_min']=profile_table.weighted_sse-profile_table.weighted_sse.min()
    profile_table.to_csv(HERE/'secondary_beta_training_profile.csv',index=False)
    assert all(sha(resolve(p))==h for p,h in source_hashes.items())
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':source_hashes,'source_archives_preserved':True,
        'frozen_numerical_source_snapshot_preserved':True,'live_numerical_code_drift':drift(),
        'numerical_code_import_root':str((HERE/'source_snapshot').relative_to(ROOT)),
        'actual_imported_model_modules':model_paths,
        'snapshot_rows':manifest()['rows'],
        'training_years':[2017,2018],'training_field_seasons':28,'training_coordinate_years':len(clusters),'training_targets':254,
        'validation_or_external_targets_used_for_fitting':False,'Corteva_outcomes_read':False,'selected_latent_days_held_fixed':latent,
        'parameter_resamples':N,'seed':SEED,'calibration_prediction_reconstruction_max_error_pp':reconstruction,
        'winning_optimizations_successful':int(result_table.winning_success.sum()),'beta_lower_boundary_count':int(result_table.beta_at_lower_boundary.sum()),
        'beta_upper_boundary_count':int(result_table.beta_at_upper_boundary.sum()),'numerical_boundary_tolerance':1e-6,
        'parameter_bounds':{'alpha':[1e-8,.1],'beta':[0,10]},'training_locations':int(data.metadata.site_id.nunique()),
        'cluster_resampling_representation':'integer multiplicity in positive source weights; each selected source field-season remains complete',
        'profile_likelihood_claimed':False,'profile_objective':'original weighted residual SSE; no variance model or likelihood-ratio cutoff',
        'uncertainty_scope':'conditional percentile bootstrap; fixed latent selection, crop calendar, phenology, weather and observation proxy'}
    (HERE/'parameter_uncertainty_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(intervals).to_string(index=False),flush=True)


if __name__=='__main__':main()
