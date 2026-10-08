#!/usr/bin/env python3
"""Independently reconstruct statistics and fitted residuals without optimization."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, sys
import numpy as np
import pandas as pd
from frozen_source import activate,resolve,drift,manifest

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/paper_study/seasonal_statistics'
ARCHIVE=ROOT/'analysis/paper_study/seasonal_calibration_v1'
SEED=20261005
checks=[]


def check(name, condition, detail=None):
    checks.append({'check':name,'passed':bool(condition),'detail':detail})
    if not condition:
        raise AssertionError(name)


def close(a,b,tol=1e-9):
    return bool(np.allclose(a,b,rtol=0,atol=tol,equal_nan=True))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parameter_receipt=json.loads((HERE/'parameter_uncertainty_receipt.json').read_text())
    for name in ['frozen_prediction_statistics_receipt.json','parameter_uncertainty_receipt.json']:
        receipt=json.loads((HERE/name).read_text())
        for path,expected in receipt['source_sha256'].items():
            check(f'{name}: retained SHA256 {path}',sha(resolve(path))==expected)
    raw=pd.read_parquet(ARCHIVE/'basf_frozen_predictions.parquet')
    raw.date=pd.to_datetime(raw.date)
    metadata=pd.read_csv(ARCHIVE/'field_season_membership.csv')
    check('forward harvest-year membership',metadata.loc[metadata.partition.eq('calibration'),'season_year'].isin([2017,2018]).all()
          and metadata.loc[metadata.partition.eq('validation'),'season_year'].eq(2019).all())
    check('all frozen numerical predictions bounded',np.isfinite(raw.predicted_percent).all() and raw.predicted_percent.between(0,100).all())
    check('source percentages bounded',np.isfinite(raw.value).all() and raw.value.between(0,100).all())
    membership=raw[['field_id','season_year','site_id','partition']].drop_duplicates()
    check('predictions retain field-season partition',len(membership)==73 and len(membership.merge(metadata,on=['field_id','season_year','site_id','partition']))==73)
    main_rows=raw[raw.model.eq('seasonal_seir')]
    check('one target per source field/date/rank',not main_rows.duplicated(['field_id','date','leaf_index']).any())
    check('all fields have an observed-window positive',main_rows.groupby('field_id').value.max().gt(0).all())
    draws=np.load(OUT/'paired_cluster_draw_indices.npz')
    stored=pd.read_parquet(OUT/'paired_final_bootstrap_draws.parquet')
    intervals=pd.read_csv(HERE/'paired_final_severity_intervals.csv')
    rng=np.random.default_rng(SEED)
    max_metric_difference=0.
    for part in ['calibration','validation']:
        for subset in ['all_ordinal_leaves','upper_three']:
            rows=raw[raw.partition.eq(part)&raw.model.isin(['seasonal_seir','calibration_only_leaf_rank_mean'])]
            if subset=='upper_three':rows=rows[rows.leaf_index.lt(3)]
            # Independent endpoint selection uses date maxima of the numerical leaf index.
            finals=rows.loc[rows.groupby(['model','field_id','leaf_index']).date.idxmax()]
            numbers={}
            for model,g in finals.groupby('model'):
                cluster_values=[]
                for coord,cluster in g.groupby('coordinate_year',sort=True):
                    field_values=[]
                    for _,field in cluster.groupby('field_id'):
                        error=(field.predicted_percent-field.value).to_numpy()
                        field_values.append([np.mean(error**2),np.mean(abs(error)),np.mean(error)])
                    cluster_values.append(np.mean(field_values,axis=0))
                numbers[model]=np.asarray(cluster_values)
            n=len(numbers['seasonal_seir']);indices=draws[f'{part}_{subset}']
            check(f'{part}/{subset}: exact seeded paired cluster draws',np.array_equal(indices,rng.integers(0,n,size=(10000,n))))
            scores={};points={}
            for model,matrix in numbers.items():
                points[model]=[np.sqrt(matrix[:,0].mean()),matrix[:,1].mean(),matrix[:,2].mean()]
                means=matrix[indices].mean(axis=1)
                scores[model]=np.column_stack([np.sqrt(means[:,0]),means[:,1],means[:,2]])
            scores['paired_model_minus_baseline']=scores['seasonal_seir']-scores['calibration_only_leaf_rank_mean']
            points['paired_model_minus_baseline']=np.array(points['seasonal_seir'])-points['calibration_only_leaf_rank_mean']
            for k,metric in enumerate(['rmse','mae','bias']):
                rows=stored[(stored.partition==part)&(stored.subset==subset)&(stored.metric==metric)].sort_values('bootstrap_id')
                check(f'{part}/{subset}/{metric}: all paired bootstrap values',close(rows.model_pp,scores['seasonal_seir'][:,k])
                      and close(rows.baseline_pp,scores['calibration_only_leaf_rank_mean'][:,k])
                      and close(rows.paired_difference_pp,scores['paired_model_minus_baseline'][:,k]))
                for model in scores:
                    q=intervals[(intervals.partition==part)&(intervals.subset==subset)&(intervals.metric==metric)&(intervals.comparison==model)].iloc[0]
                    expected=[points[model][k],*np.quantile(scores[model][:,k],[.025,.975])]
                    difference=float(np.max(abs(np.array([q.estimate_pp,q.ci95_lower_pp,q.ci95_upper_pp])-expected)))
                    max_metric_difference=max(max_metric_difference,difference)
                    check(f'{part}/{subset}/{metric}/{model}: point and percentile interval',difference<1e-9)
    onset=pd.read_csv(OUT/'onset_censoring_audit.csv')
    reconstructed=[]
    byseries={key:g.sort_values('date') for key,g in main_rows.groupby(['field_id','endpoint_series'])}
    for row in onset.itertuples():
        g=byseries[row.field_id,row.series]
        positive=g.loc[g.value.ge(row.cutoff_percent),'date']
        upper=positive.min() if len(positive) else pd.NaT
        negative=g.loc[g.value.lt(row.cutoff_percent),'date']
        lower=negative[negative.lt(upper)].max() if pd.notna(upper) else negative.max()
        censor='right' if pd.isna(upper) else 'left' if pd.isna(lower) else 'interval'
        check(f'onset bounds {row.Index}',(pd.isna(lower) and pd.isna(row.last_negative) or lower==pd.to_datetime(row.last_negative))
              and (pd.isna(upper) and pd.isna(row.first_positive) or upper==pd.to_datetime(row.first_positive)) and censor==row.censoring)
        pred=pd.to_datetime(row.predicted_visible_date)
        compatible=(pd.isna(upper) if pd.isna(pred) else
                    (pd.isna(lower) or pred>lower) and (pd.isna(upper) or pred<=upper))
        signed=np.nan if pd.isna(pred) and pd.notna(upper) else 0
        if pd.notna(pred) and not compatible:
            signed=(pred-lower).days-1 if pd.notna(lower) and pred<=lower else (pred-upper).days
        check(f'onset compatibility {row.Index}',bool(row.compatible)==compatible and close(row.bound_distance_days,signed))
        reconstructed.append(compatible)
    onset['check_compatible']=reconstructed
    onset_summary=pd.read_csv(HERE/'onset_compatibility_by_censoring.csv')
    for subset in ['all_ordinal_leaves','upper_three']:
        chosen=onset if subset=='all_ordinal_leaves' else onset[onset.leaf_index.lt(3)]
        for key,g in chosen.groupby(['model','partition','cutoff_percent','censoring']):
            cluster=g.groupby(['coordinate_year','field_id']).check_compatible.mean().groupby('coordinate_year').mean()
            boot=cluster.to_numpy()[rng.integers(0,len(cluster),size=(10000,len(cluster)))].mean(axis=1)
            model,part,cutoff,censor=key
            q=onset_summary[(onset_summary.model==model)&(onset_summary.partition==part)&(onset_summary.cutoff_percent==cutoff)
                            &(onset_summary.censoring==censor)&(onset_summary.subset==subset)].iloc[0]
            check(f'onset weighted compatibility interval {subset}/{key}',close([q.equal_cluster_compatible_fraction,q.ci95_lower,q.ci95_upper],
                                                                              [cluster.mean(),*np.quantile(boot,[.025,.975])]))
    # Verification of every calibration bootstrap residual requires simulations, but no fits.
    sys.dont_write_bytecode=True
    os.environ['NUMBA_CACHE_DIR']=str(OUT/'numba_cache')
    activate()
    from model.seasonal_septoria.field_data import prepare_fields
    from model.seasonal_septoria.calibrate import predict
    from model.seasonal_septoria.core import Parameters
    for name in ['field_data','calibrate','core','host']:
        actual=Path(sys.modules['model.seasonal_septoria.'+name].__file__).resolve()
        check(f'numerical imports use frozen source: {name}',actual==resolve('model/seasonal_septoria/'+name+'.py'))
    source=pd.read_csv(ARCHIVE/'basf_source_assessments_snapshot.csv')
    source=source[source.season_year.isin([2017,2018])]
    data=prepare_fields(source,pd.read_csv(ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv'),
                        pd.read_parquet(ROOT/'data/paper_study/field_weather/daily_weather.parquet'))
    check('numerical verification receives only calibration seasons',len(data.metadata)==28 and len(data.targets)==254
          and data.metadata.season_year.isin([2017,2018]).all())
    clusters=sorted(data.targets.coordinate_year.unique());draws=np.load(OUT/'parameter_cluster_draw_indices.npy')
    expected=np.random.default_rng(SEED).integers(0,len(clusters),size=(100,len(clusters)))
    check('exact seeded parameter bootstrap draws',np.array_equal(draws,expected))
    fits=pd.read_csv(HERE/'calibration_parameter_bootstrap.csv')
    recorded=pd.read_csv(OUT/'parameter_cluster_membership.csv')
    check('all 100 resamples represented',len(fits)==100 and set(fits.bootstrap_id)==set(range(100)) and len(recorded)==2600)
    max_sse_difference=0.
    for i,draw in enumerate(draws):
        record=json.loads((OUT/f'bootstrap_fits/fit_{i:03d}.json').read_text());fit=record['fit']
        check(f'parameter resample {i}: hashes and calibration-only provenance',record['source_sha256']==parameter_receipt['source_sha256']
              and record['calibration_only'] and not record['validation_or_external_targets_used'] and record['latent_days_fixed']==30)
        count=np.bincount(draw,minlength=len(clusters));mapping=dict(zip(clusters,count))
        factors=data.targets.coordinate_year.map(mapping).to_numpy(int)
        selected=np.flatnonzero(factors>0)
        selected_fields=np.unique(data.targets.iloc[selected].field_index)
        complete=np.flatnonzero(data.targets.field_index.isin(selected_fields))
        check(f'parameter resample {i}: complete field membership',np.array_equal(selected,complete)
              and fit['target_indices']==selected.tolist() and fit['field_indices']==selected_fields.tolist())
        check(f'parameter resample {i}: recorded cluster multiplicities',record['draw_indices']==draw.tolist()
              and record['cluster_multiplicities']=={c:int(n) for c,n in mapping.items()}
              and np.array_equal(recorded.loc[recorded.bootstrap_id.eq(i)].sort_values('coordinate_year').multiplicity,count))
        forecast,_=predict(data,Parameters(**fit['parameters']))
        weights=data.targets.weight.to_numpy()*factors
        sse=float(np.sum(weights*(forecast-data.targets.value.to_numpy())**2))
        best=fit['multistart_records'][fit['winning_start']]
        max_sse_difference=max(max_sse_difference,abs(sse-best['weighted_sse']))
        check(f'parameter resample {i}: independently reconstructed weighted SSE',close(sse,best['weighted_sse'],1e-7)
              and close(np.sqrt(sse/weights.sum()),fit['training_rmse'],1e-9))
        check(f'parameter resample {i}: multistart winner and bounds',best['weighted_sse']==min(x['weighted_sse'] for x in fit['multistart_records'])
              and 1e-8<=fit['parameters']['alpha']<=.1 and 0<=fit['parameters']['beta']<=10 and fit['parameters']['latent_days']==30)
        q=fits[fits.bootstrap_id.eq(i)].iloc[0]
        check(f'parameter resample {i}: summary matches checkpoint',close([q.alpha,q.beta,q.training_rmse_pp,q.weighted_sse],
              [fit['parameters']['alpha'],fit['parameters']['beta'],fit['training_rmse'],sse],1e-7))
    param_ci=pd.read_csv(HERE/'calibration_parameter_intervals.csv')
    for name in ['alpha','beta']:
        row=param_ci[param_ci.parameter.eq(name)].iloc[0]
        check(f'{name}: percentile parameter interval',close([row.bootstrap_median,row.ci95_lower,row.ci95_upper],
              [fits[name].median(),*fits[name].quantile([.025,.975])]))
    profile=pd.read_csv(HERE/'secondary_beta_training_profile.csv')
    check('profile uses only fixed-latency training records',profile.calibration_only.all() and profile.latent_days_fixed.eq(30).all())
    for row in profile.itertuples():
        forecast,_=predict(data,Parameters(alpha=row.alpha_optimized,beta=row.beta_fixed,latent_days=30))
        sse=float(np.sum(data.targets.weight*(forecast-data.targets.value)**2))
        check(f'profile beta={row.beta_fixed}: independently reconstructed objective',close(sse,row.weighted_sse,1e-7)
              and close(np.sqrt(sse/data.targets.weight.sum()),row.training_rmse_pp))
    primary=json.loads((ARCHIVE/'frozen_primary_only_fit.json').read_text())
    check('beta-zero profile reproduces frozen primary-only training score',close(profile.loc[profile.beta_fixed.eq(0),'training_rmse_pp'],primary['training_rmse'],1e-7))
    train=metadata[metadata.partition.eq('calibration')];valid=metadata[metadata.partition.eq('validation')]
    overlap=sorted(set(train.site_id)&set(valid.site_id))
    pd.DataFrame([{'partition':part,'field_seasons':len(g),'coordinate_year_clusters':len(g[['site_id','season_year']].drop_duplicates()),
                   'coordinate_locations':g.site_id.nunique(),'shared_train_validation_locations':len(overlap)}
                  for part,g in metadata.groupby('partition')]).to_csv(HERE/'physical_unit_inventory.csv',index=False)
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed','check_count':len(checks),'failed_checks':0,
             'checks':checks,'independent_final_metric_max_difference_pp':max_metric_difference,'bootstrap_weighted_sse_max_difference':max_sse_difference,
             'no_parameter_optimization_in_verifier':True,'Corteva_outcomes_read':False,'geographically_disjoint_forward_validation':not overlap,
             'shared_calibration_validation_location_ids':overlap,'training_only_parameter_resamples':100,'prediction_resamples':10000,
             'live_numerical_code_drift':drift(),'frozen_numerical_snapshot_hashes':manifest()['rows'],
             'executable_sha256':{p.name:sha(p) for p in HERE.glob('*.py')}}
    (HERE/'independent_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['checks','executable_sha256']},indent=2))


if __name__=='__main__':
    main()
