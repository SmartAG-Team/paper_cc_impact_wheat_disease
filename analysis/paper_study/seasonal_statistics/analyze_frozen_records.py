#!/usr/bin/env python3
"""Cluster bootstrap of archived frozen predictions, without model refitting."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/paper_study/seasonal_statistics'
SOURCE=ROOT/'analysis/paper_study/seasonal_calibration_v1'
SEED=20261005
B=10000
MODELS=['seasonal_seir','calibration_only_leaf_rank_mean']


def cluster_statistics(g):
    q=g.copy();q['error']=q.predicted_percent-q.value
    q['mse']=q.error**2;q['mae']=abs(q.error)
    field=q.groupby(['coordinate_year','field_id'])[['mse','mae','error']].mean()
    return field.groupby('coordinate_year').mean().rename(columns={'error':'bias'})


def main():
    HERE.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    watched=[SOURCE/'basf_frozen_predictions.parquet',SOURCE/'visible_onset_interval_predictions.csv',SOURCE/'field_season_membership.csv',SOURCE/'frozen_main_fit.json',SOURCE/'frozen_leaf_rank_baseline.json',SOURCE/'severity_metrics.csv']
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}
    predictions=pd.read_parquet(watched[0]);predictions['date']=pd.to_datetime(predictions.date)
    final=predictions.sort_values('date').groupby(['model','partition','field_id','endpoint_series']).tail(1).copy()
    final.to_csv(OUT/'final_leaf_predictions_snapshot.csv',index=False)
    summary=[];clusters=[];draws=[];draw_indices={};rng=np.random.default_rng(SEED)
    for partition in ['calibration','validation']:
        for subset in ['all_ordinal_leaves','upper_three']:
            q=final.loc[final.partition.eq(partition)&final.model.isin(MODELS)]
            if subset=='upper_three':q=q.loc[q.leaf_index.lt(3)]
            bymodel={m:cluster_statistics(q[q.model.eq(m)]) for m in MODELS}
            assert bymodel[MODELS[0]].index.equals(bymodel[MODELS[1]].index)
            index=bymodel[MODELS[0]].index;n=len(index);sample=rng.integers(0,n,size=(B,n));draw_indices[f'{partition}_{subset}']=sample
            bootstrap={};point={}
            for model,s in bymodel.items():
                point[model]={'rmse':float(np.sqrt(s.mse.mean())),'mae':float(s.mae.mean()),'bias':float(s.bias.mean())}
                bootstrap[model]={'rmse':np.sqrt(s.mse.to_numpy()[sample].mean(axis=1)),
                                  'mae':s.mae.to_numpy()[sample].mean(axis=1),'bias':s.bias.to_numpy()[sample].mean(axis=1)}
                v=s.reset_index();v['model']=model;v['partition']=partition;v['subset']=subset;clusters.append(v)
            for metric in ['rmse','mae','bias']:
                main_values=bootstrap[MODELS[0]][metric];base_values=bootstrap[MODELS[1]][metric];delta=main_values-base_values
                for name,estimate,values in [(MODELS[0],point[MODELS[0]][metric],main_values),(MODELS[1],point[MODELS[1]][metric],base_values),
                                            ('paired_model_minus_baseline',point[MODELS[0]][metric]-point[MODELS[1]][metric],delta)]:
                    low,high=np.quantile(values,[.025,.975]);summary.append(dict(partition=partition,subset=subset,endpoint='final_numeric_per_source_leaf',
                        metric=metric,comparison=name,estimate_pp=estimate,ci95_lower_pp=low,ci95_upper_pp=high,coordinate_year_clusters=n,
                        field_seasons=q.field_id.nunique(),source_leaf_series=q.endpoint_series.nunique(),resamples=B,seed=SEED))
                draws.extend(dict(partition=partition,subset=subset,bootstrap_id=i,metric=metric,model_pp=main_values[i],baseline_pp=base_values[i],paired_difference_pp=delta[i]) for i in range(B))
    np.savez_compressed(OUT/'paired_cluster_draw_indices.npz',**draw_indices)
    pd.concat(clusters,ignore_index=True).to_csv(OUT/'paired_final_cluster_scores.csv',index=False)
    pd.DataFrame(draws).to_parquet(OUT/'paired_final_bootstrap_draws.parquet',index=False)
    pd.DataFrame(summary).to_csv(HERE/'paired_final_severity_intervals.csv',index=False)
    # Onset compatibility is independently reconstructed with an exclusive negative bound.
    onset=pd.read_csv(watched[1]);metadata=pd.read_csv(watched[2]);metadata['coordinate_year']=metadata.site_id.astype(str)+'|'+metadata.season_year.astype(str)
    onset=onset.merge(metadata[['field_id','coordinate_year']],on='field_id',how='left',validate='many_to_one')
    targets=predictions.loc[predictions.model.eq('seasonal_seir'),['field_id','endpoint_series','leaf_index']].drop_duplicates()
    onset=onset.merge(targets.rename(columns={'endpoint_series':'series'}),on=['field_id','series'],how='left',validate='many_to_one')
    new=[]
    for row in onset.itertuples():
        lower=pd.to_datetime(row.last_negative);upper=pd.to_datetime(row.first_positive);pred=pd.to_datetime(row.predicted_visible_date)
        if pd.isna(pred):compatible=pd.isna(upper);distance=0 if compatible else np.nan
        else:
            distance=(pred-(lower+pd.Timedelta(days=1))).days if pd.notna(lower) and pred<=lower else (pred-upper).days if pd.notna(upper) and pred>upper else 0
            compatible=distance==0
        assert bool(row.compatible)==compatible
        assert np.isclose(row.delta_days,distance,equal_nan=True)
        new.append((compatible,distance,pd.isna(pred)))
    onset[['reconstructed_compatible','bound_distance_days','reconstructed_missing_prediction']]=pd.DataFrame(new,index=onset.index)
    onset.to_csv(OUT/'onset_censoring_audit.csv',index=False)
    onset_summary=[]
    for subset in ['all_ordinal_leaves','upper_three']:
        chosen=onset if subset=='all_ordinal_leaves' else onset[onset.leaf_index.lt(3)]
        for key,g in chosen.groupby(['model','partition','cutoff_percent','censoring']):
            rates=g.groupby(['coordinate_year','field_id']).reconstructed_compatible.mean().groupby('coordinate_year').mean()
            sample=rng.integers(0,len(rates),size=(B,len(rates)));values=rates.to_numpy()[sample].mean(axis=1)
            bounds=np.quantile(values,[.025,.975]);model,partition,cutoff,censoring=key
            onset_summary.append(dict(model=model,partition=partition,cutoff_percent=cutoff,censoring=censoring,subset=subset,
                source_leaf_series=len(g),field_seasons=g.field_id.nunique(),coordinate_year_clusters=len(rates),
                unweighted_compatible_count=int(g.reconstructed_compatible.sum()),unweighted_compatible_fraction=g.reconstructed_compatible.mean(),
                equal_cluster_compatible_fraction=rates.mean(),ci95_lower=bounds[0],ci95_upper=bounds[1],
                missing_predicted_events=int(g.reconstructed_missing_prediction.sum()),finite_bound_distances=int(g.bound_distance_days.notna().sum()),
                mean_absolute_finite_bound_distance_days=g.bound_distance_days.abs().mean(),
                exact_onset_error_identifiable=False,resamples=B,seed=SEED))
    pd.DataFrame(onset_summary).to_csv(HERE/'onset_compatibility_by_censoring.csv',index=False)
    archived=pd.read_csv(SOURCE/'severity_metrics.csv');diff=[]
    for row in summary:
        if row['comparison'] not in MODELS:continue
        match=archived[(archived.model==row['comparison'])&(archived.partition==row['partition'])&(archived.subset==row['subset'])&archived.endpoint.eq('final_numeric_assessment')]
        diff.append(abs(float(match.iloc[0][row['metric']])-row['estimate_pp']))
    assert max(diff)<1e-10
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in hashes.items())
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'source_sha256':hashes,'frozen_archives_preserved':True,'resamples':B,'seed':SEED,
        'final_metric_max_difference_vs_archive_pp':max(diff),'independent_onset_row_reconstructions':len(onset),
        'cluster_unit':'coordinate-year; complete fields and leaf series nested equally','paired_predictions_held_fixed':True,
        'Corteva_outcomes_read':False,'model_refitted':False,'intervals':'percentile95%; frozen-model sampling uncertainty, not parameter uncertainty'}
    (HERE/'frozen_prediction_statistics_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(summary).loc[lambda x:x.comparison.eq('paired_model_minus_baseline')].to_string(index=False))


if __name__=='__main__':main()
