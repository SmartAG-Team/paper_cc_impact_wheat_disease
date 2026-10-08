#!/usr/bin/env python3
"""Independent verification of frozen external statistics; no numerical model imports."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OUT=ROOT/'data/paper_study/external_statistics'
ARCHIVE=ROOT/'analysis/paper_study/corteva_external_seasonal_v1'
CAL=ROOT/'analysis/paper_study/seasonal_calibration_v1'
checks=[]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def check(name,condition,details=None):
    checks.append(dict(check=name,passed=bool(condition),details=details))
    if not condition:raise AssertionError(name)


def main():
    receipt=json.loads((HERE/'receipt.json').read_text())
    for source,expected in receipt['source_sha256'].items():check('immutable source '+source,sha(ROOT/source)==expected)
    source_receipt=json.loads((ARCHIVE/'receipt.json').read_text())
    for name,expected in source_receipt['frozen_parameters'].items():check('frozen fit '+name,sha(CAL/name)==expected)
    p=pd.read_parquet(ARCHIVE/'frozen_external_predictions.parquet');p.date=pd.to_datetime(p.date)
    check('frozen predictions finite/bounded',np.isfinite(p.predicted_percent).all() and p.predicted_percent.between(0,100).all())
    check('observed percentages finite/bounded',np.isfinite(p.value).all() and p.value.between(0,100).all())
    base=json.loads((CAL/'frozen_leaf_rank_baseline.json').read_text())
    b=p[p.model.eq('calibration_only_leaf_rank_mean')]
    check('baseline uses frozen calibration ranks',np.allclose(b.predicted_percent,b.leaf_index.astype(str).map(base),rtol=0,atol=1e-12))
    raw_basf=pd.read_parquet(ROOT/'data/paper_study/observations/observations.parquet',columns=['dataset_id','site_id'])
    basf_sites=set(raw_basf.loc[raw_basf.dataset_id.eq('basf-wheat-diseases'),'site_id'])
    external=p[p.model.eq('seasonal_seir')]
    check('one prediction per field/date/leaf',not external.duplicated(['field_id','date','leaf_index']).any())
    check('strict subset geographically disjoint from all BASF source points',not (set(external.loc[external.strict_location_disjoint,'site_id'])&basf_sites))
    check('all external field windows contain an observed positive',external.groupby('field_id').value.max().gt(0).all())
    drawn=np.load(OUT/'coordinate_year_draw_indices.npz');membership=pd.read_csv(OUT/'coordinate_year_membership.csv')
    rng=np.random.default_rng(receipt['seed']);ids={}
    for geography in ['full_source_transfer','strict_location_disjoint']:
        ids[geography]=membership[membership.geography.eq(geography)].sort_values('cluster_index').coordinate_year.tolist()
        n=len(ids[geography]);check('exact seeded draws '+geography,np.array_equal(drawn[geography],rng.integers(0,n,size=(10000,n))))
        check('cluster count '+geography,n==(106 if geography=='full_source_transfer' else 104))
    points=pd.read_csv(HERE/'severity_point_estimates.csv');intervals=pd.read_csv(HERE/'severity_bootstrap_intervals.csv')
    stored=pd.read_parquet(OUT/'severity_bootstrap_draws.parquet')
    group_columns=['geography','leaf_scope','sensitivity','endpoint','weighting']
    stored_groups={key:g for key,g in stored.groupby(group_columns+['metric'],sort=False)}
    max_difference=0.;original_metric_difference=0.
    archive=pd.read_csv(ARCHIVE/'severity_metrics.csv')
    for key,group in points.groupby(group_columns,sort=False):
        geography,leaves,sensitivity,endpoint,weighting=key
        q=p[p.model.isin(['seasonal_seir','calibration_only_leaf_rank_mean'])]
        if geography=='strict_location_disjoint':q=q[q.strict_location_disjoint]
        if leaves=='upper_three':q=q[q.leaf_index<3]
        if sensitivity=='exclude_2015_58_oct30':q=q[~((q.source_unit=='2015-58')&(q.date==pd.Timestamp('2015-10-30')))]
        if endpoint=='final_numeric_assessment':q=q.loc[q.groupby(['model','field_id','leaf_index']).date.idxmax()]
        matrices={};boot_values={};estimates={}
        for model,g in q.groupby('model'):
            matrix=[]
            for coord in ids[geography]:
                cg=g[g.coordinate_year.eq(coord)];totals=np.zeros(4)
                for _,field in cg.groupby('field_id'):
                    leaf_count=field.leaf_index.nunique()
                    for _,leaf in field.groupby('leaf_index'):
                        error=(leaf.predicted_percent-leaf.value).to_numpy()
                        if weighting=='archived_selected_weights' and endpoint=='all_assessments':weight=leaf.weight.to_numpy()
                        else:
                            weight=np.full(len(leaf),1/leaf_count/len(leaf))
                            if weighting!='literal_equal_field':weight=weight/cg.field_id.nunique()
                        totals+=np.asarray([np.sum(weight*error**2),np.sum(weight*abs(error)),np.sum(weight*error),np.sum(weight)])
                matrix.append(totals)
            matrix=np.asarray(matrix);total=matrix.sum(axis=0)
            estimate=np.asarray([np.sqrt(total[0]/total[3]),total[1]/total[3],total[2]/total[3]])
            record=group[group.model.eq(model)].iloc[0]
            check('independent hierarchical severity '+str(key)+'/'+model,np.allclose(estimate,record[['rmse_pp','mae_pp','bias_pp']].to_numpy(float),rtol=0,atol=1e-9))
            if sensitivity=='all_dates' and weighting=='archived_selected_weights':
                original=archive[(archive.model==model)&(archive.geography==geography)&(archive.leaf_scope==leaves)&(archive.endpoint==endpoint)].iloc[0]
                difference=float(np.max(abs(estimate-original[['rmse','mae','bias']].to_numpy(float))))
                original_metric_difference=max(original_metric_difference,difference)
                check('original archived scores remain unchanged '+str(key)+'/'+model,difference<1e-9)
            sums=matrix[drawn[geography]].sum(axis=1)
            boot_values[model]=np.column_stack([np.sqrt(sums[:,0]/sums[:,3]),sums[:,1]/sums[:,3],sums[:,2]/sums[:,3]])
            estimates[model]=estimate;matrices[model]=matrix
            if weighting=='literal_equal_coordinate_year':check('equal selected CY mass '+str(key)+'/'+model,np.allclose(matrix[:,3],1,rtol=0,atol=1e-12))
            if weighting=='literal_equal_field':check('equal selected field mass '+str(key)+'/'+model,np.isclose(matrix[:,3].sum(),g.field_id.nunique()))
        boot_values['paired_model_minus_baseline']=boot_values['seasonal_seir']-boot_values['calibration_only_leaf_rank_mean']
        estimates['paired_model_minus_baseline']=estimates['seasonal_seir']-estimates['calibration_only_leaf_rank_mean']
        for i,metric in enumerate(['rmse','mae','bias']):
            values=stored_groups[(*key,metric)].sort_values('bootstrap_id')
            for model,column in [('seasonal_seir','model_pp'),('calibration_only_leaf_rank_mean','baseline_pp'),('paired_model_minus_baseline','paired_difference_pp')]:
                error=float(np.max(abs(values[column].to_numpy()-boot_values[model][:,i])));max_difference=max(max_difference,error)
                check('all severity bootstrap draws '+str(key)+'/'+metric+'/'+model,error<1e-9)
                row=intervals
                for c,v in zip(group_columns,key):row=row[row[c]==v]
                row=row[(row.metric==metric)&(row.comparison==model)].iloc[0]
                check('severity percentile interval '+str(key)+'/'+metric+'/'+model,
                      np.allclose([row.estimate_pp,row.ci95_lower_pp,row.ci95_upper_pp],
                                  [estimates[model][i],*np.quantile(boot_values[model][:,i],[.025,.975])],rtol=0,atol=1e-9))
    onset=pd.read_csv(OUT/'onset_reconstructed_bounds.csv');onset_summaries=pd.read_csv(HERE/'onset_censoring_statistics.csv')
    for key,g in onset.groupby(['geography','sensitivity','field_id','series'],sort=False):
        geography,sensitivity,field,series=key;observations=external[(external.field_id==field)&(external.endpoint_series==series)]
        if sensitivity=='exclude_2015_58_oct30':observations=observations[~((observations.source_unit=='2015-58')&(observations.date==pd.Timestamp('2015-10-30')))]
        for r in g.itertuples():
            upper=observations.loc[observations.value.ge(r.cutoff_percent),'date'].min()
            negative=observations.loc[observations.value.lt(r.cutoff_percent),'date']
            lower=negative[negative.lt(upper)].max() if pd.notna(upper) else negative.max()
            pred=pd.to_datetime(r.predicted_visible_date)
            compatible=pd.isna(upper) if pd.isna(pred) else (pd.isna(lower) or pred>lower) and (pd.isna(upper) or pred<=upper)
            distance=np.nan if pd.isna(pred) and pd.notna(upper) else 0
            if pd.notna(pred) and not compatible:distance=(pred-lower).days-1 if pd.notna(lower) and pred<=lower else (pred-upper).days
            expected_censor='right' if pd.isna(upper) else 'left' if pd.isna(lower) else 'interval'
            check('independent onset bounds/compatibility '+str(r.Index),bool(r.compatible)==compatible and r.censoring==expected_censor
                  and np.isclose(r.bound_distance_days,distance,equal_nan=True))
    onset_stored=pd.read_parquet(OUT/'onset_bootstrap_draws.parquet')
    onset_groups={key:g for key,g in onset_stored.groupby(['geography','leaf_scope','sensitivity','model','cutoff_percent','censoring'],sort=False)}
    for r in onset_summaries.itertuples():
        key=(r.geography,r.leaf_scope,r.sensitivity,r.model,r.cutoff_percent,r.censoring)
        g=onset[(onset.geography==r.geography)&(onset.sensitivity==r.sensitivity)&(onset.model==r.model)&(onset.cutoff_percent==r.cutoff_percent)&(onset.censoring==r.censoring)]
        if r.leaf_scope=='upper_three':g=g[g.leaf_index<3]
        rates=g.groupby(['coordinate_year','field_id']).compatible.mean().groupby('coordinate_year').mean()
        member=np.asarray([coord in rates.index for coord in ids[r.geography]],int)
        numerator=rates.reindex(ids[r.geography],fill_value=0).to_numpy()[drawn[r.geography]].sum(axis=1)
        denominator=member[drawn[r.geography]].sum(axis=1)
        boot=np.divide(numerator,denominator,out=np.full(10000,np.nan),where=denominator>0)
        record=onset_groups[key].sort_values('bootstrap_id')
        check('onset cohort bootstrap draws '+str(key),np.allclose(record.compatible_fraction,boot,equal_nan=True,rtol=0,atol=1e-12))
        check('onset bootstrap denominators '+str(key),np.array_equal(record.represented_stratum_clusters,denominator))
        check('onset weighted point/interval '+str(key),np.allclose([r.equal_coordinate_year_compatible_fraction,r.ci95_lower,r.ci95_upper],
              [rates.mean(),*np.nanquantile(boot,[.025,.975])],rtol=0,atol=1e-12))
    occurrence=pd.read_csv(HERE/'observed_window_classification.csv')
    original=pd.read_csv(ARCHIVE/'observed_window_detection_metrics.csv')
    for r in occurrence.itertuples():
        q=external if r.geography=='full_source_transfer' else external[external.strict_location_disjoint]
        if r.sensitivity=='exclude_2015_58_oct30':q=q[~((q.source_unit=='2015-58')&(q.date==pd.Timestamp('2015-10-30')))]
        actual=q.groupby('field_id').value.max().ge(r.cutoff_percent)
        check('observed field threshold labels '+str(r.Index),int(actual.sum())==r.positive_fields and int((~actual).sum())==r.below_threshold_fields)
        if r.sensitivity=='all_dates':
            record=original[(original.geography==r.geography)&(original.model==r.model)&(original.cutoff_percent==r.cutoff_percent)]
            if not record.empty:check('classification archive reconstruction '+str(r.Index),r.field_seasons==record.iloc[0]['n'] and
                                     np.isclose(r.sensitivity_observed_window,record.iloc[0].sensitivity,equal_nan=True))
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),status='passed',check_count=len(checks),failed_checks=0,checks=checks,
                 bootstrap_max_difference_pp=max_difference,archive_metric_max_difference_pp=original_metric_difference,
                 parameters_refitted=False,model_imported=False,strict_source_points_disjoint=True,
                 source_archives_preserved=True,executable_sha256=sha(Path(__file__)))
    (HERE/'independent_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='checks'},indent=2),flush=True)


if __name__=='__main__':main()
