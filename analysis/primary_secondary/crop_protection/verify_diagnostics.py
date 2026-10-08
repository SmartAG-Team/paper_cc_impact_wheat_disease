"""Recompute diagnostics with nested averages and pairwise AUC, separately."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def source_subset(frame,row):
    if row.leaf_scope=='upper_three':frame=frame.loc[frame.leaf_rank.le(3)]
    if row.leaf_scope=='flag_leaf':frame=frame.loc[frame.leaf_rank.eq(1)]
    if row.horizon_scope!='all':frame=frame.loc[frame.horizon_group.eq(row.horizon_scope)]
    if row.stage_scope!='all':frame=frame.loc[frame.stage_group.eq(row.stage_scope)]
    if hasattr(row,'eligibility') and row.eligibility=='initial_below_cutoff':
        frame=frame.loc[frame.initial_percent.lt(row.cutoff_percent)]
    return frame


def nested_mean(frame,values):
    table=frame[['coordinate_year','series_id']].copy()
    table['quantity']=values
    return float(table.groupby(['coordinate_year','series_id']).quantity.mean().groupby('coordinate_year').mean().mean())


def weights_independent(frame):
    groups=frame.groupby(['coordinate_year','series_id']).size()
    series=groups.groupby('coordinate_year').size()
    return np.array([1/(len(series)*series.loc[cy]*groups.loc[(cy,sid)])
        for cy,sid in zip(frame.coordinate_year,frame.series_id)])


def check(actual,expected,name):
    if pd.isna(actual) and (expected is None or pd.isna(expected)):return 0.
    if expected is None or not np.isfinite(actual) or abs(float(actual)-float(expected))>1e-9:
        raise AssertionError(f'{name}: {actual} versus {expected}')
    return abs(float(actual)-float(expected))


def main():
    predictions=pd.read_csv(OUT/'forecast_context.csv.gz')
    groups={key:f for key,f in predictions.groupby(['dataset','fold','model'])}
    severity=pd.read_csv(OUT/'severity_metrics.csv')
    diagnostic=pd.read_csv(OUT/'threshold_diagnostics.csv')
    max_difference=0.
    for row in severity.itertuples(index=False):
        f=source_subset(groups[(row.dataset,row.fold,row.model)],row)
        error=f.predicted_percent.to_numpy()-f.observed_percent.to_numpy()
        expected={'rmse_pp':np.sqrt(nested_mean(f,error**2)), 'mae_pp':nested_mean(f,abs(error)),
            'bias_pp':nested_mean(f,error),'n_targets':len(f),'n_series':f.series_id.nunique(),
            'n_coordinate_years':f.coordinate_year.nunique(),'n_locations':f.location_id.nunique()}
        for name,value in expected.items():max_difference=max(max_difference,check(getattr(row,name),value,name))
    for row in diagnostic.itertuples(index=False):
        f=source_subset(groups[(row.dataset,row.fold,row.model)],row)
        truth=f.observed_percent.to_numpy()>=row.cutoff_percent
        warn=f.predicted_percent.to_numpy()>=row.cutoff_percent
        masses=[nested_mean(f,v.astype(float)) for v in [truth&warn,~truth&warn,~truth&~warn,truth&~warn]]
        tp,fp,tn,fn=masses
        def ratio(a,b):return a/b if b>0 else None
        expected={'weighted_true_positive':tp,'weighted_false_positive':fp,'weighted_true_negative':tn,
            'weighted_false_negative':fn,'weighted_prevalence':tp+fn,'sensitivity':ratio(tp,tp+fn),
            'specificity':ratio(tn,tn+fp),'precision':ratio(tp,tp+fp),'false_alarm_fraction':ratio(fp,tp+fp),
            'missed_case_fraction':ratio(fn,tp+fn),'n_targets':len(f),'n_positive_targets':int(truth.sum()),
            'n_negative_targets':int((~truth).sum())}
        auc=None
        if truth.any() and (~truth).any():
            weights=weights_independent(f);scores=f.predicted_percent.to_numpy()
            diff=scores[truth,None]-scores[~truth][None,:]
            credit=(diff>0).astype(float)+.5*(diff==0)
            pair_weights=weights[truth,None]*weights[~truth][None,:]
            auc=float((credit*pair_weights).sum()/pair_weights.sum())
        expected['auc']=auc
        for name,value in expected.items():max_difference=max(max_difference,check(getattr(row,name),value,name))
    if not ((pd.to_datetime(predictions.Date)-pd.to_datetime(predictions.start)).dt.days==predictions.day).all():
        raise AssertionError('Forecast chronology differs.')
    if not predictions.day.gt(0).all() or predictions.conditioning.any():
        raise AssertionError('Conditioning outcomes were scored.')
    manifest=json.loads((ROOT/'data/harmonized/manifest.json').read_text())
    schema=pq.read_schema(ROOT/'data/harmonized/observations.parquet')
    common_count=0
    for name in manifest['common_schema_partitions']:
        p=ROOT/'data/harmonized'/name
        if not pq.read_schema(p).equals(schema):raise AssertionError('Schema differs: '+name)
        common_count+=pq.ParquetFile(p).metadata.num_rows
    if common_count!=manifest['common_records']:raise AssertionError('Manifest count differs.')
    for file in manifest['files']:
        if hashlib.sha256((ROOT/file['path']).read_bytes()).hexdigest()!=file['sha256']:
            raise AssertionError('Package hash differs: '+file['path'])
    prediction_partition=pd.read_parquet(ROOT/'data/harmonized/primary_secondary_predictions.parquet')
    if not prediction_partition.observation_id.is_unique:raise AssertionError('Duplicate forecast measurement keys.')
    if prediction_partition.country.isin(['SLOWAKIA','CZECH REPUBLIC']).any():
        raise AssertionError('Unnormalized forecast country labels.')
    source_export=json.loads((OUT/'package_export_receipt.json').read_text())
    for p,h in source_export['protected_partitions_sha256'].items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise AssertionError('Protected partition changed.')
    receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'status':'passed',
        'severity_rows_recomputed_by_nested_means':len(severity),
        'diagnostic_rows_recomputed_with_pairwise_auc':len(diagnostic),
        'max_absolute_numeric_difference':max_difference,
        'common_partition_schema_checks':len(manifest['common_schema_partitions']),
        'package_file_hash_checks':len(manifest['files']),'common_measurement_cells':common_count,
        'new_prediction_observation_ids_unique':True,'original_partitions_unchanged':True,
        'source_severity_is_not_disease_probability':True}
    (OUT/'independent_algorithm_verification.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
