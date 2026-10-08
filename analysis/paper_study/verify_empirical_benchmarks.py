"""Independent arithmetic audit of frozen empirical comparisons."""

from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
DEST=ROOT/'analysis/paper_study/empirical_benchmarks'
DATA=ROOT/'data/paper_study/empirical_benchmarks'


def main():
    predictions=pd.read_parquet(DATA/'frozen_all_predictions.parquet')
    metrics=pd.read_csv(DEST/'severity_metrics.csv')
    intervals=pd.read_csv(DEST/'paired_model_comparator_intervals.csv')
    membership=pd.read_csv(ROOT/'data/paper_study/external_statistics/coordinate_year_membership.csv')
    draws=np.load(ROOT/'data/paper_study/external_statistics/coordinate_year_draw_indices.npz')
    checks=0;max_difference=0.
    def frame_for(row,model):
        frame=predictions[predictions.source.eq(row.source)&predictions.partition.eq(row.partition)&predictions.model.eq(model)].copy()
        if row.geography=='strict_location_disjoint':frame=frame[frame.strict_location_disjoint]
        if row.leaf_scope=='upper_three':frame=frame[frame.leaf_index.lt(3)]
        if row.endpoint=='final_leaf_assessment':
            latest=frame.groupby(['field_id','endpoint_series']).date.transform('max')
            frame=frame[frame.date.eq(latest)].copy()
        # Compute nested weights by explicit cluster/field/series loops.
        rows=[]
        for _,coordinate in frame.groupby('coordinate_year'):
            for _,field in coordinate.groupby('field_id'):
                for _,leaf in field.groupby('endpoint_series'):
                    part=leaf.copy()
                    part['audit_weight']=1/coordinate.field_id.nunique()/field.endpoint_series.nunique()/len(leaf)
                    rows.append(part)
        return pd.concat(rows,ignore_index=True)
    for row in metrics.itertuples():
        frame=frame_for(row,row.model)
        w=frame.audit_weight.to_numpy();error=(frame.predicted_percent-frame.value).to_numpy()
        expected=[np.sqrt(np.dot(w,error*error)/w.sum()),np.dot(w,abs(error))/w.sum(),np.dot(w,error)/w.sum()]
        difference=max(abs(expected-np.array([row.rmse,row.mae,row.bias])))
        if difference>1e-10:raise AssertionError('Weighted benchmark arithmetic differs.')
        max_difference=max(max_difference,float(difference));checks+=3
    for row in intervals[intervals.source.eq('Corteva')&intervals.endpoint.eq('final_leaf_assessment')].itertuples():
        frames=[frame_for(row,model) for model in ['seasonal_seir',row.comparator]]
        ids=membership[membership.geography.eq(row.geography)].sort_values('cluster_index').coordinate_year.tolist()
        cluster=[]
        for frame in frames:
            value={}
            for key,group in frame.groupby('coordinate_year'):
                w=group.audit_weight.to_numpy();e=(group.predicted_percent-group.value).to_numpy()
                value[key]=(np.dot(w,e*e),np.dot(w,abs(e)),w.sum())
            cluster.append(np.array([value[key] for key in ids]))
        # Multiplicity matrix, distinct from indexed draw summation in the driver.
        index=draws[row.geography];counts=np.zeros((len(index),len(ids)),int)
        np.add.at(counts,(np.arange(len(index))[:,None],index),1)
        statistics=[]
        for values in cluster:
            aggregate=counts@values
            statistics.append((np.sqrt(aggregate[:,0]/aggregate[:,2]),aggregate[:,1]/aggregate[:,2]))
        for name,k in [('rmse',0),('mae',1)]:
            ci=np.quantile(statistics[0][k]-statistics[1][k],[.025,.975])
            expected=np.array([getattr(row,name+'_difference_lower'),getattr(row,name+'_difference_upper')])
            difference=float(max(abs(ci-expected)))
            if difference>1e-10:raise AssertionError('Paired bootstrap interval differs.')
            max_difference=max(max_difference,difference);checks+=2
    receipt=json.loads((DEST/'receipt.json').read_text())
    for name,digest in receipt['frozen_comparator_sha256'].items():
        if hashlib.sha256((DEST/name).read_bytes()).hexdigest()!=digest:raise AssertionError('Frozen model hash differs.')
        checks+=1
    report=dict(status='passed',arithmetic_checks=checks,maximum_absolute_difference=max_difference,
        all_endpoint_weighting_recomputed_by_explicit_nested_groups=True,
        external_final_bootstrap_intervals_recomputed_by_cluster_multiplicities=True,
        causal_feature_tests='tests/test_empirical_benchmarks.py',field_parameters_modified=False)
    (DEST/'independent_arithmetic_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
