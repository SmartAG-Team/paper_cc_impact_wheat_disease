"""Independent hierarchy arithmetic and frozen identity verification."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(frame):
    hierarchy={}
    for index,(coordinate,field,leaf) in enumerate(frame[['coordinate_year','field_id','leaf_index']].itertuples(index=False,name=None)):
        hierarchy.setdefault(coordinate,{}).setdefault(field,{}).setdefault(leaf,[]).append(index)
    result=np.zeros(len(frame))
    for fields in hierarchy.values():
        for leaves in fields.values():
            for indices in leaves.values():result[indices]=1/(len(hierarchy)*len(fields)*len(leaves)*len(indices))
    return result


def main():
    receipt=json.loads((HERE/'receipt.json').read_text())
    for name,value in receipt['source_sha256'].items():assert digest(ROOT/name)==value,name
    for name,value in receipt['output_sha256'].items():assert digest(HERE/name)==value,name
    a=pd.read_parquet(HERE/'all_assessment_sign_predictions.parquet')
    b=pd.read_parquet(HERE/'all_first_onset_bracket_predictions.parquet')
    detection=pd.read_csv(HERE/'assessment_detection_metrics.csv');onset=pd.read_csv(HERE/'onset_bracket_metrics.csv')
    def subset(frame,row):
        part=frame
        for key in ['partition','model','scenario','detection_fraction']:part=part.loc[part[key].eq(getattr(row,key))]
        if row.leaf_scope=='top3':part=part.loc[part.leaf_index.lt(3)]
        return part
    for row in detection.itertuples():
        part=subset(a,row);w=weights(part);observed=part.observed_positive.to_numpy(bool);predicted=part.predicted_positive.to_numpy(bool)
        tp=int(sum(o and p for o,p in zip(observed,predicted)));tn=int(sum(not o and not p for o,p in zip(observed,predicted)))
        assert tp==row.true_positive and tn==row.true_negative
        sensitivity=float(sum(w[observed]*predicted[observed])/sum(w[observed])) if observed.any() else np.nan
        specificity=float(sum(w[~observed]*~predicted[~observed])/sum(w[~observed])) if (~observed).any() else np.nan
        np.testing.assert_allclose([sensitivity,specificity,.5*(sensitivity+specificity)],
            [row.sensitivity,row.specificity,row.balanced_accuracy],rtol=0,atol=1e-12,equal_nan=True)
    for row in onset.itertuples():
        part=subset(b,row);part=part.loc[part.censoring.eq(row.censoring)];w=weights(part)
        assert abs(float(sum(w*part.onset_distance_days.to_numpy()))-row.distance_days)<1e-12
        assert abs(float(sum(w*part.compatible.to_numpy()))-row.compatible_fraction)<1e-12
    scores_checked=0
    for model in ['overwinter_source_model','phenology_only']:
        directory=HERE/model;s=json.loads((directory/'selection.json').read_text())
        ia=pd.read_parquet(directory/'all_inner_assessment_predictions.parquet');ib=pd.read_parquet(directory/'all_inner_bracket_predictions.parquet')
        for record in s['candidate_scores']:
            primary=ib.loc[ib.scenario.eq(record['candidate'])&ib.leaf_index.lt(3)&ib.censoring.eq('two_sided')]
            expected=float(sum(weights(primary)*primary.onset_distance_days.to_numpy()))
            assert abs(expected-record['primary_distance_days'])<1e-12
            part=ia.loc[ia.scenario.eq(record['candidate'])&ia.leaf_index.lt(3)];w=weights(part)
            observed=part.observed_positive.to_numpy(bool);predicted=part.predicted_positive.to_numpy(bool)
            sensitivity=sum(w[observed]*predicted[observed])/sum(w[observed]);specificity=sum(w[~observed]*~predicted[~observed])/sum(w[~observed])
            error=1-.5*(sensitivity+specificity)
            assert abs(error-record['balanced_assessment_error'])<1e-12
            scores_checked+=1
        minimum=min(r['primary_distance_days'] for r in s['candidate_scores'])
        tied=[r for r in s['candidate_scores'] if abs(r['primary_distance_days']-minimum)<1e-12]
        selected=min(tied,key=lambda r:(r['balanced_assessment_error'],r['candidate_order']))
        assert selected['candidate']==s['selected_candidate']['name']
    flow=pd.read_parquet(HERE/'leaf_pathway_flow_totals.parquet')
    np.testing.assert_allclose(flow.primary_flow_total,flow.local_flow_total+flow.imported_flow_total,rtol=0,atol=1e-13)
    np.testing.assert_allclose(flow.secondary_flow_total,flow.splash_flow_total+flow.contact_flow_total,rtol=0,atol=1e-13)
    no_source=flow.loc[flow.scenario.eq('no_sources')]
    assert no_source[['local_flow_total','imported_flow_total','splash_flow_total','contact_flow_total','end_damage_fraction']].to_numpy().sum()==0.
    report=dict(status='passed',source_hashes=len(receipt['source_sha256']),output_hashes=len(receipt['output_sha256']),
        weighted_detection_metrics_checked=len(detection),weighted_onset_metrics_checked=len(onset),
        independent_candidate_scores_checked=scores_checked,flow_accounting_rows_checked=len(flow),
        no_source_state_is_zero=True,selected_candidates_unchanged=True)
    target=HERE/'independent_statistical_verification.json';content=json.dumps(report,indent=2)+'\n'
    if target.exists():assert target.read_text()==content
    else:target.write_text(content)
    print(content)


if __name__=='__main__':main()
