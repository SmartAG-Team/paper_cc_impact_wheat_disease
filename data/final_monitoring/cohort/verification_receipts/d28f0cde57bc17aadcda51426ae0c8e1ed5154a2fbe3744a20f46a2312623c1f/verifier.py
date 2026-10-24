"""Audit every fitted history member and unchanged evaluation boundary."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import distances_km,paired_interval,promotion_decision,score_predictions,score_subset
from .field_history_models import predict_ensemble
from .history import build_history
from .history_interval_hybrid import predict
from .infection_event_experiment import populations
from .inputs import sha
from .robust_models import predict_cluster_ensemble
from .verify_french_weather import write_verification_receipt

ROOT=Path(__file__).resolve().parents[2]
NAME='irish_station_candidate'


def verify(root,output):
    root,out=Path(root).resolve(),Path(output).resolve()
    protocol=json.loads((out/'registration.json').read_text())
    for path,digest in protocol['source_sha256'].items():
        assert sha(root/path)==digest,path
    for path,digest in protocol['code_sha256'].items():
        assert sha(root/path)==digest and sha(out/'code_snapshot'/path)==digest,path
    base=root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    parent=root/'outputs/disease_improvement/history_interval_20261022'
    data=root/'data/disease_improvement/station_preferred_ie_20261022'
    obs=pd.read_csv(out/'training_observation_ledger.csv')
    pd.testing.assert_frame_equal(obs,pd.read_csv(base/'training_observation_ledger.csv'))
    assert len(obs)==2304 and obs.global_target_index.is_unique
    arrays=np.load(out/'predictors.npz')
    original_arrays=np.load(base/'predictors.npz')
    frozen_arrays=np.load(data/'predictors.npz')
    for key in arrays.files:
        np.testing.assert_array_equal(arrays[key],frozen_arrays[key])
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    for key in ['history','process']:
        np.testing.assert_array_equal(arrays[key],original_arrays[key])
    original=joblib.load(parent/'fitted_candidate.joblib')
    history,provenance=build_history(obs,obs.drop(columns='observed_percent'))
    assert history.columns.tolist()==original['base']['history_names']
    np.testing.assert_array_equal(history.to_numpy(),arrays['history'])
    index=obs.set_index('global_target_index')
    positions=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    for row in provenance.itertuples(index=False):
        if row.prior_target_ids:
            assert pd.to_datetime(index.loc[row.prior_target_ids,'date']).lt(pd.Timestamp(index.loc[row.global_target_index,'date'])).all()
    splits=json.loads((out/'split_membership.json').read_text())
    assert splits==json.loads((base/'split_membership.json').read_text()) and len(splits)==20
    frames={}
    for label,filename in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]:
        frame=pd.read_csv(out/filename)
        previous=pd.read_csv(parent/filename)
        assert len(frame)==2017 and frame.global_target_index.is_unique
        pd.testing.assert_frame_equal(frame[previous.columns],previous,check_dtype=False)
        frames[label]=frame
    member_count=0

    def replay(bundle,train,test,key):
        nonlocal member_count
        assert bundle['training_ids']==train
        train_frame=index.loc[train]
        clusters=np.array(sorted(train_frame.coordinate_year.unique()))
        count=max(1,int(np.ceil(len(clusters)*.8)))
        rng=np.random.default_rng(20261022)
        assert len(bundle['members'])==20 and bundle['fraction']==.8 and bundle['seed']==20261022
        assert bundle['clusters_per_member']==count
        for member in bundle['members']:
            selected=rng.choice(clusters,size=count,replace=False)
            assert member['training_ids']==train_frame.loc[train_frame.coordinate_year.isin(selected)].index.tolist()
            assert member['spec']==dict(structure='field_split')
            assert member['n_features']==(53 if key=='station' else 68)
            for head in member['models'].values():
                if head is not None:
                    params=head.get_params()
                    for name,value in dict(max_iter=200,max_leaf_nodes=15,max_depth=3,min_samples_leaf=10,
                                           learning_rate=.05,l2_regularization=1.,early_stopping=False,random_state=20261021).items():
                        assert params[name]==value
            member_count+=1
        loc=positions.loc[test].to_numpy()
        return predict_ensemble(bundle,arrays[key][loc],history.iloc[loc],index.loc[test].reset_index().drop(columns='observed_percent'))

    minimum_distance=np.inf
    seen=dict(temporal=[],spatial_50km=[])
    for split in splits:
        train,test=split['train'],split['test']
        assert set(train).isdisjoint(test)
        assert index.loc[train,'season_year'].max()<index.loc[test,'season_year'].min()
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km':
            distance=float(distances_km(index.loc[train,['latitude','longitude']],index.loc[test,['latitude','longitude']]).min())
            assert distance>=50
            minimum_distance=min(minimum_distance,distance)
        else:
            assert train==obs.loc[obs.season_year.lt(index.loc[test,'season_year'].iloc[0]),'global_target_index'].tolist()
        part=frames[label].set_index('global_target_index').loc[test]
        first=replay(joblib.load(out/f"{split['fold']}_first.joblib"),train,test,'station')
        event=replay(joblib.load(out/f"{split['fold']}_event.joblib"),train,test,'history_events')
        np.testing.assert_allclose(part.ie_first_component,first,atol=1e-10)
        np.testing.assert_allclose(part.ie_event_component,event,atol=1e-10)
        h=history.iloc[positions.loc[test]]
        mask=(h.prior_assessment_available.eq(1)|h.other_leaf_available.eq(1)).to_numpy()
        selected=np.where(mask,event,first)
        np.testing.assert_array_equal(part.ie_uses_event_head,mask)
        np.testing.assert_allclose(part.ie_selected_history,selected,atol=1e-10)
        # Replay the unchanged parent process ensemble as well as its saved predictions.
        process=joblib.load(base/f"{split['fold']}_process.joblib")
        p=predict_cluster_ensemble(process,arrays['process'][positions.loc[test]],h,
                                  index.loc[test].reset_index().drop(columns='observed_percent'))
        np.testing.assert_allclose(part.candidate_process,p,atol=1e-10)
        np.testing.assert_allclose(part[NAME],.5*(p+selected),atol=1e-10)
        seen[label]+=test
        print('verified',split['fold'],flush=True)
    for label,ids in seen.items():
        assert len(ids)==len(set(ids))==2017 and set(ids)==set(frames[label].global_target_index)
    fitted=joblib.load(out/'fitted_candidate.joblib')
    assert fitted['training_ids']==obs.global_target_index.astype(int).tolist()
    assert joblib.hash(fitted['base']['process'])==joblib.hash(original['base']['process'])
    assert fitted['base']['history_names']==original['base']['history_names']
    ids=obs.global_target_index.astype(int).tolist()
    first=replay(fitted['base']['history'],ids,ids,'station')
    event=replay(fitted['events_history'],ids,ids,'history_events')
    # Exercise real saved component models through the qualified future-schema API.
    pick=[int(np.flatnonzero(m)[0]) for m in [
        history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(0),
        history.prior_assessment_available.eq(1),
        history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(1)]]
    q=obs.iloc[pick].drop(columns='observed_percent').copy()
    q['date'],q['season_year']='2020-06-01',2020
    h=history.iloc[pick]
    p=predict_cluster_ensemble(fitted['base']['process'],arrays['process'][pick],h,q)
    selected=np.where(h.prior_assessment_available.eq(1)|h.other_leaf_available.eq(1),event[pick],first[pick])
    np.testing.assert_allclose(predict(fitted,arrays['process'][pick],arrays['station'][pick],arrays['history_events'][pick],h,q),.5*(p+selected),atol=1e-10)
    metrics=pd.read_csv(out/'metrics.csv')
    expected_metrics={(label,name) for label in frames for name in [NAME,'history_interval','reference_replay']+(['current'] if label=='temporal' else [])}
    assert len(metrics)==len(expected_metrics) and set(zip(metrics.evaluation,metrics.model))==expected_metrics
    for row in metrics.itertuples(index=False):
        for key,value in score_predictions(frames[row.evaluation],row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row,key),value,atol=1e-9)
    intervals=pd.read_csv(out/'paired_intervals.csv')
    expected_intervals={(label,NAME,ref) for label in frames for ref in ['history_interval','reference_replay']+(['current'] if label=='temporal' else [])}
    assert len(intervals)==len(expected_intervals) and set(zip(intervals.evaluation,intervals.model,intervals.reference))==expected_intervals
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose([row.lower,row.upper],paired_interval(frames[row.evaluation],NAME,row.reference),atol=1e-9)
    f=frames['temporal'];h=history.iloc[positions.loc[f.global_target_index]].reset_index(drop=True)
    masks=populations(f,h)
    groups=pd.read_csv(out/'subgroup_metrics.csv')
    expected_groups={(key,name) for key in masks for name in [NAME,'history_interval','reference_replay']}
    assert len(groups)==len(expected_groups) and set(zip(groups.population,groups.model))==expected_groups
    for row in groups.itertuples(index=False):
        for key,value in score_subset(f,masks[row.population],row.model).items():
            if value is not None:
                np.testing.assert_allclose(getattr(row,key),value,atol=1e-9)
    decisions=json.loads((out/'adoption_decision.json').read_text())
    assert decisions['independent_external_validation'] is False and decisions['publication_model_promoted'] is False
    assert decisions['qualified_station_weather_retained_independent_of_model_score'] is True
    assert set(decisions['references'])=={'history_interval','current','reference_replay'}
    for reference,saved in decisions['references'].items():
        deltas={key:score_subset(f,masks[key],NAME)['rmse']-score_subset(f,masks[key],reference)['rmse'] for key in ['cold','warm','BASF','Corteva']}
        severe=abs(score_subset(f,masks['severe_above75'],NAME)['bias'])-abs(score_subset(f,masks['severe_above75'],reference)['bias'])
        ratio=score_subset(f,masks['new_site'],NAME)['rmse']/score_subset(f,masks['new_site'],reference)['rmse']
        expected=promotion_decision(score_predictions(f,reference),score_predictions(f,NAME),paired_interval(f,NAME,reference),
            independent=False,subgroup_deltas=deltas,severe_bias_delta=severe,new_site_ratio=ratio)
        assert saved['gates']==expected['gates'] and saved['promote']==expected['promote']
        np.testing.assert_allclose(saved['severe_absolute_bias_change'],severe,atol=1e-9)
        np.testing.assert_allclose(saved['new_site_rmse_ratio'],ratio,atol=1e-9)
        for key,value in deltas.items():
            np.testing.assert_allclose(saved['subgroup_rmse_deltas'][key],value,atol=1e-9)
    contract=json.loads((out/'fitted_candidate_lock.json').read_text())
    for name,digest in contract['sha256'].items():
        assert sha(out/name)==digest,name
    assert contract['parent_contract_sha256']==sha(parent/'fitted_candidate_lock.json')
    assert contract['inference']=='analysis.disease_improvement.history_interval_hybrid.predict'
    result=dict(verified=True,training_records=2304,evaluation_records_per_condition=2017,
        new_history_component_members_replayed=member_count,original_process_ensembles_replayed=20,
        full_process_component_identical=True,history_reconstructed_exactly=True,prior_dates_strictly_earlier=True,
        original_splits_unchanged=True,minimum_spatial_distance_km=minimum_distance,
        metrics_recomputed=len(metrics),intervals_recomputed=len(intervals),subgroup_rows_recomputed=len(groups),
        diagnostic_combinations_complete=True,adoption_decisions_recomputed=3,
        synthetic_future_schema_probe_is_accuracy_validation=False,
        independent_external_validation=False,publication_model_promoted=False,verifier_sha256=sha(Path(__file__)))
    write_verification_receipt(out,result,Path(__file__))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(ROOT,args.output),indent=2))
