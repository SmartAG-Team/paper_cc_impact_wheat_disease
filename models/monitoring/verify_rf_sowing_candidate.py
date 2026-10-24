"""Replay RF-sowing disease models, information boundaries and internal decisions."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import distances_km,paired_interval,promotion_decision,score_predictions,score_subset
from .field_history_models import predict_ensemble
from .history_interval_hybrid import predict
from .rf_sowing_finalize import assessment_populations as populations
from .inputs import sha
from .rf_sowing_predictors import designs
from .robust_models import predict_cluster_ensemble
from .verify_first_visit_screens import manual_integral,manual_weights
from .verify_french_weather import write_verification_receipt

ROOT=Path(__file__).resolve().parents[2]
NAME='rf_sowing_candidate'


def verify(output):
    output=Path(output).resolve()
    data=ROOT/'data/disease_improvement/rf_sowing_20261022'
    parent=ROOT/'data/final_monitoring/cohort'
    for folder in [data,output]:
        r=json.loads((folder/'registration.json').read_text())
        for path,digest in r['source_sha256'].items():
            assert sha(ROOT/path)==digest,path
        for path,digest in r['code_sha256'].items():
            assert digest==sha(folder/'code_snapshot'/path),path
            if path=='analysis/disease_improvement/rf_sowing_experiment.py' and (output/'finalization_registration.json').exists():
                amendment=json.loads((output/'finalization_registration.json').read_text())
                assert sha(ROOT/path)==amendment['amended_code_sha256'][path]
            else:
                assert sha(ROOT/path)==digest,path
    reg=json.loads((output/'registration.json').read_text())
    if (output/'finalization_registration.json').exists():
        finalization=json.loads((output/'finalization_registration.json').read_text())
        assert sha(output/'registration.json')==finalization['original_registration_sha256']
        for section in ['completed_fold_model_sha256','completed_prediction_sha256']:
            for name,digest in finalization[section].items():
                assert sha(output/name)==digest,name
        for path,digest in finalization['amended_code_sha256'].items():
            assert sha(ROOT/path)==digest==sha(output/'completion_code_snapshot'/path),path
    assert not reg['external_validation_required_for_this_internal_experiment']
    assert not reg['independent_external_validation'] and not reg['hyperparameter_search']
    obs=pd.read_csv(output/'training_observation_ledger.csv')
    pd.testing.assert_frame_equal(obs,pd.read_csv(parent/'training_observation_ledger.csv'))
    arrays=np.load(output/'predictors.npz');old=np.load(parent/'predictors.npz')
    assert len(obs)==2304 and obs.global_target_index.is_unique
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    np.testing.assert_array_equal(arrays['history'],old['history'])
    fields=pd.read_csv(data/'fields.csv');query=pd.read_csv(data/'label_free_queries.csv')
    np.testing.assert_array_equal(query.global_target_index,obs.global_target_index)
    assert fields.sowing_basis.eq('rf_estimated_not_observed').all() and fields.observed_sowing_date.isna().all()
    rfdates=pd.read_csv(ROOT/'outputs/disease_improvement/sowing_rf_20261022/field_estimated_sowing.csv').set_index('field_id')
    np.testing.assert_array_equal(fields.sowing_date,fields.field_id.map(rfdates.estimated_sowing_date))
    assert (pd.to_datetime(fields.sowing_date).dt.year==fields.season_year-1).all()
    fitted_sowing=joblib.load(ROOT/'outputs/disease_improvement/sowing_rf_20261022/fitted_sowing_rf.joblib')
    assert fitted_sowing['training_rows']==32232
    assert pd.Timestamp(fitted_sowing['maximum_training_date'])<pd.Timestamp('2013-01-01')
    assert fitted_sowing['feature_names']==['annual_mean_temperature_c','latitude']
    raw=pd.read_parquet(data/'raw_daily_weather.parquet')
    station=pd.read_parquet(data/'station_daily_weather.parquet')
    calendar=pd.read_csv(data/'stage_calendar.csv');hp=pd.read_json(data/'history_provenance.json')
    rebuilt,provenance=designs(raw,station,fields,query,calendar,hp)
    for key,value in rebuilt.items():
        np.testing.assert_array_equal(value,arrays[key])
    for name,ledger in provenance.items():
        pd.testing.assert_frame_equal(ledger,pd.read_csv(data/(name+'_provenance.csv')),check_dtype=False)
    amendment=json.loads((data/'provenance_amendment.json').read_text())
    assert sha(data/amendment['original_file'])==amendment['original_sha256']
    assert sha(data/amendment['effective_file'])==amendment['effective_sha256']
    effective=pd.read_csv(data/amendment['effective_file'])
    assert effective.sowing_basis.eq('rf_estimated_not_observed').all()
    pd.testing.assert_frame_equal(effective.drop(columns='sowing_basis'),provenance['stage'].drop(columns='sowing_basis'),check_dtype=False)
    # Strong cutoff checks on source/diverse severe first visits, not current-label fitting.
    perturbations=0
    for target in [1141,413,1750,1905,1000017]:
        pos=np.flatnonzero(obs.global_target_index.eq(target))
        q=query.iloc[pos]
        assert len(q)==1
        altered=[]
        for weather in [raw,station]:
            changed=weather.copy()
            mask=pd.to_datetime(changed.date).ge(pd.Timestamp(q.date.iloc[0]))
            changed.loc[mask,'rain_mm']+=999
            altered.append(changed)
        test,_=designs(*altered,fields,q,calendar,hp)
        for key,value in test.items():
            np.testing.assert_array_equal(value,arrays[key][pos])
        perturbations+=1
    parent_bundle=joblib.load(parent/'fitted_candidate.joblib')
    history=pd.DataFrame(arrays['history'],columns=parent_bundle['base']['history_names'])
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    splits=json.loads((output/'split_membership.json').read_text())
    assert (output/'split_membership.json').read_bytes()==(parent/'split_membership.json').read_bytes()
    frames={label:pd.read_csv(output/name) for label,name in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]}
    for label,name in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]:
        original=pd.read_csv(parent/name)
        pd.testing.assert_frame_equal(frames[label][original.columns],original)
        assert len(frames[label])==2017 and frames[label].global_target_index.is_unique
    members=0
    def replay(fitted,train,test,key):
        nonlocal members
        assert fitted['training_ids']==train and len(fitted['members'])==20
        assert fitted['seed']==20261022 and fitted['fraction']==.8
        pool=obs.set_index('global_target_index').loc[train]
        clusters=np.asarray(sorted(pool.coordinate_year.unique()))
        count=max(1,int(np.ceil(.8*len(clusters))))
        assert fitted['clusters_per_member']==count
        rng=np.random.default_rng(20261022)
        for member in fitted['members']:
            chosen=rng.choice(clusters,count,replace=False)
            expected=pool.loc[pool.coordinate_year.isin(chosen)].index.tolist()
            assert member['training_ids']==expected
            assert member['spec']==(dict(structure='process',process_baseline_column=-3,crossleaf=True) if key=='process' else dict(structure='field_split'))
            members+=1
        idx=position.loc[test].to_numpy();h=history.iloc[idx]
        q=obs.iloc[idx].drop(columns='observed_percent')
        fn=predict_cluster_ensemble if key=='process' else predict_ensemble
        return fn(fitted,arrays[key][idx],h,q)
    seen={label:[] for label in frames};minimum=np.inf
    for split in splits:
        train,test=split['train'],split['test']
        a,b=position.loc[train].to_numpy(),position.loc[test].to_numpy()
        assert obs.iloc[a].season_year.max()<obs.iloc[b].season_year.min()
        assert set(train).isdisjoint(test)
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km':
            distance=distances_km(obs.iloc[a][['latitude','longitude']],obs.iloc[b][['latitude','longitude']]).min()
            minimum=min(minimum,distance);assert distance>=50
        else:
            assert train==obs.loc[obs.season_year.lt(obs.iloc[b].season_year.iloc[0]),'global_target_index'].tolist()
        seen[label]+=test
        part=frames[label].set_index('global_target_index').loc[test]
        values=[]
        for head,key,column in [('process','process','rf_process_component'),('first','station','rf_first_component'),('event','history_events','rf_event_component')]:
            fitted=joblib.load(output/(split['fold']+'_'+head+'.joblib'))
            value=replay(fitted,train,test,key)
            np.testing.assert_allclose(value,part[column],atol=1e-10,rtol=0)
            values.append(value)
        h=history.iloc[b];mask=(h.prior_assessment_available.eq(1)|h.other_leaf_available.eq(1)).to_numpy()
        selected=np.where(mask,values[2],values[1])
        np.testing.assert_array_equal(mask,part.rf_uses_event_head)
        np.testing.assert_allclose(part.rf_selected_history,selected,atol=1e-10,rtol=0)
        np.testing.assert_allclose(part[NAME],.5*(values[0]+selected),atol=1e-10,rtol=0)
        print('verified',split['fold'],flush=True)
    for label,ids in seen.items():
        assert len(ids)==len(set(ids))==2017 and set(ids)==set(frames[label].global_target_index)
    full=joblib.load(output/'fitted_candidate.joblib')
    ids=obs.global_target_index.astype(int).tolist()
    p=replay(full['base']['process'],ids,ids,'process')
    a=replay(full['base']['history'],ids,ids,'station')
    b=replay(full['events_history'],ids,ids,'history_events')
    assert members==1260
    choose=[int(np.flatnonzero(mask)[0]) for mask in [history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(0),
        history.prior_assessment_available.eq(1),history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(1)]]
    q=obs.iloc[choose].drop(columns='observed_percent').copy();q['date']='2020-06-01';q['season_year']=2020
    h=history.iloc[choose];mask=(h.prior_assessment_available.eq(1)|h.other_leaf_available.eq(1)).to_numpy()
    expected=.5*(p[choose]+np.where(mask,b[choose],a[choose]))
    np.testing.assert_allclose(predict(full,arrays['process'][choose],arrays['station'][choose],arrays['history_events'][choose],h,q),expected,atol=1e-10,rtol=0)
    metrics=pd.read_csv(output/'metrics.csv');groups=pd.read_csv(output/'subgroup_metrics.csv');intervals=pd.read_csv(output/'paired_intervals.csv')
    expected_rows={(label,name) for label in frames for name in [NAME,'irish_station_candidate','reference_replay']+(['current'] if label=='temporal' else [])}
    assert len(metrics)==7 and set(zip(metrics.evaluation,metrics.model))==expected_rows
    for row in metrics.itertuples(index=False):
        frame=frames[row.evaluation]
        for key,value in score_predictions(frame,row.model).items():
            if value is not None:np.testing.assert_allclose(getattr(row,key),value,atol=1e-9,rtol=0)
        direct=np.sqrt(np.average((frame[row.model]-frame.observed_percent)**2,weights=manual_weights(frame)))
        np.testing.assert_allclose(row.rmse,direct,atol=1e-10,rtol=0)
        np.testing.assert_allclose(row.integral_rmse,manual_integral(frame,row.model),atol=1e-9,rtol=0)
    assert len(intervals)==5
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose([row.lower,row.upper],paired_interval(frames[row.evaluation],NAME,row.reference),atol=1e-10,rtol=0)
    expected_groups=set()
    for label,frame in frames.items():
        h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True);masks=populations(frame,h)
        for pop in masks:
            for name in [NAME,'irish_station_candidate','reference_replay']+(['current'] if label=='temporal' else []):
                expected_groups.add((label,pop,name))
    assert len(groups)==len(expected_groups)==63 and set(zip(groups.evaluation,groups.population,groups.model))==expected_groups
    for row in groups.itertuples(index=False):
        frame=frames[row.evaluation];h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        for key,value in score_subset(frame,populations(frame,h)[row.population],row.model).items():
            if value is not None:np.testing.assert_allclose(getattr(row,key),value,atol=1e-9,rtol=0)
    decision=json.loads((output/'assessment_decision.json').read_text())
    frame=frames['temporal'];h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True);masks=populations(frame,h)
    for reference,saved in decision['references'].items():
        delta={key:score_subset(frame,masks[key],NAME)['rmse']-score_subset(frame,masks[key],reference)['rmse'] for key in ['cold','warm','BASF','Corteva']}
        severe=abs(score_subset(frame,masks['severe_above75'],NAME)['bias'])-abs(score_subset(frame,masks['severe_above75'],reference)['bias'])
        ratio=score_subset(frame,masks['new_site'],NAME)['rmse']/score_subset(frame,masks['new_site'],reference)['rmse']
        result=promotion_decision(score_predictions(frame,reference),score_predictions(frame,NAME),paired_interval(frame,NAME,reference),independent=False,
            subgroup_deltas=delta,severe_bias_delta=severe,new_site_ratio=ratio)
        guards={k:v for k,v in result['gates'].items() if k!='independent_validation'}
        assert guards==saved['internal_guards'] and saved['passes_current_cohort_guards']==all(guards.values())
        assert saved['independent_external_validation'] is False
    for label in frames:
        newscore,oldscore=[score_predictions(frames[label],name) for name in [NAME,'irish_station_candidate']]
        assert decision['incremental_parent_guards'][label+'_rmse_noninferiority']==(newscore['rmse']<=oldscore['rmse'])
        assert decision['incremental_parent_guards'][label+'_integral_noninferiority']==(newscore['integral_rmse']<=oldscore['integral_rmse'])
    assert decision['independent_external_validation'] is False
    contract=json.loads((output/'fitted_candidate_lock.json').read_text())
    for name,digest in contract['sha256'].items():assert sha(output/name)==digest
    result=dict(verified=True,source_and_code_hashes_match=True,baseline_and_estimated_feature_arrays_exact=True,
        full_training_rows=2304,fields=218,retained_evaluation_rows_per_design=2017,disease_history35_unchanged=True,
        observed_labels_and_outer_splits_preserved=True,replayed_members=members,minimum_disease_spatial_distance_km=float(minimum),
        sowing_fit_strictly_pre2013=True,complete_annual_temperature_predictors=True,causal_cutoff_perturbations=perturbations,
        effective_stage_provenance_verified=True,metric_rows_recomputed=7,subgroup_rows_recomputed=63,paired_intervals_recomputed=5,
        internal_assessment_decisions_recomputed=3,incremental_parent_guards_recomputed=4,
        external_validation_required_for_internal_experiment=False,independent_external_validation=False,
        publication_model_promoted=False,future_inference_arithmetic_probe=True,
        finalization_bindings_verified=True,
        verifier_dependency_sha256={str((Path(__file__).parent/n).relative_to(ROOT)):sha(Path(__file__).parent/n)
            for n in ['rf_sowing_predictors.py','rf_sowing_finalize.py','verify_first_visit_screens.py','verify_french_weather.py']},
        verifier_sha256=sha(__file__))
    write_verification_receipt(output,result,Path(__file__))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(json.dumps(verify(args.output),indent=2))
