"""Reconstruct earlier-field anchors and replay every retained ensemble."""
import argparse
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .evaluation import score_predictions,score_subset,paired_interval,promotion_decision,distances_km
from .history import build_history
from .inputs import sha
from .verify_french_weather import write_verification_receipt

ROOT=Path(__file__).resolve().parents[2]


def frozen_module(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def adoption_audit(temporal,spatial):
    candidates={}
    for name in ['field_residual_mean','field_split_mean']:
        references={}
        for baseline in ['current','reference_replay']:
            ref=score_predictions(temporal,baseline);candidate=score_predictions(temporal,name)
            interval=paired_interval(temporal,name,baseline)
            masks=dict(cold=~temporal.prior_assessment_available,warm=temporal.prior_assessment_available,
                       BASF=temporal.source.eq('BASF'),Corteva=temporal.source.eq('Corteva'))
            differences={k:score_subset(temporal,mask,name)['rmse']-score_subset(temporal,mask,baseline)['rmse'] for k,mask in masks.items()}
            severe=temporal.observed_percent.gt(75)
            bias=abs(score_subset(temporal,severe,name)['bias'])-abs(score_subset(temporal,severe,baseline)['bias'])
            ratio=score_subset(temporal,temporal.new_site,name)['rmse']/score_subset(temporal,temporal.new_site,baseline)['rmse']
            decision=promotion_decision(ref,candidate,interval,independent=False,
                subgroup_deltas=differences,severe_bias_delta=bias,new_site_ratio=ratio)
            decision.update(reference=ref,candidate=candidate,paired_rmse_interval=interval,
                rmse_ratio=candidate['rmse']/ref['rmse'],new_site_rmse_ratio=ratio,
                subgroup_rmse_deltas=differences,severe_absolute_bias_change=bias)
            references[baseline]=decision
        spatial_reference=score_predictions(spatial,'reference_replay')
        spatial_candidate=score_predictions(spatial,name)
        references['spatial_sensitivity']=dict(reference=spatial_reference,candidate=spatial_candidate,
            rmse_ratio=spatial_candidate['rmse']/spatial_reference['rmse'],
            integral_noninferiority=spatial_candidate['integral_rmse']<=spatial_reference['integral_rmse'],
            descriptive_ten_percent_rmse_reduction=spatial_candidate['rmse']<=.9*spatial_reference['rmse'])
        candidates[name]=references
    return dict(candidates=candidates,independent_external_validation=False,publication_model_promoted=False,
        interpretation='Development gates use the pre-existing function and thresholds. Additional spatial ratio is descriptive. Rolling assessment permits strictly earlier disease observations at held locations.')


def verify(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    protocol=json.loads((output/'registration.json').read_text())
    for path,digest in protocol['source_sha256'].items():assert sha(root/path)==digest,path
    for path,digest in protocol['code_sha256'].items():assert sha(output/'code_snapshot'/path)==digest,path
    assert sha(root/'analysis/final_study/metrics.py')==protocol['code_sha256']['analysis/final_study/metrics.py']
    module=frozen_module(output/'code_snapshot/analysis/disease_improvement/field_history_models.py',
                         'analysis.disease_improvement.frozen_field_history')
    base_models=frozen_module(output/'code_snapshot/analysis/disease_improvement/models.py','frozen_field_base')
    module._category=base_models._category;module._estimator=base_models._estimator
    obs=pd.read_csv(output/'observation_ledger.csv')
    original=root/'outputs/disease_improvement/french_station_20261022'
    assert (output/'observation_ledger.csv').read_bytes()==(original/'observation_ledger.csv').read_bytes()
    assert (output/'spatial_membership.json').read_bytes()==(original/'spatial_membership.json').read_bytes()
    arrays=np.load(output/'predictors.npz');np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    history,provenance=build_history(obs,obs.drop(columns='observed_percent'))
    np.testing.assert_allclose(history.to_numpy(),arrays['history'],atol=1e-12)
    indexed=obs.set_index('global_target_index')
    for row in provenance.itertuples(index=False):
        if row.prior_target_ids:
            prior=indexed.loc[row.prior_target_ids]
            assert pd.to_datetime(prior.date).lt(pd.Timestamp(indexed.loc[row.global_target_index,'date'])).all()
    anchor,features=module.field_history_anchor(history,obs.drop(columns='observed_percent'))
    np.testing.assert_allclose(anchor,arrays['anchor'],atol=1e-12)
    np.testing.assert_allclose(features,arrays['anchor_features'],atol=1e-12)
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    metrics=pd.read_csv(output/'metrics.csv');intervals=pd.read_csv(output/'paired_intervals.csv')
    groups=pd.read_csv(output/'subgroup_metrics.csv');frames={}
    model_count=member_count=0;distance_min=np.inf
    def replay(path,train_ids,test_ids,expected,name):
        nonlocal model_count,member_count
        fitted=joblib.load(path);assert set(fitted['training_ids'])==set(train_ids)
        selected=position.loc[test_ids].to_numpy();eligible=obs.loc[obs.global_target_index.isin(train_ids)]
        allowed=set(train_ids)
        for member in fitted['members']:
            ids=set(member['training_ids']);assert ids<=allowed
            clusters=obs.loc[obs.global_target_index.isin(ids),'coordinate_year']
            whole=eligible.loc[eligible.coordinate_year.isin(clusters),'global_target_index']
            assert ids==set(whole)
            member_count+=1
        prediction=module.predict_ensemble(fitted,arrays['matrix'][selected],history.iloc[selected],
                                           obs.iloc[selected].drop(columns='observed_percent'))
        np.testing.assert_allclose(prediction,expected[name],atol=1e-10)
        model_count+=1
    for label,filename in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]:
        frame=pd.read_csv(output/filename);old=pd.read_csv(original/filename)
        assert set(frame.global_target_index)==set(old.global_target_index) and frame.global_target_index.is_unique
        old=old.set_index('global_target_index').loc[frame.global_target_index].reset_index()
        pd.testing.assert_frame_equal(frame[old.columns],old,check_dtype=False)
        for name in protocol['recipes']:
            np.testing.assert_allclose(frame[name+'_mean'],.5*(frame.bagged_shared+frame[name]),atol=1e-10)
        for name in list(protocol['recipes'])+[n+'_mean' for n in protocol['recipes']]+['mf_all_mean','reference_replay']:
            saved=metrics.loc[metrics.evaluation.eq(label)&metrics.model.eq(name)].iloc[0]
            for key,value in score_predictions(frame,name).items():np.testing.assert_allclose(value,saved[key],atol=1e-9)
            if name.startswith('field_'):
                saved=intervals.loc[intervals.evaluation.eq(label)&intervals.model.eq(name)].iloc[0]
                np.testing.assert_allclose(paired_interval(frame,name,'mf_all_mean'),saved[['lower','upper']].to_numpy(float),atol=1e-9)
        frames[label]=frame
        if label=='temporal':
            for year,part in frame.groupby('season_year'):
                train_ids=obs.loc[obs.season_year.lt(year),'global_target_index'].tolist()
                for name in protocol['recipes']:
                    replay(output/f'{year}_{name}.joblib',train_ids,part.global_target_index.tolist(),part,name)
        else:
            keyed=frame.set_index('global_target_index')
            for number,split in enumerate(json.loads((output/'spatial_membership.json').read_text())):
                if not split['train']:continue
                train=indexed.loc[split['train']];test=indexed.loc[split['test']]
                assert train.season_year.max()<test.season_year.min()
                distance=distances_km(train[['latitude','longitude']],test[['latitude','longitude']]).min()
                assert distance>=50;distance_min=min(distance_min,float(distance))
                for name in protocol['recipes']:
                    replay(output/f'spatial{number}_{name}.joblib',split['train'],split['test'],keyed.loc[split['test']],name)
    frame=frames['temporal'];h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
    masks=dict(cold=~frame.prior_assessment_available,warm=frame.prior_assessment_available,
        first_field=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(0),
        first_leaf_other_history=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(1),
        BASF=frame.source.eq('BASF'),Corteva=frame.source.eq('Corteva'),severe_above75=frame.observed_percent.gt(75),new_site=frame.new_site)
    for row in groups.itertuples(index=False):
        for key,value in score_subset(frame,masks[row.population],row.model).items():
            if value is not None:np.testing.assert_allclose(value,getattr(row,key),atol=1e-9)
    decision=adoption_audit(frames['temporal'],frames['spatial_50km'])
    (output/'adoption_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    result=dict(verified=True,assessments=len(frame),source_hashes=len(protocol['source_sha256']),
        frozen_code_hashes=len(protocol['code_sha256']),saved_ensembles_replayed=model_count,
        component_models_replayed=member_count,history_and_anchors_reconstructed=True,all_prior_dates_strict=True,
        observations_and_spatial_splits_unchanged=True,minimum_spatial_distance_km=distance_min,
        metrics_recomputed=len(metrics),intervals_recomputed=len(intervals),subgroup_rows_recomputed=len(groups),
        independent_external_validation=False,publication_model_promoted=False,verifier_sha256=sha(Path(__file__)))
    write_verification_receipt(output,result,Path(__file__))
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):print(json.dumps(verify(ROOT,args.output),indent=2))
