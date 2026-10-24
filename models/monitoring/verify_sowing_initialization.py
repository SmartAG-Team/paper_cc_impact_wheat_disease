"""Check unchanged child predictions, history-only routing and public inference."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval,promotion_decision,score_predictions,score_subset
from .history_interval_hybrid import predict as predict_child
from .inputs import sha
from .rf_sowing_finalize import assessment_populations
from .sowing_initialization import NAME,predict
from .verify_first_visit_screens import manual_integral,manual_weights
from .verify_french_weather import write_verification_receipt

ROOT=Path(__file__).resolve().parents[2]


def verify():
    out=ROOT/'outputs/disease_improvement/sowing_initialization_20261022'
    rf=ROOT/'outputs/disease_improvement/rf_sowing_candidate_20261022'
    parent=ROOT/'data/final_monitoring/cohort'
    reg=json.loads((out/'registration.json').read_text())
    for path,digest in reg['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in reg['code_sha256'].items():assert sha(ROOT/path)==digest==sha(out/'code_snapshot'/path),path
    assert reg['model_refits']==0 and not reg['external_validation_required_for_internal_experiment']
    obs=pd.read_csv(rf/'training_observation_ledger.csv');a=np.load(rf/'predictors.npz');old=np.load(parent/'predictors.npz')
    np.testing.assert_array_equal(a['target_ids'],obs.global_target_index)
    np.testing.assert_array_equal(a['history'],old['history'])
    bundle=joblib.load(out/'fitted_candidate.joblib')
    assert joblib.hash(bundle['calendar'])==joblib.hash(joblib.load(parent/'fitted_candidate.joblib'))
    assert joblib.hash(bundle['estimated'])==joblib.hash(joblib.load(rf/'fitted_candidate.joblib'))
    history=pd.DataFrame(a['history'],columns=bundle['calendar']['base']['history_names'])
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    frames={label:pd.read_csv(out/name) for label,name in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]}
    for label,filename in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]:
        original=pd.read_csv(rf/filename);frame=frames[label]
        pd.testing.assert_frame_equal(frame[original.columns],original)
        assert len(frame)==2017 and frame.global_target_index.is_unique
        h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        first=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(0)
        np.testing.assert_array_equal(first,frame.rf_sowing_used)
        assert first.sum()==494
        np.testing.assert_array_equal(frame.loc[first,NAME],frame.loc[first,'rf_sowing_candidate'])
        np.testing.assert_array_equal(frame.loc[~first,NAME],frame.loc[~first,'irish_station_candidate'])
    choose=[int(np.flatnonzero(m)[0]) for m in [history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(0),
        history.prior_assessment_available.eq(1),history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(1)]]
    query=obs.iloc[choose].drop(columns='observed_percent').copy();query['date']='2020-06-01';query['season_year']=2020
    h=history.iloc[choose];keys=['process','station','history_events']
    calendar={k:old[k][choose] for k in keys};estimated={k:a[k][choose] for k in keys}
    p=predict_child(bundle['calendar'],*[calendar[k] for k in keys],h,query)
    r=predict_child(bundle['estimated'],*[estimated[k] for k in keys],h,query)
    actual=predict(bundle,calendar,estimated,h,query)
    np.testing.assert_allclose(actual,[r[0],p[1],p[2]],atol=1e-10,rtol=0)
    changed={k:np.zeros_like(value) for k,value in estimated.items()}
    np.testing.assert_array_equal(predict(bundle,calendar,changed,h,query)[1:],actual[1:])
    try:
        predict(bundle,calendar,estimated,h,query.assign(observed_percent=99.))
    except ValueError:pass
    else:raise AssertionError('Query labels accepted')
    metrics=pd.read_csv(out/'metrics.csv');groups=pd.read_csv(out/'subgroup_metrics.csv');intervals=pd.read_csv(out/'paired_intervals.csv')
    expected={(e,n) for e in frames for n in [NAME,'irish_station_candidate','reference_replay']+(['current'] if e=='temporal' else [])}
    assert len(metrics)==7 and set(zip(metrics.evaluation,metrics.model))==expected
    for row in metrics.itertuples(index=False):
        f=frames[row.evaluation]
        for key,value in score_predictions(f,row.model).items():
            if value is not None:np.testing.assert_allclose(getattr(row,key),value,atol=1e-9,rtol=0)
        np.testing.assert_allclose(row.rmse,np.sqrt(np.average((f[row.model]-f.observed_percent)**2,weights=manual_weights(f))),atol=1e-10,rtol=0)
        np.testing.assert_allclose(row.integral_rmse,manual_integral(f,row.model),atol=1e-9,rtol=0)
    expected_groups=set()
    for e,f in frames.items():
        h=history.iloc[position.loc[f.global_target_index]].reset_index(drop=True)
        for pop in assessment_populations(f,h):
            for n in [NAME,'irish_station_candidate','reference_replay']+(['current'] if e=='temporal' else []):expected_groups.add((e,pop,n))
    assert len(groups)==63 and set(zip(groups.evaluation,groups.population,groups.model))==expected_groups
    for row in groups.itertuples(index=False):
        f=frames[row.evaluation];h=history.iloc[position.loc[f.global_target_index]].reset_index(drop=True)
        for key,value in score_subset(f,assessment_populations(f,h)[row.population],row.model).items():
            if value is not None:np.testing.assert_allclose(getattr(row,key),value,atol=1e-9,rtol=0)
    assert len(intervals)==5
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose([row.lower,row.upper],paired_interval(frames[row.evaluation],NAME,row.reference),atol=1e-10,rtol=0)
    decision=json.loads((out/'assessment_decision.json').read_text());f=frames['temporal']
    h=history.iloc[position.loc[f.global_target_index]].reset_index(drop=True);masks=assessment_populations(f,h)
    for reference,saved in decision['references'].items():
        delta={k:score_subset(f,masks[k],NAME)['rmse']-score_subset(f,masks[k],reference)['rmse'] for k in ['cold','warm','BASF','Corteva']}
        severe=abs(score_subset(f,masks['severe_above75'],NAME)['bias'])-abs(score_subset(f,masks['severe_above75'],reference)['bias'])
        ratio=score_subset(f,masks['new_site'],NAME)['rmse']/score_subset(f,masks['new_site'],reference)['rmse']
        d=promotion_decision(score_predictions(f,reference),score_predictions(f,NAME),paired_interval(f,NAME,reference),
            independent=False,subgroup_deltas=delta,severe_bias_delta=severe,new_site_ratio=ratio)
        guards={k:v for k,v in d['gates'].items() if k!='independent_validation'}
        assert guards==saved['internal_guards'] and saved['passes_current_cohort_guards']==all(guards.values())
    for e,f in frames.items():
        new,oldscore=[score_predictions(f,n) for n in [NAME,'irish_station_candidate']]
        assert decision['incremental_parent_guards'][e+'_rmse_noninferiority']==(new['rmse']<=oldscore['rmse'])
        assert decision['incremental_parent_guards'][e+'_integral_noninferiority']==(new['integral_rmse']<=oldscore['integral_rmse'])
    contract=json.loads((out/'fitted_candidate_lock.json').read_text())
    for name,digest in contract['sha256'].items():assert sha(out/name)==digest
    result=dict(verified=True,source_and_code_hashes_match=True,child_full_bundles_identical=True,
        current_query_label_rejected=True,real_inference_arithmetic_probe=True,estimated_weather_irrelevant_after_field_history=True,
        retained_rows_per_design=2017,rf_initialization_rows_per_design=494,parent_preserved_rows_per_design=1523,
        metric_rows_recomputed=7,subgroup_rows_recomputed=63,paired_intervals_recomputed=5,assessment_decisions_recomputed=3,
        incremental_parent_guards_recomputed=4,independent_external_validation=False,external_validation_required_for_internal_experiment=False,
        adaptive_development=True,publication_model_promoted=False,verifier_sha256=sha(__file__))
    write_verification_receipt(out,result,Path(__file__))
    return result


if __name__=='__main__':
    with threadpool_limits(limits=1):print(json.dumps(verify(),indent=2))
