"""Use estimated sowing before the first field measurement, then field history."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval,promotion_decision,score_predictions,score_subset
from .history_interval_hybrid import history_prediction,predict as predict_single
from .inputs import sha
from .rf_sowing_finalize import assessment_populations

ROOT=Path(__file__).resolve().parents[2]
NAME='rf_sowing_initialization'


def select_initialization(calendar,estimated,history,queries):
    selected,has_history=history_prediction(estimated,calendar,history,queries)
    return selected,~has_history


def predict(fitted,calendar_design,estimated_design,history,queries):
    keys=['process','station','history_events']
    calendar=predict_single(fitted['calendar'],*[calendar_design[k] for k in keys],history,queries)
    estimated=predict_single(fitted['estimated'],*[estimated_design[k] for k in keys],history,queries)
    return select_initialization(calendar,estimated,history,queries)[0]


def run(output=None):
    parent=ROOT/'data/final_monitoring/cohort'
    rf=ROOT/'outputs/disease_improvement/rf_sowing_candidate_20261022'
    output=Path(output or ROOT/'outputs/disease_improvement/sowing_initialization_20261022').resolve()
    if output.exists():raise FileExistsError('A fresh output directory is required')
    obs=pd.read_csv(rf/'training_observation_ledger.csv')
    arrays=np.load(rf/'predictors.npz')
    calendar=joblib.load(parent/'fitted_candidate.joblib')
    h=pd.DataFrame(arrays['history'],columns=calendar['base']['history_names'])
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    output.mkdir(parents=True)
    sources=[rf/n for n in ['assessment_predictions.csv','spatial_predictions.csv','fitted_candidate.joblib','verification.json',
        'predictors.npz','training_observation_ledger.csv','split_membership.json','fitted_candidate_lock.json']]
    sources += [parent/n for n in ['fitted_candidate.joblib','predictors.npz','fitted_candidate_lock.json']]
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['history_interval_hybrid','field_history_models','robust_models',
        'models','rf_sowing_finalize','evaluation']]+[ROOT/'analysis/final_study/metrics.py']
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        rule='Use full RF-sowing prediction only before ANY earlier field disease measurement; thereafter retain the existing field-observation-anchored prediction',
        motivation='Full replacement improves first-field error but degrades later-field error; estimated initialization is uncertain and earlier measured state has greater information',
        routing='Only strictlyearlier diseasehistory availability; no current disease labels, residuals, country or thresholds',
        adaptive_development=True,primary_full_sowing_experiment_negative=True,
        parameter_tuning=False,model_refits=0,preserved_evaluation_rows=2017,
        external_validation_required_for_internal_experiment=False,independent_external_validation=False,
        scientific_scope='Current dateduntreatedcohort, originalchronological/50kmdiseasefitseparation; descriptive reused-outcome development',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes})
    (output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for path in codes:
        dest=output/'code_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    metrics,groups,intervals=[],[],[];frames={}
    for label,filename in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]:
        frame=pd.read_csv(rf/filename)
        hist=h.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        query=frame.drop(columns='observed_percent')
        frame[NAME],frame['rf_sowing_used']=select_initialization(frame.irish_station_candidate,frame.rf_sowing_candidate,hist,query)
        assert frame.rf_sowing_used.sum()==494
        frame.to_csv(output/filename,index=False);frames[label]=frame
        masks=assessment_populations(frame,hist)
        for name in [NAME,'irish_station_candidate','reference_replay']+(['current'] if label=='temporal' else []):
            metrics.append(dict(evaluation=label,model=name,**score_predictions(frame,name)))
            if name!=NAME:
                lo,hi=paired_interval(frame,NAME,name);intervals.append(dict(evaluation=label,model=NAME,reference=name,lower=lo,upper=hi))
            for pop,mask in masks.items():groups.append(dict(evaluation=label,population=pop,model=name,**score_subset(frame,mask,name)))
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(output/'subgroup_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(output/'paired_intervals.csv',index=False)
    frame=frames['temporal'];hist=h.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
    masks=assessment_populations(frame,hist);decisions={}
    for reference in ['irish_station_candidate','current','reference_replay']:
        delta={k:score_subset(frame,masks[k],NAME)['rmse']-score_subset(frame,masks[k],reference)['rmse'] for k in ['cold','warm','BASF','Corteva']}
        severe=abs(score_subset(frame,masks['severe_above75'],NAME)['bias'])-abs(score_subset(frame,masks['severe_above75'],reference)['bias'])
        ratio=score_subset(frame,masks['new_site'],NAME)['rmse']/score_subset(frame,masks['new_site'],reference)['rmse']
        decision=promotion_decision(score_predictions(frame,reference),score_predictions(frame,NAME),paired_interval(frame,NAME,reference),
            independent=False,subgroup_deltas=delta,severe_bias_delta=severe,new_site_ratio=ratio)
        guards={k:v for k,v in decision['gates'].items() if k!='independent_validation'}
        decisions[reference]=dict(internal_guards=guards,passes_current_cohort_guards=all(guards.values()),
            independent_external_validation=False,subgroup_rmse_deltas=delta,severe_absolute_bias_change=severe,new_site_rmse_ratio=ratio)
    incremental={}
    for label,frame in frames.items():
        new,old=[score_predictions(frame,name) for name in [NAME,'irish_station_candidate']]
        incremental[label+'_rmse_noninferiority']=new['rmse']<=old['rmse']
        incremental[label+'_integral_noninferiority']=new['integral_rmse']<=old['integral_rmse']
    (output/'assessment_decision.json').write_text(json.dumps(dict(references=decisions,incremental_parent_guards=incremental,
        independent_external_validation=False,publication_model_promoted=False),indent=2)+'\n')
    fitted=dict(calendar=calendar,estimated=joblib.load(rf/'fitted_candidate.joblib'),training_ids=obs.global_target_index.astype(int).tolist(),minimum_assessment_year=2020)
    joblib.dump(fitted,output/'fitted_candidate.joblib')
    contract=dict(created_utc=datetime.now(timezone.utc).isoformat(),recipe=NAME,
        inference='analysis.disease_improvement.sowing_initialization.predict',minimum_assessment_year=2020,
        inputs='calendar_design andestimated_design dictionaries with process37/station53/history_events68; unchangedhistory35; qualifiedquerymetadata',
        routing=registration['rule'],sowing_model='outputs/disease_improvement/sowing_rf_20261022/fitted_sowing_rf.joblib',
        user_authorized_current_dataset_validation=True,independent_external_validation=False,publication_model_promoted=False,
        sha256={name:sha(output/name) for name in ['registration.json','assessment_predictions.csv','spatial_predictions.csv','fitted_candidate.joblib']})
    (output/'fitted_candidate_lock.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output)
