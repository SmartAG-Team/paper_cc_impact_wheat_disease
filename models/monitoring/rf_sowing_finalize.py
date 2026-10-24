"""Complete reporting/full-pool fits from saved RF-sowing outer predictions."""
from datetime import datetime,timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval,promotion_decision,score_predictions,score_subset
from .field_history_models import fit_ensemble
from .infection_event_experiment import populations
from .inputs import sha
from .robust_models import fit_cluster_ensemble

ROOT=Path(__file__).resolve().parents[2]
NAME='rf_sowing_candidate'


def assessment_populations(frame,history):
    context=frame.copy()
    if 'prior_assessment_available' not in context:
        context['prior_assessment_available']=history.prior_assessment_available.to_numpy().astype(bool)
    if 'new_site' not in context:
        spatial=context.outer_fold.astype(str).str.contains('sitefold')|context.outer_fold.astype(str).str.startswith('spatial')
        if not spatial.all():
            raise ValueError('Missing temporal site-history metadata')
        context['new_site']=True
    return populations(context,history)


def finalize(output=ROOT/'outputs/disease_improvement/rf_sowing_candidate_20261022'):
    output=Path(output)
    if (output/'metrics.csv').exists() or (output/'fitted_candidate.joblib').exists():
        raise FileExistsError('Completed assessment must not be overwritten')
    parent=ROOT/'data/final_monitoring/cohort'
    reg=json.loads((output/'registration.json').read_text())
    for path,digest in reg['source_sha256'].items():assert sha(ROOT/path)==digest
    obs=pd.read_csv(output/'training_observation_ledger.csv');arrays=np.load(output/'predictors.npz')
    original=joblib.load(parent/'fitted_candidate.joblib')
    history=pd.DataFrame(arrays['history'],columns=original['base']['history_names'])
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    frames={label:pd.read_csv(output/name) for label,name in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]}
    completed=[p for p in sorted(output.glob('*.joblib')) if p.name!='fitted_candidate.joblib']
    assert len(completed)==60 and all(len(f)==2017 for f in frames.values())
    for split in json.loads((output/'split_membership.json').read_text()):
        if not split['fold'].startswith('temporal'):
            index=obs.set_index('global_target_index')
            assert set(index.loc[split['train'],'site_id']).isdisjoint(index.loc[split['test'],'site_id'])
    amendment=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        reason='Outer training completed; reporting failed because spatial parent table lacks temporal-only history/site metadata',
        repair='Derive prior availability from unchanged H35; every held-site spatial row isnew to that diseasefit; no outcome-dependent grouping',
        outer_model_refits_performed=0,completed_fold_model_sha256={p.name:sha(p) for p in completed},
        completed_prediction_sha256={name:sha(output/name) for name in ['assessment_predictions.csv','spatial_predictions.csv']},
        amended_code_sha256={str(p.relative_to(ROOT)):sha(p) for p in [Path(__file__),Path(__file__).with_name('rf_sowing_experiment.py')]},
        original_registration_sha256=sha(output/'registration.json'))
    (output/'finalization_registration.json').write_text(json.dumps(amendment,indent=2)+'\n')
    for path in amendment['amended_code_sha256']:
        dest=output/'completion_code_snapshot'/path;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/path).read_bytes())
    metrics,groups,intervals=[],[],[]
    for label,frame in frames.items():
        h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        masks=assessment_populations(frame,h)
        for name in [NAME,'irish_station_candidate','reference_replay']+(['current'] if label=='temporal' else []):
            metrics.append(dict(evaluation=label,model=name,**score_predictions(frame,name)))
            if name!=NAME:
                lo,hi=paired_interval(frame,NAME,name);intervals.append(dict(evaluation=label,model=NAME,reference=name,lower=lo,upper=hi))
            for pop,mask in masks.items():
                groups.append(dict(evaluation=label,population=pop,model=name,**score_subset(frame,mask,name)))
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(output/'subgroup_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(output/'paired_intervals.csv',index=False)
    frame=frames['temporal'];h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
    masks=assessment_populations(frame,h);decisions={}
    for reference in ['irish_station_candidate','current','reference_replay']:
        delta={k:score_subset(frame,masks[k],NAME)['rmse']-score_subset(frame,masks[k],reference)['rmse'] for k in ['cold','warm','BASF','Corteva']}
        severe=abs(score_subset(frame,masks['severe_above75'],NAME)['bias'])-abs(score_subset(frame,masks['severe_above75'],reference)['bias'])
        ratio=score_subset(frame,masks['new_site'],NAME)['rmse']/score_subset(frame,masks['new_site'],reference)['rmse']
        result=promotion_decision(score_predictions(frame,reference),score_predictions(frame,NAME),paired_interval(frame,NAME,reference),independent=False,
            subgroup_deltas=delta,severe_bias_delta=severe,new_site_ratio=ratio)
        guards={k:v for k,v in result['gates'].items() if k!='independent_validation'}
        decisions[reference]=dict(internal_guards=guards,passes_current_cohort_guards=all(guards.values()),
            independent_external_validation=False,subgroup_rmse_deltas=delta,severe_absolute_bias_change=severe,new_site_rmse_ratio=ratio)
    incremental={}
    for label,frame in frames.items():
        a,b=[score_predictions(frame,n) for n in [NAME,'irish_station_candidate']]
        incremental[label+'_rmse_noninferiority']=a['rmse']<=b['rmse']
        incremental[label+'_integral_noninferiority']=a['integral_rmse']<=b['integral_rmse']
    (output/'assessment_decision.json').write_text(json.dumps(dict(references=decisions,incremental_parent_guards=incremental,
        independent_external_validation=False,evaluation_scope='User-authorized internallyheldcurrentcohort',publication_model_promoted=False),indent=2)+'\n')
    p=fit_cluster_ensemble(arrays['process'],history,obs,dict(structure='process',process_baseline_column=-3,crossleaf=True))
    first=fit_ensemble(arrays['station'],history,obs,dict(structure='field_split'))
    event=fit_ensemble(arrays['history_events'],history,obs,dict(structure='field_split'))
    full=dict(base=dict(original['base'],process=p,history=first),events_history=event,
        training_ids=obs.global_target_index.astype(int).tolist(),minimum_assessment_year=2020)
    joblib.dump(full,output/'fitted_candidate.joblib')
    contract=dict(created_utc=datetime.now(timezone.utc).isoformat(),recipe=NAME,
        inference='analysis.disease_improvement.history_interval_hybrid.predict',training_records=2304,minimum_assessment_year=2020,
        sowing_input='outputs/disease_improvement/sowing_rf_20261022/fitted_sowing_rf.joblib',sowing_observed=False,
        estimated_sowing_domain='GermanwinterwheatRF; non-German/latitude/temperaturetransferexplicitlyflagged',
        endpoint='Retaineduntreatedleafpercentage; EPPOalignmentconvention,Cortevamappingassumed',
        independent_external_validation=False,current_cohort_validation_authorized=True,publication_model_promoted=False,
        sha256={name:sha(output/name) for name in ['registration.json','finalization_registration.json','predictors.npz','training_observation_ledger.csv','split_membership.json','fitted_candidate.joblib']})
    (output/'fitted_candidate_lock.json').write_text(json.dumps(contract,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):finalize()
