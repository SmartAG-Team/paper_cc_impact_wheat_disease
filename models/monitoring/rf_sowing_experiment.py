"""Refit the fixed disease recipe with RF-estimated winter-wheat sowing."""
import argparse
from datetime import datetime,timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval,promotion_decision,score_predictions,score_subset
from .field_history_models import fit_ensemble,predict_ensemble
from .history_interval_hybrid import history_prediction
from .infection_event_experiment import populations
from .inputs import sha
from .robust_models import fit_cluster_ensemble,predict_cluster_ensemble
from .rf_sowing_finalize import assessment_populations

ROOT=Path(__file__).resolve().parents[2]
NAME='rf_sowing_candidate'
PROCESS_SPEC=dict(structure='process',process_baseline_column=-3,crossleaf=True)


def run(output=None):
    parent=ROOT/'data/final_monitoring/cohort'
    data=ROOT/'data/disease_improvement/rf_sowing_20261022'
    output=Path(output or ROOT/'outputs/disease_improvement/rf_sowing_candidate_20261022').resolve()
    if output.exists():
        raise FileExistsError('Fresh disease experiment destination required')
    obs=pd.read_csv(data/'training_observation_ledger.csv')
    old_obs=pd.read_csv(parent/'training_observation_ledger.csv')
    pd.testing.assert_frame_equal(obs,old_obs)
    arrays=np.load(data/'predictors.npz');original=np.load(parent/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    np.testing.assert_array_equal(arrays['history'],original['history'])
    assert len(obs)==2304 and obs.global_target_index.is_unique
    for key,width in [('process',37),('station',53),('history_events',68)]:
        assert arrays[key].shape==(2304,width) and np.isfinite(arrays[key]).all()
    original_bundle=joblib.load(parent/'fitted_candidate.joblib')
    history=pd.DataFrame(arrays['history'],columns=original_bundle['base']['history_names'])
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    splits=json.loads((parent/'split_membership.json').read_text())
    output.mkdir(parents=True)
    sources=[data/n for n in ['registration.json','baseline_reproduction.json','predictors.npz',
        'training_observation_ledger.csv','fields.csv','leaf_scoring_convention.csv']]
    sources += [parent/n for n in ['assessment_predictions.csv','spatial_predictions.csv','split_membership.json',
        'fitted_candidate.joblib','fitted_candidate_lock.json','verification.json']]
    sources += [ROOT/'outputs/disease_improvement/sowing_rf_20261022'/n for n in ['registration.json','fitted_sowing_rf.joblib','field_estimated_sowing.csv']]
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['models','robust_models','field_history_models',
        'history_interval_hybrid','infection_event_experiment','evaluation','history','rf_sowing_finalize']]
    codes += [ROOT/'analysis/final_study/metrics.py']
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Estimated sowing from independent earlier observed management improves seasonal clocks/process states relative to fixed country calendars',
        recipe='Original process37 plus station53first-field/event68earlier-field heads,20whole-coordinate-year80percent subsamples each,50:50finalweight',
        process_spec=PROCESS_SPEC,history_spec=dict(structure='field_split'),members=20,fraction=.8,seed=20261022,
        disease_training_and_evaluation_observations_preserved=True,existing_outer_splits_preserved=True,
        information='Realizedweather throughpreviousday; strictlyearlierfielddisease; pre2013sowingRF shared globally',
        sowing='Estimated, notobserved; previouscalendar-yearmeanT+latitude; German-to-Europe andweatherproducttransfer flags retained',
        eppo_convention='Leafordinal=index+1; BASFflagverified, Cortevaalignmentassumed; cultivarresistanceunreported',
        user_authorized_evaluation='Current retained dateduntreated dataset; internal chronology and50km diseasefitexclusion',
        external_validation_required_for_this_internal_experiment=False,
        independent_external_validation=False,posthoc_development=True,hyperparameter_search=False,
        internal_acceptance='Original accuracy/source/history/integral/severe/newsite guards; independent external gate separately recorded, not an execution prerequisite',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        environment={n:version(n) for n in ['numpy','pandas','scikit-learn','joblib']})
    (output/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for path in codes:
        dest=output/'code_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(path.read_bytes())
    for name in ['training_observation_ledger.csv','predictors.npz']:
        (output/name).write_bytes((data/name).read_bytes())
    (output/'split_membership.json').write_bytes((parent/'split_membership.json').read_bytes())
    frames={label:pd.read_csv(parent/name).set_index('global_target_index') for label,name in
        [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]}
    records=dict(temporal=[],spatial_50km=[])
    for split in splits:
        train,test=[position.loc[split[k]].to_numpy() for k in ['train','test']]
        assert obs.iloc[train].season_year.max()<obs.iloc[test].season_year.min()
        p=fit_cluster_ensemble(arrays['process'][train],history.iloc[train],obs.iloc[train],PROCESS_SPEC)
        first=fit_ensemble(arrays['station'][train],history.iloc[train],obs.iloc[train],dict(structure='field_split'))
        event=fit_ensemble(arrays['history_events'][train],history.iloc[train],obs.iloc[train],dict(structure='field_split'))
        for head,bundle in [('process',p),('first',first),('event',event)]:
            joblib.dump(bundle,output/(split['fold']+'_'+head+'.joblib'))
        q=obs.iloc[test].drop(columns='observed_percent');h=history.iloc[test]
        process=predict_cluster_ensemble(p,arrays['process'][test],h,q)
        a=predict_ensemble(first,arrays['station'][test],h,q)
        b=predict_ensemble(event,arrays['history_events'][test],h,q)
        selected,mask=history_prediction(a,b,h,q)
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        part=frames[label].loc[split['test']].reset_index().copy()
        part['rf_process_component'],part['rf_first_component'],part['rf_event_component']=process,a,b
        part['rf_selected_history'],part['rf_uses_event_head']=selected,mask
        part[NAME]=.5*(process+selected)
        records[label].append(part)
        print('fitted',split['fold'],flush=True)
    results={label:pd.concat(rows,ignore_index=True) for label,rows in records.items()}
    metrics,groups,intervals=[],[],[]
    for label,frame in results.items():
        assert len(frame)==2017 and frame.global_target_index.is_unique
        frame.to_csv(output/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
        names=[NAME,'irish_station_candidate','reference_replay']+(['current'] if label=='temporal' else [])
        for name in names:
            metrics.append(dict(evaluation=label,model=name,**score_predictions(frame,name)))
            if name!=NAME:
                lo,hi=paired_interval(frame,NAME,name)
                intervals.append(dict(evaluation=label,model=NAME,reference=name,lower=lo,upper=hi))
        h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        masks=assessment_populations(frame,h)
        for pop,mask in masks.items():
            for name in names:
                groups.append(dict(evaluation=label,population=pop,model=name,**score_subset(frame,mask,name)))
    pd.DataFrame(metrics).to_csv(output/'metrics.csv',index=False)
    pd.DataFrame(groups).to_csv(output/'subgroup_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(output/'paired_intervals.csv',index=False)
    frame=results['temporal'];h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
    masks=assessment_populations(frame,h);decisions={}
    for reference in ['irish_station_candidate','current','reference_replay']:
        delta={key:score_subset(frame,masks[key],NAME)['rmse']-score_subset(frame,masks[key],reference)['rmse'] for key in ['cold','warm','BASF','Corteva']}
        severe=abs(score_subset(frame,masks['severe_above75'],NAME)['bias'])-abs(score_subset(frame,masks['severe_above75'],reference)['bias'])
        ratio=score_subset(frame,masks['new_site'],NAME)['rmse']/score_subset(frame,masks['new_site'],reference)['rmse']
        external=promotion_decision(score_predictions(frame,reference),score_predictions(frame,NAME),
            paired_interval(frame,NAME,reference),independent=False,subgroup_deltas=delta,severe_bias_delta=severe,new_site_ratio=ratio)
        internal={k:v for k,v in external['gates'].items() if k!='independent_validation'}
        decisions[reference]=dict(internal_guards=internal,passes_current_cohort_guards=all(internal.values()),
            independent_external_validation=False,subgroup_rmse_deltas=delta,severe_absolute_bias_change=severe,new_site_rmse_ratio=ratio)
    incremental={}
    for label in results:
        a,b=[score_predictions(results[label],n) for n in [NAME,'irish_station_candidate']]
        incremental[label+'_rmse_noninferiority']=a['rmse']<=b['rmse']
        incremental[label+'_integral_noninferiority']=a['integral_rmse']<=b['integral_rmse']
    (output/'assessment_decision.json').write_text(json.dumps(dict(references=decisions,
        incremental_parent_guards=incremental,independent_external_validation=False,
        evaluation_scope='User-authorized internallyheldcurrentcohort',publication_model_promoted=False),indent=2)+'\n')
    p=fit_cluster_ensemble(arrays['process'],history,obs,PROCESS_SPEC)
    first=fit_ensemble(arrays['station'],history,obs,dict(structure='field_split'))
    event=fit_ensemble(arrays['history_events'],history,obs,dict(structure='field_split'))
    base=dict(original_bundle['base'],process=p,history=first)
    fitted=dict(base=base,events_history=event,training_ids=obs.global_target_index.astype(int).tolist(),minimum_assessment_year=2020)
    joblib.dump(fitted,output/'fitted_candidate.joblib')
    contract=dict(created_utc=datetime.now(timezone.utc).isoformat(),recipe=NAME,
        inference='analysis.disease_improvement.history_interval_hybrid.predict',training_records=2304,
        minimum_assessment_year=2020,sowing_input='outputs/disease_improvement/sowing_rf_20261022/fitted_sowing_rf.joblib',
        sowing_observed=False,estimated_sowing_domain='GermanwinterwheatRF; non-German/latitude/temperaturetransferexplicitlyflagged',
        endpoint='Retaineduntreatedleafpercentage, EPPOalignmentconventionwithunverifiedCortevamappinglabelledassumed',
        independent_external_validation=False,current_cohort_validation_authorized=True,
        publication_model_promoted=False,
        sha256={name:sha(output/name) for name in ['registration.json','predictors.npz','training_observation_ledger.csv','split_membership.json','fitted_candidate.joblib']})
    (output/'fitted_candidate_lock.json').write_text(json.dumps(contract,indent=2)+'\n')
    return pd.DataFrame(metrics)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(args.output).to_string(index=False))
