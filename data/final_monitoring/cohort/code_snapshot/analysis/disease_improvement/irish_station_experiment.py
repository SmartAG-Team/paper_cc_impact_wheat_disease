"""Replay the frozen conditional-head recipe with qualified Irish precipitation."""
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
from .station_candidate import load_station_design

ROOT=Path(__file__).resolve().parents[2]
NAME='irish_station_candidate'


def run(root=ROOT,output=None):
    root=Path(root).resolve()
    data=root/'data/disease_improvement/station_preferred_ie_20261022'
    base=root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    parent=root/'outputs/disease_improvement/history_interval_20261022'
    out=Path(output or root/'outputs/disease_improvement/irish_station_candidate_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    assert (data/'READY.json').exists() and (data/'manifest.json').exists()
    manifest=json.loads((data/'manifest.json').read_text())
    for key in ['source_sha256','code_sha256','output_sha256']:
        for path,digest in manifest[key].items():
            location=data/path if key=='output_sha256' else root/path
            assert sha(location)==digest,path
    obs=pd.read_csv(base/'training_observation_ledger.csv')
    arrays=np.load(data/'predictors.npz')
    old_arrays=np.load(base/'predictors.npz')
    station=load_station_design(data/'predictors.npz',obs.global_target_index.to_numpy())
    event_station=arrays['history_events']
    assert event_station.shape==(len(obs),68) and np.isfinite(event_station).all()
    np.testing.assert_array_equal(arrays['process'],old_arrays['process'])
    np.testing.assert_array_equal(arrays['history'],old_arrays['history'])
    original=joblib.load(parent/'fitted_candidate.joblib')
    history=pd.DataFrame(arrays['history'],columns=original['base']['history_names'])
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    splits=json.loads((base/'split_membership.json').read_text())
    sources=[data/n for n in ['manifest.json','READY.json','predictors.npz','daily_weather.parquet']]
    sources += [base/n for n in ['training_observation_ledger.csv','predictors.npz','split_membership.json']]
    sources += [parent/n for n in ['assessment_predictions.csv','spatial_predictions.csv','verification.json',
                                  'fitted_candidate.joblib','fitted_candidate_lock.json']]
    code=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in
        ['models','robust_models','history','field_history_models','history_interval_hybrid','infection_event_experiment','evaluation','station_candidate']]
    code += [root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        changed='Only new independently qualified Irish station precipitation; same conditional-head recipe',
        weather_selection='All 730 strictly eligible measured station total-precipitation days across five field-seasons; no source selection by model errors or performance',
        liquid_precipitation='Existing phase-assisted conversion or explicit fallback; total precipitation is not relabeled measured liquid rain',
        meteorology='Original temperature/humidity and all other nonprecipitation numerical inputs preserved',
        recipe='Original 37-column reanalysis process component; 53-column first-field station head; 68-column event-station head when any strictly earlier field history exists; final 50:50',
        component_history_spec=dict(structure='field_split'),
        ensemble=dict(members=20,fraction=.8,seed=20261022),
        data_quality_correction_independent_of_model_performance=True,
        environment={name:version(name) for name in ['numpy','pandas','scikit-learn','joblib']},
        hyperparameter_search=False,independent_external_validation=False,publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)):sha(p) for p in code})
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for path in code:
        dest=out/'code_snapshot'/path.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(path.read_bytes())
    for name in ['training_observation_ledger.csv','split_membership.json']:
        (out/name).write_bytes((base/name).read_bytes())
    (out/'predictors.npz').write_bytes((data/'predictors.npz').read_bytes())
    frames={key:pd.read_csv(parent/name).set_index('global_target_index') for key,name in
            [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]}
    parts=dict(temporal=[],spatial_50km=[])
    for split in splits:
        fold=split['fold']
        train,test=[position.loc[split[key]].to_numpy() for key in ['train','test']]
        assert obs.iloc[train].season_year.max()<obs.iloc[test].season_year.min()
        first=fit_ensemble(station[train],history.iloc[train],obs.iloc[train],dict(structure='field_split'))
        event=fit_ensemble(event_station[train],history.iloc[train],obs.iloc[train],dict(structure='field_split'))
        for name,bundle in [('first',first),('event',event)]:
            joblib.dump(bundle,out/f'{fold}_{name}.joblib')
        query=obs.iloc[test].drop(columns='observed_percent')
        old=predict_ensemble(first,station[test],history.iloc[test],query)
        new=predict_ensemble(event,event_station[test],history.iloc[test],query)
        selected,mask=history_prediction(old,new,history.iloc[test],query)
        label='temporal' if fold.startswith('temporal') else 'spatial_50km'
        part=frames[label].loc[split['test']].reset_index().copy()
        part['ie_first_component'],part['ie_event_component']=old,new
        part['ie_selected_history'],part['ie_uses_event_head']=selected,mask
        part[NAME]=.5*(part.candidate_process+selected)
        parts[label].append(part)
        print('fitted',fold,flush=True)
    metrics,intervals,subgroups=[],[],[]
    results={label:pd.concat(records,ignore_index=True) for label,records in parts.items()}
    for label,filename in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]:
        frame=results[label]
        frame.to_csv(out/filename,index=False)
        for name in [NAME,'history_interval','reference_replay']+(['current'] if label=='temporal' else []):
            metrics.append(dict(evaluation=label,model=name,**score_predictions(frame,name)))
            if name!=NAME:
                low,high=paired_interval(frame,NAME,name)
                intervals.append(dict(evaluation=label,model=NAME,reference=name,lower=low,upper=high))
        if label=='temporal':
            h=history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
            masks=populations(frame,h)
            for population,mask in masks.items():
                for name in [NAME,'history_interval','reference_replay']:
                    subgroups.append(dict(population=population,model=name,**score_subset(frame,mask,name)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    pd.DataFrame(subgroups).to_csv(out/'subgroup_metrics.csv',index=False)
    first=fit_ensemble(station,history,obs,dict(structure='field_split'))
    event=fit_ensemble(event_station,history,obs,dict(structure='field_split'))
    full_base=dict(original['base'],history=first)
    fitted=dict(base=full_base,events_history=event,training_ids=original['training_ids'],minimum_assessment_year=2020)
    joblib.dump(fitted,out/'fitted_candidate.joblib')
    contract=json.loads((parent/'fitted_candidate_lock.json').read_text())
    contract.update(created_utc=datetime.now(timezone.utc).isoformat(),recipe=NAME,
        role='Qualified Irish station precipitation addition under the frozen conditional-head development recipe',
        inference='analysis.disease_improvement.history_interval_hybrid.predict',
        parent_contract_sha256=sha(parent/'fitted_candidate_lock.json'),
        station_input_manifest=str((data/'manifest.json').relative_to(root)),
        canonical_weather_quality_does_not_depend_on_model_score=True,
        sha256={name:sha(out/name) for name in ['registration.json','predictors.npz','fitted_candidate.joblib',
                                             'training_observation_ledger.csv','split_membership.json']})
    (out/'fitted_candidate_lock.json').write_text(json.dumps(contract,indent=2)+'\n')
    frame=results['temporal'];decisions={}
    for reference in ['history_interval','current','reference_replay']:
        deltas={key:score_subset(frame,masks[key],NAME)['rmse']-score_subset(frame,masks[key],reference)['rmse'] for key in ['cold','warm','BASF','Corteva']}
        severe=abs(score_subset(frame,masks['severe_above75'],NAME)['bias'])-abs(score_subset(frame,masks['severe_above75'],reference)['bias'])
        ratio=score_subset(frame,masks['new_site'],NAME)['rmse']/score_subset(frame,masks['new_site'],reference)['rmse']
        decisions[reference]=promotion_decision(score_predictions(frame,reference),score_predictions(frame,NAME),
            paired_interval(frame,NAME,reference),independent=False,subgroup_deltas=deltas,severe_bias_delta=severe,new_site_ratio=ratio)
        decisions[reference].update(subgroup_rmse_deltas=deltas,severe_absolute_bias_change=severe,new_site_rmse_ratio=ratio)
    (out/'adoption_decision.json').write_text(json.dumps(dict(references=decisions,
        independent_external_validation=False,publication_model_promoted=False,
        qualified_station_weather_retained_independent_of_model_score=True),indent=2)+'\n')
    return pd.DataFrame(metrics)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output).to_string(index=False))
