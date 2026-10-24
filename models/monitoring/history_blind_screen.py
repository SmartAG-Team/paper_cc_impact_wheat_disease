"""Test a history-blind first-visit estimator fitted on all earlier assessments."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval,score_predictions,score_subset
from .history_blind_model import fit_blind,predict_blind
from .infection_event_experiment import populations
from .inputs import sha

ROOT=Path(__file__).resolve().parents[2]
NAME='history_blind_first'


def run(root=ROOT,output=None):
    root=Path(root).resolve()
    parent=root/'data/final_monitoring/cohort'
    out=Path(output or root/'outputs/disease_improvement/history_blind_screen_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    obs=pd.read_csv(parent/'training_observation_ledger.csv')
    arrays=np.load(parent/'predictors.npz')
    parent_bundle=joblib.load(parent/'fitted_candidate.joblib')
    history=pd.DataFrame(arrays['history'],columns=parent_bundle['base']['history_names'])
    positions=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    original=pd.read_csv(parent/'assessment_predictions.csv')
    splits=[s for s in json.loads((parent/'split_membership.json').read_text()) if s['fold'].startswith('temporal')]
    sources=[parent/n for n in ['training_observation_ledger.csv','predictors.npz','fitted_candidate.joblib',
                               'assessment_predictions.csv','split_membership.json','verification.json','fitted_candidate_lock.json']]
    code=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['history_blind_model','models',
           'field_history_models','history','infection_event_experiment','evaluation']]
    code+=[root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        hypothesis='Restricting the first-visit estimator to initial visits loses useful earlier-year severity examples; withholding disease-history inputs allows every earlier assessment to train a first-visit model',
        change='Training representation only: all prior-year observations receive exactlyzero disease-history features for first-field head fitting and prediction',
        predictors='Unchanged53stationweather/stage/nativecode columns; no six-season-feature expansion',
        preserved='Every later-field prediction, process component, weather source, cohort, outer split, estimator, coordinate-year sample and50:50 weight',
        limitation='Training all assessments changes the first-visit response distribution; evaluation guards test whether this transfer is supported',
        model=dict(structure='field_split',members=20,fraction=.8,seed=20261022),
        temporal_screen_to_spatial='Strictly lower overall and first-field RMSE; integral noninferiority; severe absolute bias noninferiority; cold/warm/BASF/Corteva RMSE changes no greater than1point',
        posthoc_development=True,hyperparameter_search=False,independent_external_validation=False,publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)):sha(p) for p in code})
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for p in code:
        dest=out/'code_snapshot'/p.relative_to(root)
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(p.read_bytes())
    records=[]
    for split in splits:
        train,test=[positions.loc[split[key]].to_numpy() for key in ['train','test']]
        assert obs.iloc[train].season_year.max()<obs.iloc[test].season_year.min()
        fitted=fit_blind(arrays['station'][train],obs.iloc[train])
        joblib.dump(fitted,out/f"{split['fold']}_first.joblib")
        prediction=predict_blind(fitted,arrays['station'][test],obs.iloc[test].drop(columns='observed_percent'))
        part=original.set_index('global_target_index').loc[split['test']].reset_index().copy()
        h=history.iloc[test]
        first=(h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(0)).to_numpy()
        np.testing.assert_array_equal(first,~part.ie_uses_event_head.to_numpy())
        part['history_blind_component']=prediction
        part[NAME]=np.where(first,.5*(part.candidate_process.to_numpy()+prediction),part.irish_station_candidate)
        records.append(part)
        print('fitted',split['fold'],flush=True)
    frame=pd.concat(records,ignore_index=True)
    frame.to_csv(out/'assessment_predictions.csv',index=False)
    metrics=pd.DataFrame([dict(model=name,**score_predictions(frame,name)) for name in
                         [NAME,'irish_station_candidate','reference_replay','current']])
    metrics.to_csv(out/'metrics.csv',index=False)
    low,high=paired_interval(frame,NAME,'irish_station_candidate')
    (out/'paired_interval.json').write_text(json.dumps(dict(lower=low,upper=high,reference='irish_station_candidate'),indent=2)+'\n')
    h=history.iloc[positions.loc[frame.global_target_index]].reset_index(drop=True)
    masks=populations(frame,h)
    groups=pd.DataFrame([dict(population=key,model=name,**score_subset(frame,mask,name))
        for key,mask in masks.items() for name in [NAME,'irish_station_candidate']])
    groups.to_csv(out/'subgroup_metrics.csv',index=False)
    old,new=[score_predictions(frame,name) for name in ['irish_station_candidate',NAME]]
    gates=dict(overall_rmse_improves=new['rmse']<old['rmse'],
        first_field_rmse_improves=score_subset(frame,masks['first_field'],NAME)['rmse']<score_subset(frame,masks['first_field'],'irish_station_candidate')['rmse'],
        integral_noninferiority=new['integral_rmse']<=old['integral_rmse'],
        severe_bias_noninferiority=abs(score_subset(frame,masks['severe_above75'],NAME)['bias'])<=abs(score_subset(frame,masks['severe_above75'],'irish_station_candidate')['bias']))
    for key in ['cold','warm','BASF','Corteva']:
        gates[key+'_rmse_noninferiority']=score_subset(frame,masks[key],NAME)['rmse']-score_subset(frame,masks[key],'irish_station_candidate')['rmse']<=1.
    (out/'screen_decision.json').write_text(json.dumps(dict(proceed_to_spatial=all(gates.values()),gates=gates,
        independent_external_validation=False,publication_model_promoted=False),indent=2)+'\n')
    return metrics


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output).to_string(index=False))
