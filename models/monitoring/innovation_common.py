"""Common frozen-cohort registration, scoring and sequential objective gates."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd

from .evaluation import score_predictions,score_subset,paired_interval
from .inputs import sha

ROOT=Path(__file__).resolve().parents[2]
PARENT=ROOT/'data/final_monitoring/cohort'
DATA=ROOT/'data/disease_improvement/station_preferred_ie_20261022'
REFERENCE='irish_station_candidate'


def load_cohort():
    obs=pd.read_csv(PARENT/'training_observation_ledger.csv',float_precision='round_trip')
    arrays=dict(np.load(PARENT/'predictors.npz'))
    np.testing.assert_array_equal(arrays['target_ids'],obs.global_target_index)
    fitted=joblib.load(PARENT/'fitted_candidate.joblib')
    history=pd.DataFrame(arrays['history'],columns=fitted['base']['history_names'])
    frames={k:pd.read_csv(PARENT/v,float_precision='round_trip').set_index('global_target_index')
        for k,v in [('temporal','assessment_predictions.csv'),('spatial_50km','spatial_predictions.csv')]}
    splits=json.loads((PARENT/'split_membership.json').read_text())
    position=pd.Series(np.arange(len(obs)),index=obs.global_target_index)
    return obs,arrays,history,frames,splits,position


def register(out,method,recipe,codes):
    if out.exists():
        raise FileExistsError('A fresh experiment destination is required')
    sources=[PARENT/n for n in ['training_observation_ledger.csv','predictors.npz',
        'split_membership.json','assessment_predictions.csv','spatial_predictions.csv','fitted_candidate.joblib']]
    sources += [DATA/n for n in ['daily_weather.parquet','fields.csv','manifest.json']]
    codes=list(dict.fromkeys([Path(__file__),*map(Path,codes),Path(__file__).with_name('evaluation.py'),
        ROOT/'analysis/final_study/metrics.py']))
    out.mkdir(parents=True)
    registration=dict(method=method,registered_utc=datetime.now(timezone.utc).isoformat(),recipe=recipe,
        thresholds=dict(temporal_rmse=15.61851480377759,spatial_rmse=15.866048678379531,
            severe_rmse_and_absolute_bias='Lower than the retained reference in both designs'),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        current_qualified_untreated_cohort=True,adaptive_internal_development=True,
        independent_external_validation=False,external_validation_execution_prerequisite=False,
        publication_model_promoted=False)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for path in codes:
        destination=out/'code_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,destination)
    shutil.copyfile(PARENT/'split_membership.json',out/'split_membership.json')
    return registration


def goal_decision(metrics,severe,model,reference=REFERENCE):
    gates={}
    for design in ['temporal','spatial_50km']:
        a=metrics.loc[(metrics.evaluation==design)&(metrics.model==model)]
        b=metrics.loc[(metrics.evaluation==design)&(metrics.model==reference)]
        sa=severe.loc[(severe.evaluation==design)&(severe.model==model)]
        sb=severe.loc[(severe.evaluation==design)&(severe.model==reference)]
        available=all(len(x)==1 for x in [a,b,sa,sb])
        gates[design+'_ten_percent_rmse']=bool(available and a.rmse.iloc[0]<=.9*b.rmse.iloc[0])
        gates[design+'_severe_rmse_improves']=bool(available and sa.rmse.iloc[0]<sb.rmse.iloc[0])
        gates[design+'_severe_absolute_bias_improves']=bool(available and abs(sa.bias.iloc[0])<abs(sb.bias.iloc[0]))
    return dict(achieved=all(gates.values()),gates=gates,independent_external_validation=False)


def assess(frames,model,out,extra_models=()):
    metrics=[];groups=[];intervals=[]
    for design,frame in frames.items():
        if len(frame)!=2017 or not frame.global_target_index.is_unique:
            raise ValueError('Original complete assessment support required')
        for name in [model,REFERENCE,*extra_models]:
            metrics.append(dict(evaluation=design,model=name,**score_predictions(frame,name)))
            masks=dict(severe_above75=frame.observed_percent.gt(75),BASF=frame.source.eq('BASF'),
                Corteva=frame.source.eq('Corteva'),first_field=~frame.ie_uses_event_head.astype(bool),
                earlier_field=frame.ie_uses_event_head.astype(bool))
            for label,mask in masks.items():
                groups.append(dict(evaluation=design,model=name,population=label,**score_subset(frame,mask,name)))
        lo,hi=paired_interval(frame,model,REFERENCE)
        intervals.append(dict(evaluation=design,model=model,reference=REFERENCE,lower=lo,upper=hi))
    m=pd.DataFrame(metrics); g=pd.DataFrame(groups)
    m.to_csv(out/'metrics.csv',index=False);g.to_csv(out/'subgroup_metrics.csv',index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv',index=False)
    decision=goal_decision(m,g.loc[g.population.eq('severe_above75')],model)
    (out/'goal_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    print(m.to_string(index=False),flush=True)
    return m,decision
