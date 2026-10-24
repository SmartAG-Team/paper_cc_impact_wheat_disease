"""Nonlinear causal error-state correction with honest training baselines."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .error_dynamics import error_states,fit_error_head,predict_error_head
from .discrepancy_experiment import forecast_baseline,CACHE
from .weather_sequence_model import design
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY='nonlinear_error_state';NAMES=[PRIMARY,'nonlinear_error_state_half'];CONTROL='error_state_unfiltered'


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/error_dynamics_20261022').resolve()
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['error_dynamics',
        'discrepancy_experiment','discrepancy_filter','crossfit_router','field_history_models',
        'models','robust_models','history_interval_hybrid','weather_sequence_model','dynamic_model']]
    registration=register(out,PRIMARY,dict(hypothesis='Nonlinear conditional persistence/sign/trend of forecast innovations can improve on stationaryGaussian discrepancy andstaticfieldclasses',
        error_state='47strictlyearlierfield/native-codeerror features: availability,counts,ages,latest/mean/std/trend,7/21/42dayEWMA,per-codeerrorcontext',
        baseline='Originalstrongprocess/historyreference; globalcorrectortraininguseshonestinnercoordinateheldoutbaselineforecasts',
        features='Original145causalpredictors+47errorstates+currentbaselinefraction;193columns',
        corrector='20meta-members80%coordinateyear,seed20261022;300iterationdepth2HGB,minleaf12,L2=10,lr.03; fitwarmqueriesonly',
        head_population='Sampledcoordinateyear membership applies toerrorhead only; cached baselineexperts remain fitted on originaloutertraining population',
        weighting='Originalhierarchyweightswithin sampledpopulation, then warm-head scalar normalization; assessmentweights unchanged',
        formulations=['Fullcorrection','Halfcorrection'],
        causal_order='Pastforecastinnovationrequirespastactualobservation only; currentdayallleafoutcomeswithheld; firstvisitexactbaseline',
        source_codes='Errorstateusesnominalnativecodeequality; inheritedbaselinecodeproximity remainsunvalidated',
        climate='Monitoringonly; no-observationforecast equalsreference',
        selection='Fixedformulations; spatial iff either improves temporalreference; controlcannotadvance',adaptive_extension=True),codes)
    for path in [CACHE/'registration.json',*sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort();parts={};records=[];started=time.monotonic()
    raw,_,_=design(arrays,history,obs.drop(columns='observed_percent'))
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=(NAMES[1],CONTROL))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES)>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both fixed nonlinearinovationcorrections fail temporalreferenceimprovement'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        prior=joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids']==split['train'] and prior['test_ids']==split['test']
        oof=.5*(prior['inner_predictions'][:,0]+prior['inner_predictions'][:,1])
        state,names=error_states(train,oof);assert state.shape[1]==47
        base={k:prior['experts'][k] for k in ['process','first','event','training_ids']}
        clusters=np.array(sorted(train.coordinate_year.unique()));rng=np.random.default_rng(20261022)
        fitted=dict(base=base,members=[],train_ids=split['train'],test_ids=split['test'],state_names=names,
            inner_memberships=prior['inner_memberships'],inner_predictions=oof,inner_states=state)
        for _ in range(20):
            selected=rng.choice(clusters,max(1,int(np.ceil(.8*len(clusters)))),replace=False)
            ix=np.flatnonzero(train.coordinate_year.isin(selected));w=hierarchical_weights(train.iloc[ix])
            fitted['members'].append(dict(model=fit_error_head(raw[ti[ix]],state[ix],train.iloc[ix],oof[ix],w),
                training_ids=train.iloc[ix].global_target_index.astype(int).tolist(),clusters=selected.tolist(),training_weight=w))
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)
        ci=np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qi=pd.Series(np.arange(len(ci)),index=ci).loc[vi].to_numpy()
        def forecast(bundle):
            baseline=forecast_baseline(bundle['base'],arrays,history,obs,ci)
            states,columns=error_states(obs.iloc[ci],baseline);assert columns==bundle['state_names']
            values={}
            for name,shrink in zip(NAMES,[1.,.5]):
                values[name]=np.mean([predict_error_head(m['model'],raw[ci],states,baseline,shrinkage=shrink)
                    for m in bundle['members']],axis=0)[qi]
            values[CONTROL]=baseline[qi];return values
        values=forecast(fitted);replay=forecast(joblib.load(path))
        for key in values:np.testing.assert_array_equal(values[key],replay[key])
        part=frames[label].loc[split['test']].reset_index().copy()
        for name in [*NAMES,CONTROL]:part[name]=values[name]
        parts.setdefault(label,[]).append(part);records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,PRIMARY,out,extra_models=(NAMES[1],CONTROL));groups=pd.read_csv(out/'subgroup_metrics.csv')
    decisions={name:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],name) for name in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,fitted_splits=len(records),
        source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output)
