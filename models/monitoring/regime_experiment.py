"""Registered predictor-routed disease-regime challenger."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .regime_model import fit_regime,predict_regime
from .weather_sequence_model import design,_baseline
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess
from .evaluation import score_predictions
from .inputs import sha

NAME='disease_regime_mixture'


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/disease_regime_20261022').resolve()
    config=dict(iterations=4,max_iter=160,min_samples_leaf=8,members=10,fraction=.8,seed=20261022)
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['regime_model',
        'weather_sequence_model','dynamic_model','field_history_models']]
    register(out,NAME,dict(config=config,architecture='Three residual severity experts jointly iterated with three predictor-dependent probability regressors',
        initialization='Outer-training labels<=10,10-50,>50%; no assessment label used for routing',
        transitions='Four mixture responsibility/expert/gate iterations;8pp sigma floor;3% responsibility refresh',
        loss='Original hierarchy-weighted fitting; no severe-case reweighting',
        output='Full-range expected severity, no fixed process blend',
        selection='Fixed recipe; ten80%whole-coordinate-year members; temporal reference-improvement screen'),codes)
    obs,arrays,history,frames,splits,position=load_cohort();parts={};records=[];started=time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},NAME,out)
            if score_predictions(frame,NAME)['rmse']>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Fails the registered temporal reference-improvement screen'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];test=obs.iloc[vi];q=test.drop(columns='observed_percent')
        assert train.season_year.max()<test.season_year.min()
        ta={k:arrays[k][ti] for k in ['process','history_events']};va={k:arrays[k][vi] for k in ta}
        x,anchor,available=design(ta,history.iloc[ti],train.drop(columns='observed_percent'))
        xv,av,hv=design(va,history.iloc[vi],q)
        fitted=dict(members=[],config=config,train_ids=split['train'],test_ids=split['test'])
        clusters=np.array(sorted(train.coordinate_year.unique()));rng=np.random.default_rng(config['seed'])
        print('start',split['fold'],flush=True)
        for member in range(config['members']):
            selected=rng.choice(clusters,max(1,int(np.ceil(config['fraction']*len(clusters)))),replace=False)
            ix=np.flatnonzero(train.coordinate_year.isin(selected));sub=train.iloc[ix]
            w=hierarchical_weights(sub);w=w/w.mean();y=sub.observed_percent.to_numpy(float)
            base=dict(pooled=float(np.average(y/100,weights=w)),means={})
            category=sub.source.astype(str)+'|'+sub.leaf_index.astype(str)
            for key in category.unique():
                mask=category.eq(key).to_numpy()
                base['means'][key]=float((np.sum(w[mask]*y[mask]/100)+10*base['pooled'])/(w[mask].sum()+10))
            b=100*_baseline(base,anchor[ix],available[ix],sub)
            fitted['members'].append(dict(base=base,model=fit_regime(x[ix],y,b,w,config),
                training_ids=sub.global_target_index.astype(int).tolist(),clusters=selected.tolist()))
        path=out/(split['fold']+'_model.joblib');joblib.dump(fitted,path)
        def forecast(bundle):
            predictions=[];probabilities=[]
            for member in bundle['members']:
                bv=100*_baseline(member['base'],av,hv,q)
                prediction,prob=predict_regime(member['model'],xv,bv,True)
                predictions.append(prediction);probabilities.append(prob)
            return np.mean(predictions,axis=0),np.mean(probabilities,axis=0)
        value,prob=forecast(fitted);again,_=forecast(joblib.load(path));np.testing.assert_array_equal(value,again)
        part=frames[label].loc[split['test']].reset_index().copy();part[NAME]=value
        for k in range(3):part[NAME+'_prob'+str(k)]=prob[:,k]
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            traces=[member['model']['trace'] for member in fitted['members']]))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'test_rmse',round(score_predictions(part,NAME)['rmse'],3),flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,decision=assess(results,NAME,out);reg=json.loads((out/'registration.json').read_text())
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,fitted_splits=len(records),
        sources_unchanged=all(sha(ROOT/p)==h for p,h in reg['source_sha256'].items()),goal_achieved=decision['achieved'],
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output)
