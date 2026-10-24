"""Adaptive field-trajectory Gaussian-process experiment on retained splits."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .field_gp_model import fit_gp,predict_gp
from .weather_sequence_model import design,_baseline
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess
from .evaluation import score_predictions
from .inputs import sha

NAME='field_trajectory_gp'


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/field_gp_20261022').resolve()
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['field_gp_model',
        'weather_sequence_model','dynamic_model','field_history_models']]
    register(out,NAME,dict(architecture='Population RBF covariance + field OU-time × native-code compound-symmetry covariance',
        features='Retained145; outer-training weighted standardization and PCA<=16',
        mean='Strict earlier measured severity; training-source/native-code shrinkage fallback',
        hyperparameters='Six parameters by outer-training marginal likelihood only;L-BFGS-B40iterations;finite bounds',
        fitting='Study hierarchy weights enter training diagonal variance; pseudo-likelihood weights are not sampling denominators',
        conditioning='Earlier held-field observations only; covariance hyperparameters frozen; all same-day codes withheld',
        native_leaf_assumption='Learned positive categorical cross-code correlation; no anatomical distance or emergence claim',
        forecast_noise='Constant learned observation variance; no future observation counts or future hierarchy weights used',
        output='Direct full-range conditional percentage mean clipped0..100',
        adaptive_extension_after_four_model_families=True,
        selection='Fixed40optimization-iteration budget; spatial if temporal reference improves'),codes)
    obs,arrays,history,frames,splits,position=load_cohort();parts={};records=[];started=time.monotonic()
    allx,allanchor,allavailable=design(arrays,history,obs.drop(columns='observed_percent'))
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},NAME,out,extra_models=(NAME+'_no_field_update',))
            if score_predictions(frame,NAME)['rmse']>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Fails the registered temporal reference-improvement screen'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];test=obs.iloc[vi];q=test.drop(columns='observed_percent')
        assert train.season_year.max()<test.season_year.min()
        w=hierarchical_weights(train);w=w/w.mean();y=train.observed_percent.to_numpy(float)
        base=dict(pooled=float(np.average(y/100,weights=w)),means={})
        category=train.source.astype(str)+'|'+train.leaf_index.astype(str)
        for key in category.unique():
            mask=category.eq(key).to_numpy()
            base['means'][key]=float((np.sum(w[mask]*y[mask]/100)+10*base['pooled'])/(w[mask].sum()+10))
        baselines=100*_baseline(base,allanchor,allavailable,obs)
        print('start',split['fold'],'train',len(train),flush=True)
        fitted=fit_gp(allx[ti],train,baselines[ti],w,max_iter=40)
        fitted.update(base=base,fold=split['fold'],train_ids=split['train'],test_ids=split['test'])
        path=out/(split['fold']+'_model.joblib');joblib.dump(fitted,path)
        args=(allx[vi],q,baselines[vi],allx,obs,baselines)
        value=predict_gp(fitted,*args)
        np.testing.assert_array_equal(value,predict_gp(joblib.load(path),*args))
        part=frames[label].loc[split['test']].reset_index().copy();part[NAME]=value
        part[NAME+'_no_field_update']=predict_gp(fitted,*args,assimilate=False)
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            converged=fitted['converged'],convergence=fitted['convergence'],iterations=fitted['iterations'],
            parameters=fitted['theta'].tolist(),trace=fitted['trace']))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',round(score_predictions(part,NAME)['rmse'],3),'converged',fitted['converged'],flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,decision=assess(results,NAME,out,extra_models=(NAME+'_no_field_update',))
    reg=json.loads((out/'registration.json').read_text())
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,fitted_splits=len(records),
        optimizer_converged_splits=sum(r['converged'] for r in records),
        sources_unchanged=all(sha(ROOT/p)==h for p,h in reg['source_sha256'].items()),goal_achieved=decision['achieved'],
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output)
