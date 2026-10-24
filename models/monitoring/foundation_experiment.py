"""Pinned local TabPFN contexts with prespecified direct and anchored targets."""
import argparse
import gc
from importlib.metadata import version as package_version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .foundation_model import transform_target,restore_prediction,contextual_regressor
from .weather_sequence_model import design,_baseline
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha


def run(model_version='v2',output=None):
    short='v2' if model_version=='v2' else 'v35fast'
    out=Path(output or ROOT/f'outputs/disease_improvement/tabpfn_{short}_20261022').resolve()
    manifest=ROOT/f'data/disease_improvement/tabpfn_weights_20261022/{short}_manifest.json'
    checkpoint=json.loads(manifest.read_text());assert sha(Path(checkpoint['path']))==checkpoint['sha256']
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    primary=f'tabpfn_{short}_anchor';direct=f'tabpfn_{short}_direct'
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['foundation_model',
        'weather_sequence_model','dynamic_model','field_history_models']]
    register(out,primary,dict(package_version=package_version('tabpfn'),checkpoint=checkpoint,
        version=model_version,device=device,ensemble_preprocessing_members=4,random_state=20261022,
        target_formulations=['direct severity','change from measured-history/training-source-code anchor'],
        features='process37 + history_events68 + causal_history35 + anchor5 =145',
        fitting='Unweighted full outer-training ICL context; upstream fit(X,y) has no sample_weight',
        evaluation='Unchanged hierarchy weights; all original rows and splits',
        tuning='No checkpoint finetuning; no automatic temperature calibration; no target/epoch/hyperparameter selection',
        output='Mean prediction restored to percentage and clipped0..100; no fixed process blend',
        continuation='Spatial fitting if either prespecified formulation improves temporal reference'),codes)
    joblib.dump(checkpoint,out/'checkpoint_identity.joblib')
    obs,arrays,history,frames,splits,position=load_cohort();parts={};records=[];started=time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},primary,out,extra_models=(direct,))
            baseline=score_predictions(frame,REFERENCE)['rmse']
            if min(score_predictions(frame,n)['rmse'] for n in [primary,direct])>=baseline:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Neither prespecified formulation improves temporal reference'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];test=obs.iloc[vi];q=test.drop(columns='observed_percent')
        assert train.season_year.max()<test.season_year.min()
        x,anchor,available=design({k:arrays[k][ti] for k in ['process','history_events']},history.iloc[ti],train.drop(columns='observed_percent'))
        xv,av,hv=design({k:arrays[k][vi] for k in ['process','history_events']},history.iloc[vi],q)
        w=hierarchical_weights(train);w=w/w.mean();y=train.observed_percent.to_numpy(float)
        base=dict(pooled=float(np.average(y/100,weights=w)),means={})
        category=train.source.astype(str)+'|'+train.leaf_index.astype(str)
        for key in category.unique():
            mask=category.eq(key).to_numpy()
            base['means'][key]=float((np.sum(w[mask]*y[mask]/100)+10*base['pooled'])/(w[mask].sum()+10))
        b=100*_baseline(base,anchor,available,train);bv=100*_baseline(base,av,hv,q)
        part=frames[label].loc[split['test']].reset_index().copy()
        record=dict(fold=split['fold'],formulations=[])
        for kind,name in [('anchor',primary),('direct',direct)]:
            print('start',model_version,split['fold'],kind,'train',len(train),'features',x.shape[1],flush=True)
            bundle=dict(context_x=x,context_y=transform_target(y,b,kind),base=base,kind=kind,
                version=model_version,checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],device=device,
                train_ids=split['train'],test_ids=split['test'],context_weights_supported=False)
            path=out/(split['fold']+'_'+kind+'_context.joblib');joblib.dump(bundle,path)
            model=contextual_regressor(bundle)
            value=restore_prediction(np.asarray(model.predict(xv),float),bv,kind)
            del model;gc.collect()
            if device=='mps':torch.mps.empty_cache()
            restored=contextual_regressor(joblib.load(path))
            again=restore_prediction(np.asarray(restored.predict(xv),float),bv,kind)
            error=float(np.max(abs(value-again)));np.testing.assert_allclose(value,again,rtol=0,atol=.001)
            del restored;gc.collect()
            if device=='mps':torch.mps.empty_cache()
            part[name]=value
            record['formulations'].append(dict(kind=kind,replay_max_abs_pp=error,
                elapsed_seconds=time.monotonic()-started,rmse=score_predictions(part,name)['rmse']))
            print('finished',split['fold'],kind,'seconds',round(time.monotonic()-started,1),
                'test_rmse',round(score_predictions(part,name)['rmse'],3),'replay',error,flush=True)
        parts.setdefault(label,[]).append(part);records.append(record)
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,decision=assess(results,primary,out,extra_models=(direct,))
    subgroups=pd.read_csv(out/'subgroup_metrics.csv')
    decisions={name:goal_decision(metrics,subgroups.loc[subgroups.population.eq('severe_above75')],name) for name in [primary,direct]}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    reg=json.loads((out/'registration.json').read_text())
    (out/'verification.json').write_text(json.dumps(dict(passed=True,all_fitted_contexts_replay_within_pp=.001,
        fitted_splits=len(records),source_unchanged=all(sha(ROOT/p)==h for p,h in reg['source_sha256'].items()),
        checkpoint_sha256=checkpoint['sha256'],goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--version',choices=['v2','v3.5-fast'],default='v2')
    p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.version,args.output)
