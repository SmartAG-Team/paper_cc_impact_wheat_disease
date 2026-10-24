"""Full-capacity local TabPFN with auditable hierarchy-weighted ICL contexts."""
import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .weighted_foundation_model import weighted_context
from .foundation_model import transform_target,restore_prediction,contextual_regressor
from .weather_sequence_model import design,_baseline
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY='tabpfn_full_weighted_anchor'
DIRECT='tabpfn_full_weighted_direct';CONTROL='tabpfn_full_unweighted_anchor';NAMES=[PRIMARY,DIRECT,CONTROL]


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/tabpfn_full_weighted_20261022').resolve()
    manifest=ROOT/'data/disease_improvement/tabpfn_weights_20261022/v35full_manifest.json'
    checkpoint=json.loads(manifest.read_text());assert sha(checkpoint['path'])==checkpoint['sha256']
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['weighted_foundation_model',
        'foundation_model','weather_sequence_model','dynamic_model','field_history_models']]
    registration=register(out,PRIMARY,dict(package_version=version('tabpfn'),checkpoint=checkpoint,
        checkpoint_version='v3.5',device=device,preprocessing_members=4,
        extension='Capacity and scientificweighting continuation of earlier fast/unweightedfoundationbenchmark; not an independentlyselectednewfamily',
        formulations=NAMES,control='Fullstandardcheckpoint, originalunweightedcontext, anchoredtarget',
        weighting='Threefixedsystematicimportance-resampledcontexts perweightedformulation,seed20261022+draw; expectedmultiplicityproportionaltooriginalhierarchyweights',
        context_size='min(2048,2*numberoforiginaloutertrainingobservations)',
        information_mass='Repeatedcontextrows encodeapproximateICLweights, notnewindependentobservations; alloriginalrows havepositiveeligibility; evaluatedsupport/weightsunchanged',
        targets='Directseverity andchangefromstrictlyearliermeasuredhistory/categoryprior; sameoriginal145features',
        tuning='No neuralweightfinetuning, calibration, epoch orouterhyperparameterselection; capacityandcontexts fixedbeforefit',
        replay='Allcontexts restoredandpredictions replaywithin.001pp; save originalIDs/multiplicities/scientificweights',
        native_codes='Inheritednumericcodeproximity remainsunvalidated, not verified anatomy',
        privacy='Localinferenceonly; no disease data uploaded',
        selection='Spatial iff anyprespecifiedformulation improves temporalreference',adaptive_internal_extension=True),codes)
    registration['source_sha256'][str(manifest.relative_to(ROOT))]=sha(manifest)
    registration['source_sha256'][str(Path(checkpoint['path']).relative_to(ROOT))]=checkpoint['sha256']
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort();parts={};records=[];started=time.monotonic()
    allx,anchor,available=design(arrays,history,obs.drop(columns='observed_percent'))
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=NAMES[1:])
            if min(score_predictions(frame,n)['rmse'] for n in NAMES)>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Allfixed fullcapacity/weightedICLformulations fail temporalreferenceimprovement'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];q=obs.iloc[vi].drop(columns='observed_percent')
        assert train.season_year.max()<obs.iloc[vi].season_year.min()
        w=hierarchical_weights(train);w/=w.mean();y=train.observed_percent.to_numpy(float)
        base=dict(pooled=float(np.average(y/100,weights=w)),means={})
        category=train.source.astype(str)+'|'+train.leaf_index.astype(str)
        for key in category.unique():
            mask=category.eq(key).to_numpy()
            base['means'][key]=float((np.sum(w[mask]*y[mask]/100)+10*base['pooled'])/(w[mask].sum()+10))
        baseline=100*_baseline(base,anchor[ti],available[ti],train)
        query_baseline=100*_baseline(base,anchor[vi],available[vi],q)
        part=frames[label].loc[split['test']].reset_index().copy();contexts=[]
        for name,kind,weighted in [(CONTROL,'anchor',False),(PRIMARY,'anchor',True),(DIRECT,'direct',True)]:
            predictions=[]
            for draw in range(3 if weighted else 1):
                target=transform_target(y,baseline,kind)
                if weighted:
                    cx,cy,ix=weighted_context(allx[ti],target,w,min(2048,2*len(train)),20261022+draw)
                else:cx,cy,ix=allx[ti],target,np.arange(len(train))
                counts=np.bincount(ix,minlength=len(train))
                bundle=dict(context_x=cx,context_y=cy,base=base,kind=kind,version='v3.5',
                    checkpoint=checkpoint['path'],checkpoint_sha256=checkpoint['sha256'],device=device,
                    train_ids=split['train'],test_ids=split['test'],context_original_positions=ix,
                    context_original_ids=train.global_target_index.to_numpy()[ix],context_multiplicity=counts,
                    original_scientific_weights=w,context_sampling_seed=20261022+draw,
                    sample_weight_api_supported=False,importance_sampling=weighted)
                path=out/f'{split["fold"]}_{name}_draw{draw}_context.joblib';joblib.dump(bundle,path)
                print('start',split['fold'],name,'draw',draw,'rows',len(ix),'device',device,flush=True)
                model=contextual_regressor(bundle)
                value=restore_prediction(np.asarray(model.predict(allx[vi]),float),query_baseline,kind)
                del model;gc.collect()
                if device=='mps':torch.mps.empty_cache()
                restored=contextual_regressor(joblib.load(path))
                replay=restore_prediction(np.asarray(restored.predict(allx[vi]),float),query_baseline,kind)
                error=float(np.max(abs(value-replay)));np.testing.assert_allclose(value,replay,rtol=0,atol=.001)
                del restored;gc.collect()
                if device=='mps':torch.mps.empty_cache()
                predictions.append(value)
                contexts.append(dict(formulation=name,draw=draw,rows=len(ix),unique_original_observations=int((counts>0).sum()),
                    total_variation_from_scientific_mass=float(.5*np.sum(abs(counts/counts.sum()-w/w.sum()))) if weighted else None,
                    replay_max_abs_pp=error,elapsed_seconds=time.monotonic()-started))
                print('context finished',split['fold'],name,draw,'seconds',round(time.monotonic()-started,1),'replay',error,flush=True)
            part[name]=np.mean(predictions,axis=0)
        parts.setdefault(label,[]).append(part);records.append(dict(fold=split['fold'],contexts=contexts))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('fold finished',split['fold'],{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,PRIMARY,out,extra_models=NAMES[1:]);groups=pd.read_csv(out/'subgroup_metrics.csv')
    decisions={name:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],name) for name in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,all_fitted_contexts_replay_within_pp=.001,
        fitted_splits=len(records),source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        checkpoint_sha256=checkpoint['sha256'],goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output)
