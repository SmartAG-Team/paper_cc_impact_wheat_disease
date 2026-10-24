"""Adaptive conditional expert combination learned exclusively inside outer training."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .crossfit_router import coordinate_folds,fit_router,predict_router
from .field_history_models import fit_ensemble,predict_ensemble
from .history_interval_hybrid import history_prediction
from .robust_models import fit_cluster_ensemble,predict_cluster_ensemble
from .offset_growth_model import fit_growth,predict_growth
from .weather_sequence_model import design,_baseline
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY='crossfit_router';NAMES=[PRIMARY,'crossfit_residual_stack']
EXPERTS=['process','history','linear_growth','logit_growth','measured_anchor']


def fit_experts(arrays,history,obs,x,anchor,available,ix):
    train=obs.iloc[ix]
    fitted=dict(process=fit_cluster_ensemble(arrays['process'][ix],history.iloc[ix],train,
            dict(structure='process',process_baseline_column=-3,crossleaf=True)),
        first=fit_ensemble(arrays['station'][ix],history.iloc[ix],train,dict(structure='field_split')),
        event=fit_ensemble(arrays['history_events'][ix],history.iloc[ix],train,dict(structure='field_split')),
        growth=[],training_ids=train.global_target_index.astype(int).tolist())
    clusters=np.asarray(sorted(train.coordinate_year.unique()));rng=np.random.default_rng(20261022)
    for _ in range(20):
        selected=rng.choice(clusters,max(1,int(np.ceil(.8*len(clusters)))),replace=False)
        idx=ix[np.flatnonzero(train.coordinate_year.isin(selected))];sub=obs.iloc[idx]
        y=sub.observed_percent.to_numpy(float);w=hierarchical_weights(sub);w/=w.mean()
        base=dict(pooled=float(np.average(y/100,weights=w)),means={})
        category=sub.source.astype(str)+'|'+sub.leaf_index.astype(str)
        for key in category.unique():
            mask=category.eq(key).to_numpy()
            base['means'][key]=float((np.sum(w[mask]*y[mask]/100)+10*base['pooled'])/(w[mask].sum()+10))
        b=100*_baseline(base,anchor[idx],available[idx],sub)
        fitted['growth'].append(dict(base=base,
            models={link:fit_growth(x[idx],y,b,w,link=link) for link in ['linear','logit']},
            training_ids=sub.global_target_index.astype(int).tolist(),clusters=selected.tolist()))
    return fitted


def expert_predictions(fitted,arrays,history,obs,x,anchor,available,ix):
    q=obs.iloc[ix].drop(columns='observed_percent');h=history.iloc[ix]
    p=predict_cluster_ensemble(fitted['process'],arrays['process'][ix],h,q)
    first=predict_ensemble(fitted['first'],arrays['station'][ix],h,q)
    event=predict_ensemble(fitted['event'],arrays['history_events'][ix],h,q)
    history_pred,_=history_prediction(first,event,h,q)
    values={link:[] for link in ['linear','logit']};baselines=[]
    for member in fitted['growth']:
        b=100*_baseline(member['base'],anchor[ix],available[ix],q);baselines.append(b)
        for link in values:values[link].append(predict_growth(member['models'][link],x[ix],b))
    return np.column_stack([p,history_pred,np.mean(values['linear'],axis=0),
        np.mean(values['logit'],axis=0),np.mean(baselines,axis=0)])


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/crossfit_router_20261022').resolve()
    state_path=ROOT/'outputs/disease_improvement/leaf_transition_20261022/state_predictors.npz'
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['crossfit_router',
        'offset_growth_model','field_history_models','models','robust_models','history_interval_hybrid',
        'weather_sequence_model','dynamic_model']]
    registration=register(out,PRIMARY,dict(hypothesis='Conditional skill varies by query; learn combining weights on honest inner-coordinate-held-out predictions',
        outer_splits='Original20, historicaltrainingonly; all2017assessmentrows retained',
        inner_splits='Five deterministic coordinate groups, all years of each coordinate held together; cross-sectional innerfits within historical outertraining, not a second temporal validation',
        experts=EXPERTS,expert_recipe='Original20member80%coordinateyear process/first/eventHGB;400treeweightedlinear/logitgrowth; measuredhistory/categoryprior',
        router='27standardizedfeatures(history14,anchor5,process3,expert5); softmaxlinear gate weightedMSE+L2=.001, L-BFGS-B300',
        residual_stack='Original158causalfeatures plusfiveinnerheldpredictions; HGB200depth2,minleaf20,L2=10,lr.03; residualaboutprocess/historymean',
        predictions='Fully refit experts on outertraining; meta-estimators fitted only toinnerheld-out error; querylabels unavailable',
        weighting='Originalhierarchyweights for expert and meta fits; original assessment weights',
        source_codes='Inherited nearest numericcode anchor/distances remain unvalidated proxies; no anatomical equivalence',
        selection='Two fixedformulations; spatial onlyif either improves temporal reference',
        adaptive_extension=True),codes)
    registration['source_sha256'][str(state_path.relative_to(ROOT))]=sha(state_path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort()
    raw,anchor,available=design(arrays,history,obs.drop(columns='observed_percent'))
    state=np.load(state_path);np.testing.assert_array_equal(state['target_ids'],obs.global_target_index)
    x=np.column_stack([raw,state['states']])
    # Compact routing context; no target label enters either query matrix.
    context=np.column_stack([history.to_numpy()[:,:14],raw[:,-5:],arrays['process'][:,-3:]])
    parts={};records=[];started=time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=NAMES[1:])
            if min(score_predictions(frame,n)['rmse'] for n in NAMES)>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both fixed inner-crossfit combination methods fail temporal reference improvement'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        oof=np.full((len(ti),len(EXPERTS)),np.nan);inner_memberships=[]
        print('start',split['fold'],flush=True)
        for k,(a,b) in enumerate(coordinate_folds(train)):
            fit=fit_experts(arrays,history,obs,x,anchor,available,ti[a])
            oof[b]=expert_predictions(fit,arrays,history,obs,x,anchor,available,ti[b])
            joblib.dump(fit,out/f'{split["fold"]}_inner{k}_experts.joblib')
            inner_memberships.append(dict(inner_fold=k,train=train.iloc[a].global_target_index.astype(int).tolist(),
                test=train.iloc[b].global_target_index.astype(int).tolist()))
            print('inner',split['fold'],k,'seconds',round(time.monotonic()-started,1),flush=True)
        assert np.isfinite(oof).all()
        w=hierarchical_weights(train);y=train.observed_percent.to_numpy(float)
        router=fit_router(np.column_stack([context[ti],oof/100]),oof,y,w)
        stack=HistGradientBoostingRegressor(max_iter=200,max_depth=2,max_leaf_nodes=7,
            min_samples_leaf=20,learning_rate=.03,l2_regularization=10,early_stopping=False,random_state=20261022)
        stack.fit(np.column_stack([x[ti],oof/100]),(y-.5*(oof[:,0]+oof[:,1]))/100,sample_weight=w/w.mean())
        outer=fit_experts(arrays,history,obs,x,anchor,available,ti)
        fitted=dict(experts=outer,router=router,stack=stack,train_ids=split['train'],test_ids=split['test'],
            inner_memberships=inner_memberships,inner_predictions=oof,inner_weight=w)
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)
        def forecast(bundle):
            p=expert_predictions(bundle['experts'],arrays,history,obs,x,anchor,available,vi)
            gate,prob=predict_router(bundle['router'],np.column_stack([context[vi],p/100]),p)
            residual=np.clip(.5*(p[:,0]+p[:,1])+100*bundle['stack'].predict(np.column_stack([x[vi],p/100])),0,100)
            return gate,residual,p,prob
        values=forecast(fitted);replay=forecast(joblib.load(path))
        for a,b in zip(values,replay):np.testing.assert_array_equal(a,b)
        part=frames[label].loc[split['test']].reset_index().copy()
        for name,value in zip(NAMES,values[:2]):part[name]=value
        for k,name in enumerate(EXPERTS):part['router_expert_'+name]=values[2][:,k];part['router_weight_'+name]=values[3][:,k]
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,router_converged=router['converged']))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,PRIMARY,out,extra_models=NAMES[1:]);groups=pd.read_csv(out/'subgroup_metrics.csv')
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
