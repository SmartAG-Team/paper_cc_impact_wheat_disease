"""Adaptive bounded-growth link experiment with preserved study weights."""
import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .offset_growth_model import fit_growth,predict_growth
from .weather_sequence_model import design,_baseline
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY='logit_growth';ALTERNATIVE='linear_growth'


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/offset_growth_20261022').resolve()
    state_path=ROOT/'outputs/disease_improvement/leaf_transition_20261022/state_predictors.npz'
    config=dict(members=20,fraction=.8,seed=20261022,rounds=400)
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['offset_growth_model',
        'weather_sequence_model','dynamic_model','field_history_models']]
    registration=register(out,PRIMARY,dict(config=config,lightgbm_version=version('lightgbm'),
        hypotheses=['Log-odds change about a causal disease anchor','Linear percentage change about the same anchor'],
        features='process37/event68/history35/anchor5/state13;158columns',
        source_code_proximity='Inherited nearest-native-numerical-code anchor and signed/distance features are unvalidated numerical proxies, not anatomical distances',
        logit_loss='Continuous-label crossentropy with baseline logit as explicit init_score; queryoffset restored exactlyonce',
        linear_loss='Original weighted squared error with explicit severityoffset',
        weighting='Original hierarchy weights within each sampledcoordinateyearpopulation',
        hyperparameters='Fixed400trees,15leaves,depth4,minleaf8,learningrate.03,L2=8,maxbin31,featurefraction.9',
        local_runtime='macOS LightGBM localwheel rpathpoints @loader_path/../../torch/lib to existinglibomp; adhocresigned, no globalpackagechange',
        adaptive_extension=True,selection='No outer tuning; spatial if either fixedformulation improves temporal reference'),codes)
    registration['source_sha256'][str(state_path.relative_to(ROOT))]=sha(state_path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort();parts={};records=[];started=time.monotonic()
    state=np.load(state_path);np.testing.assert_array_equal(state['target_ids'],obs.global_target_index)
    raw,anchor,available=design(arrays,history,obs.drop(columns='observed_percent'))
    x=np.column_stack([raw,state['states']]);assert x.shape==(2304,158)
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=(ALTERNATIVE,))
            if min(score_predictions(frame,n)['rmse'] for n in [PRIMARY,ALTERNATIVE])>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Neither fixedgrowthlink improves temporal reference'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];test=obs.iloc[vi];q=test.drop(columns='observed_percent')
        assert train.season_year.max()<test.season_year.min()
        clusters=np.array(sorted(train.coordinate_year.unique()));rng=np.random.default_rng(config['seed'])
        fitted=dict(members=[],config=config,train_ids=split['train'],test_ids=split['test'])
        print('start',split['fold'],flush=True)
        for _ in range(config['members']):
            selected=rng.choice(clusters,max(1,int(np.ceil(config['fraction']*len(clusters)))),replace=False)
            ix=ti[np.flatnonzero(train.coordinate_year.isin(selected))];sub=obs.iloc[ix]
            w=hierarchical_weights(sub);w=w/w.mean();y=sub.observed_percent.to_numpy(float)
            base=dict(pooled=float(np.average(y/100,weights=w)),means={})
            category=sub.source.astype(str)+'|'+sub.leaf_index.astype(str)
            for key in category.unique():
                mask=category.eq(key).to_numpy()
                base['means'][key]=float((np.sum(w[mask]*y[mask]/100)+10*base['pooled'])/(w[mask].sum()+10))
            b=100*_baseline(base,anchor[ix],available[ix],sub)
            fitted['members'].append(dict(base=base,
                models={link:fit_growth(x[ix],y,b,w,link=link,rounds=config['rounds']) for link in ['logit','linear']},
                training_ids=sub.global_target_index.astype(int).tolist(),clusters=selected.tolist()))
        path=out/(split['fold']+'_model.joblib');joblib.dump(fitted,path)
        def forecast(bundle):
            predictions={link:[] for link in ['logit','linear']}
            for member in bundle['members']:
                b=100*_baseline(member['base'],anchor[vi],available[vi],q)
                for link in predictions:predictions[link].append(predict_growth(member['models'][link],x[vi],b))
            return {link:np.mean(p,axis=0) for link,p in predictions.items()}
        values=forecast(fitted);replay=forecast(joblib.load(path))
        part=frames[label].loc[split['test']].reset_index().copy()
        for link,name in [('logit',PRIMARY),('linear',ALTERNATIVE)]:
            np.testing.assert_array_equal(values[link],replay[link]);part[name]=values[link]
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'logit',round(score_predictions(part,PRIMARY)['rmse'],3),'linear',round(score_predictions(part,ALTERNATIVE)['rmse'],3),flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,decision=assess(results,PRIMARY,out,extra_models=(ALTERNATIVE,))
    subgroups=pd.read_csv(out/'subgroup_metrics.csv')
    decisions={name:goal_decision(metrics,subgroups.loc[subgroups.population.eq('severe_above75')],name) for name in [PRIMARY,ALTERNATIVE]}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,fitted_splits=len(records),
        source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output)
