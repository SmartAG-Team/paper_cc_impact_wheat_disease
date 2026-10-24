"""Outcome-conserving multi-interval progression tasks from actual field records."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .multihorizon import build_contexts,context_weights
from .offset_growth_model import fit_growth,predict_growth
from .weather_sequence_model import _baseline
from .innovation_common import ROOT,DATA,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

NAMES=dict(logit='multihorizon_logit',linear='multihorizon_linear',control='point_context_linear')


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/multihorizon_20261022').resolve()
    config=dict(members=20,fraction=.8,seed=20261022,rounds=400,max_extra=3)
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['multihorizon','offset_growth_model',
        'weather_sequence_model','history','infection_event_features','field_history_models']]
    register(out,NAMES['linear'],dict(config=config,
        training_tasks='Original observations plus<=3additional earlier history-cutoff contexts per target; no invented measurements',
        cutoff_selection='Date-only evenly spaced earlier field assessment dates, excluding original latest context',
        mass='Originaltarget hierarchyweight retained;half original andhalf shared among alternativecontexts wherepresent',
        regularization='No renormalization by expandedrowcount; total effectiveweight equals originalrowcount',
        altered_features='Rebuild history35 and anchor5; recompute gapwetdose/gapthermal; other causalweather/processfeatures retained',
        excluded_features='Observation-assimilated state13 excluded because they would disclose hidden intermediate observations',
        paired_control='Same145features/learner on original latest contexts only; linear link',
        links='Logit andlinear fullrangegrowth, fixedsame400tree recipe',
        code_proximity='Inheritednearestnative numericalcode anchor remains an unvalidatedproxy, not anatomicaldistance',
        adaptive_extension=True,selection='Spatialifanyprespecifiedformulation beats temporalreference'),codes)
    obs,arrays,history,frames,splits,position=load_cohort()
    print('building additional causal contexts',flush=True)
    expanded=build_contexts(obs,arrays,history,pd.read_csv(DATA/'fields.csv'),pd.read_parquet(DATA/'daily_weather.parquet'))
    expanded['schedule'].to_csv(out/'training_context_schedule.csv',index=False)
    expanded['provenance'].to_json(out/'context_provenance.json',orient='records',indent=2)
    np.savez_compressed(out/'training_context_features.npz',x=expanded['x'],anchor=expanded['anchor'],available=expanded['available'])
    print('contexts',len(expanded['schedule']),'originaltargets',len(obs),flush=True)
    parts={};records=[];started=time.monotonic();mapping=expanded['schedule'].original_position.to_numpy(int)
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},NAMES['linear'],out,extra_models=(NAMES['logit'],NAMES['control']))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES.values())>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='No prespecifiedmultiinterval/control formulation improves temporalreference'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];q=obs.iloc[vi].drop(columns='observed_percent')
        assert train.season_year.max()<obs.iloc[vi].season_year.min()
        clusters=np.array(sorted(train.coordinate_year.unique()));rng=np.random.default_rng(config['seed'])
        fitted=dict(members=[],config=config,train_ids=split['train'],test_ids=split['test'])
        print('start',split['fold'],flush=True)
        for _ in range(config['members']):
            selected=rng.choice(clusters,max(1,int(np.ceil(config['fraction']*len(clusters)))),replace=False)
            oi=ti[np.flatnonzero(train.coordinate_year.isin(selected))];sub=obs.iloc[oi]
            w=hierarchical_weights(sub);w=w/w.mean();y=sub.observed_percent.to_numpy(float)
            base=dict(pooled=float(np.average(y/100,weights=w)),means={})
            category=sub.source.astype(str)+'|'+sub.leaf_index.astype(str)
            for key in category.unique():
                mask=category.eq(key).to_numpy()
                base['means'][key]=float((np.sum(w[mask]*y[mask]/100)+10*base['pooled'])/(w[mask].sum()+10))
            original_weight=np.zeros(len(obs));original_weight[oi]=w
            allweight=context_weights(expanded['schedule'],original_weight)
            np.testing.assert_allclose(np.bincount(mapping,weights=allweight,minlength=len(obs)),original_weight,atol=1e-12)
            ai=np.flatnonzero(allweight>0)
            b=100*_baseline(base,expanded['anchor'][ai],expanded['available'][ai],expanded['frame'].iloc[ai])
            yc=expanded['frame'].iloc[ai].observed_percent.to_numpy(float)
            models={link:fit_growth(expanded['x'][ai],yc,b,allweight[ai],link=link,
                rounds=config['rounds'],normalize_weights=False) for link in ['logit','linear']}
            bo=100*_baseline(base,expanded['anchor'][oi],expanded['available'][oi],sub)
            models['control']=fit_growth(expanded['x'][oi],y,bo,w,link='linear',rounds=config['rounds'],normalize_weights=False)
            fitted['members'].append(dict(base=base,models=models,training_ids=sub.global_target_index.astype(int).tolist(),
                clusters=selected.tolist(),context_positions=ai.tolist(),context_weights=allweight[ai]))
        path=out/(split['fold']+'_model.joblib');joblib.dump(fitted,path)
        def forecast(bundle):
            result={k:[] for k in NAMES}
            for member in bundle['members']:
                b=100*_baseline(member['base'],expanded['anchor'][vi],expanded['available'][vi],q)
                for key in result:result[key].append(predict_growth(member['models'][key],expanded['x'][vi],b))
            return {k:np.mean(v,axis=0) for k,v in result.items()}
        values=forecast(fitted);replay=forecast(joblib.load(path))
        part=frames[label].loc[split['test']].reset_index().copy()
        for key,name in NAMES.items():np.testing.assert_array_equal(values[key],replay[key]);part[name]=values[key]
        parts.setdefault(label,[]).append(part);records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{name:round(score_predictions(part,name)['rmse'],3) for name in NAMES.values()},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,NAMES['linear'],out,extra_models=(NAMES['logit'],NAMES['control']))
    groups=pd.read_csv(out/'subgroup_metrics.csv')
    decisions={name:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],name) for name in NAMES.values()}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    registration=json.loads((out/'registration.json').read_text())
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_models_replay=True,
        original_targets=len(obs),training_contexts=len(expanded['schedule']),fitted_splits=len(records),
        source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output)
