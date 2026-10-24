"""Adaptive physically informed drying/condensation input comparison."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .microclimate_features import build_microclimate
from .history import build_history
from .field_history_models import fit_ensemble,predict_ensemble
from .robust_models import fit_cluster_ensemble,predict_cluster_ensemble
from .innovation_common import ROOT,PARENT,DATA,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY='microclimate_both';NAMES=[PRIMARY,'microclimate_history_blend','microclimate_history_direct']


def insert_before_states(original,extra):
    result=np.column_stack([original[:,:-3],extra,original[:,-3:]])
    np.testing.assert_array_equal(result[:,-3:],original[:,-3:])
    return result


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/microclimate_20261022').resolve()
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['microclimate_features','history',
        'field_history_models','robust_models','models']]
    register(out,PRIMARY,dict(hypothesis='Previously unused radiation/wind/dewpoint improve representation of canopy drying/condensation conditions',
        features='43fixed microclimate summaries;7/21/42day means/maxima andcausalhistoryintervals',
        physical_interpretation='Unvalidated relative proxies; no actualcanopy leafwetness orET0 measured; nativecodeproximity remains an unvalidatedproxy',
        weather='Retained qualifiedstationrainandtemperature/RH withoriginalphase/fallback; retainedreanalysiswind/radiation/dewpoint',
        ensemble='Original20members80%coordinateyear,seed20261022',estimator='Original200iterationdepth3HGB,minleaf10,L2=1,learningrate.05',
        process_baseline='Insert addedvariables before last3processstates, preserve registered-3baselinecolumn exactly',
        history='Existing measuredanchorfield_split; all firstfield/event andhistory queries fitted',
        formulations=NAMES,selection='Spatial ifanyfixedformulation improves temporalreference',adaptive_extension=True),codes)
    obs,arrays,history,frames,splits,position=load_cohort()
    rebuilt,provenance=build_history(obs,obs.drop(columns='observed_percent'))
    np.testing.assert_array_equal(rebuilt.to_numpy(),history.to_numpy())
    micro,names,ledger=build_microclimate(pd.read_parquet(DATA/'daily_weather.parquet'),pd.read_csv(DATA/'fields.csv'),
        obs.drop(columns='observed_percent'),provenance)
    assert micro.shape==(2304,43)
    np.savez_compressed(out/'microclimate_predictors.npz',microclimate=micro,target_ids=obs.global_target_index)
    (out/'feature_schema.json').write_text(json.dumps(dict(names=names,columns=len(names)),indent=2)+'\n')
    ledger.to_csv(out/'weather_provenance.csv',index=False)
    process=insert_before_states(arrays['process'],micro)
    event=insert_before_states(arrays['history_events'],micro)
    parts={};records=[];started=time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=NAMES[1:])
            if min(score_predictions(frame,n)['rmse'] for n in NAMES)>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Allfixedmicroclimate formulations fail temporalreferenceimprovement'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];q=obs.iloc[vi].drop(columns='observed_percent')
        assert train.season_year.max()<obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        fitted=dict(process=fit_cluster_ensemble(process[ti],history.iloc[ti],train,
                dict(structure='process',process_baseline_column=-3,crossleaf=True)),
            history=fit_ensemble(event[ti],history.iloc[ti],train,dict(structure='field_split')),
            train_ids=split['train'],test_ids=split['test'])
        path=out/(split['fold']+'_model.joblib');joblib.dump(fitted,path)
        def forecast(bundle):
            p=predict_cluster_ensemble(bundle['process'],process[vi],history.iloc[vi],q)
            h=predict_ensemble(bundle['history'],event[vi],history.iloc[vi],q)
            return p,h
        p,h=forecast(fitted);pp,hh=forecast(joblib.load(path));np.testing.assert_array_equal(p,pp);np.testing.assert_array_equal(h,hh)
        part=frames[label].loc[split['test']].reset_index().copy()
        part[PRIMARY]=.5*(p+h);part[NAMES[1]]=.5*(part.candidate_process.to_numpy()+h);part[NAMES[2]]=h
        parts.setdefault(label,[]).append(part);records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{name:round(score_predictions(part,name)['rmse'],3) for name in NAMES},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,PRIMARY,out,extra_models=NAMES[1:]);groups=pd.read_csv(out/'subgroup_metrics.csv')
    decisions={name:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],name) for name in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    registration=json.loads((out/'registration.json').read_text())
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_models_replay=True,fitted_splits=len(records),
        source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args()
    with threadpool_limits(limits=1):run(args.output)
