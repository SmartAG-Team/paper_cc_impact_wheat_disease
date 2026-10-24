"""Learn prequential field/categorical-code discrepancy from honest baseline errors."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .crossfit_router import coordinate_folds
from .discrepancy_filter import fit_discrepancy,predict_discrepancy
from .field_history_models import fit_ensemble,predict_ensemble
from .history_interval_hybrid import history_prediction
from .robust_models import fit_cluster_ensemble,predict_cluster_ensemble
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY='discrepancy_linear';NAMES=[PRIMARY,'discrepancy_logit'];CONTROL='discrepancy_unfiltered'
CACHE=ROOT/'outputs/disease_improvement/crossfit_router_20261022'


def fit_baseline(arrays,history,obs,ix):
    train=obs.iloc[ix]
    return dict(process=fit_cluster_ensemble(arrays['process'][ix],history.iloc[ix],train,
            dict(structure='process',process_baseline_column=-3,crossleaf=True)),
        first=fit_ensemble(arrays['station'][ix],history.iloc[ix],train,dict(structure='field_split')),
        event=fit_ensemble(arrays['history_events'][ix],history.iloc[ix],train,dict(structure='field_split')),
        training_ids=train.global_target_index.astype(int).tolist())


def forecast_baseline(fit,arrays,history,obs,ix):
    q=obs.iloc[ix].drop(columns='observed_percent');h=history.iloc[ix]
    p=predict_cluster_ensemble(fit['process'],arrays['process'][ix],h,q)
    a=predict_ensemble(fit['first'],arrays['station'][ix],h,q)
    b=predict_ensemble(fit['event'],arrays['history_events'][ix],h,q)
    selected,_=history_prediction(a,b,h,q)
    return .5*(p+selected)


def run(output=None):
    out=Path(output or ROOT/'data/final_monitoring/folds').resolve()
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['discrepancy_filter',
        'crossfit_router','field_history_models','models','robust_models','history_interval_hybrid']]
    registration=register(out,PRIMARY,dict(hypothesis='Persistent within-field forecast discrepancy can be corrected beyond already-assimilated disease anchors',
        baseline='Original20member80%coordinateyear process andconditionalstation/eventhistory,50:50',
        honest_errors='Reuse fixed process/history innercoordinate-heldout predictions wherecached; otherwisefitidentical fiveinnercoordinatefolds withinoutertraining',
        kernel='SharedfieldMatern3/2 +same-native-codeOU; fourpositive amplitude/timeparameters; fixedunitnugget identifiesratio; nominalcodeequality only',
        fitting='WeightedprequentialMSE onhistoricalOOFbaselineerrors;L-BFGS-B60iterations,weak1e-5logparametershrinkage; noouteroutcomeuse',
        links='Twofixed formulations: additivepercentagefraction oradditivelogitdiscrepancy with.005endpointclip',
        conditioning='Onlystrictlyearlierheld-field residuals; allsame-daycodes predictedbeforeupdates; firstvisitbaselineunchanged',
        interpretation='Gaussian discrepancy proxy, not measured observation noise; originalscientifichierarchyweights,notreplicatedenominators',
        climate='Rollingmonitoringonly; no-observationpredictionequalsunfilteredbaseline',
        source_codes='Discrepancykernelcategorical; baselineinheritednumericalcodeproximity remainsunvalidated',
        selection='Spatial iff either correctedformulation improves temporalreference; controlcannotadvance experiment',
        adaptive_extension=True),codes)
    cached_paths=[CACHE/'registration.json',*sorted(CACHE.glob('*_model.joblib'))]
    for path in cached_paths:registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort();parts={};records=[];started=time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=(NAMES[1],CONTROL))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES)>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Neither fixed discrepancy correction improves temporal reference'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        cached=CACHE/f'{split["fold"]}_model.joblib'
        if cached.exists():
            prior=joblib.load(cached)
            assert prior['train_ids']==split['train'] and prior['test_ids']==split['test']
            oof=.5*(prior['inner_predictions'][:,0]+prior['inner_predictions'][:,1])
            base={k:prior['experts'][k] for k in ['process','first','event','training_ids']}
            inner=prior['inner_memberships']
        else:
            oof=np.full(len(ti),np.nan);inner=[]
            for k,(a,b) in enumerate(coordinate_folds(train)):
                fit=fit_baseline(arrays,history,obs,ti[a]);oof[b]=forecast_baseline(fit,arrays,history,obs,ti[b])
                joblib.dump(fit,out/f'{split["fold"]}_inner{k}_experts.joblib')
                inner.append(dict(inner_fold=k,train=train.iloc[a].global_target_index.astype(int).tolist(),
                    test=train.iloc[b].global_target_index.astype(int).tolist()))
            base=fit_baseline(arrays,history,obs,ti)
        assert np.isfinite(oof).all()
        w=hierarchical_weights(train)
        fits={link:fit_discrepancy(train,oof,w,link) for link in ['linear','logit']}
        fitted=dict(base=base,corrections=fits,train_ids=split['train'],test_ids=split['test'],
            inner_memberships=inner,inner_predictions=oof,inner_weight=w)
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)
        context_ix=np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        query_positions=pd.Series(np.arange(len(context_ix)),index=context_ix).loc[vi].to_numpy()
        def forecast(bundle):
            baseline=forecast_baseline(bundle['base'],arrays,history,obs,context_ix)
            values={link:predict_discrepancy(f['parameters'],obs.iloc[context_ix],baseline,link)[query_positions]
                for link,f in bundle['corrections'].items()}
            values['unfiltered']=baseline[query_positions];return values
        values=forecast(fitted);replay=forecast(joblib.load(path))
        for key in values:np.testing.assert_array_equal(values[key],replay[key])
        part=frames[label].loc[split['test']].reset_index().copy()
        for name,link in zip([*NAMES,CONTROL],['linear','logit','unfiltered']):part[name]=values[link]
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            fits={link:{k:v for k,v in f.items() if k!='parameters'}|dict(parameters=f['parameters'].tolist()) for link,f in fits.items()}))
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
