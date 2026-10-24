"""Field-level latent response classes around an honestly calibrated baseline."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .latent_field_model import fit_latent_field,predict_latent_field
from .discrepancy_experiment import forecast_baseline,CACHE
from .weather_sequence_model import design
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY='latent_field_mixture';NAMES=[PRIMARY,'latent_field_recent'];CONTROL='latent_field_unfiltered'


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/latent_field_20261022').resolve()
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['latent_field_model',
        'discrepancy_experiment','discrepancy_filter','crossfit_router','field_history_models',
        'models','robust_models','history_interval_hybrid','weather_sequence_model','dynamic_model']]
    registration=register(out,PRIMARY,dict(hypothesis='Unobserved field heterogeneity changes the conditional weather/history response, beyond a common additive discrepancy',
        architecture='Three sharedfieldlatentclasses; alltrainingobservations fromafield jointlyupdateoneEMresponsibilityvector',
        interpretation='Latentresponseproxy; not observed cultivaridentity, inoculum or protocolclassification',
        baseline='Originalstation-informedprocess/historyreference; cachedhonestinnercoordinateheld-out baselineerrors fortraining',
        experts='Three160iterationdepth2HGB residualexperts,minleaf12,L2=5,lr.05;8fixedEMiterations;originalhierarchyweights',
        priors='Learned source-specificclasspriors withoneaveragefieldmasspseudocountperclass',
        observation='Normaldiscrepancylikelihoodwith8ppstandarddeviationfloor;3%responsibilityrefresh; maxfourassessmentdayevidenceunits',
        causal_inference='Atqueryonlystrictlyearlierfielddays updateclassposterior; allsame-dayleaveswithheldtogether; firstvisitbaselineunchanged',
        formulations=['Allpastassessmentdays','Pastassessmentdays discountedexp(-age/30days)'],
        climate='Monitoringonly; no-observationpredictionequalsoriginalbaseline',
        source_codes='Nativecodesnotanatomicallyverified; inheritednumericalproximityremainsunvalidated',
        selection='Nooutertuning; spatial iff eitherfixedposteriorvariant improves temporalreference; unfilteredcontrolcannotadvance',
        adaptive_extension=True),codes)
    for path in [CACHE/'registration.json',*sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort();parts={};records=[];started=time.monotonic()
    x,_,_=design(arrays,history,obs.drop(columns='observed_percent'))
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=(NAMES[1],CONTROL))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES)>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Both fixed latentfieldposteriors fail temporalreferenceimprovement'),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        prior=joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids']==split['train'] and prior['test_ids']==split['test']
        oof=.5*(prior['inner_predictions'][:,0]+prior['inner_predictions'][:,1])
        base={k:prior['experts'][k] for k in ['process','first','event','training_ids']}
        w=hierarchical_weights(train);latent=fit_latent_field(x[ti],train,oof,w)
        fitted=dict(base=base,latent=latent,train_ids=split['train'],test_ids=split['test'],
            inner_memberships=prior['inner_memberships'],inner_predictions=oof,inner_weight=w)
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)
        ci=np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qi=pd.Series(np.arange(len(ci)),index=ci).loc[vi].to_numpy()
        def forecast(bundle):
            b=forecast_baseline(bundle['base'],arrays,history,obs,ci)
            values={}
            for name,forget in zip(NAMES,[None,30.]):
                p,prob=predict_latent_field(bundle['latent'],x[ci],obs.iloc[ci],b,forget_days=forget)
                values[name]=p[qi];values[name+'_posterior']=prob[qi]
            values[CONTROL]=b[qi];return values
        values=forecast(fitted);replay=forecast(joblib.load(path))
        for key in values:np.testing.assert_array_equal(values[key],replay[key])
        part=frames[label].loc[split['test']].reset_index().copy()
        for name in [*NAMES,CONTROL]:part[name]=values[name]
        for name in NAMES:
            for k in range(3):part[f'{name}_class{k}_probability']=values[name+'_posterior'][:,k]
        parts.setdefault(label,[]).append(part);records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            latent_training_trace=latent['trace']))
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
