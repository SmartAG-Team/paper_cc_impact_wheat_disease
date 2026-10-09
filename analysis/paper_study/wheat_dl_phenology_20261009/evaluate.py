"""Validation-only family choice followed by one retrospective station-test readout."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from model.wheat_phenology_dl.network import quantile_days
from .train import load_inputs, restore, crps_by_stage, sha

HERE=Path(__file__).resolve().parent
STAGES=[10,31,51,85]


def point_metrics(y,prediction,horizon=366):
    y=np.asarray(y);prediction=np.asarray(prediction)
    valid=y>0;matched=valid&np.isfinite(prediction)
    error=np.where(matched,prediction-y,np.nan)
    penalty=np.where(valid,np.where(matched,np.abs(error),float(horizon)),np.nan)
    rows=[]
    for j in range(y.shape[1]):
        e=error[:,j][matched[:,j]]
        rows.append(dict(observed=int(valid[:,j].sum()),matched=int(matched[:,j].sum()),
            mae_days=float(np.mean(np.abs(e))) if len(e) else None,
            rmse_days=float(np.sqrt(np.mean(e**2))) if len(e) else None,
            bias_days=float(np.mean(e)) if len(e) else None,
            penalized_mae_days=float(np.nanmean(penalty[:,j])) if valid[:,j].any() else None))
    return dict(observed=int(valid.sum()),matched=int(matched.sum()),
        pooled_mae=float(np.nanmean(np.abs(error))),
        macro_mae=float(np.mean([r['mae_days'] for r in rows])) if all(r['mae_days'] is not None for r in rows) else None,
        macro_penalized_mae=float(np.mean([r['penalized_mae_days'] for r in rows])),stages=rows)


def paired_interval(y,baseline,candidate,stations,draws=2000,seed=20261009):
    y=np.asarray(y);baseline=np.asarray(baseline);candidate=np.asarray(candidate)
    common=(y>0)&np.isfinite(baseline)&np.isfinite(candidate)
    difference=np.where(common,np.abs(candidate-y)-np.abs(baseline-y),0.)
    groups,inverse=np.unique(stations,return_inverse=True)
    sums=np.zeros((len(groups),y.shape[1]));counts=sums.copy()
    np.add.at(sums,inverse,difference);np.add.at(counts,inverse,common)
    supported=counts.sum(1)>0;sums=sums[supported];counts=counts[supported]
    if not len(sums) or (counts.sum(0)==0).any():raise ValueError('No common paired support at every requested stage')
    estimate=float(np.mean(sums.sum(0)/counts.sum(0)))
    rng=np.random.default_rng(seed);values=[]
    for _ in range(draws):
        sample=rng.integers(0,len(sums),len(sums));n=counts[sample].sum(0)
        if (n>0).all():values.append(float(np.mean(sums[sample].sum(0)/n)))
    lo,hi=np.quantile(values,[.025,.975])
    return dict(difference_days=estimate,lower95=float(lo),upper95=float(hi),
                stations=len(sums),matched_events=int(common.sum()),bootstrap_draws=len(values),
                scope='Candidate minus T-P-V absolute error on common events; station-cluster resampling conditional on fitted models')


def choose_family(validation):
    if not validation or not all(np.isfinite(x['macro_penalized_mae']) for x in validation.values()):
        raise ValueError('Finite validation scores required for every candidate')
    return min(validation,key=lambda k:(validation[k]['macro_penalized_mae'],k))


def predict_family(a,m,ids,paths,device='cpu',batch=256):
    models=[restore(p,device)[0] for p in paths]
    point=np.full((len(ids),4),np.nan);low=point.copy();high=point.copy();scores=point.copy()
    with torch.no_grad():
        for k in range(0,len(ids),batch):
            ix=ids[k:k+batch];x=torch.as_tensor(a['X'][ix],device=device);g=torch.as_tensor(a['G'][ix],device=device)
            mean=np.mean([model(x,g,m['thermal_scale'])[1].detach().cpu().numpy() for model in models],axis=0)
            if not np.isfinite(mean).all() or (np.diff(mean,axis=1)>1e-5).any() or (np.diff(mean,axis=2)<-1e-5).any():
                raise ValueError('Invalid or unordered ensemble CDF')
            for target,q in [(point,.5),(low,.05),(high,.95)]:
                target[k:k+len(ix)]=quantile_days(mean,q,a['lengths'][ix])
            scores[k:k+len(ix)]=crps_by_stage(mean,a['Y'][ix],a['lengths'][ix])
    return dict(median=point,lower90=low,upper90=high,crps=scores)


def evaluate(inputs,fits,output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);fits=Path(fits)
    a,env,m=load_inputs(inputs);protocol=json.loads((HERE/'protocol.json').read_text())
    paths={family:[fits/f'{family}_s{seed}.pt' for seed in protocol['seeds']] for family in protocol['candidates']}
    for ps in paths.values():
        for p in ps:
            r=json.loads(p.with_suffix('.json').read_text())
            assert r['status']=='complete' and r['checkpoint_sha256']==sha(p)
            assert r['input_arrays_sha256']==m['input_arrays_sha256'] and not r['test_labels_used']
    # Family selection uses only validation observations and occurs before test prediction.
    vi=np.flatnonzero(env.cohort.eq('validation').to_numpy())
    predictions={'validation':{}}
    validation={}
    for family,ps in paths.items():
        predictions['validation'][family]=predict_family(a,m,vi,ps)
        validation[family]=point_metrics(a['Y'][vi],predictions['validation'][family]['median'],m['horizon'])
    selected=choose_family(validation)
    selection=dict(selected=selected,validation_scores=validation,
        rule='Smallest validation stage-equal MAE; an unreached median incurs a 366-day diagnostic failure penalty',
        checkpoint_stopping='Stage-equal validation CRPS, fixed patience and maximum epochs',
        prediction_ensemble='Equal mixture of three fixed-seed CDFs, then median crossing',
        test_scores_used=False,protocol_sha256=sha(HERE/'protocol.json'),evaluator_sha256=sha(__file__),
        checkpoints={str(p.name):sha(p) for ps in paths.values() for p in ps})
    (output/'selection_before_test.json').write_text(json.dumps(selection,indent=2)+'\n')
    ti=np.flatnonzero(env.cohort.eq('testing').to_numpy());predictions['testing']={}
    for family,ps in paths.items():predictions['testing'][family]=predict_family(a,m,ti,ps)
    rows=[];metrics=[];common_metrics=[];intervals={};native_tables=[]
    for cohort,ids in [('validation',vi),('testing',ti)]:
        y=a['Y'][ids];baseline=a['baseline'][ids]
        values={'T-P-V':baseline,**{family:r['median'] for family,r in predictions[cohort].items()}}
        common=(y>0)
        for v in values.values():common&=np.isfinite(v)
        for model,v in values.items():
            report=point_metrics(y,v,m['horizon'])
            for j,s in enumerate(STAGES):metrics.append(dict(cohort=cohort,model=model,stage=s,**report['stages'][j]))
            common_report=point_metrics(np.where(common,y,0),v,m['horizon'])
            for j,s in enumerate(STAGES):common_metrics.append(dict(cohort=cohort,model=model,stage=s,**common_report['stages'][j]))
        intervals[cohort]={family:paired_interval(y,baseline,r['median'],env.iloc[ids].PEP_ID.to_numpy()) for family,r in predictions[cohort].items()}
        for family,r in predictions[cohort].items():
            np.savez_compressed(output/f'{cohort}_{family}_predictions.npz',indices=ids,**r)
        for k,i in enumerate(ids):
            for j,s in enumerate(STAGES):
                if not a['source_observed'][i,j]:continue
                row=dict(cohort=cohort,PEP_ID=int(env.iloc[i].PEP_ID),SOWING_DATE=env.iloc[i].SOWING_DATE,
                         BBCH=s,source_observed_position=int(a['source_observed'][i,j]),eligible=bool(y[k,j]>0),
                         weather_days=int(a['lengths'][i]),T_P_V_position=baseline[k,j],all_model_common=bool(common[k,j]))
                for family in paths:
                    for name in ['median','lower90','upper90','crps']:row[family+'_'+name]=predictions[cohort][family][name][k,j]
                rows.append(row)
        native_tables.append(dict(cohort=cohort,observed_source_events=int((a['source_observed'][ids]>0).sum()),
                                 weather_qualified_events=int((y>0).sum()),all_model_common_events=int(common.sum()),
                                 stations=int(env.iloc[ids].PEP_ID.nunique())))
    pd.DataFrame(rows).to_csv(output/'event_predictions.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    pd.DataFrame(metrics).to_csv(output/'stage_metrics.csv',index=False)
    pd.DataFrame(common_metrics).to_csv(output/'common_stage_metrics.csv',index=False)
    pd.DataFrame(native_tables).to_csv(output/'population_coverage.csv',index=False)
    baseline=point_metrics(a['Y'][ti],a['baseline'][ti],m['horizon'])
    candidate=point_metrics(a['Y'][ti],predictions['testing'][selected]['median'],m['horizon'])
    bv=point_metrics(a['Y'][vi],a['baseline'][vi],m['horizon'])
    evidence=intervals['testing'][selected]
    final_pred=predictions['testing'][selected]['median']
    dough=(a['Y'][ti,3]>0)&np.isfinite(a['baseline'][ti,3])&np.isfinite(final_pred[:,3])
    dough_delta=float(np.mean(np.abs(final_pred[dough,3]-a['Y'][ti[dough],3])-np.abs(a['baseline'][ti[dough],3]-a['Y'][ti[dough],3])))
    gate=dict(validation_improves=validation[selected]['macro_penalized_mae']<bv['macro_penalized_mae'],
              test_improves=candidate['macro_penalized_mae']<baseline['macro_penalized_mae'],
              paired_upper95_below_zero=evidence['upper95']<0,
              soft_dough_not_worse=dough_delta<=0,
              matched_coverage_not_worse=all(c['matched']>=b['matched'] for c,b in zip(candidate['stages'],baseline['stages'])))
    decision=dict(selected_family=selected,native_stage_gate=gate,native_stage_candidate_supported=all(gate.values()),
                  disease_engine_replacement=False,climate_archive_changed=False,
                  reason='Full disease-engine replacement additionally requires field-stage transfer and climate-applicability evidence.',
                  baseline_test=baseline,selected_test=candidate,paired_station_intervals=intervals,
                  evaluation_scope='Retrospective station-held-out wheat evaluation; not a new temporal or climate-extrapolation validation')
    (output/'decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    receipt=dict(status='complete',input_arrays_sha256=m['input_arrays_sha256'],selection_sha256=sha(output/'selection_before_test.json'),
                 evaluator_sha256=sha(__file__),inference_device='cpu',torch_version=torch.__version__,
                 output_sha256={p.name:sha(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(decision,indent=2));return decision


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--fits',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2);evaluate(a.inputs,a.fits,a.output)
