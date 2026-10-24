"""Sequential registered evaluation of learned daily compartment dynamics."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from analysis.final_study.metrics import hierarchical_weights
from .dynamic_data import build_sequences
from .dynamic_model import fit_dynamic,predict_dynamic
from .innovation_common import ROOT,PARENT,DATA,REFERENCE,load_cohort,register,assess
from .evaluation import score_predictions
from .inputs import sha

NAME='learned_daily_state'


def run(output=None,epochs=120,all_spatial=False):
    out=Path(output or ROOT/'outputs/disease_improvement/dynamic_state_20261022').resolve()
    config=dict(epochs=epochs,learning_rate=.02,weight_decay=.001,seed=20261022,nodes=7,hidden=8)
    codes=[Path(__file__),Path(__file__).with_name('dynamic_data.py'),Path(__file__).with_name('dynamic_model.py')]
    register(out,NAME,dict(config=config,output='Direct full-range censored-normal predictive mean',
        state='Conserved S/E1/E2/E3/V native-code proxies; all cohorts continuously active',
        field_factor='Seven-node Gaussian quadrature; one shared latent epidemic propensity; no cultivar interpretation',
        filter='Observation-density node weighting with 1% prior refresh and learned partial visible correction',
        training_objective='Hierarchy-weighted fraction MSE + .02 predictive negative log density; .0001 weak parameter shrinkage',
        selection='Fixed recipe; no outer-label hyperparameter or epoch selection',
        temporal_screen='Continue spatial when temporal RMSE improves the retained reference',
        no_observation_run='Same fitted parameters, no held-field disease assimilation'),codes)
    obs,arrays,history,frames,splits,position=load_cohort()
    weather=pd.read_parquet(DATA/'daily_weather.parquet'); fields=pd.read_csv(DATA/'fields.csv')
    parts={};records=[];started=time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            temporal=pd.concat(parts['temporal'],ignore_index=True)
            temporal.to_csv(out/'assessment_predictions.csv',index=False)
            metrics,decision=assess({'temporal':temporal},NAME,out,extra_models=(NAME+'_free',))
            candidate=score_predictions(temporal,NAME)['rmse']; baseline=score_predictions(temporal,REFERENCE)['rmse']
            if candidate>=baseline and not all_spatial:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Fixed candidate fails temporal reference-improvement screen',rmse=candidate,reference=baseline),indent=2)+'\n')
                break
        train=obs.iloc[position.loc[split['train']]].copy(); test=obs.iloc[position.loc[split['test']]].copy()
        assert train.season_year.max()<test.season_year.min()
        train_batch=build_sequences(weather,fields,train,train.drop(columns='observed_percent'))
        test_batch=build_sequences(weather,fields,obs,test.drop(columns='observed_percent'))
        print('start',split['fold'],'fields',len(train_batch['field_ids']),'days',train_batch['weather'].shape[1],flush=True)
        fitted=fit_dynamic(train_batch,train.observed_percent.to_numpy(),hierarchical_weights(train),config)
        fitted.update(fold=split['fold'],train_ids=split['train'],test_ids=split['test'])
        joblib.dump(fitted,out/(split['fold']+'_model.joblib'))
        value=predict_dynamic(fitted,test_batch);free=predict_dynamic(fitted,test_batch,assimilate=False)
        reloaded=predict_dynamic(joblib.load(out/(split['fold']+'_model.joblib')),test_batch)
        np.testing.assert_array_equal(value,reloaded)
        part=frames[label].loc[split['test']].reset_index().copy();part[NAME]=value;part[NAME+'_free']=free
        parts.setdefault(label,[]).append(part)
        record=dict(fold=split['fold'],train_rows=len(train),train_fields=train.field_id.nunique(),
            test_rows=len(test),elapsed_seconds=time.monotonic()-started,trace=fitted['trace'])
        records.append(record);(out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'train_rmse',round(fitted['trace'][-1]['training_rmse'],3),'test_rmse',round(score_predictions(part,NAME)['rmse'],3),flush=True)
    results={label:pd.concat(parts_here,ignore_index=True) for label,parts_here in parts.items()}
    for label,frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,decision=assess(results,NAME,out,extra_models=(NAME+'_free',))
    artifacts={p.name:sha(p) for p in out.iterdir() if p.is_file()}
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,
        evaluated_splits=len(records),source_hashes_unchanged=all(sha(ROOT/p)==h for p,h in
            json.loads((out/'registration.json').read_text())['source_sha256'].items()),
        artifacts_sha256=artifacts,goal_achieved=decision['achieved']),indent=2)+'\n')
    return metrics,decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    p.add_argument('--epochs',type=int,default=120);p.add_argument('--all-spatial',action='store_true')
    args=p.parse_args();run(args.output,args.epochs,args.all_spatial)
