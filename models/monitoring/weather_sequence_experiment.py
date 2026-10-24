"""Fixed-recipe weather-memory challenger on the retained disease splits."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from .dynamic_data import build_sequences
from .weather_sequence_model import fit_sequence,predict_sequence
from .innovation_common import ROOT,DATA,REFERENCE,load_cohort,register,assess
from .evaluation import score_predictions
from .inputs import sha

NAME='weather_sequence'


def run(output=None):
    out=Path(output or ROOT/'outputs/disease_improvement/weather_sequence_20261022').resolve()
    config=dict(epochs=180,members=3,learning_rate=.006,weight_decay=.02,hidden=8,seed=20261022)
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in ['dynamic_data','dynamic_model',
        'weather_sequence_model','field_history_models']]
    register(out,NAME,dict(config=config,architecture='GRU8 daily weather memory + shared48/16 decoder; process37/event68/history35/anchor5',
        loss='Original hierarchy-weighted squared error',output='Measured-history or training-category anchor plus learned full-range change',
        history='Strictly earlier source-native cross-leaf summaries; no current/future observation enters inputs',
        ensemble='Three fixed initialization seeds; full outer-training support',
        selection='No hyperparameter/epoch search; spatial only after temporal reference improvement',
        pretraining=False),codes)
    obs,arrays,history,frames,splits,position=load_cohort()
    fields=pd.read_csv(DATA/'fields.csv');weather=pd.read_parquet(DATA/'daily_weather.parquet')
    parts={};records=[];started=time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True)
            frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},NAME,out)
            if score_predictions(frame,NAME)['rmse']>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Fails the registered temporal reference-improvement screen'),indent=2)+'\n')
                break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];test=obs.iloc[vi];q=test.drop(columns='observed_percent')
        assert train.season_year.max()<test.season_year.min()
        train_batch=build_sequences(weather,fields,train,train.drop(columns='observed_percent'))
        test_batch=build_sequences(weather,fields,obs,q)
        ta={k:arrays[k][ti] for k in ['process','history_events']};va={k:arrays[k][vi] for k in ta}
        print('start',split['fold'],'fields',train.field_id.nunique(),flush=True)
        fitted=fit_sequence(train_batch,ta,history.iloc[ti],train,config)
        fitted.update(fold=split['fold'],train_ids=split['train'],test_ids=split['test'])
        path=out/(split['fold']+'_model.joblib');joblib.dump(fitted,path)
        value=predict_sequence(fitted,test_batch,va,history.iloc[vi],q)
        np.testing.assert_array_equal(value,predict_sequence(joblib.load(path),test_batch,va,history.iloc[vi],q))
        part=frames[label].loc[split['test']].reset_index().copy();part[NAME]=value
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            traces=[m['trace'] for m in fitted['members']]))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'test_rmse',round(score_predictions(part,NAME)['rmse'],3),flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():
        frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,decision=assess(results,NAME,out)
    reg=json.loads((out/'registration.json').read_text())
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,
        fitted_splits=len(records),members_per_split=3,goal_achieved=decision['achieved'],
        sources_unchanged=all(sha(ROOT/p)==h for p,h in reg['source_sha256'].items()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decision


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path)
    args=p.parse_args();run(args.output)
