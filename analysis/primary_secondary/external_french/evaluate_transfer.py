"""Frozen BASF fit transferred to independent French pycnidia endpoint."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
from .prepare_external import prepare
from calibration.primary_secondary.field_data import prepare_basf
from calibration.primary_secondary.calibrate import mechanism_prediction, fit_baseline, baseline_prediction, target_values, score, BASELINES

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--fit',default='analysis/primary_secondary/calibration_v2_age/all_development_fit.json')
    parser.add_argument('--output',default='analysis/primary_secondary/external_french/transfer_v2')
    args=parser.parse_args()
    source=ROOT/args.fit
    output=ROOT/args.output
    output.mkdir(parents=True,exist_ok=True)
    fit=json.loads(source.read_text())
    batch=prepare()
    development=prepare_basf(ROOT)
    trajectory=mechanism_prediction(batch,fit)
    records=[]
    row=batch.targets.loc[~batch.targets.conditioning].copy()
    row['observed_percent']=row.Value
    row['predicted_percent']=100*target_values(batch,trajectory,'pycnidia')
    row['model']='primary_secondary_hidden_BASF_frozen'
    records.append(row)
    for name in BASELINES:
        baseline=fit_baseline(development,name)
        prediction=row.copy()
        prediction['predicted_percent']=100*baseline_prediction(batch,baseline)
        prediction['model']=name+'_BASF_frozen'
        records.append(prediction)
    predictions=pd.concat(records,ignore_index=True)
    predictions.to_csv(output/'frozen_transfer_predictions.csv',index=False)
    metrics=[]
    for (year,model),frame in predictions.groupby(['year','model']):
        metrics.append({'year':int(year),'model':model,**score(frame)})
    table=pd.DataFrame(metrics)
    table.to_csv(output/'frozen_transfer_metrics.csv',index=False)
    receipt={'BASF_fit_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
        'BASF_fit_path':args.fit,
        'simulator_sha256':hashlib.sha256((ROOT/'model/primary_secondary/core.py').read_bytes()).hexdigest(),
        'French_targets_fitted':False,'observation_mapping':'I+R, not BASF damage N+I+R',
        'natural_field_onset_validation':False,'independent_stations':1,'station_years':2,
        'claim_scope':'external observation-domain transfer with realized ERA5 and initial pycnidial conditioning',
        'max_mass_error':float(abs(trajectory.state.sum(axis=-1)-1).max()),
        'minimum_state':float(trajectory.state.min())}
    (output/'transfer_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(table.to_string(index=False))


if __name__=='__main__': main()
