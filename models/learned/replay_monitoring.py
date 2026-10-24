"""Replay the manuscript's 20 saved monitoring folds without refitting."""
import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.disease_improvement.discrepancy_experiment import forecast_baseline
from analysis.disease_improvement.discrepancy_filter import predict_discrepancy
from analysis.disease_improvement.innovation_common import ROOT, load_cohort

DATA = ROOT / 'data/final_monitoring'


def run(output=None):
    if output is not None and Path(output).exists():
        raise FileExistsError('A fresh replay output directory is required')
    contract=json.loads((DATA/'manifest.json').read_text())
    for row in contract['files']:
        with (ROOT/row['path']).open('rb') as stream:
            assert hashlib.file_digest(stream,'sha256').hexdigest()==row['sha256'],row['path']
    observations,arrays,history,_,splits,positions=load_cohort()
    reference=pd.read_csv(DATA/'reference/data/fig1/predictions.csv',float_precision='round_trip')
    rows=[];maximum=0.
    with threadpool_limits(limits=1):
        for split in splits:
            fitted=joblib.load(DATA/'folds'/f"{split['fold']}_model.joblib")
            assert fitted['train_ids']==split['train'] and fitted['test_ids']==split['test']
            query=positions.loc[split['test']].to_numpy()
            assert observations.iloc[positions.loc[split['train']]].season_year.max()<observations.iloc[query].season_year.min()
            context=np.flatnonzero(observations.field_id.isin(observations.iloc[query].field_id.unique()))
            local=pd.Series(np.arange(len(context)),index=context).loc[query].to_numpy()
            baseline=forecast_baseline(fitted['base'],arrays,history,observations,context)
            predicted=predict_discrepancy(fitted['corrections']['logit']['parameters'],observations.iloc[context],baseline,'logit')[local]
            design='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
            expected=reference[reference.evaluation.eq(design)].set_index('global_target_index').loc[split['test'],'prediction_percent'].to_numpy()
            error=float(np.max(np.abs(predicted-expected)));maximum=max(maximum,error)
            np.testing.assert_allclose(predicted,expected,rtol=0,atol=1e-12)
            rows.append(dict(fold=split['fold'],evaluation=design,assessments=len(query),maximum_absolute_error=error))
    assert len(rows)==20 and len(reference)==4034
    result=dict(status='passed',folds=len(rows),assessments_per_design=2017,maximum_absolute_prediction_error=maximum,
        refitted=False,information='Strictly earlier own-field disease; query-day and later disease withheld',records=rows)
    if output is not None:
        output=Path(output);output.mkdir(parents=True)
        (output/'checks.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path)
    print(json.dumps(run(parser.parse_args().output),indent=2))
