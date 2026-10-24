"""Recompute the candidate-comparison table from bundled predictions and unchanged evaluation weights."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def recompute(package, output):
    package=Path(package).resolve();data=package/'data/endpoint_development'
    frame=pd.read_csv(data/'diagnostic_predictions.csv',float_precision='round_trip')
    expected=pd.read_csv(data/'candidate_changes.csv')
    models=['baseline_prediction','endpoint_joint_mean','endpoint_native_growth','endpoint_severe_priority2']
    labels=dict(zip(models[1:],['Joint prior mean','Same-code growth','Severe-priority fit']))
    scores=[];changes=[]
    for design,part in frame.groupby('evaluation'):
        assert len(part)==2017 and part.global_target_index.is_unique
        fields=part.groupby('coordinate_year').field_id.transform('nunique')
        leaves=part.groupby(['coordinate_year','field_id']).leaf_index.transform('nunique')
        records=part.groupby(['coordinate_year','field_id','leaf_index']).leaf_index.transform('size')
        rebuilt=1/(fields*leaves*records)
        np.testing.assert_array_equal(rebuilt,part.original_hierarchy_weight)
        for population,mask in [('all',np.ones(len(part),bool)),('severe',part.observed_percent.gt(75))]:
            f=part[mask];w=f.original_hierarchy_weight.to_numpy();y=f.observed_percent.to_numpy()
            for model in models:
                error=f[model].to_numpy()-y
                scores.append(dict(evaluation=design,population=population,model=model,n=len(f),
                    rmse=float(np.sqrt(np.average(error**2,weights=w))),bias=float(np.average(error,weights=w))))
    table=pd.DataFrame(scores)
    for design in frame.evaluation.unique():
        scope=table[table.evaluation.eq(design)].set_index(['population','model'])
        for model in models[1:]:
            changes.append(dict(model=labels[model],evaluation='Temporal' if design=='temporal' else 'Spatial',
                delta_rmse=scope.loc[('all',model),'rmse']-scope.loc[('all',models[0]),'rmse'],
                delta_severe_rmse=scope.loc[('severe',model),'rmse']-scope.loc[('severe',models[0]),'rmse'],
                delta_severe_bias=scope.loc[('severe',model),'bias']-scope.loc[('severe',models[0]),'bias']))
    changes=pd.DataFrame(changes)
    keys=['model','evaluation'];columns=['delta_rmse','delta_severe_rmse','delta_severe_bias']
    np.testing.assert_allclose(changes.set_index(keys).sort_index()[columns],
        expected.set_index(keys).sort_index()[columns],rtol=0,atol=1e-10)
    output=Path(output).resolve()
    if output==data or data in output.parents:raise ValueError('Write fresh receipts outside the input directory')
    output.mkdir(parents=True,exist_ok=True)
    table.to_csv(output/'scores.csv',index=False);changes.to_csv(output/'candidate_changes.csv',index=False)
    checks=dict(status='passed',records=len(frame),records_per_design=2017,models=4,
        original_evaluation_weights_rebuilt_exactly=True,table_s4_reproduced=True,
        candidate_comparison_reproduced=True,current_supplementary_table=3,fitted=False)
    (output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(checks,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();recompute(args.package,args.output)
