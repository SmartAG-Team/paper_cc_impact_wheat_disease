"""Absolute fit diagnostics under the frozen final-endpoint weighting policy."""
from pathlib import Path
import json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_empirical_benchmarks import endpoint
from analysis.paper_study.run_regional import sha


def main():
    path=ROOT/'data/paper_study/empirical_benchmarks/frozen_all_predictions.parquet'
    predictions=pd.read_parquet(path);rows=[]
    for part in ['validation','external']:
        for upper in [False,True]:
            for model in ['seasonal_seir','weighted_ridge','constrained_forest','calibration_only_leaf_rank_mean']:
                f=predictions[predictions.partition.eq(part)&predictions.model.eq(model)]
                if part=='external':f=f[f.strict_location_disjoint]
                f=endpoint(f,True,upper)
                y=f.value.to_numpy();pred=f.predicted_percent.to_numpy();w=f.weight.to_numpy(copy=True);w/=w.sum()
                avg=w@y;var=w@((y-avg)**2);err=pred-y;pm=w@pred
                r=np.sum(w*(y-avg)*(pred-pm))/np.sqrt(var*np.sum(w*(pred-pm)**2))
                rows.append(dict(partition=part,leaf_scope='upper_three' if upper else 'all_ordinal',model=model,
                    n=len(f),rmse=np.sqrt(w@(err**2)),mae=w@abs(err),bias=w@err,
                    weighted_r_squared=1-(w@(err**2))/var,weighted_pearson_r=r,
                    observed_weighted_mean=avg,predicted_weighted_mean=pm,
                    coordinate_years=f.coordinate_year.nunique(),fields=f.field_id.nunique()))
    frame=pd.DataFrame(rows)
    frame.to_csv(ROOT/'data/paper_study/publication/field_absolute_fit_diagnostics.csv',index=False)
    (ROOT/'analysis/paper_study/manuscript_review_20261006/field_diagnostics_provenance.json').write_text(json.dumps(dict(
        source_path=str(path.relative_to(ROOT)),source_sha256=sha(path),source_code_sha256=sha(Path(__file__)),
        r_squared_definition='1 - weighted squared error / weighted observed variance',
        reference_mean_is_evaluation_mean=True, reference_mean_is_operational_comparator=False,
        weights='Equal coordinate-year, field and selected source-leaf endpoint',
        field_parameters_modified=False),indent=2)+'\n')
    print(frame[frame.model.eq('seasonal_seir')].to_string(index=False))


if __name__=='__main__':main()
