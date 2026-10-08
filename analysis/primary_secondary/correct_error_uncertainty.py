"""Correct cluster dependence without recalibrating models or changing predictions."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib
import json
import pandas as pd
from calibration.primary_secondary.uncertainty import paired_location_bootstrap

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis/primary_secondary/calibration_v1'


def main():
    source=OUT/'predictions.csv'
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    predictions=pd.read_csv(source)
    records=[]
    for fold,frame in predictions.groupby('fold'):
        for candidate in ('primary_secondary_hidden','primary_secondary_selected_inner'):
            for baseline in ('persistence','global_logit_trend','thermal_logit_trend','canopy_logit_trend'):
                result=paired_location_bootstrap(frame,candidate,baseline)
                result['fold']=fold
                records.append(result)
    pd.DataFrame(records).to_csv(OUT/'paired_location_cluster_comparisons.csv',index=False)
    provenance={'completed_utc':datetime.now(timezone.utc).isoformat(),'predictions_sha256':digest,
        'predictions_unchanged':digest==hashlib.sha256(source.read_bytes()).hexdigest(),
        'older_coordinate_year_bootstrap':'superseded for inference; retained for audit',
        'scoring_estimand':'equal coordinate-year then series then targets; unchanged',
        'resampling_unit':'location with every coordinate-year retained',
        'single_location_intervals':'undefined, no CI reported',
        'interval_scope':'prediction-error uncertainty conditional on fitted models; not process or parameter intervals'}
    (OUT/'uncertainty_correction.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':main()
