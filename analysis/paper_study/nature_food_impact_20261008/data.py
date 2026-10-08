"""Pooled reporting of existing evaluation predictions; no model refitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.paper_study.nature_food_fix_20261007.validation.statistics import hierarchy_weights, paired_bootstrap

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
BASE=ROOT/'analysis/paper_study/nature_food_fix_20261007/validation'
GRID=ROOT/'analysis/paper_study/full_grid_climate_20261008/results'
PARTITIONS=['reused_BASF2019','reused_strict_Corteva']
FIELD_PARTITIONS=['reused_development_2019','reused_external_strict']
DERIVED=HERE/'derived'
CANOPY='GS65_85_lost_had3'
SEVERITY='GS65_85_functional_lost_fraction'
ONSET='F1_symptom_relative_anthesis_days'
FREQUENCY='F1_symptom_before85'
YIELD='conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai'
MODELS=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
SCENARIOS=['ssp126','ssp245','ssp585']
REGIONS=['Atlantic','Continental','Boreal','Mediterranean','Steppic','Pannonian','Alpine','Black Sea']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_pooled():
    DERIVED.mkdir(exist_ok=True)
    paths=[BASE/'paired_onset_membership.csv',BASE/'source_stage_and_prediction_membership.csv',
           ROOT/'analysis/paper_study/full_validation_20261007/field_stage_scores_all.csv']
    onset=pd.read_csv(paths[0])
    onset=onset[onset.partition.isin(PARTITIONS)&onset.censoring.eq('two_sided')&onset.leaf_index.lt(3)].copy()
    onset['model']=onset.onset_distance_days
    onset['benchmark']=onset.onset_distance_days_benchmark
    severity=pd.read_csv(paths[1])
    severity=severity[severity.partition.isin(PARTITIONS)].copy()
    detection=severity.copy()
    severity['model']=severity.frozen_damage_percent
    severity['benchmark']=severity.leaf_mean_percent
    severity['observed']=severity.observed_percent
    training=pd.read_csv(paths[1]);training=training[training.partition.eq('calibration')]
    assert not set(severity.field_id)&set(training.field_id)
    # Coordinate-year groups retain their existing geographic identities. A
    # cross-source overlap would be one group rather than a fabricated replicate.
    records=[];rng=np.random.default_rng(20261008)
    for endpoint,frame,kind in [('symptom_onset',onset,'mean'),('severity',severity,'severity'),
                              ('symptom_detection',detection,'detection')]:
        groups=sorted(frame.coordinate_year.unique())
        draws=rng.integers(0,len(groups),size=(20000,len(groups)),dtype=np.int32)
        point,reps=paired_bootstrap(frame,draws,kind=kind)
        for metric,value in point.items():
            finite=reps[metric][np.isfinite(reps[metric])]
            lo,hi=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
            records.append(dict(endpoint=endpoint,metric=metric,value=value,lower95=lo,upper95=hi,
                records=len(frame),fields=frame.field_id.nunique(),coordinate_years=len(groups),
                requested_replicates=20000,valid_replicates=len(finite)))
        if kind=='severity':
            weights=hierarchy_weights(frame);obs=frame.observed.to_numpy(float)
            var=np.sum(weights*(obs-np.sum(weights*obs))**2)
            for label in ['model','benchmark']:
                r2=1-np.sum(weights*(frame[label].to_numpy()-obs)**2)/var
                records.append(dict(endpoint=endpoint,metric=label+'_r2',value=r2,lower95=np.nan,upper95=np.nan,
                    records=len(frame),fields=frame.field_id.nunique(),coordinate_years=len(groups),
                    requested_replicates=0,valid_replicates=0))
        frame.to_csv(DERIVED/f'pooled_{endpoint}_membership.csv',index=False)
        np.savez_compressed(DERIVED/f'pooled_{endpoint}_bootstrap.npz',group_ids=np.array(groups),draws=draws,**reps)
    result=pd.DataFrame(records);result.to_csv(DERIVED/'pooled_evaluation_metrics.csv',index=False)
    field=pd.read_csv(paths[2]);field=field[field.forcing_scope.eq('full_season')].copy()
    field['cohort']=np.where(field.partition.eq('calibration'),'Calibration',
                             np.where(field.partition.isin(FIELD_PARTITIONS),'Field evaluation','Excluded'))
    field=field[field.cohort.ne('Excluded')]
    field.to_csv(DERIVED/'pooled_field_stage_records.csv',index=False)
    stages=[]
    for (cohort,event),part in field.groupby(['cohort','event']):
        q=part[part.genuinely_bracketed].copy()
        if len(q):
            q['leaf_index']=0;w=hierarchy_weights(q)
            error=float(w@q.distance_days.to_numpy())
        else:error=np.nan
        stages.append(dict(cohort=cohort,event=event,fields=part.field_id.nunique(),bracketed=len(q),
            compatible=int(q.compatible.sum()),coordinate_years=q.coordinate_year.nunique(),
            mean_interval_excess_days=error,median_interval_width_days=q.bracket_width_days.median() if len(q) else np.nan))
    pd.DataFrame(stages).to_csv(DERIVED/'pooled_field_stage_metrics.csv',index=False)
    receipt=dict(status='complete',model_refitted=False,source_identifiers_retained=True,
        calibration_excluded_from_evaluation=True,weighting='Equal coordinate-year, field, source leaf and assessment within its parent',
        pooling_scope='Fixed retrospective evaluation predictions; source percentage scoring conventions retained',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in paths})
    (DERIVED/'pooled_evaluation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return result


def pooled(endpoint,metric):
    d=pd.read_csv(DERIVED/'pooled_evaluation_metrics.csv')
    return d[d.endpoint.eq(endpoint)&d.metric.eq(metric)].iloc[0]


def grid_ready():
    return (GRID/'completion_receipt.json').is_file()


def grid_data():
    receipt=json.loads((GRID/'completion_receipt.json').read_text())
    if receipt.get('annual_jobs')!=810 or not receipt.get('full_grid_census'):
        raise ValueError('Complete full-grid climate comparison required.')
    for name in ['full_grid_ensemble_paired_changes.parquet','country_changes.csv','environmental_region_changes.csv']:
        if receipt['outputs_sha256'].get(name)!=sha(GRID/name):
            raise ValueError('Grid summary changed: '+name)
    cells=pd.read_parquet(GRID/'full_grid_ensemble_paired_changes.parquet')
    regions=pd.read_csv(GRID/'environmental_region_changes.csv')
    countries=pd.read_csv(GRID/'country_changes.csv')
    return cells,regions,countries
