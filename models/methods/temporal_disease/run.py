"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_temporal_validation_20261010')
import datetime as dt
import json
import platform
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from analysis.methods.temporal_disease import study, evaluation
from analysis.methods.joint_disease import run as joint

def fit_year(partition, z, year, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    training, validation = (partition['training'], partition['validation'])
    train_bounds, test_bounds = (partition['train_bounds'], partition['test_bounds'])
    if not training.season_year.lt(year).all() or not validation.season_year.eq(year).all():
        raise ValueError('Invalid forward-year training boundary')
    if not set(train_bounds.field_id) == set(training.field_id) or not set(test_bounds.field_id) == set(validation.field_id):
        raise ValueError('Onset bounds outside the appropriate training or test fields')
    study.write_json(output / 'inner_assignment.json', partition['assignment'])
    selected = joint.select(training, train_bounds, z, partition['assignment'], output)
    fitted = joint.models.fit(z['features'], z['valid'], training, train_bounds, **selected)
    fitted.update(validation_sites=sorted(validation.site_id.unique()), validation_targets=sorted(validation.global_target_index.astype(int)), test_year=year, training_years=sorted(training.season_year.unique().astype(int).tolist()))
    study.write_json(output / 'joint_fit.json', fitted)
    out, bounds, daily = joint.extract_prediction(fitted, z, validation, test_bounds)
    np.savez_compressed(output / 'daily_predictions.npz', **daily)
    leaf = joint.benchmarks.fit_leaf_mean(training)
    occurrence = joint.benchmarks.fit_leaf_mean(training.assign(observed_percent=joint.e.event_labels(training)))
    out['fold_leaf_mean'] = joint.benchmarks.predict_leaf_mean(leaf, out.leaf_index)
    out['fold_occurrence'] = joint.benchmarks.predict_leaf_mean(occurrence, out.leaf_index)
    out['event_label'] = joint.e.event_labels(validation)
    calendar = joint.benchmarks.fit_calendar(train_bounds)
    bounds['calendar_day'] = joint.benchmarks.predict_calendar(calendar, bounds.leaf_index)
    bounds['calendar_distance'] = joint.e.interval_distance(bounds, bounds.calendar_day)
    boosting = joint.benchmarks.fit_boosting(z['features'], training)
    out['boosting'] = joint.benchmarks.predict_boosting(boosting, z['features'], out)
    joblib.dump(boosting, output / 'boosting.joblib')
    study.write_json(output / 'benchmarks.json', dict(leaf=leaf, occurrence=occurrence, calendar=calendar, boosting_settings=joint.benchmarks.BOOSTING, test_year=year, training_targets=sorted(training.global_target_index.astype(int)), validation_targets=sorted(validation.global_target_index.astype(int)), training_sites=sorted(training.site_id.unique()), validation_sites=sorted(validation.site_id.unique())))
    out.to_csv(output / 'assessments.csv', index=False)
    bounds.to_csv(output / 'intervals.csv', index=False)
    print(year, 'completed:', len(training), 'earlier assessments;', len(out), 'future assessments', flush=True)
    return (out, bounds)

def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('A new result directory is required')
    protocol = study.check_protocol()
    observations, fields, bounds, z = study.load_inputs()
    output.mkdir(parents=True)
    start = time.time()
    study.write_json(output / 'registration.json', dict(started_before_fitting=dt.datetime.now(dt.timezone.utc).isoformat(), protocol_sha256=study.sha(study.HERE / 'protocol.json'), source_sha256=protocol['source_sha256'], python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__))
    all_assessments, all_bounds, membership = ([], [], [])
    for year in study.YEARS:
        p = study.partition(observations, fields, bounds, year)
        for role in ['training', 'validation']:
            membership.append(p[role][['global_target_index', 'field_id', 'field_index', 'site_id', 'coordinate_year', 'source', 'season_year']].assign(test_year=year, role=role))
        assessments, intervals = fit_year(p, z, year, output / f'year_{year}')
        all_assessments.append(assessments)
        all_bounds.append(intervals)
    assessments, intervals = (pd.concat(all_assessments, ignore_index=True), pd.concat(all_bounds, ignore_index=True))
    if len(assessments) != 2018 or assessments.global_target_index.duplicated().any() or set(assessments.global_target_index) != set(observations.loc[observations.season_year.ge(2015), 'global_target_index']):
        raise ValueError('Incomplete future-year population')
    assessments.to_csv(output / 'assessment_predictions.csv', index=False)
    intervals.to_csv(output / 'onset_predictions.csv', index=False)
    pd.concat(membership, ignore_index=True).to_csv(output / 'membership.csv', index=False)
    for name, table in evaluation.evaluate(assessments, intervals, replicates=protocol['bootstrap_replicates']).items():
        filename = name + ('.csv.gz' if name == 'paired_draws' else '.csv')
        compression = {'method': 'gzip', 'mtime': 0} if name == 'paired_draws' else None
        table.to_csv(output / filename, index=False, compression=compression)
    study.check_protocol()
    receipt = dict(status='completed', elapsed_seconds=time.time() - start, joint_fits=65, test_years=list(study.YEARS), future_assessments=len(assessments), new_site_assessments=int(assessments.new_site.sum()), output_sha256={str(p.relative_to(output)): study.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    study.write_json(output / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'output_sha256'}, indent=2), flush=True)
