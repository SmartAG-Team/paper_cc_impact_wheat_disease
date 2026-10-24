"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analysis.methods.joint_disease import models as joint
from analysis.methods.joint_disease.data import site_assignments
from analysis.methods.disease_trajectory import models as trajectory
from analysis.methods.residual_disease import config

def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def fit_pair(training, bounds, z, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if set(bounds.field_id) != set(training.field_id):
        raise ValueError('Training onset fields do not match')
    training.to_csv(output / 'training_membership.csv', index=False)
    j = joint.fit(z['features'], z['valid'], training, bounds, **config.JOINT)
    write(output / 'joint_fit.json', j)
    o = trajectory.fit_onset(z['features'], z['valid'], z['names'], training, bounds, family='full', ridge=config.TRAJECTORY['onset_ridge'])
    write(output / 'onset_fit.json', o)
    occurrence = trajectory.predict_onset(o, z['features'], z['valid'])
    s = trajectory.fit_severity(z['features'], z['valid'], z['names'], training, occurrence, family='full', ridge=config.TRAJECTORY['severity_ridge'], integral_weight=config.TRAJECTORY['integral_weight'])
    write(output / 'severity_fit.json', s)
    return dict(joint=j, onset=o, severity=s)

def load_pair(folder):
    folder = Path(folder)
    return dict(joint=json.loads((folder / 'joint_fit.json').read_text()), onset=json.loads((folder / 'onset_fit.json').read_text()), severity=json.loads((folder / 'severity_fit.json').read_text()))

def predict_pair(fitted, z, frame, bounds=None):
    fields = np.sort(frame.field_index.unique().astype(int))
    lookup = {int(k): i for i, k in enumerate(fields)}
    x = z['features'][fields]
    valid = z['valid'][fields]
    j = joint.predict(fitted['joint'], x, valid)
    occurrence = trajectory.predict_onset(fitted['onset'], x, valid)
    severity = trajectory.predict_severity(fitted['severity'], x, occurrence)
    index = frame.field_index.map(lookup).to_numpy(int)
    day = frame.day_index.to_numpy(int) - 1
    leaf = frame.leaf_index.to_numpy(int)
    out = frame.copy()
    for name, values in [('joint_base', j['severity']), ('joint_occurrence', j['occurrence']), ('trajectory_base', severity), ('trajectory_occurrence', occurrence)]:
        out[name] = values[index, day, leaf]
    events = None
    if bounds is not None:
        events = bounds.copy()
        bi = events.field_index.map(lookup).to_numpy(int)
        bl = events.leaf_index.to_numpy(int)
        for name, prob in [('joint', j['occurrence']), ('trajectory', occurrence)]:
            days, crossed = trajectory.median_day(prob, valid)
            events[name + '_base_day'] = days[bi, bl]
            events[name + '_base_crossed'] = crossed[bi, bl]
    return (out, events)

def crossfit(training, bounds, z, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if training.global_target_index.duplicated().any():
        raise ValueError('Duplicate training target')
    nfold = min(3, training.site_id.nunique())
    if nfold < 2:
        raise ValueError('At least two training sites required for residual cross-fitting')
    assignment = site_assignments(training, nfold, config.SEED)
    write(output / 'site_assignment.json', assignment)
    parts = []
    for fold in range(nfold):
        train = training.loc[training.site_id.map(assignment).ne(fold)]
        validation = training.loc[training.site_id.map(assignment).eq(fold)]
        if set(train.site_id) & set(validation.site_id):
            raise ValueError('Residual cross-fit site leakage')
        b = bounds.loc[bounds.field_id.isin(train.field_id)]
        folder = output / f'fold_{fold}'
        fitted = fit_pair(train, b, z, folder)
        write(folder / 'held_sites.json', sorted(validation.site_id.unique()))
        pred, _ = predict_pair(fitted, z, validation)
        pred['residual_fold'] = fold
        pred.to_csv(folder / 'held_predictions.csv', index=False)
        parts.append(pred)
    combined = pd.concat(parts, ignore_index=True)
    if combined.global_target_index.duplicated().any() or set(combined.global_target_index) != set(training.global_target_index):
        raise ValueError('Incomplete residual cross-fitting targets')
    combined = combined.set_index('global_target_index').loc[training.global_target_index].reset_index()
    combined['joint_residual'] = combined.observed_percent - combined.joint_base
    combined['trajectory_residual'] = combined.observed_percent - combined.trajectory_base
    combined.to_csv(output / 'residual_training.csv', index=False)
    return combined
