"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
from pathlib import Path
from datetime import datetime, timezone
import time
import json
import numpy as np
import pandas as pd
from analysis.methods.residual_disease import study, pipeline, evaluation

def select(frame):
    rows = []
    for spec in study.candidate_specs():
        prediction = frame[f"candidate_{spec['candidate_id']}"]
        result = study.loss(frame, prediction)
        rows.append(dict(**spec, **result, point_score=result['assessment_mse'] / 10000))
    choices = {'selected_pipeline': study.choose(rows)}
    for family in sorted({row['family'] for row in rows}):
        choices[family] = study.choose([row for row in rows if row['family'] == family])
    choices['joint_rf_point'] = study.choose([row for row in rows if row['family'] == 'joint_rf'], 'point_score')
    return (pd.DataFrame(rows), choices)

def fit_partition(partition, fields, z, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train, test = (partition['training'], partition['validation'])
    bounds, test_bounds = (partition['train_bounds'], partition['test_bounds'])
    assignment = partition['assignment']
    study.write(output / 'inner_assignment.json', assignment)
    train.to_csv(output / 'training_membership.csv', index=False)
    test.to_csv(output / 'test_membership.csv', index=False)
    inner = []
    for fold in range(3):
        fitting = train.loc[train.site_id.map(assignment).ne(fold)].copy()
        validation = train.loc[train.site_id.map(assignment).eq(fold)].copy()
        prediction, _ = pipeline.candidates(fitting, validation, bounds.loc[bounds.field_id.isin(fitting.field_id)].copy(), bounds.loc[bounds.field_id.isin(validation.field_id)].copy(), fields, z, output / f'inner_{fold}')
        inner.append(prediction.assign(inner_fold=fold))
        print(output.name, 'inner', fold, 'complete', flush=True)
    combined = pd.concat(inner, ignore_index=True)
    if combined.global_target_index.duplicated().any() or set(combined.global_target_index) != set(train.global_target_index):
        raise ValueError('Incomplete inner validation population')
    combined.to_csv(output / 'inner_predictions.csv', index=False)
    scores, choices = select(combined)
    scores.to_csv(output / 'selection_scores.csv', index=False)
    study.write(output / 'selected_choices.json', choices)
    prediction, events = pipeline.selected(train, test, bounds, test_bounds, fields, z, choices, output / 'outer')
    print(output.name, 'selected', choices['selected_pipeline']['family'], choices['selected_pipeline']['multiplier'], flush=True)
    return (prediction, events)

def attach_reference(prediction, reference):
    aligned = reference.set_index('global_target_index').loc[prediction.global_target_index]
    for key in ['field_id', 'site_id', 'source', 'observed_percent', 'day_index', 'leaf_index', 'new_site']:
        if not np.array_equal(prediction[key].to_numpy(), aligned[key].to_numpy()):
            raise ValueError('Archived target mismatch: ' + key)
    for col in ['joint', 'boosting', 'full_window', 'full_point', 'calendar_window']:
        prediction[col] = aligned[col].to_numpy()
    np.testing.assert_allclose(prediction.fold_leaf_mean, aligned.fold_leaf_mean, atol=1e-10, rtol=1e-12)
    return prediction

def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('New empty result destination required')
    protocol = study.check_protocol()
    observations, fields, bounds, z, reference = study.load()
    output.mkdir(parents=True)
    start = time.time()
    study.write(output / 'registration.json', dict(started_before_fitting=datetime.now(timezone.utc).isoformat(), protocol_sha256=study.sha(study.HERE / 'protocol.json'), source_sha256=protocol['source_sha256']))
    temporal, source, events = ([], [], [])
    for year in study.YEARS:
        partition = study.temporal_partition(observations, bounds, year)
        if not partition['training'].season_year.lt(year).all():
            raise ValueError('Chronology leak')
        predicted, onset = fit_partition(partition, fields, z, output / f'year_{year}')
        temporal.append(attach_reference(predicted, reference))
        events.append(onset.assign(test_year=year))
    for held in sorted(observations.source.unique()):
        partition = study.source_partition(observations, bounds, held)
        predicted, onset = fit_partition(partition, fields, z, output / f'source_{held}')
        source.append(predicted)
        events.append(onset.assign(held_source=held))
    temporal = pd.concat(temporal, ignore_index=True)
    source = pd.concat(source, ignore_index=True)
    eligible = observations.loc[observations.season_year.isin(study.YEARS)]
    for actual, expected in [(temporal, eligible), (source, observations)]:
        if actual.global_target_index.duplicated().any() or set(actual.global_target_index) != set(expected.global_target_index):
            raise ValueError('Lost or duplicated outer targets')
    temporal.to_csv(output / 'assessment_predictions.csv', index=False)
    source.to_csv(output / 'source_predictions.csv', index=False)
    pd.concat(events, ignore_index=True).to_csv(output / 'base_onset_predictions.csv', index=False)
    tables = evaluation.evaluate(temporal, source, replicates=protocol['bootstrap_replicates'])
    for name, table in tables.items():
        if name == 'decision':
            study.write(output / 'decision.json', table)
        else:
            table.to_csv(output / (name + '.csv'), index=False)
    study.check_protocol()
    receipt = dict(status='completed', elapsed_seconds=time.time() - start, base_fits=len(list(output.rglob('*fit.json'))), heads=len(list(output.rglob('*.joblib'))), temporal_assessments=len(temporal), new_site_assessments=int(temporal.new_site.sum()), source_assessments=len(source), output_sha256={str(p.relative_to(output)): study.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    study.write(output / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'output_sha256'}, indent=2), flush=True)
