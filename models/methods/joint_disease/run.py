"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_joint_forecast_20261010')
import datetime
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
from analysis.methods.joint_disease import data, models, benchmarks, evaluation as e
from analysis.methods.observation_model import observation as metrics

def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def assert_boundary(training, validation):
    metrics.assert_disjoint(training, validation)

def choose_setting(rows):
    return min(rows, key=lambda r: (0.5 * (r['window_rmse'] / 25 + r['onset_distance'] / 10), -r['ridge'], -r['onset_weight']))

def check_inputs():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def extract_prediction(fitted, z, observations, intervals):
    fields = np.sort(observations.field_index.unique().astype(int))
    lookup = {k: i for i, k in enumerate(fields)}
    predicted = models.predict(fitted, z['features'][fields], z['valid'][fields])
    med, crossed = models.median_day(predicted['occurrence'], z['horizons'][fields])
    out = observations.copy()
    index = out.field_index.map(lookup).to_numpy(int)
    day = out.day_index.to_numpy(int) - 1
    leaf = out.leaf_index.to_numpy(int)
    out['joint'] = predicted['severity'][index, day, leaf]
    out['occurrence'] = predicted['occurrence'][index, day, leaf]
    bounds = intervals.copy()
    bi = bounds.field_index.map(lookup).to_numpy(int)
    bl = bounds.leaf_index.to_numpy(int)
    bounds['joint_day'] = med[bi, bl]
    bounds['joint_crossed'] = crossed[bi, bl]
    bounds['joint_distance'] = e.interval_distance(bounds, bounds.joint_day)
    return (out, bounds, dict(**predicted, field_indices=fields, median_day=med, crossed=crossed))

def select(training, intervals, z, assignment, output):
    if set(assignment) != set(training.site_id):
        raise ValueError('Inner assignment has missing or extra sites')
    rows = []
    all_predictions = []
    all_intervals = []
    for setting_id, setting in enumerate(models.GRID):
        predictions = []
        bounds = []
        for fold in range(3):
            development = training.loc[training.site_id.map(assignment).ne(fold)]
            validation = training.loc[training.site_id.map(assignment).eq(fold)]
            assert_boundary(development, validation)
            bi = intervals.loc[intervals.site_id.isin(development.site_id)]
            vi = intervals.loc[intervals.site_id.isin(validation.site_id)]
            fitted = models.fit(z['features'], z['valid'], development, bi, **setting)
            fitted['validation_sites'] = sorted(validation.site_id.unique())
            write_json(output / f'inner_{setting_id}_{fold}.json', fitted)
            p, b, _ = extract_prediction(fitted, z, validation, vi)
            predictions.append(p.assign(inner_fold=fold))
            bounds.append(b.assign(inner_fold=fold))
        combined = pd.concat(predictions, ignore_index=True)
        events = pd.concat(bounds, ignore_index=True)
        if set(combined.global_target_index) != set(training.global_target_index) or len(combined) != len(training):
            raise ValueError('Incomplete inner out-of-fold predictions')
        windows = metrics.windows(combined, ['joint'])
        score = metrics.scores(windows, 'joint')
        genuine = events.loc[events.genuine]
        if genuine.empty:
            raise ValueError('No inner genuine onset support')
        distance = float(np.average(genuine.joint_distance, weights=metrics.hierarchical_weights(genuine)))
        rows.append(dict(setting_id=setting_id, **setting, window_rmse=score['rmse'], onset_distance=distance, selection_score=0.5 * (score['rmse'] / 25 + distance / 10), unresolved=int((~events.joint_crossed).sum())))
        all_predictions.append(combined.assign(setting_id=setting_id))
        all_intervals.append(events.assign(setting_id=setting_id))
        print(output.name, 'setting', setting_id, 'converged', flush=True)
    pd.concat(all_predictions, ignore_index=True).to_csv(output / 'inner_assessments.csv', index=False)
    pd.concat(all_intervals, ignore_index=True).to_csv(output / 'inner_intervals.csv', index=False)
    pd.DataFrame(rows).to_csv(output / 'selection_scores.csv', index=False)
    chosen = choose_setting(rows)
    write_json(output / 'selection.json', dict(selected=chosen, training_sites=sorted(training.site_id.unique()), rule='0.5*(window-mean RMSE/25 + genuine interval error/10); exact ties prefer stronger regularization then onset weight'))
    return {key: chosen[key] for key in ['ridge', 'onset_weight']}

def fit_partition(training, validation, intervals, z, assignment, output):
    output.mkdir(parents=True)
    assert_boundary(training, validation)
    train_bounds = intervals.loc[intervals.site_id.isin(training.site_id)]
    test_bounds = intervals.loc[intervals.site_id.isin(validation.site_id)]
    selected = select(training, train_bounds, z, assignment, output)
    fitted = models.fit(z['features'], z['valid'], training, train_bounds, **selected)
    fitted['validation_sites'] = sorted(validation.site_id.unique())
    write_json(output / 'joint_fit.json', fitted)
    out, bounds, daily = extract_prediction(fitted, z, validation, test_bounds)
    np.savez_compressed(output / 'daily_predictions.npz', **daily)
    leaf = benchmarks.fit_leaf_mean(training)
    out['fold_leaf_mean'] = benchmarks.predict_leaf_mean(leaf, out.leaf_index)
    event_training = training.assign(observed_percent=e.event_labels(training))
    occurrence = benchmarks.fit_leaf_mean(event_training)
    out['fold_occurrence'] = benchmarks.predict_leaf_mean(occurrence, out.leaf_index)
    out['event_label'] = e.event_labels(validation)
    calendar = benchmarks.fit_calendar(train_bounds)
    bounds['calendar_day'] = benchmarks.predict_calendar(calendar, bounds.leaf_index)
    bounds['calendar_distance'] = e.interval_distance(bounds, bounds.calendar_day)
    boosted = benchmarks.fit_boosting(z['features'], training)
    out['boosting'] = benchmarks.predict_boosting(boosted, z['features'], out)
    joblib.dump(boosted, output / 'boosting.joblib')
    write_json(output / 'benchmarks.json', dict(leaf=leaf, occurrence=occurrence, calendar=calendar, boosting_settings=benchmarks.BOOSTING, training_targets=sorted(training.global_target_index.astype(int).tolist()), training_sites=sorted(training.site_id.unique()), validation_targets=sorted(validation.global_target_index.astype(int).tolist()), validation_sites=sorted(validation.site_id.unique())))
    out.to_csv(output / 'assessments.csv', index=False)
    bounds.to_csv(output / 'intervals.csv', index=False)
    print(output.name, 'completed', len(validation), 'held-out assessments', flush=True)
    return (out, bounds)

def fixed_references(assessments, intervals):
    old = pd.read_csv(data.ROOT / data.SOURCE_PATHS['assessments'])
    cols = ['global_target_index', 'raw_retained', 'mapped_retained', 'raw_lstm_extended_q50', 'leaf_mean']
    old = old[cols].rename(columns={'leaf_mean': 'archived_leaf_mean'})
    frame = assessments.merge(old, on='global_target_index', how='left', validate='one_to_one')
    onsets = pd.read_csv(data.ROOT / 'data/final_study/evidence/phenology_disease_coupling/results/first_positive_intervals.csv')
    keys = ['field_id', 'endpoint_series', 'leaf_index']
    cols = keys + ['retained_distance_days', 'lstm_extended_q50_distance_days']
    onsets = onsets[cols].rename(columns={'retained_distance_days': 'retained_distance', 'lstm_extended_q50_distance_days': 'lstm_extended_q50_distance'})
    intervals = intervals.merge(onsets, on=keys, how='left', validate='one_to_one')
    if frame[e.PREDICTIONS].isna().any().any() or intervals[['retained_distance', 'lstm_extended_q50_distance']].isna().any().any():
        raise ValueError('Missing frozen comparison')
    return (frame, intervals)

def evaluate(assessments, intervals, output):
    frame, bounds = fixed_references(assessments, intervals)
    frame.to_csv(output / 'assessment_predictions.csv', index=False)
    bounds.to_csv(output / 'onset_predictions.csv', index=False)
    for population in ['all_records', 'original_evaluation']:
        part = frame if population == 'all_records' else frame.loc[frame.partition.ne('calibration')]
        onset = bounds if population == 'all_records' else bounds.loc[bounds.partition.ne('calibration')]
        windows, summary = e.summarize(part, e.PREDICTIONS)
        scores = e.onset_scores(onset)
        complement, bins = e.complementary(part)
        for name, value in [('windows', windows), ('summary', summary), ('onset_summary', scores), ('complementary', complement), ('calibration', bins)]:
            value.to_csv(output / f'{population}_{name}.csv', index=False)
        if population == 'original_evaluation':
            if len(part) != 2033 or len(windows) != 573:
                raise ValueError('Primary comparison population changed')
            comparisons, draws = e.magnitude_comparisons(part, windows)
            alt, alt_draws = e.complementary_bootstrap(part, onset)
            comparisons.to_csv(output / 'paired_comparisons.csv', index=False)
            draws.to_csv(output / 'paired_draws.csv.gz', index=False)
            alt.to_csv(output / 'complementary_intervals.csv', index=False)
            alt_draws.to_csv(output / 'complementary_draws.csv.gz', index=False)
            write_json(output / 'decision.json', e.decide(summary, comparisons, scores, alt))

def run(output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('New result directory required')
    check_inputs()
    output.mkdir(parents=True)
    start = time.time()
    sources = [p for p in data.HERE.glob('*.py') if not p.name.startswith('test_')]
    write_json(output / 'registration.json', dict(registered_before_outer_fitting=datetime.datetime.now(datetime.timezone.utc).isoformat(), protocol_sha256=data.sha(data.HERE / 'protocol.json'), source_sha256={str(p.relative_to(data.ROOT)): data.sha(p) for p in sources}, reference_onset_sha256=data.sha(data.ROOT / 'data/final_study/evidence/phenology_disease_coupling/results/first_positive_intervals.csv'), python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, torch=torch.__version__))
    inputs = data.HERE / 'inputs'
    observations = pd.read_csv(inputs / 'observations.csv')
    intervals = pd.read_csv(inputs / 'onset_intervals.csv')
    folds = json.loads((inputs / 'folds.json').read_text())
    with np.load(inputs / 'features.npz', allow_pickle=False) as arrays:
        z = {k: arrays[k] for k in arrays.files}
    all_predictions = []
    all_intervals = []
    for fold in range(5):
        test = observations.loc[observations.site_id.map(folds['outer']).eq(fold)]
        train = observations.loc[observations.site_id.map(folds['outer']).ne(fold)]
        p, b = fit_partition(train, test, intervals, z, folds['inner'][str(fold)], output / f'outer_{fold}')
        all_predictions.append(p.assign(outer_fold=fold))
        all_intervals.append(b.assign(outer_fold=fold))
    evaluate(pd.concat(all_predictions, ignore_index=True), pd.concat(all_intervals, ignore_index=True), output)
    transport_predictions = []
    transport_intervals = []
    for source in sorted(observations.source.unique()):
        train = observations.loc[observations.source.ne(source)]
        test = observations.loc[observations.source.eq(source)]
        p, b = fit_partition(train, test, intervals, z, folds['leave_source_out_inner'][source], output / f'transport_{source}')
        transport_predictions.append(p.assign(held_source=source))
        transport_intervals.append(b.assign(held_source=source))
    transport = output / 'transport'
    transport.mkdir()
    evaluate(pd.concat(transport_predictions, ignore_index=True), pd.concat(transport_intervals, ignore_index=True), transport)
    check_inputs()
    receipt = dict(status='completed', elapsed_seconds=time.time() - start, outer_folds=5, source_holdouts=2, joint_fits=7 * (len(models.GRID) * 3 + 1), original_evaluation_assessments=2033, original_evaluation_windows=573, all_assessments=len(observations), output_sha256={str(p.relative_to(output)): data.sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    write_json(output / 'receipt.json', receipt)
    print(json.dumps({k: v for k, v in receipt.items() if k != 'output_sha256'}, indent=2), flush=True)
