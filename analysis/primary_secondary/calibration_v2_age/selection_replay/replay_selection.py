"""Retrospective replay of archived v2 training-only hyperparameter selection."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parent
PROJECT = HERE.parents[3]
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.numba_cache')
sys.dont_write_bytecode = True
sys.path.insert(0, str(PROJECT))

import numpy as np
import pandas as pd

from calibration.primary_secondary.field_data import prepare_basf
from calibration.primary_secondary.calibrate import make_folds, inner_folds, fit_mechanism, prediction_frame, score

ABSOLUTE_TOLERANCE_PP = 1e-8
RELATIVE_TOLERANCE = 1e-10
METRICS = ['rmse_pp', 'mae_pp', 'bias_pp']
COUNTS = ['n_coordinate_years', 'n_series', 'n_targets']
SEED = 20261004
STARTS = 3


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def watched_files():
    validation = json.loads((ARCHIVE / 'validation.json').read_text())
    paths = {PROJECT / name for name in validation['source_sha256']}
    paths.update(p for p in ARCHIVE.rglob('*') if p.is_file() and HERE not in p.parents)
    paths.update(p for p in (PROJECT / 'model/primary_secondary').rglob('*')
                 if p.is_file() and '__pycache__' not in p.parts)
    return {str(p.relative_to(PROJECT)): digest(p) for p in sorted(paths)}


def source_checks():
    validation = json.loads((ARCHIVE / 'validation.json').read_text())
    for relative, expected in validation['source_sha256'].items():
        if digest(PROJECT / relative) != expected:
            raise AssertionError(f'Archived source hash differs: {relative}')
    for filename in ['core.py', 'field_data.py', 'calibrate.py', 'uncertainty.py']:
        assert digest(PROJECT / 'model/primary_secondary' / filename) == digest(ARCHIVE / 'source_snapshot' / filename)


def independently_score(frame):
    f = frame.copy()
    e = f.predicted_percent - f.observed_percent
    f['sq'], f['absolute'], f['signed'] = e ** 2, abs(e), e
    series = f.groupby(['coordinate_year', 'series_id'])[['sq', 'absolute', 'signed']].mean()
    coordinate_year = series.groupby(level='coordinate_year').mean()
    return dict(rmse_pp=float(np.sqrt(coordinate_year.sq.mean())),
        mae_pp=float(coordinate_year.absolute.mean()), bias_pp=float(coordinate_year.signed.mean()),
        n_coordinate_years=len(coordinate_year), n_series=f.series_id.nunique(), n_targets=len(f))


def close(a, b):
    return bool(abs(a - b) <= ABSOLUTE_TOLERANCE_PP + RELATIVE_TOLERANCE * abs(b))


def archive_memberships(batch, folds):
    members, targets = [], []
    for outer in folds:
        working = batch.subset(outer['train'])
        for i, (train_ids, test_ids) in enumerate(inner_folds(working.episodes)):
            assert not (train_ids & test_ids)
            assert train_ids | test_ids == outer['train']
            assert not ((train_ids | test_ids) & outer['test'])
            train_locations = set(working.episodes.loc[working.episodes.series_id.isin(train_ids), 'location_id'])
            test_locations = set(working.episodes.loc[working.episodes.series_id.isin(test_ids), 'location_id'])
            assert not train_locations & test_locations
            for side, ids in [('train', train_ids), ('test', test_ids)]:
                for record in working.episodes.loc[working.episodes.series_id.isin(ids)].itertuples():
                    members.append(dict(outer_fold=outer['name'], inner_fold=i, side=side,
                        series_id=record.series_id, location_id=record.location_id,
                        coordinate_year=record.coordinate_year, trial_id=int(record.TrialId),
                        leaf_rank=int(record.leaf_rank), year=int(record.year),
                        start=str(record.start.date()), end=str(record.end.date())))
                f = working.targets.loc[working.targets.series_id.isin(ids),
                    ['source_row', 'series_id', 'location_id', 'coordinate_year', 'Date', 'day',
                     'conditioning', 'Value', 'leaf_rank']].copy()
                f['outer_fold'], f['inner_fold'], f['side'] = outer['name'], i, side
                targets.append(f)
    pd.DataFrame(members).to_csv(HERE / 'inner_membership.csv', index=False)
    pd.concat(targets, ignore_index=True).to_csv(HERE / 'inner_target_membership.csv', index=False)
    return len(members), sum(len(f) for f in targets)


def run(resume=False):
    source_checks()
    before = watched_files()
    versions = {package: importlib.metadata.version(package) for package in
                ['numpy', 'pandas', 'pyarrow', 'scipy', 'numba']}
    configuration = dict(evidence_type='retrospective reproducibility replay', seed=SEED, starts=STARTS,
        latent_grid_days=[10., 20., 30.], infectious_grid_days=[14., 21., 28.],
        metric_absolute_tolerance_pp=ABSOLUTE_TOLERANCE_PP, metric_relative_tolerance=RELATIVE_TOLERANCE,
        count_and_winner_tolerance='exact', python=platform.python_version(), package_versions=versions,
        replay_code_sha256=digest(Path(__file__)), original_files_sha256=before,
        numba_cache_directory=str((HERE / '.numba_cache').relative_to(PROJECT)))
    config_path = HERE / 'replay_configuration.json'
    if resume and config_path.exists():
        assert json.loads(config_path.read_text()) == configuration, 'Resume configuration/source hashes differ'
    else:
        config_path.write_text(json.dumps(configuration, indent=2) + '\n')
    started = time.monotonic()
    batch = prepare_basf(PROJECT)
    folds = make_folds(batch.episodes)
    assert len(folds) == 13
    assert all(len(inner_folds(batch.subset(fold['train']).episodes)) == 3 for fold in folds)
    n_members, n_target_members = archive_memberships(batch, folds)
    archived = pd.read_csv(ARCHIVE / 'training_inner_selection.csv')
    assert len(archived) == 117
    fits_archive = json.loads((ARCHIVE / 'fits.json').read_text())
    original_selected = {f['fold']: f for f in fits_archive if f['model'] == 'primary_secondary_selected_inner'}
    candidates, all_fits, comparisons, winners = [], [], [], []
    completed_fits = 0
    checkpoint_directory = HERE / 'candidate_checkpoints'
    prediction_directory = HERE / 'predictions'
    checkpoint_directory.mkdir(exist_ok=True)
    prediction_directory.mkdir(exist_ok=True)
    for outer in folds:
        training = batch.subset(outer['train'])
        split = inner_folds(training.episodes)
        outer_records = []
        for latent in [10., 20., 30.]:
            for infectious in [14., 21., 28.]:
                key = f"{outer['name']}__L{latent:g}_I{infectious:g}"
                fit_file = checkpoint_directory / (key + '.json')
                prediction_file = prediction_directory / (key + '.csv.gz')
                if resume and fit_file.exists() and prediction_file.exists():
                    checkpoint = json.loads(fit_file.read_text())
                    assert checkpoint['configuration_sha256'] == digest(config_path)
                    assert checkpoint['prediction_sha256'] == digest(prediction_file)
                    fitted_records = checkpoint['inner_fits']
                    predictions = pd.read_csv(prediction_file)
                else:
                    fitted_records, prediction_frames = [], []
                    for inner_index, (train_ids, test_ids) in enumerate(split):
                        inner_train, inner_test = training.subset(train_ids), training.subset(test_ids)
                        assert set(inner_train.episodes.series_id) <= outer['train']
                        assert not set(inner_train.episodes.series_id) & outer['test']
                        fitted = fit_mechanism(inner_train, 'primary_secondary_hidden',
                            latent_days=latent, infectious_days=infectious, starts=STARTS, seed=SEED)
                        fitted_records.append(dict(outer_fold=outer['name'], inner_fold=inner_index,
                            latent_days=latent, infectious_days=infectious, seed=SEED,
                            n_training_series=len(inner_train.episodes),
                            n_training_targets=int((~inner_train.targets.conditioning).sum()),
                            n_test_series=len(inner_test.episodes),
                            n_test_targets=int((~inner_test.targets.conditioning).sum()), **fitted))
                        predictions_inner = prediction_frame(inner_test, fitted)
                        predictions_inner['outer_fold'], predictions_inner['inner_fold'] = outer['name'], inner_index
                        predictions_inner['latent_days'], predictions_inner['infectious_days'] = latent, infectious
                        prediction_frames.append(predictions_inner)
                    predictions = pd.concat(prediction_frames, ignore_index=True)
                    predictions.to_csv(prediction_file, index=False, compression=dict(method='gzip', mtime=0))
                    checkpoint = dict(configuration_sha256=digest(config_path), inner_fits=fitted_records,
                        prediction_file=str(prediction_file.relative_to(HERE)), prediction_sha256=digest(prediction_file))
                    fit_file.write_text(json.dumps(checkpoint, indent=2) + '\n')
                metrics = score(predictions)
                independently_computed = independently_score(predictions)
                assert all(close(metrics[k], independently_computed[k]) for k in METRICS)
                assert all(metrics[k] == independently_computed[k] for k in COUNTS)
                row = dict(outer_fold=outer['name'], latent_days=latent, infectious_days=infectious, **metrics)
                candidates.append(row)
                outer_records.append(row)
                all_fits.extend(fitted_records)
                completed_fits += len(fitted_records)
                original = archived.loc[archived.outer_fold.eq(outer['name']) &
                    archived.latent_days.eq(latent) & archived.infectious_days.eq(infectious)]
                assert len(original) == 1
                original = original.iloc[0]
                comparison = dict(outer_fold=outer['name'], latent_days=latent, infectious_days=infectious)
                for metric in METRICS + COUNTS:
                    comparison[f'original_{metric}'] = float(original[metric])
                    comparison[f'replay_{metric}'] = metrics[metric]
                    comparison[f'difference_{metric}'] = float(metrics[metric] - original[metric])
                    comparison[f'matches_{metric}'] = close(metrics[metric], original[metric]) if metric in METRICS else bool(metrics[metric] == original[metric])
                comparison['all_match'] = all(comparison[f'matches_{metric}'] for metric in METRICS + COUNTS)
                comparisons.append(comparison)
                (HERE / 'progress.json').write_text(json.dumps(dict(status='running_retrospective_replay',
                    completed_inner_fits=completed_fits, total_inner_fits=351,
                    completed_candidates=len(candidates), total_candidates=117,
                    last_candidate=key, comparison_matches=comparison['all_match'],
                    runtime_seconds=time.monotonic() - started), indent=2) + '\n')
        winner = pd.DataFrame(outer_records).sort_values(['rmse_pp', 'latent_days', 'infectious_days']).iloc[0]
        original = original_selected[outer['name']]['parameters']
        winners.append(dict(outer_fold=outer['name'], replay_latent_days=float(winner.latent_days),
            replay_infectious_days=float(winner.infectious_days), original_latent_days=original['latent_days'],
            original_infectious_days=original['infectious_days'],
            exact_match=bool(winner.latent_days == original['latent_days'] and winner.infectious_days == original['infectious_days'])))
        print(json.dumps(dict(outer_fold=outer['name'], completed_inner_fits=completed_fits,
            candidate_matches=sum(c['all_match'] for c in comparisons), candidates_checked=len(comparisons),
            selected_winner_matches=winners[-1]['exact_match'], elapsed_seconds=round(time.monotonic() - started, 1))), flush=True)
    pd.DataFrame(candidates).to_csv(HERE / 'replay_candidate_scores.csv', index=False)
    pd.DataFrame(comparisons).to_csv(HERE / 'candidate_comparison.csv', index=False)
    pd.DataFrame(winners).to_csv(HERE / 'selected_winner_comparison.csv', index=False)
    (HERE / 'inner_fits.json').write_text(json.dumps(all_fits, indent=2) + '\n')
    after = watched_files()
    changed = {name: dict(before=old, after=after.get(name)) for name, old in before.items() if after.get(name) != old}
    candidate_match = all(c['all_match'] for c in comparisons)
    winner_match = all(w['exact_match'] for w in winners)
    receipt = dict(status='passed' if candidate_match and winner_match and not changed else 'failed',
        evidence_type='retrospective reproducibility replay, not contemporaneous original execution evidence',
        completed_utc=datetime.now(timezone.utc).isoformat(), runtime_seconds=time.monotonic() - started,
        outer_folds=len(folds), inner_location_splits_per_outer_fold=3, inner_fits=completed_fits,
        candidates_checked=len(candidates), candidates_matching=sum(c['all_match'] for c in comparisons),
        selected_winners_checked=len(winners), selected_winners_matching=sum(w['exact_match'] for w in winners),
        maximum_absolute_metric_difference_pp={k: max(abs(c[f'difference_{k}']) for c in comparisons) for k in METRICS},
        metric_absolute_tolerance_pp=ABSOLUTE_TOLERANCE_PP, metric_relative_tolerance=RELATIVE_TOLERANCE,
        counts_match_exactly=all(c[f'matches_{k}'] for c in comparisons for k in COUNTS),
        membership_records=n_members, target_membership_records=n_target_members,
        inner_fits_use_only_outer_training=True, inner_locations_disjoint=True,
        source_code_matches_archived_snapshot=True, original_files_preserved=not changed,
        original_file_count=len(before), changed_original_files=changed,
        package_versions=versions, fixed_initial_optimizer_starts=3,
        seed_note='seed=20261004; starts=3 uses the same three deterministic initial vectors and no extra random starts')
    (HERE / 'replay_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    (HERE / 'progress.json').write_text(json.dumps(dict(status=receipt['status'],
        completed_inner_fits=completed_fits, total_inner_fits=351, completed_candidates=len(candidates),
        total_candidates=117), indent=2) + '\n')
    artifacts = sorted(p for p in HERE.rglob('*') if p.is_file() and '.numba_cache' not in p.parts
                       and '__pycache__' not in p.parts and p.name != 'SHA256SUMS')
    (HERE / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.relative_to(PROJECT)}\n' for p in artifacts))
    print(json.dumps(receipt, indent=2), flush=True)
    if receipt['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true', help='reuse matching retrospective replay checkpoints')
    arguments = parser.parse_args()
    run(arguments.resume)
