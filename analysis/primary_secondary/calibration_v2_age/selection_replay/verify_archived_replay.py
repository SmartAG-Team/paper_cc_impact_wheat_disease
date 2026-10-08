"""Verify the persisted replay artifacts without refitting or importing the model."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[3]


def verify():
    configuration = json.loads((HERE / 'replay_configuration.json').read_text())
    membership = pd.read_csv(HERE / 'inner_membership.csv')
    target_membership = pd.read_csv(HERE / 'inner_target_membership.csv')
    candidate_table = pd.read_csv(HERE / 'replay_candidate_scores.csv')
    original_table = pd.read_csv(HERE.parent / 'training_inner_selection.csv')
    all_fits = json.loads((HERE / 'inner_fits.json').read_text())
    assert len(all_fits) == 351 and len(candidate_table) == len(original_table) == 117
    assert all(f['starts'] == 3 and f['seed'] == 20261004 for f in all_fits)
    config_hash = hashlib.sha256((HERE / 'replay_configuration.json').read_bytes()).hexdigest()
    comparisons = []
    persisted_fits = []
    prediction_rows = 0
    for path in sorted((HERE / 'candidate_checkpoints').glob('*.json')):
        checkpoint = json.loads(path.read_text())
        persisted_fits.extend(checkpoint['inner_fits'])
        assert checkpoint['configuration_sha256'] == config_hash
        file = HERE / checkpoint['prediction_file']
        assert hashlib.sha256(file.read_bytes()).hexdigest() == checkpoint['prediction_sha256']
        f = pd.read_csv(file)
        prediction_rows += len(f)
        outer = f.outer_fold.unique().tolist()
        assert len(outer) == 1
        outer = outer[0]
        latent, infectious = float(f.latent_days.iloc[0]), float(f.infectious_days.iloc[0])
        assert f.latent_days.eq(latent).all() and f.infectious_days.eq(infectious).all()
        assert not f.conditioning.any()
        for i, inner in f.groupby('inner_fold'):
            selected = membership.loc[membership.outer_fold.eq(outer) & membership.inner_fold.eq(i)]
            train = selected.loc[selected.side.eq('train')]
            test = selected.loc[selected.side.eq('test')]
            assert not set(train.location_id) & set(test.location_id)
            assert not set(train.series_id) & set(test.series_id)
            assert set(inner.series_id) == set(test.series_id)
            expected = target_membership.loc[target_membership.outer_fold.eq(outer) &
                target_membership.inner_fold.eq(i) & target_membership.side.eq('test') &
                ~target_membership.conditioning]
            assert set(zip(inner.source_row, inner.series_id)) == set(zip(expected.source_row, expected.series_id))
            matched = inner.merge(expected[['source_row', 'series_id', 'Value']],
                on=['source_row', 'series_id'], suffixes=('_replay', '_membership'), validate='one_to_one')
            np.testing.assert_array_equal(matched.observed_percent, matched.Value_membership)
            fit = [r for r in checkpoint['inner_fits'] if r['inner_fold'] == i]
            assert len(fit) == 1 and fit[0]['n_test_targets'] == len(inner)
            assert fit[0]['n_training_series'] == len(train)
        residual = f.predicted_percent - f.observed_percent
        g = f[['coordinate_year', 'series_id']].copy()
        g['sq'], g['absolute'], g['signed'] = residual ** 2, abs(residual), residual
        per_series = g.groupby(['coordinate_year', 'series_id']).mean(numeric_only=True)
        per_coordinate = per_series.groupby(level='coordinate_year').mean()
        metrics = dict(rmse_pp=float(np.sqrt(per_coordinate.sq.mean())),
            mae_pp=float(per_coordinate.absolute.mean()), bias_pp=float(per_coordinate.signed.mean()),
            n_coordinate_years=len(per_coordinate), n_series=f.series_id.nunique(), n_targets=len(f))
        for name, table in [('replay', candidate_table), ('original', original_table)]:
            target = table.loc[table.outer_fold.eq(outer) & table.latent_days.eq(latent) & table.infectious_days.eq(infectious)]
            assert len(target) == 1
            target = target.iloc[0]
            differences = {}
            for key in ['rmse_pp', 'mae_pp', 'bias_pp']:
                differences[key] = float(metrics[key] - target[key])
                assert abs(differences[key]) <= configuration['metric_absolute_tolerance_pp'] + configuration['metric_relative_tolerance'] * abs(target[key])
            for key in ['n_coordinate_years', 'n_series', 'n_targets']:
                assert metrics[key] == target[key]
            comparisons.append(dict(outer_fold=outer, latent_days=latent, infectious_days=infectious,
                                    comparison=name, **differences))
    assert len(comparisons) == 234
    def fit_key(fit):
        return (fit['outer_fold'], fit['latent_days'], fit['infectious_days'], fit['inner_fold'])
    assert len({fit_key(fit) for fit in all_fits}) == 351
    assert {fit_key(fit): fit for fit in all_fits} == {fit_key(fit): fit for fit in persisted_fits}
    originals = {row['fold']: row for row in json.loads((HERE.parent / 'fits.json').read_text())
                 if row['model'] == 'primary_secondary_selected_inner'}
    winners = pd.read_csv(HERE / 'selected_winner_comparison.csv')
    for outer, scores in candidate_table.groupby('outer_fold'):
        best = scores.sort_values(['rmse_pp', 'latent_days', 'infectious_days']).iloc[0]
        archived = originals[outer]['parameters']
        assert (best.latent_days, best.infectious_days) == (archived['latent_days'], archived['infectious_days'])
        row = winners.loc[winners.outer_fold.eq(outer)].iloc[0]
        assert row.exact_match and (row.replay_latent_days, row.replay_infectious_days) == (best.latent_days, best.infectious_days)
    changed = {p: h for p, h in configuration['original_files_sha256'].items()
               if hashlib.sha256((PROJECT / p).read_bytes()).hexdigest() != h}
    assert not changed
    receipt = dict(status='passed', evidence_type='independent audit of persisted retrospective replay artifacts',
        completed_utc=datetime.now(timezone.utc).isoformat(), fitted_parameter_records_checked=351,
        candidate_prediction_archives_checked=117, prediction_rows_checked=prediction_rows,
        independently_reconstructed_selected_winners=len(originals),
        score_table_comparisons=234, original_files_preserved=True,
        maximum_absolute_score_reconstruction_difference_pp={key: max(abs(c[key]) for c in comparisons)
            for key in ['rmse_pp', 'mae_pp', 'bias_pp']},
        exact_membership_and_observed_value_checks=True, all_inner_locations_disjoint=True)
    (HERE / 'persisted_artifact_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    verify()
