"""Reconcile retained German phenology events and archived metrics without source access."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    directory = Path(directory).resolve()
    provenance = json.loads((directory / 'PROVENANCE.json').read_text())
    for item in provenance['source_files_retained']:
        assert digest(directory / item['destination_path']) == item['sha256']
    for name, expected in provenance['derived_files'].items():
        assert digest(directory / name) == expected
    split = json.loads((directory / 'source_evidence/station_split.json').read_text())
    calibration = json.loads((directory / 'source_evidence/calibration.json').read_text())
    evaluation = json.loads((directory / 'source_evidence/evaluation_sites.json').read_text())
    assert set(calibration['fit_station_ids']) == set(split['train'])
    assert not set(split['train']) & set(split['val'])
    assert not set(split['train']) & set(split['test'])
    assert not set(split['val']) & set(split['test'])
    assert set(evaluation['station_ids']) | set(evaluation['excluded_station_ids']) == set(split['test'])
    table = pd.read_csv(directory / 'metrics_long.csv', dtype={'stage': str})
    keys = ['PEP_ID', 'SOWING_DATE', 'BBCH']
    comparisons = 0
    total_rows = 0
    years = {}
    for cohort, group in [('calibration', 'train'), ('validation', 'val'), ('testing', 'test')]:
        frame = pd.read_csv(directory / (cohort + '_all_model_event_dates.csv.gz'),
            parse_dates=['SOWING_DATE', 'END_DATE', 'true_date', 'predicted_date'])
        total_rows += len(frame)
        assert not frame.duplicated(keys + ['model']).any()
        assert frame.groupby(keys).size().eq(3).all()
        assert set(frame.model) == {'GDD', 'T-P', 'T-P-V'}
        sites = evaluation['station_ids'] if cohort == 'testing' else split[group]
        assert set(frame.PEP_ID) == set(sites)
        error = (frame.predicted_date - frame.true_date).dt.days
        np.testing.assert_allclose(error, frame.error_days, rtol=0, atol=0, equal_nan=True)
        matched = frame.predicted_date.notna() & frame.true_date.notna()
        shared = matched.groupby([frame[k] for k in keys]).transform('all')
        assert shared.equals(frame.common_matched)
        years[cohort] = sorted(frame.true_date.dt.year.dropna().astype(int).unique().tolist())
        for row in table.loc[table.cohort == cohort].itertuples():
            selected = frame.loc[frame.model == row.model]
            if row.stage != 'overall':
                selected = selected.loc[selected.BBCH == int(row.stage)]
            assert selected.true_date.notna().sum() == row.observed
            assert (selected.true_date.notna() & selected.predicted_date.isna()).sum() == row.missing_predictions
            if row.denominator == 'three_model_shared':
                selected = selected.loc[selected.common_matched]
            error = (selected.predicted_date - selected.true_date).dt.days.dropna().astype(float)
            assert len(error) == row.matched
            np.testing.assert_allclose(len(error) / row.observed, row.coverage_fraction, rtol=0, atol=1e-12)
            for value, name in [(error.abs().mean(), 'mae_days'),
                (np.sqrt((error ** 2).mean()), 'rmse_days'), (error.mean(), 'bias_days')]:
                np.testing.assert_allclose(value, getattr(row, name), rtol=0, atol=1e-12)
                comparisons += 1
    figure = pd.read_csv(directory / 'heldout_test_figure_data.csv', dtype={'stage': str})
    expected = table.loc[(table.cohort == 'testing')
        & (table.denominator == 'three_model_shared') & (table.stage != 'overall')]
    pd.testing.assert_frame_equal(figure.reset_index(drop=True), expected.reset_index(drop=True))
    figure_provenance = json.loads((directory / 'figure_provenance.json').read_text())
    assert digest(directory / figure_provenance['source_file']) == figure_provenance['source_sha256']
    for name, expected_hash in figure_provenance['artifacts'].items():
        assert digest(directory / name) == expected_hash
    receipt = dict(status='verified', source_access_required=False,
        retained_event_rows=total_rows, recomputed_numeric_metrics=comparisons,
        metric_rows=len(table), plotted_metric_rows=len(figure),
        source_files_sha256_checked=len(provenance['source_files_retained']),
        derived_files_sha256_checked=len(provenance['derived_files']),
        train_val_test_station_sets_disjoint=True,
        withheld_year_validation=False,
        test_years_absent_from_training=sorted(set(years['testing']) - set(years['calibration'])),
        units='calendar days', BBCH85='soft dough')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(verify(args.directory), indent=2))
