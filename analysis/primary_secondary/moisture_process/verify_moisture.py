"""Independent source-cell, prediction and weighted-score reconciliation."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd
from scipy.special import expit, gammainc

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE = PROJECT / 'data/public_septoria/primary_evidence/boixel-moisture/data_moisture_regimes_Zymoseptoria_tritici_Boixel_et_al._2020.xlsx'


def grouped_score(frame, predicted):
    """Nested arithmetic averages, independent of fitter's vectorized weights."""
    f = frame.copy()
    residual = f.pycnidial_area_percent.to_numpy(float) - predicted
    f['absolute_error'] = np.abs(residual)
    f['squared_error'] = residual ** 2
    block = f.groupby(['series', 'population', 'isolate', 'regime', 'dpi', 'block'])[['absolute_error', 'squared_error']].mean()
    cell = block.groupby(level=['series', 'population', 'isolate', 'regime', 'dpi']).mean()
    isolate = cell.groupby(level=['series', 'population', 'isolate']).mean()
    population = isolate.groupby(level=['series', 'population']).mean()
    series = population.groupby(level='series').mean()
    value = series.mean()
    return float(value.absolute_error), float(np.sqrt(value.squared_error))


def independent_predict(parameter, f):
    name = parameter['model']
    regimes = ['C0', 'C1', 'C2', 'C3']
    r = np.array([regimes.index(v) for v in f.regime])
    day = f.dpi.to_numpy(float)
    if name == 'global_constant':
        return np.full(len(f), parameter['value'])
    if name == 'regime_constant':
        return np.array(parameter['value'])[r]
    if name.startswith('gamma_'):
        return np.array(parameter['capacity_percent'])[r] * gammainc(parameter['shape'], day / parameter['progression_scale_days'])
    coefficients = np.array(parameter['coefficients'])
    if name == 'time_linear':
        r = np.zeros(len(f), int)
    value = coefficients[r, 0] + coefficients[r, 1] * (day - 14)
    return expit(value) * 100 if name == 'regime_logistic_100' else np.minimum(100, np.maximum(0, value))


def verify():
    workbook = openpyxl.load_workbook(SOURCE, data_only=True, read_only=True)
    rows = list(workbook['data_globales_R'].values)
    assert rows[0] == ('serie', 'bloc', 'duration_HR', 'pop', 'iso', 'rep', 'DPI', 'PYC')
    exported = pd.read_csv(HERE / 'source_records.csv', keep_default_na=False)
    assert len(exported) == len(rows) - 1 == 3456
    for original, record in zip(rows[1:], exported.itertuples()):
        assert original[:7] == (record.series, record.block, record.regime, record.population,
                              record.isolate, record.repetition_label, record.dpi)
        if original[7] == 'na':
            assert record.quality_status == 'missing' and record.pycnidial_area_percent == ''
        else:
            assert record.quality_status == 'valid'
            assert float(record.pycnidial_area_percent) == float(original[7])
    f = pd.read_csv(HERE / 'source_records.csv')
    assert f.groupby('isolate').series.nunique().max() == 1
    assert f.loc[f.pycnidial_area_percent.notna()].isolate.nunique() == 47
    training = f.loc[f.series.lt(3) & f.dpi.isin([14, 17]) & f.pycnidial_area_percent.notna()]
    assert len(training) == 1456 and training.isolate.nunique() == 31
    outcomes = pd.read_csv(HERE / 'heldout_predictions.csv')
    parameters = json.loads((HERE / 'fitted_parameters.json').read_text())
    metrics = pd.read_csv(HERE / 'heldout_metrics.csv')
    tested = 0
    for (partition, model), group in outcomes.groupby(['partition', 'model']):
        original = group.merge(f[['source_row', 'pycnidial_area_percent', 'partition']], on='source_row',
            suffixes=('_prediction', '_source'), validate='one_to_one')
        np.testing.assert_array_equal(original.pycnidial_area_percent_prediction, original.pycnidial_area_percent_source)
        assert original.partition_source.eq(partition).all()
        np.testing.assert_allclose(group.predicted_percent, independent_predict(parameters[model], group), atol=1e-11, rtol=0)
        mae, rmse = grouped_score(group, group.predicted_percent.to_numpy())
        row = metrics.loc[metrics.partition.eq(partition) & metrics.model.eq(model)]
        assert len(row) == 1
        np.testing.assert_allclose([mae, rmse], row[['mae_pp', 'rmse_pp']].to_numpy()[0], atol=1e-11, rtol=0)
        tested += 1
    cv = pd.read_csv(HERE / 'shape_selection_training_only.csv')
    selected = int(cv.groupby('shape').mse_pp2.mean().idxmin())
    assert selected == parameters['gamma_selected_shape']['shape'] == 24
    assert cv.heldout_training_series.isin([1, 2]).all() and len(cv) == 12
    onset = pd.read_csv(HERE / 'repetition_onset_intervals.csv')
    for record in onset.itertuples():
        raw = f.loc[(f.series == record.series) & (f.block == record.block) &
            (f.regime == record.regime) & (f.population == record.population) &
            (f.isolate == record.isolate) & (f.repetition_label == record.repetition_label)]
        raw = raw.loc[raw.pycnidial_area_percent.notna()].sort_values('dpi')
        positive = raw.loc[raw.pycnidial_area_percent > 0]
        if positive.empty and raw.empty:
            assert record.onset_status == 'unobserved' and pd.isna(record.lower_dpi)
        elif positive.empty:
            assert record.onset_status == 'right_censored_late_or_never'
            assert record.lower_dpi == raw.dpi.max() and np.isinf(record.upper_dpi)
        else:
            first = positive.dpi.iloc[0]
            negative = raw.loc[(raw.dpi < first) & (raw.pycnidial_area_percent == 0)]
            low = negative.dpi.max() if len(negative) else 0
            assert record.upper_dpi == first and record.lower_dpi == low
            assert record.onset_status == ('interval_censored' if low else 'left_censored')
        assert bool(record.nonmonotone_area) == bool((raw.pycnidial_area_percent.diff().dropna() < 0).any())
    assert len(onset) == 1152 and onset.nonmonotone_area.sum() == 25
    bands = pd.read_csv(HERE / 'coverage_curve_bootstrap.csv')
    assert (bands.lower_percent <= bands.upper_percent).all()
    assert bands[['lower_percent', 'upper_percent', 'median_percent', 'point_percent']].min().min() >= 0
    assert bands[['lower_percent', 'upper_percent', 'median_percent', 'point_percent']].max().max() <= 100
    for _, curve in bands.groupby('regime'):
        assert curve.sort_values('dpi').point_percent.diff().dropna().ge(-1e-10).all()
    plotted_means = pd.read_csv(HERE / 'figure_observed_means.csv')
    for row in plotted_means.itertuples():
        sample = f.loc[(f.series == row.series) & (f.regime == row.regime) &
                       (f.dpi == row.dpi) & f.pycnidial_area_percent.notna()]
        block = sample.groupby(['population', 'isolate', 'block']).pycnidial_area_percent.mean()
        isolate = block.groupby(level=['population', 'isolate']).mean()
        population = isolate.groupby(level='population').mean()
        assert abs(population.mean() - row.balanced_mean_percent) < 1e-11
        assert len(sample) == row.n_valid_scores
    plotted_curves = pd.read_csv(HERE / 'figure_curve_points.csv')
    for model, curves in plotted_curves.groupby('model'):
        np.testing.assert_allclose(curves.predicted_percent, independent_predict(parameters[model], curves), atol=1e-11, rtol=0)
    receipt = dict(status='passed', completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(), exact_source_records_checked=3456,
        independent_prediction_and_hierarchical_metric_groups=tested,
        onset_interval_records_checked=1152, selected_shape_training_cv=selected,
        plotted_source_means_checked=len(plotted_means), plotted_curve_points_checked=len(plotted_curves),
        raw_outcome_isolates=47, training_isolates=31,
        checks=['all original Excel identities and PYC values', 'predictions from independent equations',
                'metrics from independently nested averages', 'source-label onset censoring',
                'disjoint source isolates across series', 'bounded monotone coverage curves'])
    (HERE / 'independent_verification.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    verify()
