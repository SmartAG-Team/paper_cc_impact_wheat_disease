"""Source-backed pycnidial-area prediction with isolated series/time holdouts."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, gammainc, logit

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]
SOURCE = PROJECT / 'data/public_septoria/primary_evidence/boixel-moisture/data_moisture_regimes_Zymoseptoria_tritici_Boixel_et_al._2020.xlsx'
MANIFEST = PROJECT / 'data/public_septoria/primary_evidence/boixel_download_manifest.json'
REGIMES = ['C0', 'C1', 'C2', 'C3']
SHAPES = [3, 6, 12, 24, 48, 96]
MODELS = ['global_constant', 'regime_constant', 'time_linear', 'regime_linear',
          'regime_logistic_100', 'gamma_shape3', 'gamma_selected_shape']
CELL = ['series', 'population', 'isolate', 'regime', 'dpi']
TRAJECTORY = ['series', 'block', 'regime', 'population', 'isolate', 'repetition_label']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source():
    manifest = json.loads(MANIFEST.read_text())
    raw = SOURCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['sha256']
    assert hashlib.md5(raw).hexdigest() == manifest['expected_digest']
    assert len(raw) == manifest['expected_bytes']
    frame = pd.read_excel(SOURCE, sheet_name='data_globales_R', keep_default_na=False)
    frame = frame.rename(columns=dict(serie='series', bloc='block', duration_HR='regime',
        pop='population', iso='isolate', rep='repetition_label', DPI='dpi', PYC='raw_pyc'))
    frame['source_sheet'] = 'data_globales_R'
    frame['source_row'] = np.arange(2, len(frame) + 2)
    frame['pycnidial_area_percent'] = pd.to_numeric(frame.raw_pyc, errors='coerce')
    frame['quality_status'] = np.where(frame.pycnidial_area_percent.isna(), 'missing', 'valid')
    assert len(frame) == 3456 and frame.pycnidial_area_percent.notna().sum() == 3326
    assert set(frame.loc[frame.quality_status.eq('missing'), 'raw_pyc']) == {'na'}
    assert frame.pycnidial_area_percent.dropna().between(0, 100).all()
    assert not frame.duplicated(TRAJECTORY + ['dpi']).any()
    assert frame.groupby('isolate').series.nunique().eq(1).all()
    assert frame.groupby(['series', 'population']).isolate.nunique().eq(8).all()
    frame['trajectory_id'] = frame[TRAJECTORY].astype(str).agg('|'.join, axis=1)
    frame['partition'] = np.select([
        frame.series.isin([1, 2]) & frame.dpi.le(17),
        frame.series.eq(3) & frame.dpi.le(17),
        frame.series.isin([1, 2]) & frame.dpi.eq(20)],
        ['training_early', 'heldout_series3_early', 'future20_series12'],
        default='heldout_series3_future20')
    return frame


def weights(frame):
    """Equal series/population/isolate/cell/block, then valid repetition scores."""
    x = frame.reset_index(drop=True)
    x['_cell_id'] = x.groupby(CELL, sort=False).ngroup()
    denominator = (x.series.nunique() *
        x.groupby('series').population.transform('nunique') *
        x.groupby(['series', 'population']).isolate.transform('nunique') *
        x.groupby(['series', 'population', 'isolate'])._cell_id.transform('nunique') *
        x.groupby(CELL).block.transform('nunique') *
        x.groupby(CELL + ['block']).block.transform('size'))
    w = 1 / denominator.to_numpy(float)
    assert np.isclose(w.sum(), 1)
    return w


def fit_model(frame, model, shape=None):
    f = frame.loc[frame.pycnidial_area_percent.notna()].reset_index(drop=True)
    w = weights(f)
    y = f.pycnidial_area_percent.to_numpy(float)
    t = f.dpi.to_numpy(float)
    r = f.regime.map({v: i for i, v in enumerate(REGIMES)}).to_numpy(int)
    if model == 'global_constant':
        return dict(model=model, value=float(w @ y))
    if model == 'regime_constant':
        return dict(model=model, value=[float(w[r == j] @ y[r == j] / w[r == j].sum()) for j in range(4)])
    if model in ('time_linear', 'regime_linear'):
        groups = [np.ones(len(f), bool)] if model == 'time_linear' else [r == j for j in range(4)]
        coefficients = []
        for mask in groups:
            X = np.column_stack([np.ones(mask.sum()), t[mask] - 14])
            z = np.sqrt(w[mask])
            coefficients.append(np.linalg.lstsq(X * z[:, None], y[mask] * z, rcond=None)[0].tolist())
        return dict(model=model, coefficients=coefficients, prediction_clip_percent=[0, 100])
    if model == 'regime_logistic_100':
        coefficients = []
        for j in range(4):
            means = [np.average(y[(r == j) & (t == d)], weights=w[(r == j) & (t == d)]) for d in [14, 17]]
            q = logit(np.clip(np.array(means) / 100, .005, .995))
            coefficients.append([float(q[0]), float((q[1] - q[0]) / 3)])
        return dict(model=model, coefficients=coefficients, asymptote_percent=100,
                    logit_mean_clip_percent=[.5, 99.5])
    k = int(3 if model == 'gamma_shape3' else shape)
    def loss(mean, full=False):
        progress = gammainc(k, k * t / mean)
        capacity = np.array([np.clip(np.sum(w[r == j] * progress[r == j] * y[r == j]) /
            np.sum(w[r == j] * progress[r == j] ** 2), 0, 100) for j in range(4)])
        residual = y - capacity[r] * progress
        objective = float(w @ (residual ** 2))
        return (objective, capacity) if full else objective
    grid = np.geomspace(1, 100, 180)
    i = int(np.argmin([loss(v) for v in grid]))
    solution = minimize_scalar(loss, bounds=(grid[max(0, i - 1)], grid[min(len(grid) - 1, i + 1)]),
                               method='bounded', options={'xatol': 1e-9})
    mean = float(solution.x)
    objective, capacity = loss(mean, True)
    return dict(model=model, shape=k, progression_mean_days=mean,
        progression_scale_days=mean / k, capacity_percent=capacity.tolist(),
        training_mse_pp2=objective, mean_parameter_bounds_days=[1, 100],
        zero_time_origin='inoculation, not observed penetration',
        interpretation='phenomenological pycnidial-area coverage curve; not infection-time distribution')


def predict(parameters, frame):
    model = parameters['model']
    t = frame.dpi.to_numpy(float)
    r = frame.regime.map({v: i for i, v in enumerate(REGIMES)}).to_numpy(int)
    if model == 'global_constant':
        return np.repeat(parameters['value'], len(frame))
    if model == 'regime_constant':
        return np.array(parameters['value'])[r]
    if model in ('time_linear', 'regime_linear', 'regime_logistic_100'):
        coef = np.array(parameters['coefficients'])
        rr = np.zeros_like(r) if model == 'time_linear' else r
        z = coef[rr, 0] + coef[rr, 1] * (t - 14)
        return 100 * expit(z) if model == 'regime_logistic_100' else np.clip(z, 0, 100)
    return np.array(parameters['capacity_percent'])[r] * gammainc(parameters['shape'],
        parameters['shape'] * t / parameters['progression_mean_days'])


def select_and_fit(records):
    # This predicate alone defines every parameter and hyperparameter input.
    train = records.loc[records.series.isin([1, 2]) & records.dpi.le(17) &
                        records.pycnidial_area_percent.notna()].copy()
    rows = []
    for k in SHAPES:
        for heldout in [1, 2]:
            parameter = fit_model(train.loc[train.series.ne(heldout)], 'gamma_selected_shape', k)
            validation = train.loc[train.series.eq(heldout)].reset_index(drop=True)
            e = validation.pycnidial_area_percent.to_numpy() - predict(parameter, validation)
            rows.append(dict(shape=k, heldout_training_series=heldout,
                n_valid_scores=len(validation), n_isolates=validation.isolate.nunique(),
                mse_pp2=float(weights(validation) @ (e ** 2))))
    cv = pd.DataFrame(rows)
    score = cv.groupby('shape').mse_pp2.mean()
    selected = int(score.sort_values(kind='stable').index[0])
    parameters = {name: fit_model(train, name, selected) for name in MODELS}
    return train, cv, selected, parameters


def cluster_losses(frame, predictions):
    x = frame.reset_index(drop=True)
    w = weights(x)
    rows = []
    for name, yhat in predictions.items():
        e = x.pycnidial_area_percent.to_numpy() - yhat
        for (series, population, isolate), group in x.groupby(['series', 'population', 'isolate']):
            z = w[group.index]
            rows.append(dict(model=name, series=int(series), population=population, isolate=isolate,
                mae_pp=float(np.average(np.abs(e[group.index]), weights=z)),
                mse_pp2=float(np.average(e[group.index] ** 2, weights=z))))
    return pd.DataFrame(rows)


def evaluate(records, parameters, rng, repetitions):
    test_rows, metric_rows, cluster_rows, comparison_rows = [], [], [], []
    for partition in ['heldout_series3_early', 'future20_series12', 'heldout_series3_future20']:
        f = records.loc[records.partition.eq(partition) & records.pycnidial_area_percent.notna()].reset_index(drop=True)
        predictions = {name: predict(p, f) for name, p in parameters.items()}
        cluster = cluster_losses(f, predictions)
        cluster['partition'] = partition
        cluster_rows.append(cluster)
        pivot_mse = cluster.pivot(index=['series', 'population', 'isolate'], columns='model', values='mse_pp2')
        pivot_mae = cluster.pivot(index=['series', 'population', 'isolate'], columns='model', values='mae_pp')
        strata = [np.flatnonzero((pivot_mse.index.get_level_values(0) == s) &
                    (pivot_mse.index.get_level_values(1) == p)) for s, p in
                    sorted(set(zip(pivot_mse.index.get_level_values(0), pivot_mse.index.get_level_values(1))))]
        boot_mse, boot_mae = [], []
        mse_array = pivot_mse.to_numpy()
        mae_array = pivot_mae.to_numpy()
        point_mse = np.mean([mse_array[ix].mean(axis=0) for ix in strata], axis=0)
        point_mae = np.mean([mae_array[ix].mean(axis=0) for ix in strata], axis=0)
        for _ in range(repetitions):
            indices = [rng.choice(ix, len(ix), replace=True) for ix in strata]
            boot_mse.append(np.mean([mse_array[ix].mean(axis=0) for ix in indices], axis=0))
            boot_mae.append(np.mean([mae_array[ix].mean(axis=0) for ix in indices], axis=0))
        boot_rmse = np.sqrt(np.array(boot_mse))
        boot_mae = np.array(boot_mae)
        for j, name in enumerate(pivot_mse.columns):
            metric_rows.append(dict(partition=partition, model=name, n_valid_scores=len(f),
                n_isolates=f.isolate.nunique(), n_series=f.series.nunique(),
                mae_pp=float(point_mae[j]), rmse_pp=float(np.sqrt(point_mse[j])),
                mae_ci_low_pp=float(np.quantile(boot_mae[:, j], .025)),
                mae_ci_high_pp=float(np.quantile(boot_mae[:, j], .975)),
                rmse_ci_low_pp=float(np.quantile(boot_rmse[:, j], .025)),
                rmse_ci_high_pp=float(np.quantile(boot_rmse[:, j], .975)),
                uncertainty='95% percentile isolate-cluster bootstrap; fixed training fit and observed series'))
        a = list(pivot_mse.columns).index('gamma_selected_shape')
        for reference in ['regime_linear', 'regime_logistic_100', 'gamma_shape3']:
            b = list(pivot_mse.columns).index(reference)
            delta = boot_rmse[:, a] - boot_rmse[:, b]
            comparison_rows.append(dict(partition=partition, contrast=f'gamma_selected_shape minus {reference}',
                rmse_difference_pp=float(np.sqrt(point_mse[a]) - np.sqrt(point_mse[b])),
                ci_low_pp=float(np.quantile(delta, .025)), ci_high_pp=float(np.quantile(delta, .975)),
                paired_resampling_unit='isolate, stratified within series and population'))
        for name, prediction in predictions.items():
            out = f[['source_row', 'series', 'population', 'isolate', 'block', 'regime',
                     'repetition_label', 'dpi', 'pycnidial_area_percent']].copy()
            out['partition'], out['model'], out['predicted_percent'] = partition, name, prediction
            out['residual_pp'] = out.pycnidial_area_percent - prediction
            test_rows.append(out)
    return pd.concat(test_rows), pd.DataFrame(metric_rows), pd.concat(cluster_rows), pd.DataFrame(comparison_rows)


def bootstrap_curve(train, selected, rng, repetitions):
    curve_grid = pd.DataFrame([(r, float(d)) for r in REGIMES for d in np.linspace(0, 30, 121)], columns=['regime', 'dpi'])
    curves, parameters = [], []
    for b in range(repetitions):
        blocks = []
        for (series, population), group in train.groupby(['series', 'population']):
            isolates = sorted(group.isolate.unique())
            for j, isolate in enumerate(rng.choice(isolates, len(isolates), replace=True)):
                block = group.loc[group.isolate.eq(isolate)].copy()
                block['isolate'] = f'{population}|bootstrap{j}'
                blocks.append(block)
        fit = fit_model(pd.concat(blocks), 'gamma_selected_shape', selected)
        curves.append(predict(fit, curve_grid))
        parameters.append(dict(replicate=b, progression_mean_days=fit['progression_mean_days'],
                               **{f'capacity_{r}_percent': v for r, v in zip(REGIMES, fit['capacity_percent'])}))
    draws = np.array(curves)
    curve_grid['lower_percent'] = np.quantile(draws, .025, axis=0)
    curve_grid['upper_percent'] = np.quantile(draws, .975, axis=0)
    curve_grid['median_percent'] = np.median(draws, axis=0)
    return curve_grid, pd.DataFrame(parameters)


def onset_records(records):
    rows = []
    for key, group in records.groupby(TRAJECTORY):
        f = group.loc[group.pycnidial_area_percent.notna()].sort_values('dpi')
        positive = f.loc[f.pycnidial_area_percent.gt(0)]
        if len(positive):
            upper = int(positive.dpi.iloc[0])
            negatives = f.loc[f.dpi.lt(upper) & f.pycnidial_area_percent.eq(0)]
            lower = int(negatives.dpi.max()) if len(negatives) else 0
            status = 'interval_censored' if lower else 'left_censored'
        elif len(f):
            lower, upper, status = int(f.dpi.max()), np.inf, 'right_censored_late_or_never'
        else:
            lower, upper, status = np.nan, np.nan, 'unobserved'
        rows.append(dict(zip(TRAJECTORY, key), onset_status=status, lower_dpi=lower, upper_dpi=upper,
            n_valid_dates=len(f), nonmonotone_area=bool((f.pycnidial_area_percent.diff().dropna() < 0).any()),
            positive_to_zero=bool(len(positive) and ((f.dpi > upper) & f.pycnidial_area_percent.eq(0)).any())))
    return pd.DataFrame(rows)


def plot_figure(records, parameters, bands):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.7), sharex=True, sharey=True)
    plotted_means, plotted_curves = [], []
    line_styles = [('gamma_selected_shape', '-', '#126b5f', 'Selected gamma coverage'),
                   ('gamma_shape3', ':', '#777777', 'Gamma shape 3'),
                   ('regime_linear', '--', '#6846a1', 'Regime linear'),
                   ('regime_logistic_100', '-.', '#b65b1d', 'Regime logistic (100% limit)')]
    for ax, regime in zip(axes.flat, REGIMES):
        grid = pd.DataFrame(dict(regime=regime, dpi=np.linspace(0, 26, 160)))
        band = bands.loc[bands.regime.eq(regime) & bands.dpi.le(26)]
        ax.fill_between(band.dpi.to_numpy(), band.lower_percent.to_numpy(), band.upper_percent.to_numpy(), color='#126b5f', alpha=.12)
        for model, style, color, label in line_styles:
            prediction = predict(parameters[model], grid)
            ax.plot(grid.dpi, prediction, style, color=color, lw=1.4, label=label)
            points = grid.copy()
            points['model'], points['predicted_percent'] = model, prediction
            plotted_curves.append(points)
        f = records.loc[records.regime.eq(regime) & records.pycnidial_area_percent.notna()]
        for series, marker, color in [(1, 'o', '#32618a'), (2, '^', '#ad822a'), (3, 's', '#bf3c48')]:
            sample = f.loc[f.series.eq(series)]
            for dpi, date in sample.groupby('dpi'):
                value = weights(date.reset_index(drop=True)) @ date.pycnidial_area_percent.to_numpy()
                heldout = series == 3 or dpi == 20
                plotted_means.append(dict(series=series, regime=regime, dpi=int(dpi),
                    balanced_mean_percent=float(value), n_valid_scores=len(date),
                    n_isolates=date.isolate.nunique(), fitting_date=not heldout))
                ax.scatter(dpi, value, marker=marker, color=color, facecolors='none' if heldout else color,
                           s=38, linewidths=1.2, zorder=5, label=f'Series {series}' if dpi == 14 else None)
        ax.set_title(f'{regime}: {REGIMES.index(regime)} day(s) bagging', fontsize=10)
        ax.set_xlim(10, 25)
        ax.set_ylim(0, 103)
        ax.grid(axis='y', alpha=.2)
    for ax in axes[-1]:
        ax.set_xlabel('Days after inoculation')
    for ax in axes[:, 0]:
        ax.set_ylabel('Pycnidial area (% inoculated area)')
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=8.2, bbox_to_anchor=(.51, .98))
    fig.text(.5, .015, 'Filled markers: fitting observations (series 1–2, days 14/17). Open markers: excluded targets.\nBand: 95% training-isolate bootstrap interval, conditional on selected gamma shape; series effects are not sampled.',
             ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .065, 1, .88))
    fig.savefig(HERE / 'pycnidial_coverage_transfer.png', dpi=220)
    fig.savefig(HERE / 'pycnidial_coverage_transfer.pdf')
    plt.close(fig)
    pd.DataFrame(plotted_means).to_csv(HERE / 'figure_observed_means.csv', index=False)
    pd.concat(plotted_curves).to_csv(HERE / 'figure_curve_points.csv', index=False)


def run(parameter_bootstrap=200, metric_bootstrap=2000):
    records = load_source()
    records.to_csv(HERE / 'source_records.csv', index=False)
    dictionary = pd.read_excel(SOURCE, sheet_name='notice', header=4, usecols='A:B')
    dictionary.to_csv(HERE / 'source_dictionary.csv', index=False)
    (HERE / 'source_download_manifest.json').write_bytes(MANIFEST.read_bytes())
    contract = dict(dataset_doi='10.15454/FK7WHW', response='PYC: percentage of inoculated area covered by pycnidia',
        training='series 1–2, DPI 14/17', shape_selection='leave-one-training-series-out at DPI 14/17 only',
        heldout=['series 3 at DPI14/17', 'series 1–2 at DPI20', 'series 3 at DPI20'],
        gamma_candidate_shapes=SHAPES, gamma_mean_bounds_days=[1, 100], gamma_capacity_bounds_percent=[0, 100],
        baselines=MODELS, weights='equal series/population/isolate/regime-date/available block/valid repetition',
        uncertain_physical_identity='F1–F3 are source repetition labels; leaf/plant identifiers are not independently documented',
        bootstrap_unit='entire isolate cluster, preserving blocks, regimes and repeated F-label trajectories',
        parameter_bootstrap=parameter_bootstrap, score_bootstrap=metric_bootstrap,
        parameter_seed=5102026, score_seed=5102027,
        unperformed_claims=['temperature response', 'infection efficiency', 'natural primary inoculum arrival',
                           'universal thermal latent duration', 'transfer to field epidemics'])
    (HERE / 'evaluation_contract.json').write_text(json.dumps(contract, indent=2) + '\n')
    train, cv, selected, parameters = select_and_fit(records)
    cv['selected_shape'] = selected
    cv.to_csv(HERE / 'shape_selection_training_only.csv', index=False)
    (HERE / 'fitted_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
    # Mutating every excluded target proves neither fitting nor selection reads it.
    perturbed = records.copy()
    mask = ~(perturbed.series.isin([1, 2]) & perturbed.dpi.le(17))
    perturbed.loc[mask, 'pycnidial_area_percent'] = 97.25
    _, cv_check, selected_check, parameters_check = select_and_fit(perturbed)
    assert selected_check == selected and parameters_check == parameters
    np.testing.assert_array_equal(cv_check.mse_pp2.to_numpy(), cv.mse_pp2.to_numpy())
    predictions, metrics, clusters, comparisons = evaluate(records, parameters,
        np.random.default_rng(5102027), metric_bootstrap)
    predictions.to_csv(HERE / 'heldout_predictions.csv', index=False)
    metrics.to_csv(HERE / 'heldout_metrics.csv', index=False)
    clusters.to_csv(HERE / 'heldout_isolate_losses.csv', index=False)
    comparisons.to_csv(HERE / 'paired_model_comparisons.csv', index=False)
    bands, parameter_draws = bootstrap_curve(train, selected,
        np.random.default_rng(5102026), parameter_bootstrap)
    bands['point_percent'] = predict(parameters['gamma_selected_shape'], bands)
    bands.to_csv(HERE / 'coverage_curve_bootstrap.csv', index=False)
    parameter_draws.to_csv(HERE / 'parameter_bootstrap_draws.csv', index=False)
    onset = onset_records(records)
    onset.to_csv(HERE / 'repetition_onset_intervals.csv', index=False)
    summary = records.groupby(['series', 'population', 'regime', 'dpi']).pycnidial_area_percent.agg(['size', 'count', 'mean', 'median', 'std']).reset_index()
    summary.to_csv(HERE / 'descriptive_coverage.csv', index=False)
    plot_figure(records, parameters, bands)
    parameter_summary = dict(selected_gamma_shape=selected,
        point_estimates=parameters['gamma_selected_shape'],
        bootstrap_percentiles={name: dict(lower=float(parameter_draws[name].quantile(.025)),
            median=float(parameter_draws[name].median()), upper=float(parameter_draws[name].quantile(.975)))
            for name in parameter_draws if name != 'replicate'},
        capacity_boundary_fraction={r: float((parameter_draws[f'capacity_{r}_percent'] >= 99.999).mean()) for r in REGIMES},
        uncertainty_scope='conditional on chosen shape and two observed training series; resampled isolates, not independent leaves',
        identifiability='Two training dates cannot separately identify latent progression, onset lag and moisture-dependent final capacity. Capacity extrapolates beyond the observation window.')
    (HERE / 'coverage_parameter_summary.json').write_text(json.dumps(parameter_summary, indent=2) + '\n')
    weather_constraints = dict(dataset_doi='10.15454/FK7WHW',
        response=dict(name='PYC', definition='percentage of inoculated area covered by pycnidia',
                      unit='percent', percent_to_fraction_multiplier=.01),
        time_unit='calendar days after inoculation', treatment_window_days=[0, 3],
        treatments={r: dict(bagging_duration_days=i, source_mean_relative_humidity_percent=rh)
                    for i, (r, rh) in enumerate(zip(REGIMES, [88.3, 92.2, 96.1, 100.]))},
        temperature_c=None, hours_RH_above_90=None, leaf_wetness_hours=None,
        inoculum_spore_type=None, inoculum_concentration=None, deposited_viable_spores=None,
        wheat_cultivar=None,
        constraint_scope='categorical early-moisture regimes and observed pycnidial area; bagging and mean RH covary',
        unsupported_mappings=['continuous RH response', 'RH90-hour threshold calibration',
            'rain-splash transmission', 'spore establishment efficiency',
            'thermal latent duration', 'natural airborne primary infection hazard', 'fungicide decision thresholds'])
    (HERE / 'weather_coverage_constraints.json').write_text(json.dumps(weather_constraints, indent=2) + '\n')
    receipt = dict(status='verified_source_and_target_isolation', completed_utc=datetime.now(timezone.utc).isoformat(),
        source_path=str(SOURCE.relative_to(PROJECT)), source_sha256=sha(SOURCE), manifest_sha256=sha(MANIFEST),
        original_acquisition_manifest=json.loads(MANIFEST.read_text()), records=len(records), valid_scores=3326,
        missing_scores=130, isolates=48, n_label_trajectories=len(onset),
        isolates_with_valid_scores=int(records.loc[records.pycnidial_area_percent.notna(), 'isolate'].nunique()),
        training_isolates_with_valid_scores=int(train.isolate.nunique()),
        decreasing_label_trajectories=int(onset.nonmonotone_area.sum()),
        onset_status_counts=onset.onset_status.value_counts().to_dict(),
        selected_shape=selected, training_valid_scores=len(train), target_perturbation_test='passed: all excluded response values changed without affecting selection or parameters',
        source_unchanged=sha(SOURCE) == json.loads(MANIFEST.read_text())['sha256'],
        fit_uses_temperature=False, raw_targets_inspected_for_schema_and_descriptive_audit=True)
    (HERE / 'verification_receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(dict(status=receipt['status'], selected_shape=selected,
        records=len(records), valid_scores=3326, training_valid_scores=len(train),
        metrics=metrics[['partition', 'model', 'mae_pp', 'rmse_pp']].to_dict('records')), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--parameter-bootstrap', type=int, default=200)
    parser.add_argument('--metric-bootstrap', type=int, default=2000)
    args = parser.parse_args()
    run(args.parameter_bootstrap, args.metric_bootstrap)
