"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/canopy_senescence_20261010')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.methods.canopy_senescence import model
HERE = evidence_dir('analysis/paper_study/canopy_senescence_20261010')
YEARS = (2015, 2016, 2017)
KINDS = ('training_mean', 'calendar_logistic', 'thermal_logistic')
PREDICTION_COLUMNS = ['observation_id', 'plot_id', 'year', 'cultivar_id', 'treatment', 'date', 'heading_date', 'elapsed_days', 'degree_days', 'green_fraction']

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def split_prediction(train, test, kind, phase, fold):
    parameters = model.fit(train, kind)
    prediction = test[PREDICTION_COLUMNS].copy()
    prediction['prediction'] = model.predict(test, parameters)
    prediction['model'] = kind
    prediction['phase'] = phase
    prediction['fold'] = fold
    prediction['training_years'] = '|'.join(map(str, parameters['training_years']))
    prediction['cultivar_seen_in_training'] = test.cultivar_id.isin(train.cultivar_id)
    for feature in ('elapsed_days', 'degree_days'):
        prediction[feature + '_outside_training_range'] = ~test[feature].between(train[feature].min(), train[feature].max())
    metrics = dict(phase=phase, fold=fold, model=kind, **model.scores(test, prediction.prediction))
    return (prediction, metrics, dict(phase=phase, fold=fold, **parameters))

def cross_validate(frame):
    if set(frame.year.unique()) != set(YEARS):
        raise ValueError('Exactly the prescribed historical years are required')
    predictions = []
    metrics = []
    fits = []
    for year in YEARS:
        train = frame[frame.year.ne(year)]
        test = frame[frame.year.eq(year)]
        for kind in KINDS:
            p, s, f = split_prediction(train, test, kind, 'historical_cv', year)
            predictions.append(p)
            metrics.append(s)
            fits.append(f)
    return (pd.concat(predictions, ignore_index=True), pd.DataFrame(metrics), fits)

def choose_historical(metrics):
    scores = {}
    for kind in KINDS[1:]:
        part = metrics[metrics.phase.eq('historical_cv') & metrics.model.eq(kind)]
        if set(part.fold) != set(YEARS) or part.fold.duplicated().any():
            raise ValueError('Complete prescribed historical years are required')
        if not np.isfinite(part.mse).all():
            raise ValueError('Finite historical errors required')
        scores[kind] = float(part.mse.mean())
    selected = min(KINDS[1:], key=lambda name: scores[name])
    return dict(selected_model=selected, historical_mse=scores, selection_years=list(YEARS), evaluation_year=2018, tie_rule='Calendar first on exact unrounded tie', evaluation_used_for_selection=False)

def paired_bootstrap(predictions, comparisons, replicates=2000, seed=20261010):
    if predictions.year.nunique() != 1:
        raise ValueError('One evaluation year required')
    rows = []
    identity = ['plot_id', 'cultivar_id', 'year', 'date', 'green_fraction']
    for candidate, reference in comparisons:
        a = predictions[predictions.model.eq(candidate)].set_index('observation_id').sort_index()
        b = predictions[predictions.model.eq(reference)].set_index('observation_id').sort_index()
        if a.empty or a.index.duplicated().any() or b.index.duplicated().any() or (not a.index.equals(b.index)):
            raise ValueError('Models must use the same observation population')
        if not a[identity].equals(b[identity]):
            raise ValueError('Models must use the same observation population')
        if not np.isfinite(a[['prediction', 'green_fraction']]).all().all() or not np.isfinite(b.prediction).all():
            raise ValueError('Finite paired values required')
        ea = a.prediction - a.green_fraction
        eb = b.prediction - b.green_fraction
        part = a[identity].assign(a_squared=ea ** 2, b_squared=eb ** 2, absolute_difference=abs(ea) - abs(eb), signed_difference=ea - eb)
        plot = part.groupby(['cultivar_id', 'plot_id'])[['a_squared', 'b_squared', 'absolute_difference', 'signed_difference']].mean()
        window_rows = []
        for keys, g in part.groupby(['cultivar_id', 'plot_id']):
            if len(g) < 2:
                continue
            g = g.sort_values('date')
            gap = np.diff(pd.to_datetime(g.date).to_numpy(dtype='datetime64[D]').astype(int))
            if np.any(gap <= 0):
                raise ValueError('Distinct observation dates required')
            va = ea.loc[g.index].to_numpy()
            vb = eb.loc[g.index].to_numpy()
            error = abs(np.sum(gap * (va[:-1] + va[1:]) / 2)) - abs(np.sum(gap * (vb[:-1] + vb[1:]) / 2))
            window_rows.append(dict(cultivar_id=keys[0], integral_difference=error, integral_plots=1))
        grouped = plot.assign(plots=1).groupby('cultivar_id').sum()
        if window_rows:
            windows = pd.DataFrame(window_rows).groupby('cultivar_id').sum()
            grouped = grouped.join(windows).fillna({'integral_difference': 0.0, 'integral_plots': 0})
        else:
            grouped['integral_difference'] = 0.0
            grouped['integral_plots'] = 0
        n = len(grouped)
        if n < 2:
            raise ValueError('At least two informative cultivars required for resampling')
        rng = np.random.default_rng(seed)
        draws = rng.multinomial(n, np.ones(n) / n, size=replicates)
        denominator = draws @ grouped.plots.to_numpy()
        difference = np.sqrt(draws @ grouped.a_squared.to_numpy() / denominator) - np.sqrt(draws @ grouped.b_squared.to_numpy() / denominator)
        row = dict(model=candidate, comparator=reference, rmse_difference=float(np.sqrt(plot.a_squared.mean()) - np.sqrt(plot.b_squared.mean())), lower95=float(np.quantile(difference, 0.025)), upper95=float(np.quantile(difference, 0.975)), cultivars=n, plots=len(plot), records=len(a), replicates=replicates, seed=seed, limited_cluster_support=n < 30, scope='Conditional on fixed fits and one site-year; cultivar resampling')
        for metric, column in [('mae', 'absolute_difference'), ('bias', 'signed_difference')]:
            values = draws @ grouped[column].to_numpy() / denominator
            row[metric + '_difference'] = float(plot[column].mean())
            row[metric + '_lower95'] = float(np.quantile(values, 0.025))
            row[metric + '_upper95'] = float(np.quantile(values, 0.975))
        support = draws @ grouped.integral_plots.to_numpy()
        valid = support > 0
        n_integrals = int(grouped.integral_plots.sum())
        n_integral_cultivars = int(grouped.integral_plots.gt(0).sum())
        values = draws[valid] @ grouped.integral_difference.to_numpy() / support[valid]
        row.update(integral_plots=n_integrals, integral_cultivars=n_integral_cultivars, integral_limited_cluster_support=n_integral_cultivars < 30, integral_valid_draws=int(valid.sum()), integral_mae_difference=float(grouped.integral_difference.sum() / n_integrals) if n_integrals else None, integral_lower95=float(np.quantile(values, 0.025)) if n_integral_cultivars >= 2 else None, integral_upper95=float(np.quantile(values, 0.975)) if n_integral_cultivars >= 2 else None)
        rows.append(row)
    return pd.DataFrame(rows)

def integrals(predictions):
    rows = []
    for keys, part in predictions.groupby(['phase', 'fold', 'model', 'plot_id'], sort=True):
        part = part.sort_values('date')
        if len(part) < 2:
            continue
        times = pd.to_datetime(part.date).to_numpy(dtype='datetime64[D]').astype(int)
        gaps = np.diff(times)
        if np.any(gaps <= 0):
            raise ValueError('Distinct observation dates required')

        def area(name):
            v = part[name].to_numpy(float)
            return float(np.sum(gaps * (v[:-1] + v[1:]) / 2))
        observed, predicted = (area('green_fraction'), area('prediction'))
        rows.append(dict(zip(['phase', 'fold', 'model', 'plot_id'], keys), cultivar_id=part.cultivar_id.iloc[0], observations=len(part), first_date=part.date.iloc[0], last_date=part.date.iloc[-1], duration_days=int(times[-1] - times[0]), observed_green_fraction_days=observed, predicted_green_fraction_days=predicted, signed_error=predicted - observed, absolute_error=abs(predicted - observed)))
    return pd.DataFrame(rows)

def verify_inputs():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def implementation():
    return {name: sha(HERE / name) for name in ['features.py', 'model.py', 'run.py']}
