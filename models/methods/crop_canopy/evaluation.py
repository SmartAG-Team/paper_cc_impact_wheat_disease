"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_canopy_coupling_20261011')
import numpy as np
import pandas as pd
from analysis.methods.crop_canopy.models import weights, KINDS
YEARS = (2015, 2016, 2017)

def integrals(frame, prediction):
    rows = []
    for (year, plot), part in frame.assign(prediction=np.asarray(prediction)).groupby(['year', 'plot_id'], sort=True):
        part = part.sort_values('date')
        if len(part) < 2:
            continue
        times = pd.to_datetime(part.date).to_numpy(dtype='datetime64[D]').astype(int)
        gap = np.diff(times)
        if np.any(gap <= 0):
            raise ValueError('Distinct ordered observation dates required')
        area = lambda v: float(np.sum(gap * (v[:-1] + v[1:]) / 2))
        observed, predicted = (area(part.green_fraction.to_numpy()), area(part.prediction.to_numpy()))
        rows.append(dict(year=int(year), plot_id=plot, cultivar_id=part.cultivar_id.iloc[0], observations=len(part), first_date=part.date.iloc[0], last_date=part.date.iloc[-1], duration_days=int(times[-1] - times[0]), observed_green_fraction_days=observed, predicted_green_fraction_days=predicted, signed_error=predicted - observed, absolute_error=abs(predicted - observed)))
    return pd.DataFrame(rows)

def scores(frame, prediction):
    y = frame.green_fraction.to_numpy(float)
    p = np.asarray(prediction, float)
    if p.shape != y.shape or not np.isfinite(p).all() or (not np.isfinite(y).all()):
        raise ValueError('Finite paired observations required')
    w = weights(frame)
    error = p - y
    windows = integrals(frame, p)
    integral = float(windows.groupby('year').absolute_error.mean().mean()) if len(windows) else None
    return dict(records=len(frame), plots=frame.plot_id.nunique(), years=frame.year.nunique(), mse=float(w @ error ** 2), rmse=float(np.sqrt(w @ error ** 2)), mae=float(w @ abs(error)), bias=float(w @ error), integral_plots=len(windows), integral_mae_green_fraction_days=integral)

def choose_historical(metrics):
    result = {}
    for kind in KINDS[1:]:
        part = metrics.loc[metrics.phase.eq('historical_cv') & metrics.model.eq(kind)]
        if len(part) != 3 or set(part.fold) != set(YEARS) or (not np.isfinite(part.mse).all()):
            raise ValueError('Complete prescribed historical folds required')
        result[kind] = float(part.mse.mean())
    return dict(selected_model=min(KINDS[1:], key=lambda x: result[x]), historical_mse=result, selection_years=list(YEARS), evaluation_year=2018, evaluation_used_for_selection=False, tie_rule='calendar, median, marginal on exact unrounded tie')

def paired_bootstrap(predictions, comparisons, replicates=2000, seed=20261011):
    if predictions.year.nunique() != 1:
        raise ValueError('One evaluation year required')
    summaries, all_draws = ([], [])
    identity = ['plot_id', 'cultivar_id', 'year', 'date', 'green_fraction']
    for candidate, reference in comparisons:
        a = predictions.loc[predictions.model.eq(candidate)].set_index('observation_id').sort_index()
        b = predictions.loc[predictions.model.eq(reference)].set_index('observation_id').sort_index()
        if a.empty or a.index.duplicated().any() or b.index.duplicated().any() or (not a.index.equals(b.index)) or (not a[identity].equals(b[identity])):
            raise ValueError('Models must use the same observation population')
        ea, eb = (a.prediction - a.green_fraction, b.prediction - b.green_fraction)
        plot = a[identity].assign(a2=ea ** 2, b2=eb ** 2, mae=abs(ea) - abs(eb), bias=ea - eb).groupby(['cultivar_id', 'plot_id'])[['a2', 'b2', 'mae', 'bias']].mean()
        ia, ib = (integrals(a, a.prediction), integrals(b, b.prediction))
        windows = ia[['cultivar_id', 'plot_id']].assign(integral=ia.absolute_error.to_numpy() - ib.absolute_error.to_numpy(), integral_plots=1)
        grouped = plot.assign(plots=1).groupby('cultivar_id', sort=True).sum()
        grouped = grouped.join(windows.groupby('cultivar_id')[['integral', 'integral_plots']].sum()).fillna(0.0)
        n = len(grouped)
        if n < 2:
            raise ValueError('At least two cultivars required')
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, np.full(n, 1 / n), size=replicates)
        den = counts @ grouped.plots.to_numpy()
        draws = pd.DataFrame(dict(replicate=np.arange(replicates), rmse_difference=np.sqrt(counts @ grouped.a2.to_numpy() / den) - np.sqrt(counts @ grouped.b2.to_numpy() / den), mae_difference=counts @ grouped.mae.to_numpy() / den, bias_difference=counts @ grouped.bias.to_numpy() / den, integral_mae_difference=counts @ grouped.integral.to_numpy() / (counts @ grouped.integral_plots.to_numpy())))
        values = dict(rmse=np.sqrt(plot.a2.mean()) - np.sqrt(plot.b2.mean()), mae=plot.mae.mean(), bias=plot.bias.mean(), integral_mae=windows.integral.mean())
        row = dict(model=candidate, comparator=reference, cultivars=n, plots=len(plot), records=len(a), replicates=replicates, seed=seed, scope='Fixed fits and one site-year; whole-cultivar resampling')
        for name, value in values.items():
            row[name + '_difference'] = float(value)
            row[name + '_lower95'], row[name + '_upper95'] = np.quantile(draws[name + '_difference'], [0.025, 0.975])
        summaries.append(row)
        all_draws.append(draws.assign(model=candidate, comparator=reference))
    return (pd.DataFrame(summaries), pd.concat(all_draws, ignore_index=True))
