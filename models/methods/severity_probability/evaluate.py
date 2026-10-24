"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/severity_probability_20261011')
import numpy as np
import pandas as pd
from analysis.methods.severity_probability import data, models
PROBABILITY_MODELS = ('ordinal_full', 'ordinal_weather', 'ordinal_calendar', 'leaf_frequency')
POINT_MODELS = PROBABILITY_MODELS + ('archived_joint', 'archived_leaf_mean', 'archived_boosting')
EPSILON = 1e-12

def losses(y, probability):
    y = np.asarray(y, float)
    p = np.asarray(probability, float)
    if p.shape != (len(y), 4) or not np.isfinite(p).all() or (p < 0).any() or (not np.allclose(p.sum(1), 1, rtol=0, atol=1e-12)):
        raise ValueError('Exhaustive four-class probabilities required')
    labels = models.classes(y)
    actual = p[np.arange(len(y)), labels]
    rps = np.mean((models.exceedance(p) - (y[:, None] >= models.THRESHOLDS)) ** 2, axis=1)
    return (rps, -np.log(np.maximum(actual, EPSILON)), int((actual < EPSILON).sum()))

def discrimination(label, score, weight):
    y = np.asarray(label, int)
    s = np.asarray(score, float)
    w = np.asarray(weight, float)
    pos = np.sum(w * y)
    neg = np.sum(w * (1 - y))
    if pos <= 0 or neg <= 0:
        return dict(auc=None, average_precision=None, prevalence=float(pos / (pos + neg)), status='single_class')
    _, inv = np.unique(s, return_inverse=True)
    pm = np.bincount(inv, weights=w * y)
    nm = np.bincount(inv, weights=w * (1 - y))
    auc = np.sum(pm * (np.cumsum(nm) - nm / 2)) / (pos * neg)
    tp = np.cumsum(pm[::-1])
    fp = np.cumsum(nm[::-1])
    den = tp + fp
    precision = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    return dict(auc=float(auc), average_precision=float(np.sum(pm[::-1] / pos * precision)), prevalence=float(pos / (pos + neg)), status='estimated')

def bootstrap_losses(frame, values, *, replicates=2000, seed=data.SEED):
    w = data.weights(frame)
    sites, inv = np.unique(frame.site_id, return_inverse=True)
    denominator = np.bincount(inv, weights=w)
    rng = np.random.default_rng(seed)
    count = np.array([np.bincount(rng.integers(0, len(sites), len(sites)), minlength=len(sites)) for _ in range(replicates)])
    point = {k: float(np.average(v, weights=w)) for k, v in values.items()}
    draws = {k: count @ np.bincount(inv, weights=w * np.asarray(v, float)) / (count @ denominator) for k, v in values.items()}
    return (point, pd.DataFrame({'replicate': np.arange(replicates), **draws}))

def score_magnitude(frame):
    w = data.weights(frame)
    y = frame.observed_percent.to_numpy(float)
    e = frame.prediction.to_numpy(float) - y
    mse = np.average(e * e, weights=w)
    variance = np.average((y - np.average(y, weights=w)) ** 2, weights=w)
    return dict(n=len(frame), fields=frame.field_id.nunique(), sites=frame.site_id.nunique(), rmse=float(np.sqrt(mse)), mae=float(np.average(abs(e), weights=w)), bias=float(np.average(e, weights=w)), r2=float(1 - mse / variance) if variance > 0 else np.nan)

def windows(predictions):
    rows = []
    for (model, field, series), part in predictions.groupby(['model', 'field_id', 'endpoint_series'], sort=True):
        part = part.sort_values('day_index')
        days = part.day_index.to_numpy(float)
        if len(part) < 2:
            continue
        if np.any(np.diff(days) <= 0):
            raise ValueError('Window assessment dates must be unique')
        first = part.iloc[0]
        duration = days[-1] - days[0]
        observed = float(np.trapezoid(part.observed_percent, days))
        predicted = float(np.trapezoid(part.prediction, days))
        meta = {k: first[k] for k in ['site_id', 'coordinate_year', 'leaf_index', 'source', 'test_year', 'new_site']}
        rows.append(dict(model=model, field_id=field, endpoint_series=series, window_id=field + '::' + series, duration_days=duration, observed_integral=observed, predicted_integral=predicted, observed_mean=observed / duration, predicted_mean=predicted / duration, **meta))
    return pd.DataFrame(rows)

def groups(frame):
    yield ('pooled', frame)
    for name, part in frame.groupby('source', sort=True):
        yield (name, part)

def paired(predictions, window_table, replicates):
    results = []
    samples = []
    for population in ['all_future_records', 'new_sites']:
        pp = predictions if population == 'all_future_records' else predictions.loc[predictions.new_site]
        ww = window_table if population == 'all_future_records' else window_table.loc[window_table.new_site]
        for source, part in groups(pp):
            base = part.loc[part.model.eq('ordinal_full')].sort_values('global_target_index')
            ids = base.global_target_index
            w = data.weights(base)
            values = {}
            for model in POINT_MODELS:
                sample = part.loc[part.model.eq(model)].set_index('global_target_index').loc[ids]
                values['mse_' + model] = (sample.prediction.to_numpy() - base.observed_percent.to_numpy()) ** 2
                if model in PROBABILITY_MODELS:
                    p = sample[['p0', 'p1', 'p2', 'p3']].to_numpy(float)
                    values['rps_' + model] = losses(base.observed_percent, p)[0]
                    for j, t in enumerate(models.THRESHOLDS):
                        values[f'brier_{int(t)}_' + model] = (models.exceedance(p)[:, j] - (base.observed_percent.to_numpy() >= t)) ** 2
            point, draw = bootstrap_losses(base, values, replicates=replicates)
            window = ww if source == 'pooled' else ww.loc[ww.source.eq(source)]
            wb = window.loc[window.model.eq('ordinal_full')].sort_values('window_id')
            sites = np.unique(base.site_id)
            site_index = {s: i for i, s in enumerate(sites)}
            wi = wb.site_id.map(site_index).to_numpy(int)
            weights = data.weights(wb)
            rng = np.random.default_rng(data.SEED)
            counts = np.array([np.bincount(rng.integers(0, len(sites), len(sites)), minlength=len(sites)) for _ in range(replicates)])
            den = np.bincount(wi, weights=weights, minlength=len(sites))
            den_draw = counts @ den
            for model in POINT_MODELS:
                sample = window.loc[window.model.eq(model)].set_index('window_id').loc[wb.window_id]
                error = (sample.predicted_integral.to_numpy() - wb.observed_integral.to_numpy()) ** 2
                num = np.bincount(wi, weights=weights * error, minlength=len(sites))
                key = 'integral_mse_' + model
                point[key] = float(np.average(error, weights=weights))
                draw[key] = np.divide(counts @ num, den_draw, out=np.full(replicates, np.nan), where=den_draw > 0)
            comparisons = {}
            for ref in ['leaf_frequency', 'ordinal_calendar', 'ordinal_weather']:
                comparisons['rps_difference__' + ref] = (point['rps_ordinal_full'] - point['rps_' + ref], draw.rps_ordinal_full - draw['rps_' + ref])
            for t in models.THRESHOLDS:
                key = f'brier_{int(t)}_'
                comparisons[key + 'difference__leaf_frequency'] = (point[key + 'ordinal_full'] - point[key + 'leaf_frequency'], draw[key + 'ordinal_full'] - draw[key + 'leaf_frequency'])
            for ref in ['archived_leaf_mean', 'archived_joint', 'archived_boosting']:
                for prefix in ['', 'integral_']:
                    a = prefix + 'mse_ordinal_full'
                    b = prefix + 'mse_' + ref
                    comparisons[prefix + 'rmse_difference__' + ref] = (np.sqrt(point[a]) - np.sqrt(point[b]), np.sqrt(draw[a]) - np.sqrt(draw[b]))
            for name, (value, values_draw) in comparisons.items():
                metric, reference = name.split('__')
                finite = np.asarray(values_draw)[np.isfinite(values_draw)]
                lower, upper = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
                results.append(dict(population=population, source=source, metric=metric, reference=reference, value=float(value), lower=lower, upper=upper, replicates=replicates, valid_replicates=len(finite), sites=len(sites)))
                samples.append(pd.DataFrame(dict(population=population, source=source, metric=metric, reference=reference, replicate=np.arange(replicates), value=values_draw)))
    return (pd.DataFrame(results), pd.concat(samples, ignore_index=True))

def decide(paired_intervals):
    criteria = []
    for population in ['all_future_records', 'new_sites']:
        part = paired_intervals.loc[paired_intervals.population.eq(population)]
        for ref in ['leaf_frequency', 'ordinal_calendar']:
            row = part.loc[part.source.eq('pooled') & part.metric.eq('rps_difference') & part.reference.eq(ref)].iloc[0]
            criteria.append(dict(criterion='probability', population=population, source='pooled', reference=ref, metric='rps_upper_95', value=float(row.upper), passed=bool(row.upper < 0 and row.valid_replicates >= 0.95 * row.replicates)))
        for row in part.loc[part.source.ne('pooled') & part.metric.eq('rps_difference') & part.reference.eq('leaf_frequency')].itertuples():
            criteria.append(dict(criterion='probability', population=population, source=row.source, reference='leaf_frequency', metric='source_average_rps_difference', value=float(row.value), passed=bool(row.value <= 0)))
        for metric in ['rmse_difference', 'integral_rmse_difference']:
            row = part.loc[part.source.eq('pooled') & part.metric.eq(metric) & part.reference.eq('archived_leaf_mean')].iloc[0]
            criteria.append(dict(criterion='magnitude', population=population, source='pooled', reference='archived_leaf_mean', metric=metric + '_upper_95', value=float(row.upper), passed=bool(row.upper < 0 and row.valid_replicates >= 0.95 * row.replicates)))
    return dict(probability_support=all((c['passed'] for c in criteria if c['criterion'] == 'probability')), magnitude_support=all((c['passed'] for c in criteria if c['criterion'] == 'magnitude')), criteria=criteria, prospective_validation=False, independent_validation=False, new_onset_predictions=False, climate_adoption=False, functional_canopy_validated=False, yield_validated=False)

def evaluate(predictions, replicates=2000):
    if predictions.duplicated(['global_target_index', 'model']).any():
        raise ValueError('Duplicate target-model predictions')
    keys = set(predictions.loc[predictions.model.eq('ordinal_full'), 'global_target_index'])
    for m in POINT_MODELS:
        if set(predictions.loc[predictions.model.eq(m), 'global_target_index']) != keys:
            raise ValueError('Unequal model evaluation membership')
    win = windows(predictions)
    magnitude = []
    probability = []
    risk = []
    calibration = []
    cohorts = []
    for pop in ['all_future_records', 'new_sites']:
        pp = predictions if pop == 'all_future_records' else predictions.loc[predictions.new_site]
        ww = win if pop == 'all_future_records' else win.loc[win.new_site]
        for year in [0] + list(data.YEARS):
            a = pp if year == 0 else pp.loc[pp.test_year.eq(year)]
            b = ww if year == 0 else ww.loc[ww.test_year.eq(year)]
            for source, part in groups(a):
                meta = dict(population=pop, test_year=year, source=source)
                base = part.loc[part.model.eq('ordinal_full')]
                cohorts.append(dict(**meta, assessments=len(base), fields=base.field_id.nunique(), sites=base.site_id.nunique(), unseen_leaf=int(base.unseen_leaf.sum())))
                for model, sample in part.groupby('model', sort=True):
                    magnitude.append(dict(**meta, model=model, endpoint='assessment', **score_magnitude(sample)))
                    w = data.weights(sample)
                    y = sample.observed_percent.to_numpy(float)
                    exceed = None
                    if model in PROBABILITY_MODELS:
                        p = sample[['p0', 'p1', 'p2', 'p3']].to_numpy(float)
                        rps, ll, clipped = losses(y, p)
                        exceed = models.exceedance(p)
                        probability.append(dict(**meta, model=model, n=len(sample), rps=float(np.average(rps, weights=w)), log_loss=float(np.average(ll, weights=w)), log_clip_count=clipped))
                    for j, threshold in enumerate(models.THRESHOLDS):
                        label = y >= threshold
                        score = exceed[:, j] if exceed is not None else sample.prediction.to_numpy(float)
                        record = dict(**meta, model=model, threshold=int(threshold), n=len(sample), positive_n=int(label.sum()), negative_n=int((~label).sum()), brier=float(np.average((score - label) ** 2, weights=w)) if exceed is not None else np.nan, **discrimination(label, score, w))
                        risk.append(record)
                        if exceed is not None:
                            bins = np.minimum((score * 10).astype(int), 9)
                            for index in range(10):
                                mask = bins == index
                                mass = w[mask].sum()
                                calibration.append(dict(**meta, model=model, threshold=int(threshold), bin=index, lower=index / 10, upper=(index + 1) / 10, n=int(mask.sum()), weight_fraction=float(mass / w.sum()), predicted=float(np.average(score[mask], weights=w[mask])) if mask.any() else np.nan, observed=float(np.average(label[mask], weights=w[mask])) if mask.any() else np.nan))
                sample = b if source == 'pooled' else b.loc[b.source.eq(source)]
                for model, window in sample.groupby('model', sort=True):
                    for endpoint, suffix in [('window_mean', 'mean'), ('window_integral', 'integral')]:
                        f = window.assign(observed_percent=window['observed_' + suffix], prediction=window['predicted_' + suffix])
                        magnitude.append(dict(**meta, model=model, endpoint=endpoint, **score_magnitude(f)))
    intervals, draws = paired(predictions, win, replicates)
    return dict(magnitude=pd.DataFrame(magnitude), probability=pd.DataFrame(probability), discrimination=pd.DataFrame(risk), calibration=pd.DataFrame(calibration), cohorts=pd.DataFrame(cohorts), windows=win, paired_intervals=intervals, paired_draws=draws)
