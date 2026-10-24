"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_joint_forecast_20261010')
import numpy as np
import pandas as pd
from analysis.methods.observation_model import observation as metrics
from analysis.methods.crop_disease.evaluation import source_groups, endpoint_frames
PREDICTIONS = ['joint', 'fold_leaf_mean', 'boosting', 'raw_retained', 'mapped_retained', 'raw_lstm_extended_q50', 'archived_leaf_mean']

def event_labels(frame):
    copy = frame.reset_index(drop=True).assign(_position=np.arange(len(frame)))
    ordered = copy.sort_values(['field_id', 'endpoint_series', 'day_index'])
    values = ordered.observed_percent.gt(0).groupby([ordered.field_id, ordered.endpoint_series]).cummax().astype(int)
    out = np.empty(len(frame), int)
    out[ordered._position.to_numpy(int)] = values
    return out

def interval_distance(frame, prediction):
    predicted = np.asarray(prediction, float)
    lower = frame.lower_day.to_numpy(float)
    upper = frame.upper_day.to_numpy(float)
    return np.maximum(lower + 1 - predicted, 0) + np.where(np.isfinite(upper), np.maximum(predicted - upper, 0), 0)

def weighted_midrank(values, weight):
    values = np.asarray(values)
    weight = np.asarray(weight, float)
    _, inverse = np.unique(values, return_inverse=True)
    mass = np.bincount(inverse, weights=weight)
    return (np.cumsum(mass) - mass / 2)[inverse]

def rank_correlation(observed, predicted, weight):
    weight = np.asarray(weight, float)
    a = weighted_midrank(observed, weight)
    b = weighted_midrank(predicted, weight)
    a -= np.average(a, weights=weight)
    b -= np.average(b, weights=weight)
    denominator = np.sqrt(np.sum(weight * a * a) * np.sum(weight * b * b))
    return float(np.sum(weight * a * b) / denominator) if denominator > 0 else None

def discrimination(label, score, weight):
    y = np.asarray(label, int)
    score = np.asarray(score, float)
    w = np.asarray(weight, float)
    pos = np.sum(w * y)
    neg = np.sum(w * (1 - y))
    if pos <= 0 or neg <= 0:
        return dict(auc=None, average_precision=None, prevalence=float(pos / (pos + neg)))
    _, inverse = np.unique(score, return_inverse=True)
    pm = np.bincount(inverse, weights=w * y)
    nm = np.bincount(inverse, weights=w * (1 - y))
    auc = np.sum(pm * (np.cumsum(nm) - nm / 2)) / (pos * neg)
    tp = np.cumsum(pm[::-1])
    fp = np.cumsum(nm[::-1])
    den = tp + fp
    precision = np.divide(tp, den, out=np.zeros_like(tp), where=den > 0)
    ap = np.sum(pm[::-1] / pos * precision)
    return dict(auc=float(auc), average_precision=float(ap), prevalence=float(pos / (pos + neg)))

def calibration(frame):
    w = metrics.hierarchical_weights(frame)
    prob = frame.occurrence.to_numpy(float)
    y = frame.event_label.to_numpy(float)
    bins = np.minimum((prob * 10).astype(int), 9)
    rows = []
    for index in range(10):
        take = bins == index
        if take.any():
            rows.append(dict(bin=index, lower=index / 10, upper=(index + 1) / 10, n=int(take.sum()), mass=float(w[take].sum()), predicted=float(np.average(prob[take], weights=w[take])), observed=float(np.average(y[take], weights=w[take]))))
    return pd.DataFrame(rows)

def onset_scores(intervals):
    rows = []
    for scope, frame in [('all_available', intervals), ('genuine_two_sided', intervals.loc[intervals.genuine])]:
        for source, part in source_groups(frame):
            if part.empty:
                continue
            weight = metrics.hierarchical_weights(part)
            for model in ['joint', 'calendar', 'retained', 'lstm_extended_q50']:
                distance = part[model + '_distance'].to_numpy(float)
                rows.append(dict(scope=scope, source=source, model=model, n=len(part), sites=part.site_id.nunique(), mean_distance=float(np.average(distance, weights=weight)), coverage=float(np.average(distance == 0, weights=weight)), unresolved=int((~part.joint_crossed).sum()) if model == 'joint' else 0))
    return pd.DataFrame(rows)

def complementary(frame):
    rows = []
    bins = []
    for source, part in source_groups(frame):
        w = metrics.hierarchical_weights(part)
        y = part.observed_percent.to_numpy(float)
        for model in PREDICTIONS:
            rank = rank_correlation(y, part[model], w)
            for threshold in [10, 25, 50]:
                stats = discrimination(y >= threshold, part[model], w)
                rows.append(dict(endpoint='severity_discrimination', source=source, model=model, threshold=threshold, n=len(part), sites=part.site_id.nunique(), positive_n=int((y >= threshold).sum()), negative_n=int((y < threshold).sum()), rank_correlation=rank, brier=None, **stats))
        for model in ['occurrence', 'fold_occurrence']:
            error = part[model].to_numpy(float) - part.event_label.to_numpy(float)
            stats = discrimination(part.event_label, part[model], w)
            rows.append(dict(endpoint='first_positive_occurrence', source=source, model=model, threshold=None, n=len(part), sites=part.site_id.nunique(), positive_n=int(part.event_label.sum()), negative_n=int((1 - part.event_label).sum()), rank_correlation=None, brier=float(np.average(error ** 2, weights=w)), **stats))
        bins.append(calibration(part).assign(source=source))
    return (pd.DataFrame(rows), pd.concat(bins, ignore_index=True))

def magnitude_comparisons(assessments, windows, replicates=2000):
    rows = []
    samples = []
    for endpoint, frame in endpoint_frames(assessments, windows, PREDICTIONS):
        for source, part in source_groups(frame):
            for benchmark in ['raw_retained', 'archived_leaf_mean', 'fold_leaf_mean', 'boosting']:
                point, draws = metrics.bootstrap_pair(part, 'joint', benchmark, replicates=replicates, seed=20261010)
                identity = dict(endpoint=endpoint, source=source, candidate='joint', benchmark=benchmark)
                intervals = {}
                for metric in ['rmse', 'mae', 'bias']:
                    lo, hi = np.quantile(draws[metric + '_difference'], [0.025, 0.975])
                    intervals.update({metric + '_difference_lower': float(lo), metric + '_difference_upper': float(hi)})
                rows.append(dict(**identity, n=len(part), sites=part.site_id.nunique(), **point, **intervals))
                samples.append(pd.DataFrame(draws).assign(**identity, replicate=np.arange(replicates)))
    return (pd.DataFrame(rows), pd.concat(samples, ignore_index=True))

def complementary_bootstrap(assessments, intervals, replicates=2000):
    rows = []
    drawrows = []
    for source, part in source_groups(assessments):
        base = metrics.hierarchical_weights(part)
        sites, inverse = np.unique(part.site_id, return_inverse=True)
        rng = np.random.default_rng(20261010)
        y = part.observed_percent.to_numpy(float)
        event = part.event_label.to_numpy(float)
        candidate = part.joint.to_numpy(float)
        reference = part.fold_leaf_mean.to_numpy(float)
        eb = (part.occurrence.to_numpy(float) - event) ** 2 - (part.fold_occurrence.to_numpy(float) - event) ** 2

        def reduce(weight):
            out = {'brier_difference': float(np.average(eb, weights=weight))}
            for threshold in [10, 25, 50]:
                a = discrimination(y >= threshold, candidate, weight)['auc']
                b = discrimination(y >= threshold, reference, weight)['auc']
                out[f'auc_{threshold}'] = a
                out[f'auc_difference_{threshold}'] = None if a is None or b is None else a - b
            return out
        point = reduce(base)
        draws = []
        for replicate in range(replicates):
            count = np.bincount(rng.integers(0, len(sites), len(sites)), minlength=len(sites))
            draws.append(dict(replicate=replicate, **reduce(base * count[inverse])))
        samples = pd.DataFrame(draws)
        for metric, value in point.items():
            finite = samples[metric].dropna()
            lo, hi = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
            rows.append(dict(source=source, metric=metric, value=value, lower=float(lo), upper=float(hi), valid_replicates=len(finite), replicates=replicates, sites=len(sites)))
        drawrows.append(samples.assign(source=source))
    genuine = intervals.loc[intervals.genuine]
    for source, part in source_groups(genuine):
        base = metrics.hierarchical_weights(part)
        sites, inverse = np.unique(part.site_id, return_inverse=True)
        error = part.joint_distance.to_numpy(float) - part.calendar_distance.to_numpy(float)
        rng = np.random.default_rng(20261010)
        values = []
        for _ in range(replicates):
            count = np.bincount(rng.integers(0, len(sites), len(sites)), minlength=len(sites))
            values.append(float(np.average(error, weights=base * count[inverse])))
        lo, hi = np.quantile(values, [0.025, 0.975])
        rows.append(dict(source=source, metric='onset_distance_difference', value=float(np.average(error, weights=base)), lower=float(lo), upper=float(hi), valid_replicates=replicates, replicates=replicates, sites=len(sites)))
        drawrows.append(pd.DataFrame(dict(replicate=np.arange(replicates), source=source, onset_distance_difference=values)))
    return (pd.DataFrame(rows), pd.concat(drawrows, ignore_index=True))

def decide(summary, comparisons, onset, complementary_intervals):

    def one(frame, column, **where):
        for key, value in where.items():
            frame = frame.loc[frame[key].eq(value)]
        if len(frame) != 1 or not np.isfinite(frame[column].iloc[0]):
            raise ValueError('Missing decision evidence')
        return float(frame[column].iloc[0])
    criteria = []
    alternative = []
    for source in ['pooled', 'BASF', 'Corteva']:
        for benchmark in ['raw_retained', 'archived_leaf_mean']:
            upper = one(comparisons, 'rmse_difference_upper', source=source, endpoint='window_integral', benchmark=benchmark)
            criteria.append(dict(source=source, endpoint='window_integral', benchmark=benchmark, value=upper, passed=upper < 0))
            for endpoint in ['assessment', 'window_mean']:
                delta = one(summary, 'rmse', source=source, endpoint=endpoint, model='joint') - one(summary, 'rmse', source=source, endpoint=endpoint, model=benchmark)
                criteria.append(dict(source=source, endpoint=endpoint, benchmark=benchmark, value=delta, passed=delta <= 0))
        delta = one(onset, 'mean_distance', source=source, scope='genuine_two_sided', model='joint') - one(onset, 'mean_distance', source=source, scope='genuine_two_sided', model='retained')
        criteria.append(dict(source=source, endpoint='genuine_onset', benchmark='retained', value=delta, passed=delta <= 0))
        for metric in ['brier_difference', 'onset_distance_difference']:
            upper = one(complementary_intervals, 'upper', source=source, metric=metric)
            alternative.append(dict(source=source, endpoint=metric, value=upper, passed=upper < 0))
        for threshold in [10, 25, 50]:
            lo = one(complementary_intervals, 'lower', source=source, metric=f'auc_{threshold}')
            diff = one(complementary_intervals, 'lower', source=source, metric=f'auc_difference_{threshold}')
            alternative.append(dict(source=source, endpoint=f'severity_{threshold}', auc_lower=lo, difference_lower=diff, passed=lo > 0.5 and diff > 0))
    endpoints = sorted(set((x['endpoint'] for x in alternative)))
    return dict(quantitative_criteria_pass=all((x['passed'] for x in criteria)), criteria=criteria, complementary_endpoint_pass={endpoint: all((x['passed'] for x in alternative if x['endpoint'] == endpoint)) for endpoint in endpoints}, complementary_criteria=alternative, climate_adoption=False, independent_validation=False, assessment_weather_lead_days=0, interpretation='Retrospective site-held-out prediction; source transport reported separately; no quantitative canopy or yield validation')
