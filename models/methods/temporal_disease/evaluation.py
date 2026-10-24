"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_temporal_validation_20261010')
import numpy as np
import pandas as pd
from analysis.methods.joint_disease import evaluation as original
from analysis.methods.observation_model import observation as metrics
PREDICTIONS = ['joint', 'fold_leaf_mean', 'boosting']
THRESHOLDS = [10, 25, 50]

def complementary(frame):
    rows, bins = ([], [])
    for source, part in original.source_groups(frame):
        if part.empty:
            continue
        weight = metrics.hierarchical_weights(part)
        observed = part.observed_percent.to_numpy(float)
        for model in PREDICTIONS:
            rank = original.rank_correlation(observed, part[model], weight)
            for threshold in THRESHOLDS:
                positive = observed >= threshold
                rows.append(dict(endpoint='severity_discrimination', source=source, model=model, threshold=threshold, n=len(part), sites=part.site_id.nunique(), positive_n=int(positive.sum()), negative_n=int((~positive).sum()), rank_correlation=rank, brier=None, **original.discrimination(positive, part[model], weight)))
        for model in ['occurrence', 'fold_occurrence']:
            error = part[model].to_numpy(float) - part.event_label.to_numpy(float)
            rows.append(dict(endpoint='first_positive_occurrence', source=source, model=model, threshold=None, n=len(part), sites=part.site_id.nunique(), positive_n=int(part.event_label.sum()), negative_n=int((1 - part.event_label).sum()), rank_correlation=None, brier=float(np.average(error ** 2, weights=weight)), **original.discrimination(part.event_label, part[model], weight)))
        bins.append(original.calibration(part).assign(source=source))
    return (pd.DataFrame(rows), pd.concat(bins, ignore_index=True))

def onset_scores(intervals):
    rows = []
    sources = ['pooled'] + sorted(intervals.source.unique())
    for scope in ['all_available', 'genuine_two_sided']:
        for source in sources:
            part = intervals if source == 'pooled' else intervals.loc[intervals.source.eq(source)]
            if scope == 'genuine_two_sided':
                part = part.loc[part.genuine]
            for model in ['joint', 'calendar']:
                weight = metrics.hierarchical_weights(part) if len(part) else None
                distance = part[model + '_distance'].to_numpy(float)
                rows.append(dict(scope=scope, source=source, model=model, n=len(part), sites=part.site_id.nunique(), mean_distance=float(np.average(distance, weights=weight)) if len(part) else np.nan, coverage=float(np.average(distance == 0, weights=weight)) if len(part) else np.nan, unresolved=int((~part.joint_crossed).sum()) if model == 'joint' else 0, status='estimated' if len(part) else 'no_eligible_brackets'))
    return pd.DataFrame(rows)

def site_bootstrap(frame, reduce, *, replicates=2000, seed=20261010):
    base = metrics.hierarchical_weights(frame)
    sites, inverse = np.unique(frame.site_id, return_inverse=True)
    rng = np.random.default_rng(seed)
    point = reduce(base)
    draws = []
    for replicate in range(replicates):
        count = np.bincount(rng.integers(0, len(sites), len(sites)), minlength=len(sites))
        draws.append(dict(replicate=replicate, **reduce(base * count[inverse])))
    return (point, pd.DataFrame(draws))

def paired_intervals(assessments, windows, intervals, *, replicates=2000):
    rows, samples = ([], [])
    for endpoint, frame in original.endpoint_frames(assessments, windows, PREDICTIONS):
        for source, part in original.source_groups(frame):
            for benchmark in ['fold_leaf_mean', 'boosting']:
                point, draws = metrics.bootstrap_pair(part, 'joint', benchmark, replicates=replicates, seed=20261010)
                for metric in ['rmse', 'mae', 'bias']:
                    identity = dict(endpoint=endpoint, source=source, benchmark=benchmark, metric=metric + '_difference')
                    values = np.asarray(draws[metric + '_difference'])
                    lower, upper = np.quantile(values, [0.025, 0.975])
                    rows.append(dict(**identity, value=point[metric + '_difference'], lower=lower, upper=upper, valid_replicates=replicates, replicates=replicates, sites=part.site_id.nunique()))
                    samples.append(pd.DataFrame(dict(**identity, replicate=np.arange(replicates), value=values)))
    for source, part in original.source_groups(assessments):
        y = part.observed_percent.to_numpy(float)
        event = part.event_label.to_numpy(float)
        error_joint = (part.occurrence.to_numpy(float) - event) ** 2
        error_leaf = (part.fold_occurrence.to_numpy(float) - event) ** 2

        def reduce(weight):
            result = dict(brier=float(np.average(error_joint, weights=weight)), brier_difference=float(np.average(error_joint - error_leaf, weights=weight)))
            for threshold in THRESHOLDS:
                auc = original.discrimination(y >= threshold, part.joint, weight)['auc']
                reference = original.discrimination(y >= threshold, part.fold_leaf_mean, weight)['auc']
                result[f'auc_{threshold}'] = auc
                result[f'auc_difference_{threshold}'] = None if auc is None or reference is None else auc - reference
            return result
        point, draws = site_bootstrap(part, reduce, replicates=replicates)
        for metric, value in point.items():
            benchmark = 'fold_occurrence' if metric == 'brier_difference' else 'fold_leaf_mean' if metric.startswith('auc_difference_') else 'none'
            identity = dict(endpoint='complementary', source=source, benchmark=benchmark, metric=metric)
            finite = pd.to_numeric(draws[metric], errors='coerce').dropna()
            lower, upper = np.quantile(finite, [0.025, 0.975]) if len(finite) else [np.nan, np.nan]
            rows.append(dict(**identity, value=value, lower=lower, upper=upper, valid_replicates=len(finite), replicates=replicates, sites=part.site_id.nunique()))
            samples.append(draws[['replicate', metric]].rename(columns={metric: 'value'}).assign(**identity))
    for source in ['pooled'] + sorted(intervals.source.unique()):
        part = intervals.loc[intervals.genuine]
        if source != 'pooled':
            part = part.loc[part.source.eq(source)]
        identity = dict(endpoint='genuine_onset', source=source, benchmark='calendar', metric='distance_difference')
        if part.empty:
            rows.append(dict(**identity, value=np.nan, lower=np.nan, upper=np.nan, valid_replicates=0, replicates=replicates, sites=0))
            continue
        difference = part.joint_distance.to_numpy(float) - part.calendar_distance.to_numpy(float)
        point, draws = site_bootstrap(part, lambda w: dict(value=float(np.average(difference, weights=w))), replicates=replicates)
        lower, upper = np.quantile(draws.value, [0.025, 0.975])
        rows.append(dict(**identity, value=point['value'], lower=lower, upper=upper, valid_replicates=replicates, replicates=replicates, sites=part.site_id.nunique()))
        samples.append(draws.assign(**identity))
    return (pd.DataFrame(rows), pd.concat(samples, ignore_index=True))

def evaluate(assessments, intervals, *, replicates=2000):
    result = {key: [] for key in ['summary', 'complementary', 'calibration', 'onset_summary', 'windows', 'paired_intervals', 'paired_draws', 'cohorts']}
    for population in ['all_future_records', 'new_sites']:
        frame = assessments if population == 'all_future_records' else assessments.loc[assessments.new_site]
        bounds = intervals if population == 'all_future_records' else intervals.loc[intervals.new_site]
        for year in [0] + sorted(frame.test_year.unique()):
            part = frame if year == 0 else frame.loc[frame.test_year.eq(year)]
            events = bounds if year == 0 else bounds.loc[bounds.test_year.eq(year)]
            windows, summary = original.summarize(part, PREDICTIONS)
            complement, bins = complementary(part)
            onset = onset_scores(events)
            for key, table in [('summary', summary), ('complementary', complement), ('calibration', bins), ('onset_summary', onset)]:
                result[key].append(table.assign(population=population, test_year=year))
            for source, sample in original.source_groups(part):
                result['cohorts'].append(pd.DataFrame([dict(population=population, test_year=year, source=source, assessments=len(sample), fields=sample.field_id.nunique(), sites=sample.site_id.nunique(), coordinate_years=sample.coordinate_year.nunique(), formerly_calibration=int(sample.partition.eq('calibration').sum()), unseen_leaf_assessments=int(sample.unseen_leaf.sum()))]))
            if year == 0:
                paired, draws = paired_intervals(part, windows, events, replicates=replicates)
                result['windows'].append(windows.assign(population=population))
                result['paired_intervals'].append(paired.assign(population=population))
                result['paired_draws'].append(draws.assign(population=population))
    return {key: pd.concat(tables, ignore_index=True) for key, tables in result.items()}
