"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_residual_ml_20261011')
import numpy as np
import pandas as pd
from analysis.methods.crop_disease.evaluation import summarize, source_groups, endpoint_frames
from analysis.methods.observation_model import observation as metrics
from analysis.methods.joint_disease.evaluation import discrimination
from analysis.methods.severity_probability.data import weights
from analysis.methods.residual_disease import config
MODELS = ['selected_pipeline', 'joint_rf_point', *config.FAMILY_ORDER]
TEMPORAL_MODELS = MODELS + ['joint', 'boosting', 'full_window', 'full_point', 'calendar_window']
REFERENCES = ['joint', 'fold_leaf_mean', 'boosting', 'full_window', 'joint_base', 'trajectory_base']

def paired(frame, candidate, reference, *, replicates=2000):
    point, draws = metrics.bootstrap_pair(frame, candidate, reference, replicates=replicates, seed=config.SEED)
    rows = []
    for metric in ['rmse', 'mae', 'bias']:
        low, high = np.quantile(draws[metric + '_difference'], [0.025, 0.975])
        rows.append(dict(metric=metric + '_difference', value=point[metric + '_difference'], lower=float(low), upper=float(high), replicates=replicates))
    return pd.DataFrame(rows)

def aucs(frame, models):
    rows = []
    for source, part in source_groups(frame):
        for model in models:
            for threshold in [10, 25, 50]:
                rows.append(dict(source=source, model=model, threshold=threshold, n=len(part), sites=part.site_id.nunique(), **discrimination(part.observed_percent.ge(threshold), part[model], weights(part))))
    return pd.DataFrame(rows)

def decide(summary, intervals):
    criteria = []
    for population in ['all_future_records', 'new_sites']:
        for endpoint in ['assessment', 'window_integral']:
            for reference in ['joint', 'fold_leaf_mean']:
                part = intervals.loc[intervals.population.eq(population) & intervals.endpoint.eq(endpoint) & intervals.source.eq('pooled') & intervals.reference.eq(reference) & intervals.candidate.eq('selected_pipeline') & intervals.metric.eq('rmse_difference')]
                if len(part) != 1:
                    raise ValueError('Incomplete primary interval')
                value = float(part.upper.iloc[0])
                criteria.append(dict(population=population, endpoint=endpoint, source='pooled', reference=reference, rule='paired_upper_below_zero', value=value, passed=value < 0))
            for source in ['BASF', 'Corteva']:
                part = summary.loc[summary.population.eq(population) & summary.test_year.eq(0) & summary.endpoint.eq(endpoint) & summary.source.eq(source)].set_index('model')
                value = float(part.loc['selected_pipeline', 'rmse'] - part.loc['fold_leaf_mean', 'rmse'])
                criteria.append(dict(population=population, endpoint=endpoint, source=source, reference='fold_leaf_mean', rule='point_error_does_not_increase', value=value, passed=value <= 0))
    return dict(primary_candidate='selected_pipeline', magnitude_support=all((row['passed'] for row in criteria)), independent_validation=False, prospective_validation=False, climate_adoption=False, functional_canopy_validated=False, yield_validated=False, onset_changed_by_ml=False, criteria=criteria)

def evaluate(temporal, source, *, replicates=2000):
    tables = {key: [] for key in ['summary', 'windows', 'discrimination', 'paired_intervals']}
    for population, frame, models, refs in [('all_future_records', temporal, TEMPORAL_MODELS, REFERENCES), ('new_sites', temporal.loc[temporal.new_site], TEMPORAL_MODELS, REFERENCES), ('source_transfer', source, MODELS, ['fold_leaf_mean', 'joint_base', 'trajectory_base'])]:
        for year in [0] + sorted(frame.test_year.unique()) if population != 'source_transfer' else [0]:
            part = frame if year == 0 else frame.loc[frame.test_year.eq(year)]
            windows, summary = summarize(part, models)
            tables['summary'].append(summary.assign(population=population, test_year=year))
            tables['discrimination'].append(aucs(part, models).assign(population=population, test_year=year))
            if year != 0:
                continue
            tables['windows'].append(windows.assign(population=population))
            for endpoint, data in endpoint_frames(part, windows, models):
                for label, sample in source_groups(data):
                    for candidate in ['selected_pipeline', 'joint_rf', 'joint_rf_geo', 'trajectory_rf', 'joint_hgb', 'direct_rf', 'joint_rf_point', 'leaf_bias']:
                        for reference in refs:
                            tables['paired_intervals'].append(paired(sample, candidate, reference, replicates=replicates).assign(population=population, endpoint=endpoint, source=label, candidate=candidate, reference=reference, n=len(sample), sites=sample.site_id.nunique()))
    result = {name: pd.concat(parts, ignore_index=True) for name, parts in tables.items()}
    result['decision'] = decide(result['summary'], result['paired_intervals'])
    return result
