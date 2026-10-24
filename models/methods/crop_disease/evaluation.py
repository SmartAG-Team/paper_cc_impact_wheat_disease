"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_disease_coupling_20261010')
import numpy as np
import pandas as pd
from analysis.methods.observation_model import observation as metrics
PRIMARY = 'raw_lstm_extended_q50'

def source_groups(frame):
    yield ('pooled', frame)
    yield from frame.groupby('source', sort=True)

def endpoint_frames(assessments, windows, prediction_columns):
    yield ('assessment', assessments)
    yield ('window_mean', windows)
    integral = windows.copy()
    integral['observed_percent'] = integral.observed_audpc
    for col in prediction_columns:
        integral[col] = integral[col + '_audpc']
    yield ('window_integral', integral)

def summarize(frame, prediction_columns):
    windows = metrics.windows(frame, prediction_columns)
    rows = []
    for endpoint, data in endpoint_frames(frame, windows, prediction_columns):
        for source, part in source_groups(data):
            for model in prediction_columns:
                rows.append(dict(endpoint=endpoint, source=source, model=model, n=len(part), fields=part.field_id.nunique(), sites=part.site_id.nunique(), **metrics.scores(part, model)))
    return (windows, pd.DataFrame(rows))

def compare(frame, candidate, benchmark, *, replicates=2000, seed=20261010):
    return metrics.bootstrap_pair(frame, candidate, benchmark, replicates=replicates, seed=seed)

def all_comparisons(assessments, windows, prediction_columns):
    rows, samples = ([], [])
    for endpoint, frame in endpoint_frames(assessments, windows, prediction_columns):
        for source, part in source_groups(frame):
            for mapping in ['raw', 'mapped']:
                candidate = mapping + '_lstm_extended_q50'
                for benchmark in [mapping + '_retained', 'leaf_mean']:
                    point, draws = compare(part, candidate, benchmark)
                    identity = dict(endpoint=endpoint, source=source, candidate=candidate, benchmark=benchmark)
                    intervals = {}
                    for metric in ['rmse', 'mae', 'bias']:
                        lo, hi = np.quantile(draws[metric + '_difference'], [0.025, 0.975])
                        intervals[metric + '_difference_lower'] = float(lo)
                        intervals[metric + '_difference_upper'] = float(hi)
                    rows.append(dict(**identity, n=len(part), sites=part.site_id.nunique(), **point, **intervals))
                    samples.append(pd.DataFrame(draws).assign(**identity, replicate=np.arange(2000)))
    return (pd.DataFrame(rows), pd.concat(samples, ignore_index=True))

def decide(summary, comparisons):
    criteria = []

    def scalar(frame, key, **selectors):
        selected = frame
        for col, value in selectors.items():
            selected = selected.loc[selected[col].eq(value)]
        if len(selected) != 1 or not np.isfinite(selected[key].iloc[0]):
            raise ValueError('Missing, duplicate or nonfinite advancement evidence')
        return float(selected[key].iloc[0])
    for source in ['pooled', 'BASF', 'Corteva']:
        for benchmark in ['raw_retained', 'leaf_mean']:
            upper = scalar(comparisons, 'rmse_difference_upper', source=source, endpoint='window_integral', candidate=PRIMARY, benchmark=benchmark)
            criteria.append(dict(source=source, endpoint='window_integral', benchmark=benchmark, criterion='Paired raw integral RMSE upper difference below zero', value=upper, passed=upper < 0))
            for endpoint in ['assessment', 'window_mean']:
                difference = scalar(summary, 'rmse', source=source, endpoint=endpoint, model=PRIMARY) - scalar(summary, 'rmse', source=source, endpoint=endpoint, model=benchmark)
                criteria.append(dict(source=source, endpoint=endpoint, benchmark=benchmark, criterion='Raw point RMSE does not increase', value=difference, passed=difference <= 0))
    return dict(primary_candidate=PRIMARY, advance_host_evaluation=all((c['passed'] for c in criteria)), climate_adoption=False, independent_validation=False, biological_parameters_refitted=False, observation_mapping_refitted=False, evaluation_outcomes_previously_known=True, criteria=criteria)
