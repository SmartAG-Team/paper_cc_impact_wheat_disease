"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/disease_coupled_ml_20261012')
import numpy as np
import pandas as pd
NAMES = ['prior_assessment_available', 'latest_severity_fraction', 'days_since_latest_scaled', 'prior_assessment_count_scaled', 'first_severity_fraction', 'prior_mean_severity_fraction', 'previous_severity_change_per_30d', 'days_since_first_scaled']
KEYS = ['field_id', 'endpoint_series']

def design_for_queries(context, queries):
    if context.global_target_index.duplicated().any() or context.duplicated(KEYS + ['date']).any() or queries.global_target_index.duplicated().any():
        raise ValueError('Unique assessment and query identities required')
    context = context.reset_index(drop=True).copy()
    context['_date'] = pd.to_datetime(context.date, errors='coerce').dt.normalize()
    dates = pd.to_datetime(queries.date, errors='coerce').dt.normalize()
    y = context.observed_percent.to_numpy(float) / 100
    if context._date.isna().any() or dates.isna().any() or (not np.isfinite(y).all()) or np.any((y < 0) | (y > 1)):
        raise ValueError('Valid dates and observed history percentages required')
    groups = {}
    for identity, part in context.groupby(KEYS, sort=True):
        if any((part[name].nunique() != 1 for name in ['source', 'leaf_index', 'site_id'])):
            raise ValueError('Source-series metadata is inconsistent')
        groups[identity] = part.sort_values('_date')
    values = np.zeros((len(queries), len(NAMES)))
    provenance = []
    for position, item in enumerate(queries.itertuples(index=False)):
        part = groups.get((item.field_id, item.endpoint_series))
        prior_targets = []
        latest_date = None
        if part is not None:
            if part.source.iloc[0] != item.source or part.leaf_index.iloc[0] != item.leaf_index:
                raise ValueError('Query/source-series mismatch')
            past = part.loc[part._date.lt(dates.iloc[position])]
            if len(past):
                severity = past.observed_percent.to_numpy(float) / 100
                elapsed = (dates.iloc[position] - past._date.iloc[-1]).total_seconds() / 86400
                since_first = (dates.iloc[position] - past._date.iloc[0]).total_seconds() / 86400
                rate = 0.0
                if len(past) >= 2:
                    gap = (past._date.iloc[-1] - past._date.iloc[-2]).total_seconds() / 86400
                    rate = 30 * (severity[-1] - severity[-2]) / gap
                values[position] = [1, severity[-1], elapsed / 30, len(past) / 10, severity[0], severity.mean(), rate, since_first / 30]
                prior_targets = past.global_target_index.astype(int).tolist()
                latest_date = str(past.date.iloc[-1])
        provenance.append(dict(global_target_index=int(item.global_target_index), field_id=item.field_id, endpoint_series=item.endpoint_series, date=str(item.date), prior_targets=prior_targets, latest_prior_date=latest_date))
    if not np.isfinite(values).all():
        raise ValueError('Nonfinite history features')
    return (values, pd.DataFrame(provenance))

def design(frame):
    return design_for_queries(frame, frame)
