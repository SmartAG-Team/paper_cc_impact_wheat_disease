"""Strictly earlier same-field history; source leaf categories remain explicit."""
import numpy as np
import pandas as pd

SAME_NAMES = ['prior_assessment_available', 'latest_severity_fraction',
    'days_since_latest_scaled', 'prior_assessment_count_scaled', 'first_severity_fraction',
    'prior_mean_severity_fraction', 'previous_severity_change_per_30d', 'days_since_first_scaled']
OTHER_NAMES = ['other_leaf_available', 'other_leaf_latest_mean', 'other_leaf_latest_max',
    'other_leaf_mean', 'other_leaf_age_scaled', 'other_leaf_count_scaled']
OTHER_NAMES += [f'other_code_{code}_{kind}' for code in range(7) for kind in ['available', 'severity', 'age']]


def build_history(observations, queries, config=None):
    required = ['global_target_index', 'field_id', 'site_id', 'source', 'leaf_index', 'endpoint_series', 'date']
    for frame in [observations, queries]:
        if frame[required].isna().any().any() or frame.global_target_index.duplicated().any():
            raise ValueError('Complete unique history and query identities required')
    if not observations.observed_percent.between(0, 100).all():
        raise ValueError('Finite observed severity required')
    context = observations.copy()
    context['_date'] = pd.to_datetime(context.date, errors='coerce', format='mixed').dt.normalize()
    query_dates = pd.to_datetime(queries.date, errors='coerce', format='mixed').dt.normalize()
    if context._date.isna().any() or query_dates.isna().any():
        raise ValueError('Valid history and query dates required')
    if context.duplicated(['field_id', 'endpoint_series', '_date']).any():
        raise ValueError('Duplicate normalized history date')
    groups = dict(tuple(context.groupby('field_id', sort=False)))
    values, provenance = [], []
    for position, row in enumerate(queries.itertuples(index=False)):
        now = query_dates.iloc[position]
        field = groups.get(row.field_id, context.iloc[:0])
        for key in ['site_id', 'source']:
            if len(field) and (field[key].nunique() != 1 or field[key].iloc[0] != getattr(row, key)):
                raise ValueError('Conflicting field metadata')
        past = field.loc[field._date.lt(now)].sort_values('_date')
        same = past.loc[past.endpoint_series.eq(row.endpoint_series)]
        if len(same) and not same.leaf_index.eq(row.leaf_index).all():
            raise ValueError('Conflicting leaf metadata')
        other = past.loc[past.leaf_index.ne(row.leaf_index)]
        v = np.zeros(len(SAME_NAMES) + len(OTHER_NAMES))
        latest_date = None
        if len(same):
            y = same.observed_percent.to_numpy(float)/100
            gap = (now-same._date.iloc[-1]).days
            rate = 30*(y[-1]-y[-2])/(same._date.iloc[-1]-same._date.iloc[-2]).days if len(y) >= 2 else 0
            v[:8] = [1, y[-1], gap/30, len(y)/10, y[0], y.mean(), rate,
                     (now-same._date.iloc[0]).days/30]
            latest_date = same.date.iloc[-1]
        if len(other):
            last = other.loc[other._date.eq(other._date.max())]
            v[8:14] = [1, last.observed_percent.mean()/100, last.observed_percent.max()/100,
                       other.observed_percent.mean()/100, (now-other._date.max()).days/30, len(other)/10]
            for code in range(7):
                part = other.loc[other.leaf_index.eq(code)]
                if len(part):
                    recent = part.iloc[-1]
                    v[14+3*code:17+3*code] = [1, recent.observed_percent/100, (now-recent['_date']).days/30]
        values.append(v)
        provenance.append(dict(global_target_index=int(row.global_target_index),
            prior_target_ids=past.global_target_index.astype(int).tolist(),
            latest_same_date=latest_date, latest_any_date=str(past.date.iloc[-1]) if len(past) else None,
            anatomical_direction_used=False))
    result = pd.DataFrame(values, columns=SAME_NAMES + OTHER_NAMES)
    if not np.isfinite(result.to_numpy()).all():
        raise ValueError('Finite history features required')
    return result, pd.DataFrame(provenance)
