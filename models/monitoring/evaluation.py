"""Complete-support scoring, chronology/spatial isolation and adoption gates."""
import hashlib

import numpy as np
import pandas as pd

from analysis.final_study.metrics import disease_scores, hierarchical_weights, observed_windows


def score_predictions(frame, prediction_column):
    if not len(frame) or not np.isfinite(frame[prediction_column]).all():
        raise ValueError('Complete finite predictions required')
    result = disease_scores(frame, prediction_column)
    support = frame.copy()
    if 'partition' not in support:
        support['partition'] = 'development'
    windows = observed_windows(support, [prediction_column])
    result.update(n=len(frame), fields=frame.field_id.nunique(), sites=frame.site_id.nunique(),
                  windows=len(windows), integral_rmse=None)
    if len(windows):
        w = hierarchical_weights(windows)
        e = windows[prediction_column+'_audpc']-windows.observed_audpc
        result['integral_rmse'] = float(np.sqrt(np.average(e**2, weights=w)))
    return result


def distances_km(first, second):
    a, b = np.deg2rad(np.asarray(first, float)), np.deg2rad(np.asarray(second, float))
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != 2 or b.shape[1] != 2:
        raise ValueError('Latitude/longitude pairs required')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite spatial coordinates required')
    lat = a[:, None, 0]-b[None, :, 0]; lon = a[:, None, 1]-b[None, :, 1]
    h = np.sin(lat/2)**2 + np.cos(a[:, None, 0])*np.cos(b[None, :, 0])*np.sin(lon/2)**2
    return 6371*2*np.arcsin(np.sqrt(np.clip(h, 0, 1)))


def temporal_spatial_splits(frame, years, folds=3, buffer_km=0):
    if folds < 2 or buffer_km < 0:
        raise ValueError('At least two folds and nonnegative buffer required')
    if frame.groupby('site_id')[['latitude', 'longitude']].nunique().gt(1).any().any():
        raise ValueError('A site has inconsistent coordinates')
    sites = sorted(frame.site_id.unique())
    # Stable balanced identity-only assignment; outcomes cannot affect membership.
    ordered = sorted(sites, key=lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    assignment = {s: i % folds for i, s in enumerate(ordered)}
    for year in years:
        for fold in range(folds):
            test = np.flatnonzero(frame.season_year.eq(year) & frame.site_id.map(assignment).eq(fold))
            if not len(test):
                continue
            train = np.flatnonzero(frame.season_year.lt(year) & frame.site_id.map(assignment).ne(fold))
            if buffer_km and len(train):
                distance = distances_km(frame.iloc[train][['latitude', 'longitude']],
                                        frame.iloc[test][['latitude', 'longitude']])
                train = train[distance.min(axis=1) >= buffer_km]
            yield f'{year}:sitefold{fold}:buffer{buffer_km:g}km', train, test


def score_subset(frame, mask, prediction_column):
    """Outcome-defined diagnostic strata keep the full-cohort record weights.

    Integrals are undefined here: removing intermediate assessments would
    create different windows and bridge omitted observations.
    """
    mask = np.asarray(mask, bool)
    part = frame.loc[mask]
    w = hierarchical_weights(frame)[mask]
    if not len(part) or not np.isfinite(part[[prediction_column, 'observed_percent']]).all().all():
        raise ValueError('Complete finite diagnostic subset required')
    y = part.observed_percent.to_numpy(); e = part[prediction_column].to_numpy()-y
    mean = np.average(y, weights=w); variance = np.average((y-mean)**2, weights=w)
    mse = np.average(e**2, weights=w)
    return dict(rmse=float(np.sqrt(mse)), mae=float(np.average(abs(e), weights=w)),
        bias=float(np.average(e, weights=w)), r2=float(1-mse/variance) if variance > 0 else None,
        observed_mean=float(mean), prediction_mean=float(np.average(part[prediction_column], weights=w)),
        n=len(part), fields=part.field_id.nunique(), sites=part.site_id.nunique(),
        integral_rmse=None, windows=0, full_cohort_weight_fraction=float(w.sum()/hierarchical_weights(frame).sum()))


def paired_interval(frame, candidate, reference, repeats=2000, seed=20261021, weights=None):
    if not np.isfinite(frame[[candidate, reference, 'observed_percent']]).all().all():
        raise ValueError('Complete paired predictions required')
    w = hierarchical_weights(frame) if weights is None else np.asarray(weights, float)
    if w.shape != (len(frame),) or not np.isfinite(w).all() or np.any(w <= 0):
        raise ValueError('Aligned positive bootstrap weights required')
    ids, sites = pd.factorize(frame.site_id, sort=True)
    first = np.bincount(ids, weights=w*(frame[candidate]-frame.observed_percent)**2)
    second = np.bincount(ids, weights=w*(frame[reference]-frame.observed_percent)**2)
    mass = np.bincount(ids, weights=w)
    rng = np.random.default_rng(seed)
    draws = rng.multinomial(len(sites), np.full(len(sites), 1/len(sites)), size=repeats)
    differences = np.sqrt(draws@first/(draws@mass))-np.sqrt(draws@second/(draws@mass))
    return np.quantile(differences, [.025, .975]).tolist()


def promotion_decision(reference, candidate, interval, *, independent,
                       subgroup_deltas, severe_bias_delta, new_site_ratio=None):
    required = {'cold', 'warm', 'BASF', 'Corteva'}
    groups_available = isinstance(subgroup_deltas, dict) and required <= set(subgroup_deltas)
    group_values = [subgroup_deltas[name] for name in sorted(required)] if groups_available else []
    finite = lambda v: v is not None and bool(np.isfinite(v))
    integral_available = finite(candidate.get('integral_rmse')) and finite(reference.get('integral_rmse'))
    gates = dict(independent_validation=bool(independent),
        ten_percent_rmse_reduction=bool(finite(candidate.get('rmse')) and finite(reference.get('rmse'))
            and reference['rmse'] > 0 and 0 <= candidate['rmse'] <= .9*reference['rmse']),
        paired_interval_below_zero=bool(len(interval) == 2 and all(finite(v) for v in interval)
            and interval[0] <= interval[1] < 0),
        subgroup_noninferiority=bool(groups_available and all(finite(v) for v in group_values)
                                    and max(group_values) <= 1),
        severe_bias_improves=bool(finite(severe_bias_delta) and severe_bias_delta < 0),
        integral_noninferiority=bool(integral_available
            and 0 <= candidate['integral_rmse'] <= reference['integral_rmse']),
        new_site_ten_percent_reduction=bool(finite(new_site_ratio) and 0 <= new_site_ratio <= .9))
    return dict(promote=all(gates.values()), gates=gates)
