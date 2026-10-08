"""Censoring-aware evaluation of seasonal field observations."""

import numpy as np
import pandas as pd


def onset_distance(predicted_date, last_zero_date, first_positive_date):
    """Signed distance from a predicted daily onset to its censoring interval.

    A recorded negative date is excluded from the possible onset interval.
    Missing predicted events remain explicit rather than being dropped from
    the compatibility denominator. Both missing observation bounds are invalid.
    """
    lower = None if pd.isna(last_zero_date) else pd.Timestamp(last_zero_date).normalize()
    upper = None if pd.isna(first_positive_date) else pd.Timestamp(first_positive_date).normalize()
    if lower is None and upper is None:
        raise ValueError('Onset needs an observed negative or positive bound.')
    if lower is not None:
        lower += pd.Timedelta(days=1)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError('Positive onset must occur after the last negative date.')
    if pd.isna(predicted_date):
        return dict(compatible=upper is None, delta_days=0 if upper is None else None,
                    missing_prediction=True)
    predicted = pd.Timestamp(predicted_date).normalize()
    delta = 0
    if lower is not None and predicted < lower:
        delta = (predicted-lower).days
    elif upper is not None and predicted > upper:
        delta = (predicted-upper).days
    return dict(compatible=delta == 0, delta_days=delta, missing_prediction=False)


def chronological_partition(frame, validation_years):
    """Reserve complete harvest years, excluding subsequent years from fitting."""
    years = pd.to_numeric(frame['season_year'], errors='raise')
    reserved = np.asarray(validation_years, float)
    if (reserved.size == 0 or not np.isfinite(reserved).all()
            or np.any(reserved != reserved.astype(int)) or years.isna().any()
            or np.any(years != years.astype(int))):
        raise ValueError('Partition years must be finite integers.')
    result = pd.Series('excluded_future', index=frame.index, name='partition')
    result.loc[years < reserved.min()] = 'calibration'
    result.loc[years.isin(reserved)] = 'validation'
    return result


def occurrence_scores(observed, probability, weights=None, threshold=.5):
    """Weighted binary scores; undefined one-class scores remain missing."""
    actual = np.asarray(observed, float)
    predicted = np.asarray(probability, float)
    weight = np.ones_like(actual) if weights is None else np.asarray(weights, float)
    if (actual.ndim != 1 or actual.size == 0 or predicted.shape != actual.shape
            or weight.shape != actual.shape or not np.isfinite(actual).all()
            or not np.isfinite(predicted).all() or not np.isfinite(weight).all()
            or np.any((actual != 0) & (actual != 1))
            or np.any((predicted < 0) | (predicted > 1))
            or np.any(weight < 0) or weight.sum() <= 0
            or not np.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError('Observed0/1, probability0–1 and valid nonnegative weights required.')
    positive = actual == 1
    detected = predicted >= threshold
    pos_weight, neg_weight = weight[positive].sum(), weight[~positive].sum()
    sensitivity = float(weight[positive & detected].sum()/pos_weight) if pos_weight else None
    specificity = float(weight[~positive & ~detected].sum()/neg_weight) if neg_weight else None
    return dict(n=int(actual.size), positive_n=int(positive.sum()), negative_n=int((~positive).sum()),
                prevalence=float(np.average(actual, weights=weight)), sensitivity=sensitivity,
                specificity=specificity,
                false_warning_rate=None if specificity is None else 1-specificity,
                balanced_accuracy=None if sensitivity is None or specificity is None
                    else (sensitivity+specificity)/2,
                brier_score=float(np.average((predicted-actual)**2, weights=weight)))
