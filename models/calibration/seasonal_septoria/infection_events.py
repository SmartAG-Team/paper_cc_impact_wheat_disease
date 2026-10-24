"""Observed symptom intervals, onset scoring and hierarchical evaluation weights."""

import numpy as np
import pandas as pd


def symptom_brackets(targets, *, upper_three=True):
    """First-positive brackets using only zero/positive labels.

    A last preceding zero gives an open lower bound; the first positive gives a
    closed upper bound. No preceding zero implies left censoring; no positive
    implies right censoring at the last zero. Later returns to zero are reported
    as persistence violations and do not replace the first-positive bracket.
    """
    required = ["field_id", "field_index", "coordinate_year", "leaf_index",
                "endpoint_series", "day_index", "value"]
    if not set(required).issubset(targets):
        raise ValueError("Complete ordinal-leaf assessment coordinates are required.")
    raw = targets.value.to_numpy(float)
    days = targets.day_index.to_numpy(float)
    leaves = targets.leaf_index.to_numpy(float)
    if (not np.isfinite(raw).all() or np.any(raw < 0)
            or not np.isfinite(days).all() or np.any(days != np.floor(days)) or np.any(days < 0)
            or not np.isfinite(leaves).all() or np.any(leaves != np.floor(leaves))
            or np.any((leaves < 0) | (leaves >= 7))):
        raise ValueError("Finite nonnegative disease signs and integer assessment coordinates required.")
    selected = targets.loc[targets.leaf_index.lt(3)].copy() if upper_three else targets.copy()
    rows = []
    key = ["field_id", "endpoint_series", "leaf_index"]
    for _, group in selected.groupby(key, sort=True):
        group = group.sort_values("day_index")
        if group.day_index.duplicated().any():
            raise ValueError("Duplicate leaf assessment dates cannot identify a unique bracket.")
        first = group.iloc[0]
        positive = group.value.gt(0).to_numpy()
        observed_days = group.day_index.to_numpy(int)
        if positive.any():
            first_index = int(np.flatnonzero(positive)[0])
            upper = int(observed_days[first_index])
            prior_zero = observed_days[:first_index]
            lower = int(prior_zero[-1]) if len(prior_zero) else np.nan
            censoring = "two_sided" if len(prior_zero) else "left"
            persistence_violations = int((~positive[first_index+1:]).sum())
        else:
            lower, upper, censoring = int(observed_days[-1]), np.nan, "right"
            persistence_violations = 0
        row = {name: first[name] for name in ("field_id", "field_index", "coordinate_year",
               "leaf_index", "endpoint_series", "site_id", "season_year", "source") if name in group}
        row.update(lower_day=lower, upper_day=upper, censoring=censoring,
            observed_positive=bool(positive.any()), first_assessment_day=int(observed_days[0]),
            last_assessment_day=int(observed_days[-1]), assessment_count=len(group),
            zero_assessment_count=int((~positive).sum()), positive_assessment_count=int(positive.sum()),
            persistence_violations=persistence_violations)
        rows.append(row)
    return pd.DataFrame(rows)


def onset_distance(brackets, symptom_days, forcing_end_days, *, missed_positive_penalty=30.):
    """Distance in calendar days to the admissible first-detection interval.

    A missed positive receives at least 30 days of loss, or the distance beyond
    its last admissible date to the forcing end plus one, whichever is larger.
    Right-censored no-events are compatible through their final assessment.
    """
    onset, end = np.asarray(symptom_days), np.asarray(forcing_end_days)
    if (onset.shape != (len(brackets),) or end.shape != onset.shape
            or not np.isfinite(onset).all() or not np.isfinite(end).all()
            or np.any(onset != np.floor(onset)) or np.any((onset < -1) | (onset > end))
            or np.any(end < 0) or not np.isfinite(missed_positive_penalty)
            or missed_positive_penalty <= 0):
        raise ValueError("Valid event boundaries, forcing windows and positive missed-event penalty required.")
    lower = brackets.lower_day.to_numpy(float)
    upper = brackets.upper_day.to_numpy(float)
    observed_positive = np.isfinite(upper)
    loss = np.zeros(len(brackets), float)
    present = onset >= 0
    after_lower = present & np.isfinite(lower)
    loss[after_lower] = np.maximum(lower[after_lower]+1-onset[after_lower], 0.)
    before_upper = present & observed_positive
    loss[before_upper] += np.maximum(onset[before_upper]-upper[before_upper], 0.)
    missed = ~present & observed_positive
    loss[missed] = np.maximum(missed_positive_penalty, end[missed]+1-upper[missed])
    return loss


def equal_hierarchy_weights(frame):
    """Equal coordinate-year mass, then equal fields and leaves/assessments."""
    required = ["coordinate_year", "field_id", "leaf_index"]
    if frame.empty or not set(required).issubset(frame):
        raise ValueError("Nonempty coordinate-year, field and leaf membership required.")
    fields = frame.groupby("coordinate_year").field_id.transform("nunique").to_numpy(float)
    leaves = frame.groupby(["coordinate_year", "field_id"]).leaf_index.transform("nunique").to_numpy(float)
    rows = frame.groupby(required).leaf_index.transform("size").to_numpy(float)
    return 1/(frame.coordinate_year.nunique()*fields*leaves*rows)
