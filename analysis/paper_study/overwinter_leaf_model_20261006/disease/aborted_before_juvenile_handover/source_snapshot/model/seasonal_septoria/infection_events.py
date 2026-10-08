"""Deterministic effective infection events conditional on inoculum presence.

Duration dose and its dry-gap reset are field-scale hypotheses, not laboratory
infection thresholds. Positive disease assessments identify symptom detection,
not infection dates. First symptom onset is persistent for bracket scoring.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit


@dataclass(frozen=True)
class EventParameters:
    dose_threshold: float = .5
    dry_gap_days: int = 1
    delay_reference_days: float = 20.


@dataclass
class EventTrajectory:
    infection_day: np.ndarray
    symptom_day: np.ndarray
    infected: np.ndarray
    symptomatic: np.ndarray


@njit(cache=True)
def _event_dates(t, exposure, active, threshold, gap, delay, phenology_only):
    fields, days, leaves = active.shape
    infection = np.full((fields, leaves), -1, np.int64)
    symptom = np.full_like(infection, -1)
    for field in range(fields):
        for leaf in range(leaves):
            dose, thermal, dry = 0., 0., 0
            for day in range(days):
                if infection[field, leaf] >= 0:
                    thermal += max(t[field, day], 0.)/18.
                    if symptom[field, leaf] < 0 and thermal+1e-10 >= delay:
                        symptom[field, leaf] = day+1
                elif active[field, day, leaf]:
                    if exposure[field, day] > 0:
                        dose += exposure[field, day]
                        dry = 0
                    else:
                        dry += 1
                        if dry >= gap:
                            dose = 0.
                    if phenology_only or dose+1e-12 >= threshold:
                        infection[field, leaf] = day+1
                else:
                    # Exposure before the ordinal leaf exists cannot establish
                    # infection on that leaf or contribute to its future dose.
                    dose, dry = 0., 0
    return infection, symptom


def simulate_events(temperature, exposure, host_active, parameters=EventParameters(),
                    *, phenology_only=False):
    """Return first events and persistent indicators on daily end boundaries.

    Day zero precedes the first supplied forcing day. Infection at boundary d
    begins thermal waiting on the next forcing day. A missing event is -1.
    The phenology comparator shares the same host and thermal delay.
    """
    t, e = np.asarray(temperature, float), np.asarray(exposure, float)
    active = np.asarray(host_active)
    p = parameters
    if (t.ndim != 2 or not t.size or e.shape != t.shape
            or active.ndim != 3 or active.shape[:2] != t.shape or not active.shape[2]
            or not np.isfinite(t).all() or not np.isfinite(e).all()
            or np.any((e < 0) | (e > 1)) or np.any((active != 0) & (active != 1))
            or not np.isfinite([p.dose_threshold, p.dry_gap_days, p.delay_reference_days]).all()
            or p.dose_threshold <= 0 or p.dry_gap_days < 1
            or p.dry_gap_days != int(p.dry_gap_days) or p.delay_reference_days <= 0):
        raise ValueError("Valid daily forcing, binary host and positive event parameters required.")
    infection, symptom = _event_dates(t, e, active.astype(bool), p.dose_threshold,
        int(p.dry_gap_days), p.delay_reference_days, bool(phenology_only))
    day = np.arange(t.shape[1]+1)[None, :, None]
    return EventTrajectory(infection, symptom,
        (infection[:, None, :] >= 0) & (day >= infection[:, None, :]),
        (symptom[:, None, :] >= 0) & (day >= symptom[:, None, :]))


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
