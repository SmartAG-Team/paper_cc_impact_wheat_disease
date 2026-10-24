"""Deterministic effective infection events conditional on inoculum presence.

Duration dose and its dry-gap reset are field-scale hypotheses, not laboratory
infection thresholds. Positive disease assessments identify symptom detection,
not infection dates. First symptom onset is persistent for bracket scoring.
"""

from dataclasses import dataclass

import numpy as np
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


