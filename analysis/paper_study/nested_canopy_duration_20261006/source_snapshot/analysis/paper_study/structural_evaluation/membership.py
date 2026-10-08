"""Location-history folds and explicit nested membership invariants."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EvaluationFold:
    name: str
    kind: str
    train: np.ndarray
    test: np.ndarray
    excluded: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    test_year: int | None = None


def _indices(values, length):
    raw = np.asarray(values)
    if (raw.ndim != 1 or not np.issubdtype(raw.dtype, np.integer)
            or np.any(raw < 0) or np.any(raw >= length)
            or len(np.unique(raw)) != len(raw)):
        raise ValueError("Unique valid row indices are required.")
    return raw.astype(int)


def _complete_fields(frame, indices):
    selected = frame.iloc[indices]
    entire = np.flatnonzero(frame.field_id.isin(selected.field_id.unique()))
    if not np.array_equal(np.sort(indices), entire):
        raise ValueError("Every selected field-season needs its complete assessment history.")


def _disjoint(frame, train, test):
    if set(train) & set(test):
        raise ValueError("Training and validation row memberships overlap.")
    if set(frame.iloc[train].site_id) & set(frame.iloc[test].site_id):
        raise ValueError("Training and validation location histories overlap.")
    _complete_fields(frame, train)
    _complete_fields(frame, test)


def guard_outer_membership(frame, fold):
    train, test = _indices(fold.train, len(frame)), _indices(fold.test, len(frame))
    if not len(train) or not len(test):
        raise ValueError("Nonempty training and validation partitions are required.")
    _disjoint(frame, train, test)
    if fold.kind == "spatial":
        if set(train) | set(test) != set(range(len(frame))):
            raise ValueError("Spatial folds need complete location-history partitions.")
    elif fold.kind == "forward":
        if (fold.test_year is None
                or not frame.iloc[train].season_year.lt(fold.test_year).all()
                or not frame.iloc[test].season_year.eq(fold.test_year).all()):
            raise ValueError("Chronological training contains future or validation years.")
        expected_train = np.flatnonzero(frame.season_year.lt(fold.test_year))
        prior_sites = set(frame.iloc[expected_train].site_id)
        expected_test = np.flatnonzero(frame.season_year.eq(fold.test_year) & ~frame.site_id.isin(prior_sites))
        if not np.array_equal(np.sort(train), expected_train) or not np.array_equal(np.sort(test), expected_test):
            raise ValueError("Forward folds require all preceding years and only previously absent locations.")
    else:
        raise ValueError("Unknown outer evaluation fold kind.")


def guard_nested_membership(frame, outer, inner):
    guard_outer_membership(frame, outer)
    train, test = _indices(inner.train, len(frame)), _indices(inner.test, len(frame))
    if not len(train) or not len(test):
        raise ValueError("Nonempty inner training and validation partitions are required.")
    if not (set(train) | set(test)) <= set(outer.train):
        raise ValueError("Inner membership contains an outer holdout or excluded observation.")
    _disjoint(frame, train, test)
    if set(train) | set(test) != set(outer.train):
        raise ValueError("Inner folds must partition the complete outer training membership.")


def _site_folds(frame, indices, n_splits, seed, kind, prefix):
    indices = _indices(indices, len(frame))
    if n_splits < 2:
        raise ValueError("At least two location folds are required.")
    sites = np.array(sorted(frame.iloc[indices].site_id.unique()))
    if len(sites) < n_splits:
        raise ValueError("Fewer training locations than requested folds.")
    sites = np.random.default_rng(seed).permutation(sites)
    membership = {site: number % n_splits for number, site in enumerate(sites)}
    assignments = frame.iloc[indices].site_id.map(membership).to_numpy()
    return [EvaluationFold(f"{prefix}{number}", kind,
        indices[assignments != number], indices[assignments == number])
        for number in range(n_splits)]


def spatial_folds(frame, n_splits=5, seed=20261006):
    folds = _site_folds(frame, np.arange(len(frame)), n_splits, seed, "spatial", "spatial_")
    for fold in folds:
        guard_outer_membership(frame, fold)
    return folds


def chronological_folds(frame, years=(2016, 2017, 2018, 2019)):
    years = np.asarray(years)
    if (years.ndim != 1 or not len(years) or not np.isfinite(years).all()
            or np.any(years != years.astype(int)) or len(np.unique(years)) != len(years)):
        raise ValueError("Unique finite integer evaluation years are required.")
    folds = []
    for year in years.astype(int):
        train = np.flatnonzero(frame.season_year.lt(year))
        prior_sites = set(frame.iloc[train].site_id)
        test = np.flatnonzero(frame.season_year.eq(year) & ~frame.site_id.isin(prior_sites))
        excluded = np.setdiff1d(np.arange(len(frame)), np.r_[train, test])
        fold = EvaluationFold(f"forward_{year}", "forward", train, test, excluded, int(year))
        guard_outer_membership(frame, fold)
        folds.append(fold)
    return folds


def inner_folds(frame, outer, n_splits=3, seed=20261006):
    guard_outer_membership(frame, outer)
    folds = _site_folds(frame, outer.train, n_splits, seed, "inner", "inner_")
    for fold in folds:
        guard_nested_membership(frame, outer, fold)
    return folds


def membership_records(frame, outer, inners):
    """Record every target, including observations excluded from forward tests."""
    rows = []
    columns = [name for name in ("field_id", "dataset_id", "site_id", "season_year",
        "endpoint_series", "date", "coordinate_year") if name in frame]
    for fold, level in [(outer, "outer"), *[(item, "inner") for item in inners]]:
        for role, indices in [("training", fold.train), ("validation", fold.test), ("excluded", fold.excluded)]:
            part = frame.iloc[indices][columns].copy()
            part["target_index"] = indices
            part["outer_fold"], part["fold"], part["level"], part["role"] = outer.name, fold.name, level, role
            rows.append(part)
    return pd.concat(rows, ignore_index=True)
