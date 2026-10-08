"""Event constraints retain the informative side of incomplete stage records."""

import importlib

import numpy as np
import pandas as pd
import pytest


def study():
    try:
        return importlib.import_module(
            "analysis.paper_study.phenology_assumptions_20261006.partial_endpoint_inclusive.run"
        )
    except ModuleNotFoundError:
        pytest.fail("The partial-endpoint-inclusive phenology sensitivity is not implemented")


def rows(intervals):
    return pd.DataFrame([dict(dataset_id="basf-wheat-diseases", physical_unit="a",
        season_year=2017, site_id="x", date=date, stage_from=lower, stage_to=upper,
        country="DE", latitude=50., longitude=8., value=99.)
        for date, lower, upper in intervals])


def test_lone_endpoint_contributes_only_its_supported_event_bound():
    module = study()
    raw = rows([("2017-05-01", np.nan, 30.), ("2017-05-02", 31., np.nan),
        ("2017-05-03", np.nan, 51.), ("2017-05-04", 49., np.nan),
        ("2017-05-05", 65., np.nan)])
    normalized, rejected = module.normalize_stage_dates(module.base.project_stage_rows(raw, "BASF"))
    assert len(normalized) == 5 and rejected.empty
    constraints = module.base.make_constraints(normalized)
    stem = constraints.loc[constraints.event.eq(31)].iloc[0]
    assert stem.lower_exclusive == pd.Timestamp("2017-05-01")
    assert stem.upper_inclusive == pd.Timestamp("2017-05-02")
    heading = constraints.loc[constraints.event.eq(51)].iloc[0]
    assert heading.lower_exclusive == pd.Timestamp("2017-05-01")
    assert heading.upper_inclusive == pd.Timestamp("2017-05-05")
    terminal = constraints.loc[constraints.event.eq(85)].iloc[0]
    assert terminal.lower_exclusive == pd.Timestamp("2017-05-03")
    assert pd.isna(terminal.upper_inclusive)


def test_partial_from_below_event_and_partial_to_at_event_are_uninformative():
    module = study()
    raw = rows([("2017-05-01", 49., np.nan), ("2017-05-02", np.nan, 51.)])
    normalized, _ = module.normalize_stage_dates(module.base.project_stage_rows(raw, "BASF"))
    heading = module.base.make_constraints(normalized, events=[51]).iloc[0]
    assert not heading.eligible_constraint
    assert heading.reason == "no_event_bound"


def test_date_intersection_uses_partial_information_and_rejects_conflicts():
    module = study()
    raw = rows([("2017-05-01", 31., np.nan), ("2017-05-01", 30., 39.),
        ("2017-05-02", 55., np.nan), ("2017-05-02", np.nan, 49.)])
    normalized, rejected = module.normalize_stage_dates(module.base.project_stage_rows(raw, "BASF"))
    assert len(normalized) == 1
    assert normalized.iloc[0].stage_from == 31.
    assert normalized.iloc[0].stage_to == 39.
    assert normalized.iloc[0].n_stage_from_only_rows == 1
    assert len(rejected) == 2
    assert rejected.reason.eq("conflicting_date_stage_ranges").all()


def test_missing_both_and_reversed_intervals_are_rejected():
    module = study()
    raw = rows([("2017-05-01", np.nan, np.nan), ("2017-05-02", 39., 31.),
        ("2017-05-03", 39., np.nan)])
    normalized, rejected = module.normalize_stage_dates(module.base.project_stage_rows(raw, "BASF"))
    assert len(normalized) == 1
    assert set(rejected.reason) == {"missing_both_stage_endpoints", "reversed_stage_interval"}


def test_partial_endpoint_q_fit_is_disease_independent_and_rejects_2019():
    module = study()
    raw = rows([("2017-05-01", np.nan, 30.), ("2017-05-08", 49., 49.),
                ("2017-05-15", 65., np.nan)])
    metadata = pd.DataFrame([dict(field_index=0, field_id="basf-wheat-diseases|a|2017",
        sowing_date="2017-05-01", forcing_days=15, last_forcing_date="2017-05-15")])
    results = []
    for version in [raw, pd.concat([raw.assign(value=-999.), raw.assign(season_year=2019,
        physical_unit="heldout", value=1e20, stage_from=90., stage_to=np.nan)])]:
        selected = module.base.calibration_stage_rows(module.base.project_stage_rows(version, "BASF"))
        normalized, _ = module.normalize_stage_dates(selected)
        results.append(module.base.fit_grid(module.base.make_constraints(normalized),
            np.arange(15, dtype=float)[None, :] * 10., metadata, {31: 1., 51: 100., 85: 200.}))
    assert results[0][0] == results[1][0]
    pd.testing.assert_frame_equal(results[0][1], results[1][1])
