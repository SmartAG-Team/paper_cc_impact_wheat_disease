"""Stage-only phenology calibration guards and censoring semantics."""

import importlib

import numpy as np
import pandas as pd
import pytest


def study():
    try:
        return importlib.import_module(
            "analysis.paper_study.phenology_assumptions_20261006.run"
        )
    except ModuleNotFoundError:
        pytest.fail("The bounded stage-only phenology study is not implemented")


def raw_rows():
    return pd.DataFrame([
        dict(dataset_id="basf-wheat-diseases", physical_unit="a", season_year=2017,
             site_id="x", date="2017-05-01", stage_from=30., stage_to=32.,
             country="DE", latitude=50., longitude=8., value=7.),
        dict(dataset_id="basf-wheat-diseases", physical_unit="a", season_year=2017,
             site_id="x", date="2017-05-01", stage_from=31., stage_to=33.,
             country="DE", latitude=50., longitude=8., value=99.),
        dict(dataset_id="basf-wheat-diseases", physical_unit="a", season_year=2017,
             site_id="x", date="2017-05-08", stage_from=49., stage_to=49.,
             country="DE", latitude=50., longitude=8., value=12.),
        dict(dataset_id="basf-wheat-diseases", physical_unit="a", season_year=2017,
             site_id="x", date="2017-05-15", stage_from=55., stage_to=55.,
             country="DE", latitude=50., longitude=8., value=31.),
    ])


def test_repeated_leaf_stages_are_intersected_once_per_field_date():
    module = study()
    stages = module.project_stage_rows(raw_rows(), "BASF")
    assert "value" not in stages
    normalized, rejected = module.normalize_stage_dates(stages)
    assert len(normalized) == 3
    assert rejected.empty
    assert normalized.iloc[0].stage_from == 31.
    assert normalized.iloc[0].stage_to == 32.
    assert normalized.iloc[0].source_row_count == 2


def test_reversed_and_conflicting_ranges_are_rejected():
    module = study()
    raw = raw_rows()
    raw.loc[0, ["stage_from", "stage_to"]] = [39., 31.]
    raw.loc[1, ["stage_from", "stage_to"]] = [20., 20.]
    raw = pd.concat([raw, raw.iloc[[1]].assign(stage_from=30., stage_to=30.)],
                    ignore_index=True)
    valid, rejected = module.normalize_stage_dates(module.project_stage_rows(raw, "BASF"))
    assert valid.date.dt.strftime("%Y-%m-%d").tolist() == ["2017-05-08", "2017-05-15"]
    assert set(rejected.reason) == {"reversed_stage_interval", "conflicting_date_stage_ranges"}


def test_event_bounds_are_exclusive_lower_and_inclusive_upper():
    module = study()
    valid, _ = module.normalize_stage_dates(module.project_stage_rows(raw_rows(), "BASF"))
    constraints = module.make_constraints(valid, events=[31, 51, 85])
    heading = constraints.loc[constraints.event.eq(51)].iloc[0]
    assert heading.censoring == "interval"
    assert heading.lower_exclusive == pd.Timestamp("2017-05-08")
    assert heading.upper_inclusive == pd.Timestamp("2017-05-15")
    assert module.censoring_distance(pd.Timestamp("2017-05-08"),
        heading.lower_exclusive, heading.upper_inclusive,
        pd.Timestamp("2017-05-20"))["signed_distance_days"] == -1.
    assert module.censoring_distance(pd.Timestamp("2017-05-15"),
        heading.lower_exclusive, heading.upper_inclusive,
        pd.Timestamp("2017-05-20"))["compatible"] is True
    terminal = constraints.loc[constraints.event.eq(85)].iloc[0]
    assert terminal.censoring == "right"
    assert terminal.lower_exclusive == pd.Timestamp("2017-05-15")
    assert pd.isna(terminal.upper_inclusive)


def test_stage_regression_does_not_create_a_spurious_bracket():
    module = study()
    raw = raw_rows()
    raw.loc[2, ["stage_from", "stage_to"]] = [55., 55.]
    raw.loc[3, ["stage_from", "stage_to"]] = [49., 49.]
    valid, _ = module.normalize_stage_dates(module.project_stage_rows(raw, "BASF"))
    heading = module.make_constraints(valid, events=[51]).iloc[0]
    assert not heading.eligible_constraint
    assert heading.reason == "temporally_inconsistent_event_bounds"


def test_missing_predicted_events_are_explicitly_censored_and_scored():
    module = study()
    end = pd.Timestamp("2017-05-20")
    missing = module.censoring_distance(None, pd.Timestamp("2017-05-08"),
        pd.Timestamp("2017-05-15"), end)
    assert missing["prediction_status"] == "not_reached_by_forcing_end"
    assert missing["signed_distance_days"] == 6.
    assert not missing["compatible"]
    permitted = module.censoring_distance(None, pd.Timestamp("2017-05-15"), None, end)
    assert permitted["compatible"]
    assert permitted["distance_days"] == 0.
    assert permitted["distance_is_lower_bound"]


def test_first_event_uses_scaled_accumulation_and_ignores_padding():
    module = study()
    metadata = pd.DataFrame([dict(field_index=0, field_id="a", sowing_date="2017-01-01",
        forcing_days=3, last_forcing_date="2017-01-03")])
    accumulated = np.array([[0., 10., 20., 9999.]])
    predictions = module.first_event_dates(accumulated, metadata, {31: 30., 51: 60.}, 1.5)
    assert predictions.loc[predictions.event.eq(31), "predicted_date"].iloc[0] == pd.Timestamp("2017-01-03")
    assert pd.isna(predictions.loc[predictions.event.eq(51), "predicted_date"].iloc[0])


def test_equal_field_loss_does_not_weight_a_field_by_its_number_of_events():
    module = study()
    scores = pd.DataFrame(dict(field_id=["a", "a", "a", "b"], distance_days=[0., 0., 0., 10.]))
    assert module.equal_field_event_loss(scores) == 5.


def test_ties_choose_q_nearest_one_then_smaller_q():
    module = study()
    curve = pd.DataFrame(dict(q=[.6, .9, 1.1, 1.4], loss_days=[4., 2., 2., 3.]))
    assert module.choose_q(curve) == .9


def test_stage_calibration_excludes_2019_and_disease_values():
    module = study()
    rows = raw_rows()
    changed = pd.concat([rows.assign(value=-999.), rows.assign(season_year=2019,
        physical_unit="heldout", stage_from=90., stage_to=90., value=1e20)])
    first = module.calibration_stage_rows(module.project_stage_rows(rows, "BASF"))
    second = module.calibration_stage_rows(module.project_stage_rows(changed, "BASF"))
    pd.testing.assert_frame_equal(first.reset_index(drop=True), second.reset_index(drop=True))
    placeholders = module.input_placeholders(module.project_stage_rows(changed, "BASF"))
    assert placeholders.value.eq(0.).all()
    assert placeholders.organ.eq("F1").all()


def test_external_membership_is_exact_registry_membership():
    module = study()
    rows = raw_rows().assign(dataset_id="corteva-2014-2018-external")
    rows = pd.concat([rows, rows.assign(physical_unit="overlap")], ignore_index=True)
    stage = module.project_stage_rows(rows, "Corteva", strict_units={"a"})
    assert stage.physical_unit.unique().tolist() == ["a"]
    assert stage.partition.unique().tolist() == ["reused_external_strict"]


def test_original_field_eligibility_and_forcing_horizon_are_preserved():
    module = study()
    rows = module.project_stage_rows(raw_rows(), "BASF")
    membership = pd.DataFrame([dict(field_id="basf-wheat-diseases|a|2017",
        first_forcing_date="2016-10-01", last_forcing_date="2017-05-08")])
    selected, excluded = module.restrict_membership(rows, membership)
    assert len(selected) == 3
    assert len(excluded) == 1
    assert excluded.reason.tolist() == ["outside_original_forcing_horizon"]
    wrong, excluded = module.restrict_membership(rows, membership.assign(field_id="other"))
    assert wrong.empty
    assert excluded.reason.eq("outside_original_model_field_membership").all()


def test_full_q_grid_is_invariant_to_disease_scores_and_added_2019_stages():
    module = study()
    raw = raw_rows()
    changed = pd.concat([raw.assign(value=1e20), raw.assign(season_year=2019,
        physical_unit="heldout", stage_from=1., stage_to=1., value=-999.)])
    metadata = pd.DataFrame([dict(field_index=0, field_id="basf-wheat-diseases|a|2017",
        sowing_date="2017-05-01", forcing_days=15, last_forcing_date="2017-05-15")])
    accumulated = np.arange(15, dtype=float)[None, :] * 10.
    results = []
    for rows in [raw, changed]:
        selected = module.calibration_stage_rows(module.project_stage_rows(rows, "BASF"))
        stages, _ = module.normalize_stage_dates(selected)
        constraints = module.make_constraints(stages)
        results.append(module.fit_grid(constraints, accumulated, metadata,
            {31: 1., 51: 100., 85: 200.}))
    assert results[0][0] == results[1][0]
    assert len(results[0][1]) == 401
    pd.testing.assert_frame_equal(results[0][1], results[1][1])


def test_fitting_boundary_rejects_a_heldout_stage_constraint():
    module = study()
    stages, _ = module.normalize_stage_dates(module.project_stage_rows(raw_rows(), "BASF"))
    constraints = module.make_constraints(stages).assign(season_year=2019)
    with pytest.raises(ValueError, match="2017-2018"):
        module.fit_grid(constraints, None, None, None)
