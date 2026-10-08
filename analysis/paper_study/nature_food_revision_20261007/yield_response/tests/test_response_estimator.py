"""Behavioral safeguards against leakage, unequal environment weights and clipping."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

MODULE = Path(__file__).resolve().parents[1] / "response_estimator.py"
spec = importlib.util.spec_from_file_location("response_estimator", MODULE)
estimator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = estimator
spec.loader.exec_module(estimator)


def test_many_arms_cannot_increase_a_trials_total_weight():
    df = pd.DataFrame({"record": ["A", "B", "B", "B"], "R": [0, 1, 1, 1]})
    weights = estimator.trial_weights(df)
    assert weights.tolist() == pytest.approx([1, 1 / 3, 1 / 3, 1 / 3])
    assert weights[df.record == "A"].sum() == pytest.approx(1)
    assert weights[df.record == "B"].sum() == pytest.approx(1)


def test_withheld_outcomes_cannot_change_training_coefficient_or_baseline():
    df = pd.DataFrame({
        "row_id": ["A1", "B1", "B2", "C1"],
        "record": ["A", "B", "B", "C"],
        "control_id": ["A0", "B0", "B0", "C0"],
        "x": [1, 1, 1, 2], "R": [0.1, 0.2, 0.4, 0.8],
    })
    predictions, folds, membership = estimator.leave_one_trial_out(df, ["x"])
    withheld = predictions[predictions.record == "C"]
    # Trial A contributes .1, trial B contributes mean(.2,.4)=.3.
    assert withheld.prediction.tolist() == pytest.approx([0.4])
    assert withheld.baseline_prediction.tolist() == pytest.approx([0.2])
    altered = df.copy()
    altered.loc[altered.record == "C", "R"] = -999
    altered_predictions, _, _ = estimator.leave_one_trial_out(altered, ["x"])
    assert altered_predictions.loc[altered_predictions.record == "C", "prediction"].tolist() == pytest.approx([0.4])
    assert altered_predictions.loc[altered_predictions.record == "C", "baseline_prediction"].tolist() == pytest.approx([0.2])
    assert all(f["held_out_trial"] not in f["training_trials"] for f in folds)
    train = membership[(membership.fold == "C") & (membership.role == "train")]
    assert set(train.record) == {"A", "B"}
    assert set(train.row_id) == {"A1", "B1", "B2"}


def test_arms_sharing_control_are_never_split_across_train_and_test():
    df = pd.DataFrame({"row_id": ["A1", "A2", "B1", "B2"],
                       "record": ["A", "A", "B", "B"],
                       "control_id": ["A0", "A0", "B0", "B0"],
                       "x": [1, 2, 1, 2], "R": [.1, .2, .2, .4]})
    _, _, membership = estimator.leave_one_trial_out(df, ["x"])
    for _, fold in membership.groupby("fold"):
        assert fold.groupby("control_id").role.nunique().max() == 1
        assert fold.loc[fold.role == "train"].groupby("record").weight.sum().tolist() == pytest.approx([1])


def test_duplicate_arms_in_one_trial_do_not_change_fit_or_baseline():
    df = pd.DataFrame({"record": ["A", "B"], "x": [1, 2], "R": [.1, .4]})
    inflated = pd.concat([df.iloc[[0]]] + [df.iloc[[1]]] * 30, ignore_index=True)
    fit = estimator.fit_origin(df, ["x"])
    duplicate_fit = estimator.fit_origin(inflated, ["x"])
    assert fit["coefficients"] == pytest.approx([.18])
    assert fit["baseline"] == pytest.approx(.25)
    assert duplicate_fit["coefficients"] == pytest.approx([.18])
    assert duplicate_fit["baseline"] == pytest.approx(.25)


def test_signed_disease_differences_and_negative_responses_are_retained():
    df = pd.DataFrame({"record": ["A", "B"], "x": [-2, 1], "R": [-.4, .2]})
    fit = estimator.fit_origin(df, ["x"])
    assert fit["coefficients"] == pytest.approx([.2])
    assert fit["negative_feature_values"]["x"] == 1
    assert estimator.predict(df, ["x"], fit).tolist() == pytest.approx([-.4, .2])
    assert estimator.signed_response(np.array([11., 8.]), np.array([10., 10.])).tolist() == pytest.approx([-.1, .2])


def test_negative_slope_constraint_and_null_predictor_are_reported():
    df = pd.DataFrame({"record": ["A", "B"], "x": [1, 2], "R": [-.1, -.2]})
    assert estimator.fit_origin(df, ["x"])["coefficients"] == pytest.approx([0])
    df["x"] = 0
    fit = estimator.fit_origin(df, ["x"])
    assert fit["coefficients"] == pytest.approx([0])
    assert fit["null_predictors"] == ["x"]


def test_integral_uses_only_common_dates_and_actual_day_spacing():
    untreated = pd.DataFrame({"date": ["2020-06-01", "2020-06-04", "2020-06-08", "2020-06-11"],
                              "stb": [0., 10., 99., 20.]})
    treated = pd.DataFrame({"date": ["2020-06-01", "2020-06-04", "2020-06-11", "2020-06-20"],
                            "stb": [0., 5., 5., 99.]})
    integral, knots = estimator.paired_integral(untreated, treated)
    assert integral["paired_points"] == 3
    assert integral["window_days"] == 10
    assert integral["auc_untreated_percentage_days"] == pytest.approx(120)
    assert integral["auc_treated_percentage_days"] == pytest.approx(42.5)
    assert integral["delta_stb_percentage_days"] == pytest.approx(77.5)
    assert knots.date.tolist() == ["2020-06-01", "2020-06-04", "2020-06-11"]


def test_additive_model_does_not_turn_missing_disease_into_zero():
    df = pd.DataFrame({"record": ["A", "B"], "R": [.1, .2], "stb_delta": [1., 2.],
                       "rust_delta": [np.nan, 0.]})
    with pytest.raises(ValueError, match="finite"):
        estimator.fit_origin(df, ["stb_delta", "rust_delta"])


def test_measured_zero_stb_is_eligible_but_missing_or_positive_copathogen_is_not():
    df = pd.DataFrame({"stb": [0., 15., 15., 15., 15.], "stb_treated": [0., 5., 5., 5., 5.],
                       "no_positive_nontarget_at_any_recorded_date": [True, True, True, True, False]})
    for disease in ["yellow_rust", "brown_rust", "mildew", "leaf_spot"]:
        df[disease] = 0.
        df[disease + "_treated"] = 0.
    df.loc[2, "brown_rust_treated"] = 1.
    df.loc[3, "brown_rust_treated"] = np.nan
    assert estimator.endpoint_eligibility(df).tolist() == [True, True, False, False, False]
    assert estimator.endpoint_eligibility(df, require_positive_untreated_stb=True).tolist() == [False, True, False, False, False]


def test_duplicate_html_table_ids_do_not_replace_indexed_grain_table():
    from bs4 import BeautifulSoup
    verification_spec = importlib.util.spec_from_file_location("verify_arithmetic", MODULE.parent / "verify_arithmetic.py")
    verification = importlib.util.module_from_spec(verification_spec)
    verification_spec.loader.exec_module(verification)
    soup = BeautifulSoup('<table id="duplicate"><tr><td>98</td></tr></table><table id="duplicate"><tr><td>20</td></tr></table>', "html.parser")
    cells = verification.read_indexed_source_table(soup, 1, "duplicate")
    assert cells[0, 0] == "20"


def test_exponential_damage_prediction_preserves_signed_effect_and_origin():
    df = pd.DataFrame({"record": ["A", "B"], "x": [-2., 1.],
                       "untreated_yield_native": [np.exp(.4), np.exp(-.2)],
                       "treated_yield_native": [1., 1.]})
    df["R"] = [1-np.exp(.4), 1-np.exp(-.2)]
    fit = estimator.fit_damage(df, ["x"])
    assert fit["coefficients"] == pytest.approx([.2])
    result = estimator.predict_damage(pd.DataFrame({"x": [-2., 0., 1.]}), ["x"], fit)
    assert result.tolist() == pytest.approx([-.49182469764127035, 0., .18126924692201818])
    assert fit["baseline"] == pytest.approx(-.1552777253596261)


def test_exponential_damage_approaches_one_without_clipping_linear_score():
    frame = pd.DataFrame({"x": [10., 100.]})
    fit = {"features": ["x"], "coefficients": [.2]}
    predicted = estimator.predict_damage(frame, ["x"], fit)
    assert predicted.tolist() == pytest.approx([.8646647167633873, .9999999979388464])
    assert 0 < predicted[0] < predicted[1] < 1


def test_damage_fold_never_uses_withheld_yields_for_coefficients_or_baseline():
    df = pd.DataFrame({"row_id": ["A1", "B1", "C1"], "record": ["A", "B", "C"],
                       "control_id": ["A0", "B0", "C0"], "x": [1., 1., 2.],
                       "untreated_yield_native": [np.exp(-.1), np.exp(-.3), np.exp(-.8)],
                       "treated_yield_native": [1., 1., 1.]})
    df["R"] = 1-df.untreated_yield_native
    first, _, _ = estimator.leave_one_trial_out(df, ["x"], family="damage")
    changed = df.copy()
    changed.loc[changed.record == "C", "untreated_yield_native"] = 50
    changed.loc[changed.record == "C", "R"] = -49
    second, _, _ = estimator.leave_one_trial_out(changed, ["x"], family="damage")
    assert first.loc[first.record == "C", "prediction"].tolist() == pytest.approx([.3296799539643607])
    assert second.loc[second.record == "C", "prediction"].tolist() == pytest.approx([.3296799539643607])
    assert second.loc[second.record == "C", "baseline_prediction"].tolist() == pytest.approx(first.loc[first.record == "C", "baseline_prediction"].tolist())
