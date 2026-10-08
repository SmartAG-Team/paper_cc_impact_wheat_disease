"""Small boundary checks for the post-evaluation deployment refit."""

import importlib

import numpy as np
import pandas as pd
import pytest


def refit_module():
    try:
        return importlib.import_module("analysis.paper_study.structural_publication_20261006.full_refit.run_full_refit")
    except ModuleNotFoundError:
        pytest.fail("The isolated deployment refit is not implemented")


def test_bundle_retains_fixed_registered_component_weights_and_saved_operators():
    module = refit_module()
    from model.seasonal_septoria.publication_model import load_publication_model

    daily = dict(parameters=dict(initiation=.02, amplification=0., latent_days=20.),
        host_parameters=dict(flag_fraction=.3, leaf_interval=120., expansion_units=100.),
        weather_preprocessing=dict(weather_operator="daily_or"))
    duration = dict(parameters=dict(initiation=.2, amplification=0., latent_days=30.),
        host_parameters=dict(flag_fraction=.6, leaf_interval=80., expansion_units=100.),
        weather_preprocessing=dict(weather_operator="duration_proxy", rain_rate_mm_hour=.5))
    bundle = module.make_bundle({"daily_or": daily, "duration_proxy": duration},
        {"protocol_sha256": "registered-protocol"})
    loaded = load_publication_model(bundle)
    assert loaded["model_id"] == "canopy_score_ensemble_v6_20261006"
    assert [component["weight"] for component in loaded["components"]] == [.5, .5]
    assert loaded["components"][1]["fitted"]["weather_preprocessing"]["rain_rate_mm_hour"] == .5
    assert loaded["deployment_refit_after_retrospective_nested_evaluation"] is True
    assert loaded["independent_validation"] is False


def test_in_sample_predictions_are_labelled_fit_diagnostics_and_use_fixed_average():
    module = refit_module()
    target = pd.DataFrame(dict(field_id=["field", "field"], value=[10., 20.]))
    result = module.diagnostic_predictions(target,
        {"daily_or": np.array([12., 18.]), "duration_proxy": np.array([16., 22.])})
    ensemble = result[result.model.eq("canopy_score_ensemble_v6_20261006")]
    assert ensemble.predicted_percent.tolist() == [14., 20.]
    assert result.diagnostic_label.unique().tolist() == ["full_data_fit_diagnostics"]
    assert result.independent_validation.eq(False).all()
    assert "validation_label" not in result
    assert target.columns.tolist() == ["field_id", "value"]


def test_bundle_rejects_missing_or_mismatched_saved_weather_operators():
    module = refit_module()
    with pytest.raises(ValueError, match="operator|component"):
        module.make_bundle({"daily_or": {"weather_preprocessing": {"weather_operator": "duration_proxy"}}}, {})


def test_registered_weights_cannot_be_changed_during_deployment_refit():
    module = refit_module()
    with pytest.raises(ValueError, match="weight|registered"):
        module.check_ensemble_protocol(dict(weights={"daily_or": .6, "duration_proxy": .4},
            weights_fitted_to_outer_outcomes=False, pooled_nested_scores_read_before_registration=False))
