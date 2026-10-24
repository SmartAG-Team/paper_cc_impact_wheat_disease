"""Native event-likelihood and historical monitor contracts without fitting."""
import importlib

import numpy as np
import pandas as pd
import pytest
import torch


def module(name):
    spec = importlib.util.find_spec("analysis.methods.agc_training." + name)
    assert spec is not None, "Retained native AGC training method is missing"
    return importlib.import_module(spec.name)


def test_native_interval_and_right_censor_objective_matches_conditioned_logistic_mass():
    training, objective = module("training"), module("objective")
    event = training.supervision(np.asarray([[1, 5, -1, -1]]), horizon=3)
    np.testing.assert_array_equal(event["event_kind"], [[1, 2, 0, 0]])
    batch = {key: torch.as_tensor(value) for key, value in event.items()}
    batch.update(known_future=torch.zeros((1, 3, 14)), future_valid=torch.ones((1, 3), dtype=torch.bool))
    progress = torch.tensor([[1., 2., 3.]], dtype=torch.float64, requires_grad=True)
    output = dict(objective="process_residual_truncated_logistic_event_nll", corrected_progress=progress,
        process_thresholds=torch.tensor([[2., 3., 4., 5.]], dtype=torch.float64),
        residual_temperature=torch.ones(1, dtype=torch.float64), log_correction=torch.zeros((1, 3), dtype=torch.float64))
    result = objective.ablation_loss(output, batch)
    logistic = lambda value: 1 / (1 + np.exp(-value))
    mass = (logistic(0) - logistic(-1)) / logistic(2)
    survival = logistic(0) / logistic(3)
    assert result["loss"].item() == pytest.approx((-np.log(mass) - np.log(survival)) / 2, abs=1e-14)
    assert result["event_count"].item() == 2
    assert result["supported_stages"].item() == 2
    result["loss"].backward()
    assert torch.isfinite(progress.grad).all()
    with pytest.raises(ValueError, match="objective"):
        objective.ablation_loss({**output, "objective": "unrelated_baseline"}, batch)


def test_epoch_monitor_has_unique_stations_and_rejects_post2015_labels():
    training = module("training")
    records = pd.DataFrame(dict(partition=["val"] * 6, harvest_year=[2015] * 6,
        PEP_ID=[1, 1, 2, 2, 3, 3], SOWING_DATE=["2014-10-01", "2013-10-01"] * 3))
    chosen = training.monitor_membership(records, np.arange(6), maximum=2)
    assert len(chosen) == records.iloc[chosen].PEP_ID.nunique() == 2
    np.testing.assert_array_equal(chosen, training.monitor_membership(records, np.arange(6), maximum=2))
    records.loc[0, "harvest_year"] = 2016
    with pytest.raises(ValueError, match="historical"):
        training.monitor_membership(records, np.arange(6))


def test_training_modules_import_local_architecture_and_recorded_constants():
    training, adapter, device = module("training"), module("adapter"), module("device")
    assert training.SEEDS == [17, 29, 43]
    assert training.SETTINGS["encoder"] == "gated_delta"
    assert callable(training.fit_one) and callable(adapter.fit_preprocessing)
    model = adapter.make_model(training.SETTINGS)
    assert device.mixed_neural_device(model, "cpu") is model


def test_established_crop_crps_masks_missing_stages_and_retains_beyond_horizon_events():
    spec = importlib.util.find_spec("analysis.methods.crop_training")
    assert spec is not None, "Retained established crop training method is missing"
    crop = importlib.import_module(spec.name)
    cdf = torch.zeros((1, 4, 330), dtype=torch.float64)
    target = torch.tensor([[0, -1, 400, 2]])
    assert crop.torch_loss(cdf, target, torch.ones(4)).item() == pytest.approx(658 / 3)
    values = crop.crps_values(cdf.numpy(), target.numpy())
    np.testing.assert_array_equal(values[:, [0, 2, 3]], [[330, 0, 328]])
    assert np.isnan(values[0, 1])
    assert crop.SEEDS == [17, 29, 43]
    assert crop.MODES == ["ordered_process", "ordered_lstm", "stage_attention"]
