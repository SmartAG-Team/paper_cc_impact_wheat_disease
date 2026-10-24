"""Field-input isolation, weather preprocessing, and frozen-result identity."""
import numpy as np
import pandas as pd
import pytest

from models.methods.field_inputs import run as field_inputs

def runner_module():
    return field_inputs


def test_duration_rate_uses_unique_training_weather_and_ignores_heldout_values():
    module = runner_module()
    metadata = pd.DataFrame([
        dict(field_index=0, site_id="train", first_forcing_date="2017-01-01", last_forcing_date="2017-01-03"),
        dict(field_index=1, site_id="train", first_forcing_date="2017-01-01", last_forcing_date="2017-01-03"),
    ])
    weather = pd.DataFrame([
        dict(location_id="train", date="2017-01-01", tmean_c=10., precipitation_mm=4., rain_hours_gt_0_1mm=2.),
        dict(location_id="train", date="2017-01-02", tmean_c=10., precipitation_mm=2., rain_hours_gt_0_1mm=2.),
        dict(location_id="train", date="2017-01-03", tmean_c=0., precipitation_mm=100., rain_hours_gt_0_1mm=1.),
        dict(location_id="heldout", date="2017-01-01", tmean_c=10., precipitation_mm=900., rain_hours_gt_0_1mm=1.),
    ])
    rate = module.estimate_training_rain_rate(metadata, weather)
    assert rate["rain_rate_mm_hour"] == pytest.approx(1.5)
    assert rate["unique_training_location_dates"] == 3
    assert rate["eligible_positive_rain_days"] == 2
    altered = weather.copy()
    altered.loc[altered.location_id.eq("heldout"), "precipitation_mm"] = 99999.
    assert module.estimate_training_rain_rate(metadata, altered) == rate


def test_fit_subsets_remove_withheld_targets_and_remap_all_field_indices():
    from models.calibration.seasonal_septoria.field_data import FieldData

    module = runner_module()
    targets = pd.DataFrame([
        dict(field_index=0, field_id="old", value=90., day_index=2, leaf_index=0),
        dict(field_index=1, field_id="heldout", value=80., day_index=2, leaf_index=0),
        dict(field_index=2, field_id="selected", value=5., day_index=1, leaf_index=0),
        dict(field_index=2, field_id="selected", value=15., day_index=2, leaf_index=0),
    ])
    metadata = pd.DataFrame(dict(field_index=[0, 1, 2], field_id=["old", "heldout", "selected"]))
    array = np.array([[1., 2.], [3., 4.], [5., 6.]])
    data = FieldData(array, array, array, np.ones((3, 2, 8), bool),
        np.zeros((3, 2, 8)), metadata, targets, pd.DataFrame())
    isolated = module.subset_fields(data, np.array([2, 3]))
    assert isolated.targets.value.tolist() == [5., 15.]
    assert isolated.targets.field_index.tolist() == [0, 0]
    assert isolated.targets.global_target_index.tolist() == [2, 3]
    assert isolated.metadata.field_id.tolist() == ["selected"]
    assert isolated.temperature.tolist() == [[5., 6.]]
    with pytest.raises(ValueError, match="complete"):
        module.subset_fields(data, np.array([2]))
    redacted = module.subset_fields(data, np.array([2, 3]), redact_values=True)
    assert redacted.targets.value.isna().all()
    assert targets.value.tolist() == [90., 80., 5., 15.]


def test_completed_fit_recovery_requires_exact_hash_identity(tmp_path):
    module = runner_module()
    path = tmp_path / "fit.json"
    identity = {"input_sha256": "abc", "membership_sha256": "def"}
    assert module.read_completed_fit(path, identity) is None
    module.write_frozen_json(path, dict(status="fit_complete", identity=identity,
        fitted=dict(parameters=dict(initiation=.001))))
    assert module.read_completed_fit(path, identity)["parameters"]["initiation"] == .001
    with pytest.raises(ValueError, match="identity|hash"):
        module.read_completed_fit(path, dict(identity, input_sha256="changed"))
    with pytest.raises(FileExistsError):
        module.write_frozen_json(path, {})


def test_frozen_csv_recovery_does_not_duplicate_results(tmp_path):
    module = runner_module()
    path = tmp_path / "metrics.csv"
    frame = pd.DataFrame(dict(model=["canopy", "seir"], rmse=[12., 15.]))
    assert hasattr(module, "save_frozen_csv"), "CSV recovery must verify a frozen result rather than append it"
    module.save_frozen_csv(path, frame)
    module.save_frozen_csv(path, frame)
    assert pd.read_csv(path).rmse.tolist() == [12., 15.]
    with pytest.raises(ValueError, match="frozen|differ"):
        module.save_frozen_csv(path, pd.DataFrame(dict(model=["canopy", "seir"], rmse=[99., 15.])))


def test_prediction_boundary_rejects_unredacted_disease_values():
    from types import SimpleNamespace

    module = runner_module()
    data = SimpleNamespace(targets=pd.DataFrame(dict(value=[1., 3.])))
    with pytest.raises(ValueError, match="redacted"):
        module.predict_record("original_seir", data, None, None, {})
