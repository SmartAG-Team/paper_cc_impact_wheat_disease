"""Relocated scientific recipes against the retained final evidence."""
import importlib
import numpy as np
import pandas as pd


def test_scientific_modules_import_without_publication_builders():
    for name in ["observation_model.observation", "severity_probability.data",
                 "joint_disease.models", "crop_canopy.models", "swiss_heading.models",
                 "winter_crop_bounds.bounds", "production_exposure.exposure"]:
        importlib.import_module("analysis.methods." + name)


def test_relocated_frozen_disease_inputs_preserve_original_population():
    from analysis.methods.severity_probability.data import load
    observations, fields, features, names, archived = load()
    assert len(observations) == 2287
    assert len(fields) == 216
    assert len(archived) == 2018
    assert features.shape[0] == 216
    assert len(names) == features.shape[2]


def test_original_joint_assessment_error_matches_retained_final_table():
    from analysis.final_study.paths import resolve
    from analysis.methods.observation_model.observation import hierarchical_weights
    base = "analysis/paper_study/disease_temporal_validation_20261010/results/"
    rows = pd.read_csv(resolve(base + "assessment_predictions.csv"))
    summary = pd.read_csv(resolve(base + "summary.csv"))
    selected = summary[(summary.population == "all_future_records") & (summary.model == "joint")
                       & (summary.endpoint == "assessment") & (summary.source == "pooled")
                       & (summary.test_year == 0)]
    assert len(selected) == 1
    w = hierarchical_weights(rows)
    rmse = np.sqrt(np.average((rows.joint - rows.observed_percent) ** 2, weights=w))
    np.testing.assert_allclose(rmse, selected.rmse.iloc[0], rtol=0, atol=1e-10)


def test_stage_sources_use_authenticated_compact_corteva_input():
    from analysis.final_study.paths import ROOT, resolve
    from analysis.methods.field_stages.data import load_sources
    registry, expanded, legacy = load_sources(ROOT)
    assert len(registry) == 216
    assert not expanded.empty
    assert not legacy.empty
    archived = pd.read_csv(resolve("analysis/paper_study/field_stage_extension_20261010/results/expanded_source_stage_rows.csv"))
    for frame in (expanded, archived):
        frame["date"] = pd.to_datetime(frame.date)
    pd.testing.assert_frame_equal(expanded.reset_index(drop=True), archived, check_dtype=False)
