"""Tests of retained climate inputs and exact final-census sample replay."""
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_cli_exposes_input_check_and_census_rerun():
    result = subprocess.run(
        [sys.executable, "-m", "models.climate", "--help"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--check-inputs" in result.stdout
    assert "--all" in result.stdout


def test_scientific_manifest_has_exact_final_forcing_support():
    from analysis.final_climate.run import read_manifest, required_forcing
    manifest = read_manifest()
    assert len(manifest["forcing"]) == 687
    assert len(required_forcing(manifest)) == 687
    assert {row["model"] for row in manifest["forcing"]} == {
        "ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0"
    }


def test_missing_input_is_reported_with_its_path(tmp_path):
    from analysis.final_climate.run import check_file
    path = tmp_path / "absent.parquet"
    try:
        check_file(path, {"bytes": 1}, verify_hash=False)
    except FileNotFoundError as error:
        assert str(path) in str(error)
    else:
        raise AssertionError("Missing forcing was accepted")


def test_original_sixty_two_cell_results_are_preserved():
    from analysis.final_climate.run import replay_sample
    result = replay_sample()
    assert result["cells"] == 62
    assert result["matches_frozen_final_census"] is True
    assert result["maximum_numeric_difference"] < 1e-10


def test_retained_monthly_statistics_reproduce_published_era5_targets():
    import numpy as np
    import pandas as pd
    from analysis.final_climate.alignment import reference_monthly_means
    means = reference_monthly_means()
    cells = pd.read_parquet(ROOT / "data/final_climate/cells.parquet")
    published = pd.read_csv(ROOT / "data/final_climate/published_monthly_target_cell_detail.csv")
    positions = pd.Index(cells.cell_id).get_indexer(published.cell_id)
    assert (positions >= 0).all()
    for variable, rows in published.groupby("variable"):
        selected = pd.Index(cells.cell_id).get_indexer(rows.cell_id)
        actual = means[variable][selected, rows.month.to_numpy(int) - 1]
        np.testing.assert_allclose(actual, rows.reference_monthly_mean, rtol=0, atol=1e-10)
