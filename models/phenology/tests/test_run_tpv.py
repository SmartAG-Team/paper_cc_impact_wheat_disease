"""Forcing and output contracts of the small copied-model command wrapper."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def runner():
    path = Path(__file__).resolve().parents[1] / 'run_tpv.py'
    assert path.is_file(), 'Runnable frozen-parameter T-P-V entry point is missing'
    spec = importlib.util.spec_from_file_location('tpv_runner', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(tmp_path, supplied_gdd=False):
    weather = pd.DataFrame(dict(PEP_ID=[1, 1, 1],
        DATE=['2001-01-01', '2001-01-02', '2001-01-03'],
        LAT=[50., 50., 50.], t_mean=[-5., 5., 25.], t_max=[0., 10., 30.]))
    if supplied_gdd:
        weather['GDD'] = [1., 2., 3.]
    weather_path = tmp_path / 'weather.csv'
    weather.to_csv(weather_path, index=False)
    records = pd.DataFrame(dict(PEP_ID=[1], SOWING_DATE=['2001-01-01'],
        SOWING_KNOWN_AT=['2001-01-01']))
    sowing_path = tmp_path / 'sowing.csv'
    records.to_csv(sowing_path, index=False)
    return weather_path, sowing_path


def test_supplied_gdd_is_authoritative_and_outputs_have_no_observed_truth(tmp_path):
    weather, sowing = inputs(tmp_path, supplied_gdd=True)
    output = tmp_path / 'output'
    report = runner().run(weather, sowing, output)
    daily = pd.read_csv(output / 'daily_features.csv')
    assert daily.Cumulative_t_pp_v_GDD.tolist() == [1., 3., 6.]
    events = pd.read_csv(output / 'event_dates.csv')
    assert events.true_date.isna().all()
    assert report['status'] == 'complete'
    assert report['observed_validation'] is False
    assert json.loads((output / 'run_manifest.json').read_text())['gdd_convention'] == 'supplied'


def test_weather_requires_supplied_gdd_unless_derivation_is_explicit(tmp_path):
    weather, sowing = inputs(tmp_path)
    with pytest.raises(ValueError, match='GDD'):
        runner().run(weather, sowing, tmp_path / 'output')
    assert not (tmp_path / 'output').exists()


def test_explicit_data_derived_gdd_convention_caps_hot_days_and_freezing(tmp_path):
    weather, sowing = inputs(tmp_path)
    output = tmp_path / 'output'
    runner().run(weather, sowing, output, gdd_convention='clipped_mean_0_20')
    daily = pd.read_csv(output / 'daily_features.csv')
    assert daily.GDD.tolist() == [0., 5., 20.]
    assert daily.Cumulative_t_pp_v_GDD.tolist() == [0., 5., 25.]


def test_incomplete_weather_cycle_has_missing_exact_event_dates(tmp_path):
    weather, sowing = inputs(tmp_path, supplied_gdd=True)
    frame = pd.read_csv(weather).drop(index=1)
    frame['GDD'] = 5000.
    frame.to_csv(weather, index=False)
    output = tmp_path / 'output'
    report = runner().run(weather, sowing, output)
    events = pd.read_csv(output / 'event_dates.csv')
    assert events.loc[events.BBCH != 0, 'T-P-V_date'].isna().all()
    assert report['incomplete_cycles'] == 1


def test_existing_output_is_preserved(tmp_path):
    weather, sowing = inputs(tmp_path, supplied_gdd=True)
    output = tmp_path / 'output'
    output.mkdir()
    marker = output / 'marker.txt'
    marker.write_text('preserve')
    with pytest.raises(FileExistsError):
        runner().run(weather, sowing, output)
    assert marker.read_text() == 'preserve'


def test_reconstructed_archived_forcing_retains_decline_above_30_celsius(tmp_path):
    weather, sowing = inputs(tmp_path)
    frame = pd.read_csv(weather)
    frame['t_mean'] = [-5., 25., 35.]
    frame['t_max'] = [0., 30., 40.]
    frame.to_csv(weather, index=False)
    output = tmp_path / 'output'
    runner().run(weather, sowing, output, gdd_convention='archived_temperature_response')
    daily = pd.read_csv(output / 'daily_features.csv')
    assert daily.GDD.tolist() == [0., 20., 10.]
    assert daily.Cumulative_t_pp_v_GDD.tolist() == [0., 20., 30.]


def test_reconstructed_forcing_rejects_negative_thermal_progress_above_40(tmp_path):
    weather, sowing = inputs(tmp_path)
    frame = pd.read_csv(weather)
    frame.loc[2, 't_mean'] = 41.
    frame.to_csv(weather, index=False)
    with pytest.raises(ValueError, match='40'):
        runner().run(weather, sowing, tmp_path / 'output',
            gdd_convention='archived_temperature_response')
