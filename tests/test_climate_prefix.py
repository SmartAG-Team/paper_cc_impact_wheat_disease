import json
from pathlib import Path

import numpy as np
import pandas as pd


def _case():
    from models.monitoring.innovation_common import ROOT
    d = ROOT/'data/final_climate'
    calibration = json.loads((d/'phenology_parameters.json').read_text())
    fit = json.loads((d/'disease_parameters.json').read_text())['fitted']
    threshold = {int(k): v for k, v in json.loads((d/'stage_thresholds.json').read_text())['all_stage_thresholds'].items()}
    dates = pd.date_range('2017-09-01', periods=520)
    temperature = np.full((1, len(dates)), 15.)
    weather = dict(tmean_c=temperature, tmax_c=temperature+5., rh_mean_pct=np.full_like(temperature, 90.),
        precipitation_mm=np.full_like(temperature, 2.))
    return dates, weather, np.array(['2017-09-05'], dtype='datetime64[D]'), np.array([50.]), calibration, fit, threshold


def test_daily_prefix_does_not_require_future_season_completion():
    from models.monitoring.climate_prefix import simulate_prefix
    dates, w, sow, lat, calibration, fit, threshold = _case()
    value = simulate_prefix(dates[:80], {k: a[:, :80] for k, a in w.items()}, sow, lat, calibration, fit, threshold)
    assert value['status'][0] == 'in_progress'
    assert value['mask'].sum() == 76
    assert value['trajectory'].state.shape[1] == 81
    assert np.max(np.abs(value['trajectory'].state.sum(axis=-1)-1.)) < 1e-10


def test_completed_season_prefix_matches_frozen_climate_arithmetic_exactly():
    from models.monitoring.climate_prefix import simulate_prefix
    from models.climate.kernel import simulate_draw_seasons
    dates, w, sow, lat, calibration, fit, threshold = _case()
    _, frozen = simulate_draw_seasons(dates, w, sow, lat, calibration, fit, threshold, return_daily=True)
    assert frozen['complete_count'] == 1
    value = simulate_prefix(dates, w, sow, lat, calibration, fit, threshold)
    np.testing.assert_array_equal(value['trajectory'].state, frozen['trajectory'].state)
    np.testing.assert_array_equal(value['host'].active, frozen['host'].active)
    short = simulate_prefix(dates[:80], {k: a[:, :80] for k, a in w.items()}, sow, lat, calibration, fit, threshold)
    np.testing.assert_array_equal(short['trajectory'].state, frozen['trajectory'].state[:, :81])


def test_future_weather_and_invalid_tail_cannot_change_prefix_states():
    from models.monitoring.climate_prefix import simulate_prefix
    dates, w, sow, lat, calibration, fit, threshold = _case()
    original = simulate_prefix(dates, w, sow, lat, calibration, fit, threshold)
    changed = {k: a.copy() for k, a in w.items()}
    for a in changed.values():
        a[:, 80:] = np.nan
    altered = simulate_prefix(dates, changed, sow, lat, calibration, fit, threshold)
    np.testing.assert_array_equal(original['trajectory'].state[:, :81], altered['trajectory'].state[:, :81])
