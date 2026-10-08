"""Verified sowing dates must survive ERA5-to-process forcing preparation."""
import importlib.util
from pathlib import Path

import pandas as pd
import pytest


def adapter():
    path = Path(__file__).resolve().parents[2] / 'analysis/phenology/run_field_transfer.py'
    assert path.is_file(), 'ERA5 field-transfer adapter is missing'
    spec = importlib.util.spec_from_file_location('field_transfer_adapter', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inputs(tmp_path):
    pd.DataFrame(dict(location_id=['grid1']*3,
        date=['2017-12-26', '2017-12-27', '2017-12-28'],
        tmean_c=[5., 5., 35.], tmax_c=[10., 10., 40.], tmin_c=[0., 0., 30.]
    )).to_csv(tmp_path / 'weather.csv', index=False)
    pd.DataFrame(dict(dataset_id=['fixture'], site_id=['site'], season_year=[2018],
        requested_latitude=[36.5], requested_longitude=[9.], location_id=['grid1'],
        coordinate_source=['verified trial coordinate']
    )).to_csv(tmp_path / 'links.csv', index=False)
    pd.DataFrame(dict(PEP_ID=[1], case_id=['test2018'], dataset_id=['fixture'],
        weather_link_dataset_id=['fixture'], site_id=['site'], season_year=[2018],
        SOWING_DATE=['2017-12-27'], SOWING_KNOWN_AT=['2017-12-27'],
        sowing_source=['https://example.test/verified_methods'], wheat_type=['durum'],
        transfer_status=['uncalibrated cultivar/wheat-type transfer']
    )).to_csv(tmp_path / 'sowing.csv', index=False)
    return tmp_path / 'weather.csv', tmp_path / 'links.csv', tmp_path / 'sowing.csv'


def test_adapter_retains_documented_sowing_and_excludes_prior_weather(tmp_path):
    weather, links, sowing = inputs(tmp_path)
    output = tmp_path / 'output'
    report = adapter().run(weather, links, sowing, output)
    forcing = pd.read_csv(output / 'canonical_weather.csv')
    assert forcing.DATE.tolist() == ['2017-12-27', '2017-12-28']
    assert forcing.LAT.tolist() == [36.5, 36.5]
    daily = pd.read_csv(output / 'simulation/daily_features.csv')
    assert daily.GDD.tolist() == [5., 10.]
    assert daily.Cumulative_t_pp_v_GDD.tolist() == [5., 15.]
    assert report['observed_validation'] is False


def test_adapter_rejects_sowing_without_an_exact_site_season_link(tmp_path):
    weather, links, sowing = inputs(tmp_path)
    frame = pd.read_csv(links).assign(season_year=2019)
    frame.to_csv(links, index=False)
    with pytest.raises(ValueError, match='link'):
        adapter().run(weather, links, sowing, tmp_path / 'output')
