import io
import zipfile

import numpy as np
import pandas as pd


def zipped(text):
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, 'w') as z:
        z.writestr('produkt_test.txt', text)
    return stream.getvalue()


def test_dwd_inventory_and_nearest_station_use_coverage_and_distance():
    from models.monitoring.dwd_station import parse_inventory, match_stations
    text = ('header\nseparator\n00001 20000101 20251231 10 50.0 10.0 Station One\n'
            '00002 20180101 20251231 10 50.01 10.0 Too New\n'
            '00003 20000101 20251231 10 55.0 10.0 Far Away\n')
    inventory = parse_inventory(text)
    sites = pd.DataFrame(dict(site_id=['x'], latitude=[50.01], longitude=[10.]))
    matched = match_stations(sites, inventory, '20130101', '20191231', 50)
    assert matched.station_id.item() == '00001'
    assert matched.distance_km.item() < 2


def test_hourly_qc_does_not_turn_missing_rain_into_zero():
    from models.monitoring.dwd_station import parse_hourly, daily_aggregate
    lines = ['STATIONS_ID;MESS_DATUM;QN_8;R1;RS_IND;WRTR;eor']
    for hour, stamp in enumerate(pd.date_range('2017-06-01 01:00', periods=24, freq='h')):
        value = -999 if hour == 0 else 1
        lines.append(f'1;{stamp:%Y%m%d%H};3;{value};1;-999;eor')
    hourly = parse_hourly(zipped('\n'.join(lines)), 'RR')
    daily = daily_aggregate(hourly, 'RR')
    assert daily.rain_hours_available.item() == 23
    assert pd.isna(daily.precipitation_mm.item())


def test_rain_accumulation_ending_midnight_belongs_to_previous_utc_day():
    from models.monitoring.dwd_station import parse_hourly, daily_aggregate
    lines = ['STATIONS_ID;MESS_DATUM;QN_8;R1;RS_IND;WRTR;eor']
    for hour, stamp in enumerate(pd.date_range('2017-06-01 01:00', periods=24, freq='h')):
        lines.append(f'1;{stamp:%Y%m%d%H};3;{4 if hour==23 else 0};1;-999;eor')
    daily = daily_aggregate(parse_hourly(zipped('\n'.join(lines)), 'RR'), 'RR')
    assert len(daily) == 1
    assert str(daily.date.item())[:10] == '2017-06-01'
    assert daily.precipitation_mm.item() == 4
    assert daily.station_precip_hours_gt_0_1mm.item() == 1


def test_hourly_temperature_humidity_units_and_utc_day():
    from models.monitoring.dwd_station import parse_hourly, daily_aggregate
    lines = ['STATIONS_ID;MESS_DATUM;QN_9;TT_TU;RF_TU;eor']
    lines += [f'1;20170601{h:02d};3;20;90;eor' for h in range(24)]
    hourly = parse_hourly(zipped('\n'.join(lines)), 'TU')
    daily = daily_aggregate(hourly, 'TU')
    assert daily.tmean_c.item() == 20
    assert daily.rh_mean_pct.item() == 90
    assert daily.rh_hours_ge_90pct.item() == 24
    assert np.isfinite(daily.rh_hours_ge_90pct).all()
    assert str(daily.date.item())[:10] == '2017-06-01'


def test_pairing_names_forcing_precip_even_for_temperature_station_table():
    from models.monitoring.dwd_station import join_station_daily
    forcing = pd.DataFrame(dict(date=pd.to_datetime(['2017-06-01']), tmean_c=[20.], precipitation_mm=[2.]))
    station = pd.DataFrame(dict(date=pd.to_datetime(['2017-06-01']), tmean_c=[19.]))
    paired = join_station_daily(forcing, station)
    assert paired.precipitation_mm_forcing.item() == 2
    assert paired.tmean_c_station.item() == 19


def test_hybrid_weather_requires_elevation_screen_and_retains_liquid_rain():
    from models.monitoring.repairs import hybrid_weather, recorded_leaf_features
    weather = pd.DataFrame(dict(location_id=['s', 's'], date=pd.to_datetime(['2017-06-01','2017-06-02']),
        tmean_c=[20.,21.], tmax_c=[25.,26.], tmin_c=[10.,11.],
        rh_mean_pct=[60.,61.], rh_hours_ge_90pct=[1.,2.], rain_mm=[1.,2.]))
    pairs = pd.DataFrame(dict(site_id=['s','s'], date=weather.date,
        tmean_c_station=[18.,np.nan], tmax_c_station=[24.,np.nan], tmin_c_station=[9.,np.nan],
        rh_mean_pct_station=[80.,np.nan], rh_hours_ge_90pct_station=[8.,np.nan], precipitation_mm_station=[10.,10.]))
    changed, ledger = hybrid_weather(weather, pairs, eligible_sites={'s'})
    assert changed.tmean_c.tolist() == [18.,21.]
    assert changed.rh_mean_pct.tolist() == [80.,61.]
    assert changed.rain_mm.tolist() == [1.,2.]
    assert ledger.loc[ledger.variable.eq('tmean_c'), 'replaced_days'].item() == 1
    unchanged, _ = hybrid_weather(weather, pairs, eligible_sites=set())
    pd.testing.assert_frame_equal(unchanged, weather)
    frame = pd.DataFrame(dict(source=['BASF','BASF'], leaf_index=[2,4]))
    source_only, _ = recorded_leaf_features(frame, shared=False)
    shared, names = recorded_leaf_features(frame, shared=True)
    # An unseen source still has usable recorded-code columns from another source.
    other, _ = recorded_leaf_features(frame.assign(source='Corteva'), shared=True)
    np.testing.assert_array_equal(shared[:,14:], other[:,14:])
    assert shared.shape[1] == source_only.shape[1]+8 == len(names)
    assert not np.array_equal(shared[0,14:], shared[1,14:])


def test_station_rain_prefers_measurement_with_explicit_phase_and_missing_fallback():
    from models.monitoring.repairs import hybrid_weather
    dates=pd.date_range('2017-06-01',periods=5)
    weather=pd.DataFrame(dict(location_id=['s']*5,date=dates,tmin_c=[5.,-2.,-2.,-2.,5.],
        precipitation_mm=[0.,10.,0.,1.,1.],rain_mm=[0.,2.,0.,1.,1.],rain_hours_gt_0_1mm=[0.,4.,0.,2.,2.]))
    pairs=pd.DataFrame(dict(site_id=['s']*5,date=dates,precipitation_mm_station=[8.,20.,4.,0.,np.nan]))
    changed, ledger=hybrid_weather(weather,pairs,{'s'},rainfall=True,meteorology=False)
    np.testing.assert_allclose(changed.precipitation_mm,[8.,20.,4.,0.,1.])
    np.testing.assert_allclose(changed.rain_mm,[8.,4.,0.,0.,1.])
    assert changed.rain_hours_gt_0_1mm.iloc[3]==0
    assert changed.DWD_liquid_phase_assisted.tolist()==[True,True,False,True,False]
    assert ledger.loc[ledger.variable.eq('precipitation_mm'),'replaced_days'].item()==4


def test_observed_snow_suppresses_warm_liquid_assumption():
    from models.monitoring.repairs import hybrid_weather
    dates=pd.to_datetime(['2017-06-01'])
    weather=pd.DataFrame(dict(location_id=['s'],date=dates,tmin_c=[5.],
        precipitation_mm=[10.],rain_mm=[2.],rain_hours_gt_0_1mm=[1.]))
    pairs=pd.DataFrame(dict(site_id=['s'],date=dates,precipitation_mm_station=[20.],
        solid_precipitation_observed=[True]))
    changed,_=hybrid_weather(weather,pairs,{'s'},rainfall=True,meteorology=False)
    assert changed.rain_mm.item()==4
    assert not changed.DWD_warm_liquid_assumption.item()
