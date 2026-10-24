"""NOAA GSOD source semantics and outcome-independent station matching."""
import numpy as np
import pandas as pd
import pytest


def test_gsod_units_missing_and_mean_count_quality():
    from models.monitoring.noaa_station import parse_gsod
    raw = pd.DataFrame({'DATE':['2017-01-01','2017-01-02','2017-01-03'],
        'TEMP':[50,9999.9,50], 'TEMP_ATTRIBUTES':[24,24,17],
        'DEWP':[32,9999.9,32], 'DEWP_ATTRIBUTES':[24,24,17],
        'MAX':[68,9999.9,68], 'MIN':[32,9999.9,32],
        'PRCP':[1,99.99,0], 'PRCP_ATTRIBUTES':['G','G','H']})
    parsed = parse_gsod(raw)
    assert parsed.tmean_c.iloc[0] == pytest.approx(10)
    assert parsed.dewpoint_mean_c.iloc[0] == 0
    assert parsed.precipitation_mm.iloc[0] == pytest.approx(25.4)
    assert parsed.loc[1, ['tmean_c','dewpoint_mean_c','tmax_c','tmin_c','precipitation_mm']].isna().all()
    assert np.isnan(parsed.tmean_c.iloc[2]) and np.isnan(parsed.dewpoint_mean_c.iloc[2])
    assert np.isnan(parsed.precipitation_mm.iloc[2])
    assert not any('rh' in c.lower() for c in parsed.columns)


def test_only_complete_precipitation_periods_are_retained():
    from models.monitoring.noaa_station import parse_gsod
    raw = pd.DataFrame({'DATE':pd.date_range('2017-01-01', periods=9),
        'TEMP':[50]*9, 'TEMP_ATTRIBUTES':[24]*9, 'DEWP':[32]*9,
        'DEWP_ATTRIBUTES':[24]*9, 'MAX':[68]*9, 'MIN':[32]*9,
        'PRCP':[.1]*9, 'PRCP_ATTRIBUTES':list('ABCDEFGHI')})
    p = parse_gsod(raw)
    assert set(raw.loc[p.precipitation_mm.notna(),'PRCP_ATTRIBUTES']) == {'D','F','G'}


def test_station_ranking_honors_period_and_radius_and_not_outcomes():
    from models.monitoring.noaa_station import rank_stations
    stations = pd.DataFrame({'USAF':['111111','222222','333333','444444'],
        'WBAN':['99999']*4,'LAT':[50,50.05,50.1,52], 'LON':[1]*4,
        'BEGIN':['20180101','20100101','20100101','20100101'],
        'END':['20201231']*4, 'STATION NAME':['badperiod','near','far','outside']})
    a = rank_stations(stations, 50., 1., '2016-01-01','2017-12-31')
    assert a.station_id.tolist() == ['22222299999','33333399999']
    assert a.distance_km.is_monotonic_increasing
    assert a.within_25km.all()


def test_site_weather_union_keeps_gaps_and_excludes_germany():
    from models.monitoring.noaa_station import eligible_windows, select_site_weather
    fields=pd.DataFrame({'field_id':['a','b','c'],'site_id':['s','s','g'],
        'country':['FRANCE','FRANCE','GERMANY'], 'latitude':[50]*3,'longitude':[1]*3,
        'first_forcing_date':['2016-01-01','2016-01-05','2016-01-01'],
        'last_forcing_date':['2016-01-02','2016-01-06','2016-01-06']})
    windows=eligible_windows(fields)
    weather=pd.DataFrame({'location_id':['s']*6,'date':pd.date_range('2016-01-01',periods=6)})
    selected=select_site_weather(weather,windows,'s')
    assert selected.date.dt.day.tolist()==[1,2,5,6]
    assert windows.site_id.unique().tolist()==['s']


def test_metrics_sign_and_wet_day_threshold_are_explicit():
    from models.monitoring.noaa_station import pair_metrics
    pairs=pd.DataFrame({'forcing':[0.,2.,4.], 'station':[0.,1.,2.]})
    m=pair_metrics(pairs,expected_days=4,precip=True)
    assert m['n_pairs']==3 and m['coverage']==.75
    assert m['bias_forcing_minus_station']==1.
    assert m['rmse']==pytest.approx(np.sqrt(5/3))
    assert m['wet_day_agreement_ge_1mm']==1.


def test_station_elevation_is_preserved_as_representativeness_caution():
    from models.monitoring.noaa_station import rank_stations
    stations = pd.DataFrame({'USAF':['119300'],'WBAN':['99999'],'LAT':[49.2],
        'LON':[20.217],'BEGIN':['19520101'],'END':['20250824'],
        'ELEV(M)':['+2635.0'],'STATION NAME':['LOMNICKY STIT']})
    s=rank_stations(stations,49.2,20.5,'2018-10-01','2019-07-10')
    assert s.station_elevation_m.item()==2635.
    assert s.high_elevation_caution.item()


def test_elevation_screen_chooses_nearest_representative_station_without_outcomes():
    from models.monitoring.noaa_station import rank_stations
    stations = pd.DataFrame({'USAF':['119300','119340'],'WBAN':['99999']*2,
        'LAT':[49.2,49.067],'LON':[20.217,20.25],
        'BEGIN':['19520101']*2,'END':['20250824']*2,
        'ELEV(M)':['2635','694'],'STATION NAME':['SUMMIT','POPRAD']})
    s=rank_stations(stations,49.2,20.5,'2018-10-01','2019-07-10',target_elevation_m=678.)
    assert s.station_id.tolist()==['11934099999']
    assert s.elevation_difference_m.item()==16.
    assert rank_stations(stations,49.2,20.5,'2018-10-01','2019-07-10',target_elevation_m=np.nan).empty


def test_dem_payload_preserves_site_order_and_marks_unknown():
    from models.monitoring.noaa_station import elevation_table
    sites=pd.DataFrame({'site_id':['a','b'],'latitude':[49.,50.],'longitude':[20.,21.]})
    table=elevation_table(sites,{'elevation':[678.,None]})
    assert table.site_id.tolist()==['a','b']
    assert table.target_dem_elevation_m.iloc[0]==678.
    assert table.elevation_status.tolist()==['available','unknown']
    with pytest.raises(ValueError,match='length'):
        elevation_table(sites,{'elevation':[5.]})
