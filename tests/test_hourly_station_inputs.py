import pandas as pd


def test_precipitation_closing_midnight_uses_previous_hour_start_temperature():
    from models.monitoring.hourly_station_inputs import align_station_hours
    temperature=pd.DataFrame(dict(time=pd.to_datetime(['2020-01-01 23:00','2020-01-02 00:00']),
        temperature_c=[10.,30.],humidity_pct=[80.,20.]))
    precipitation=pd.DataFrame(dict(time=pd.to_datetime(['2020-01-02 00:00']),precipitation_mm=[2.]))
    aligned=align_station_hours(temperature,precipitation).set_index('hour_start')
    last=aligned.loc[pd.Timestamp('2020-01-01 23:00')]
    assert last.precipitation_mm==2. and last.temperature_c==10. and last.humidity_pct==80.


def test_station_cache_follows_qualified_collection_when_station_has_two_copies(tmp_path):
    import json
    from models.monitoring.hourly_station_inputs import load_station_hours
    base=tmp_path/'outputs/weather_validation_20261022'
    for name in ['dwd','meteo_france','coordinate_additions']:(base/name).mkdir(parents=True)
    pd.DataFrame(columns=['site_id','station_id','within_distance','variable_group']).to_csv(base/'dwd/station_matches.csv',index=False)
    (base/'dwd/receipt.json').write_text(json.dumps(dict(receipts=[])))
    pd.DataFrame(dict(site_id=['s'],station_id=['00000001'])).to_csv(base/'meteo_france/qualified_station_matches.csv',index=False)
    (base/'meteo_france/receipt.json').write_text('{}')
    for collection,rain in [('meteo_france',.5),('coordinate_additions',1.)]:
        pd.DataFrame(dict(time=pd.date_range('2020-01-01',periods=25,freq='h'),
            RR1=rain,T=10.,U=80.)).to_parquet(base/collection/'00000001_hourly.parquet',index=False)
    weather=pd.DataFrame(dict(location_id=['s'],date=['2020-01-01'],origin_precipitation_mm=['METEO_FRANCE'],
        origin_tmean_c=['METEO_FRANCE'],origin_rh_mean_pct=['METEO_FRANCE'],station_id_precipitation_mm=[None]))
    result,paths,_=load_station_hours(tmp_path,weather)
    assert result.precipitation_mm.sum()==12.
    assert base/'meteo_france/00000001_hourly.parquet' in paths
    assert base/'coordinate_additions/00000001_hourly.parquet' not in paths


def test_variable_specific_humidity_station_overrides_precipitation_station(tmp_path):
    import json
    from models.monitoring.hourly_station_inputs import load_station_hours
    base=tmp_path/'outputs/weather_validation_20261022'
    for name in ['dwd','meteo_france','coordinate_additions']:(base/name).mkdir(parents=True)
    pd.DataFrame(columns=['site_id','station_id','within_distance','variable_group']).to_csv(base/'dwd/station_matches.csv',index=False)
    (base/'dwd/receipt.json').write_text(json.dumps(dict(receipts=[])))
    pd.DataFrame(columns=['site_id','station_id']).to_csv(base/'meteo_france/qualified_station_matches.csv',index=False)
    (base/'meteo_france/receipt.json').write_text('{}')
    for station,rh in [('00000001',20.),('00000002',80.)]:
        pd.DataFrame(dict(time=pd.date_range('2020-01-01',periods=25,freq='h'),
            RR1=.5,T=10.,U=rh)).to_parquet(base/'coordinate_additions'/f'{station}_hourly.parquet',index=False)
    weather=pd.DataFrame(dict(location_id=['s'],date=['2020-01-01'],origin_precipitation_mm=['METEO_FRANCE'],
        origin_tmean_c=['METEO_FRANCE'],origin_rh_mean_pct=['METEO_FRANCE'],station_id_precipitation_mm=['00000001'],
        station_id_tmean_c=['00000001'],station_id_rh_mean_pct=['00000002']))
    result,paths,mapping=load_station_hours(tmp_path,weather)
    assert result.humidity_pct.mean()==80.
    assert mapping.humidity_station_id.iloc[0]=='00000002'
