"""Reuse archived, independently matched station hours under retained daily qualification."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .dwd_station import parse_hourly
from .inputs import sha


def align_station_hours(temperature,precipitation):
    t=temperature.copy();r=precipitation.copy()
    t['hour_start']=pd.to_datetime(t.pop('time'))
    r['hour_start']=pd.to_datetime(r.pop('time'))-pd.Timedelta(hours=1)
    return t.merge(r,on='hour_start',how='outer',validate='one_to_one').sort_values('hour_start').reset_index(drop=True)


def load_station_hours(root,weather):
    root=Path(root);dwd=root/'outputs/weather_validation_20261022/dwd'
    mf=root/'outputs/weather_validation_20261022/meteo_france'
    additions=root/'outputs/weather_validation_20261022/coordinate_additions'
    cache=root/'data/external_weather_validation_20261022/dwd'
    dwd_matches=pd.read_csv(dwd/'station_matches.csv',dtype={'station_id':str})
    french=pd.read_csv(mf/'qualified_station_matches.csv',dtype={'station_id':str})
    french_map=dict(zip(french.site_id,french.station_id.str.zfill(8)))
    receipt=json.loads((dwd/'receipt.json').read_text())
    dwd_hash={Path(item['path']).name:item['sha256'] for item in receipt['receipts']}
    paths={dwd/'receipt.json',dwd/'station_matches.csv',mf/'qualified_station_matches.csv',mf/'receipt.json'}
    if (additions/'variable_station_matches.csv').exists():paths.add(additions/'variable_station_matches.csv')
    archive_tables={};french_tables={};rows=[];mapping=[]
    weather=weather.copy();weather['date']=pd.to_datetime(weather.date)
    if weather.duplicated(['location_id','date']).any():raise ValueError('Unique retained location/day weather required')

    def dwd_table(variable,station):
        key=(variable,station)
        if key not in archive_tables:
            archives=sorted(cache.glob(f'stundenwerte_{variable}_{station}_*_hist.zip'))
            if len(archives)!=1:raise ValueError(f'Unique already-qualified DWD archive required: {key}')
            path=archives[0]
            if path.name not in dwd_hash or sha(path)!=dwd_hash[path.name]:raise ValueError('Archived DWD source hash mismatch')
            frame=parse_hourly(path.read_bytes(),variable)
            frame=frame.loc[frame.time.between('2013-01-01','2020-01-01')].copy()
            if variable=='TU':frame=frame.rename(columns={'TT_TU':'temperature_c','RF_TU':'humidity_pct'})[['time','temperature_c','humidity_pct']]
            else:frame=frame.rename(columns={'R1':'precipitation_mm'})[['time','precipitation_mm']]
            archive_tables[key]=frame;paths.add(path)
        return archive_tables[key]

    for provider in ['DWD','METEO_FRANCE']:
        sites=weather.loc[weather.origin_precipitation_mm.eq(provider),'location_id'].unique()
        for site in sites:
            days=weather.loc[weather.location_id.eq(site)].set_index('date').sort_index()
            if provider=='DWD':
                match=dwd_matches.loc[dwd_matches.site_id.eq(site)&dwd_matches.within_distance]
                stations={str(row.variable_group):str(row.station_id).zfill(5) for row in match.itertuples(index=False)}
                if set(stations)!= {'TU','RR'}:raise ValueError('Qualified DWD TU/RR match missing')
                tu,rr=stations['TU'],stations['RR'];humidity_station=tu
                aligned=align_station_hours(dwd_table('TU',tu),dwd_table('RR',rr))
            else:
                supplied=days.loc[days.origin_precipitation_mm.eq(provider),'station_id_precipitation_mm'].dropna().astype(str).unique()
                candidate=[str(v).split('.')[0].zfill(8) for v in supplied]
                if site in french_map:candidate.append(french_map[site])
                candidate=set(candidate)
                if len(candidate)!=1:raise ValueError('Unique retained French station identity required')
                station=candidate.pop();rr=station
                def variable_station(column,origin):
                    if column in days:
                        supplied=days.loc[days[origin].eq(provider),column].dropna().astype(str).unique()
                        selected={str(v).split('.')[0].zfill(8) for v in supplied}
                        if len(selected)>1:raise ValueError('Variable station changes require explicit daily routing')
                        if selected:return selected.pop()
                    return station
                tu=variable_station('station_id_tmean_c','origin_tmean_c')
                humidity_station=variable_station('station_id_rh_mean_pct','origin_rh_mean_pct')
                # The later coordinate audit can retain another copy of an
                # original station. Follow the collection that qualified this
                # location rather than treating unrelated copies as candidates.
                collection=mf if site in french_map else additions
                def french_frame(selected):
                    path=collection/f'{selected}_hourly.parquet'
                    if not path.exists():raise ValueError('Qualified French collection cache missing')
                    if path not in french_tables:
                        french_tables[path]=pd.read_parquet(path);paths.add(path)
                    return french_tables[path]
                temperature=french_frame(tu)[['time','T']].rename(columns={'T':'temperature_c'})
                humidity=french_frame(humidity_station)[['time','U']].rename(columns={'U':'humidity_pct'})
                temperature=temperature.merge(humidity,on='time',how='outer',validate='one_to_one')
                rain=french_frame(rr)[['time','RR1']].rename(columns={'RR1':'precipitation_mm'})
                aligned=align_station_hours(temperature,rain)
            hours=pd.date_range(days.index.min(),days.index.max()+pd.Timedelta(hours=23),freq='h')
            aligned=aligned.set_index('hour_start').reindex(hours)
            for measure,origin in [('precipitation_mm','origin_precipitation_mm'),
                    ('temperature_c','origin_tmean_c'),('humidity_pct','origin_rh_mean_pct')]:
                qualified=days[origin].eq(provider).reindex(hours.normalize(),fill_value=False).to_numpy()
                aligned.loc[~qualified,measure]=np.nan
            aligned.index.name='hour_start';aligned=aligned.reset_index()
            aligned['location_id']=site;aligned['provider']=provider
            aligned['precipitation_station_id']=rr;aligned['temperature_station_id']=tu
            aligned['humidity_station_id']=humidity_station
            rows.append(aligned)
            mapping.append(dict(location_id=site,provider=provider,precipitation_station_id=rr,
                temperature_station_id=tu,humidity_station_id=humidity_station,hours=len(aligned),valid_precipitation_hours=int(aligned.precipitation_mm.notna().sum()),
                valid_temperature_hours=int(aligned.temperature_c.notna().sum()),valid_humidity_hours=int(aligned.humidity_pct.notna().sum()),
                time_policy='Precipitation ending at t assigned interval start t-1h; T/RH observed at interval start; retained daily source qualification required',
                interpretation='Total precipitation, not measured liquid rain; air humidity, not canopy leafwetness'))
            print('station hours',provider,site,len(aligned),flush=True)
    result=pd.concat(rows,ignore_index=True)
    if result.duplicated(['location_id','hour_start']).any():raise ValueError('Conflicting station-hour provenance')
    return result,sorted(paths),pd.DataFrame(mapping)
