"""Independent NOAA NCEI GSOD station checks against original field forcing.

Daily summaries are NOAA-hosted ISD-derived records exchanged through WMO
programmes; they are not WMO-hosted observations. Station selection never uses
assessment severity or prediction error. All bias signs are forcing minus station.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
from urllib.parse import urlencode

import numpy as np
import pandas as pd
import requests

INVENTORY_URL = 'https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv'
README_URL = 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/doc/readme.txt'
GSOD_URL = 'https://www.ncei.noaa.gov/data/global-summary-of-the-day/access/{year}/{station}.csv'
VARIABLES = ['tmean_c', 'tmin_c', 'tmax_c', 'dewpoint_mean_c', 'precipitation_mm']


def parse_gsod(raw: pd.DataFrame) -> pd.DataFrame:
    """Convert documented units; >=18 mean samples and full-period rain only."""
    out = pd.DataFrame({'date': pd.to_datetime(raw.DATE).dt.normalize()})
    for source, variable in [('TEMP','tmean_c'),('DEWP','dewpoint_mean_c'),
                             ('MAX','tmax_c'),('MIN','tmin_c')]:
        values = pd.to_numeric(raw[source], errors='coerce').replace(9999.9, np.nan)
        if source in {'TEMP','DEWP'}:
            counts = pd.to_numeric(raw[f'{source}_ATTRIBUTES'], errors='coerce')
            values = values.where(counts.ge(18))
            out[f'{source.lower()}_observation_count'] = counts
        out[variable] = (values - 32) * 5 / 9
    rain = pd.to_numeric(raw.PRCP, errors='coerce').replace(99.99, np.nan)
    flags = raw.PRCP_ATTRIBUTES.astype(str).str.strip()
    out['precipitation_mm'] = (rain * 25.4).where(flags.isin(['D','F','G']))
    out['precipitation_flag'] = flags
    return out


def rank_stations(inventory, latitude, longitude, start, end, max_km=50., target_elevation_m=None):
    """Rank period-covering stations by distance, independent of outcomes."""
    s = inventory.copy()
    begin = pd.to_datetime(s.BEGIN.astype(str), format='%Y%m%d', errors='coerce')
    finish = pd.to_datetime(s.END.astype(str), format='%Y%m%d', errors='coerce')
    s['LAT'] = pd.to_numeric(s.LAT, errors='coerce')
    s['LON'] = pd.to_numeric(s.LON, errors='coerce')
    s = s.loc[begin.le(pd.Timestamp(start)) & finish.ge(pd.Timestamp(end))
              & s.LAT.between(-90,90) & s.LON.between(-180,180)].copy()
    lat = np.deg2rad(s.LAT.to_numpy())
    lat0 = np.deg2rad(latitude)
    dl = np.deg2rad(s.LON.to_numpy()-longitude)
    a = np.sin((lat-lat0)/2)**2 + np.cos(lat)*np.cos(lat0)*np.sin(dl/2)**2
    s['distance_km'] = 6371.0088 * 2 * np.arcsin(np.sqrt(np.clip(a,0,1)))
    s['station_id'] = s.USAF.astype(str).str.zfill(6) + s.WBAN.astype(str).str.zfill(5)
    s['within_25km'] = s.distance_km.le(25)
    s['station_elevation_m'] = pd.to_numeric(s.get('ELEV(M)', pd.Series(index=s.index,dtype=float)), errors='coerce')
    s['high_elevation_caution'] = s.station_elevation_m.ge(1500)
    if target_elevation_m is not None:
        s['elevation_difference_m']=(s.station_elevation_m-target_elevation_m).abs()
        s=s.loc[s.elevation_difference_m.le(300)].copy()
    return s.loc[s.distance_km.le(max_km)].sort_values(['distance_km','station_id']).reset_index(drop=True)


def elevation_table(sites, payload):
    """Map the documented ordered DEM response, retaining failed/missing values."""
    values=payload.get('elevation',[None]*len(sites))
    if len(values)!=len(sites):
        raise ValueError('DEM response length differs from requested coordinate count')
    table=sites.copy().reset_index(drop=True)
    table['target_dem_elevation_m']=pd.to_numeric(pd.Series(values),errors='coerce')
    table['elevation_status']=np.where(table.target_dem_elevation_m.notna(),'available','unknown')
    table['source']='Open-Meteo elevation API; Copernicus DEM 2021 GLO-90'
    table['coordinate_caveat']='Rounded field coordinates; point DEM may differ from actual field elevation.'
    return table


def eligible_windows(fields):
    windows = fields.loc[~fields.country.str.upper().eq('GERMANY')].copy()
    for col in ['first_forcing_date','last_forcing_date']:
        windows[col] = pd.to_datetime(windows[col]).dt.normalize()
    if not windows.first_forcing_date.ge('2013-01-01').all() or not windows.last_forcing_date.le('2019-12-31').all():
        raise ValueError('Eligible windows must lie within 2013–2019')
    return windows


def select_site_weather(weather, windows, site_id):
    w = weather.loc[weather.location_id.eq(site_id)].copy()
    w['date'] = pd.to_datetime(w.date).dt.normalize()
    keep = pd.Series(False, index=w.index)
    for row in windows.loc[windows.site_id.eq(site_id)].itertuples():
        keep |= w.date.between(row.first_forcing_date, row.last_forcing_date)
    return w.loc[keep].sort_values('date').drop_duplicates(['location_id','date'])


def pair_metrics(pairs, expected_days, precip=False):
    valid = pairs[['forcing','station']].dropna()
    n = len(valid)
    diff = valid.forcing-valid.station
    result = {'n_pairs': n, 'expected_days': int(expected_days),
              'coverage': n/expected_days if expected_days else np.nan,
              'bias_forcing_minus_station': float(diff.mean()) if n else np.nan,
              'rmse': float(np.sqrt((diff**2).mean())) if n else np.nan,
              'mae': float(diff.abs().mean()) if n else np.nan,
              'correlation': float(valid.forcing.corr(valid.station)) if n>1 and valid.forcing.std()>0 and valid.station.std()>0 else np.nan}
    result['adequate_coverage'] = bool(n>=30 and result['coverage']>=.7)
    if precip:
        fw, sw = valid.forcing.ge(1), valid.station.ge(1)
        result['wet_day_agreement_ge_1mm'] = float((fw==sw).mean()) if n else np.nan
        result['wet_day_frequency_forcing'] = float(fw.mean()) if n else np.nan
        result['wet_day_frequency_station'] = float(sw.mean()) if n else np.nan
        result['mean_forcing'] = float(valid.forcing.mean()) if n else np.nan
        result['mean_station'] = float(valid.station.mean()) if n else np.nan
    return result


def fetch_source(url, path):
    """Cache raw public bytes and a receipt, including unsuccessful requests."""
    path = Path(path)
    meta_path = path.with_suffix(path.suffix+'.receipt.json')
    if meta_path.exists():
        previous = json.loads(meta_path.read_text())
        if previous['status_code']==200 and path.exists():
            if hashlib.sha256(path.read_bytes()).hexdigest()!=previous['sha256']:
                raise ValueError(f'Cached source hash mismatch: {path}')
            return previous
        if previous['status_code']==404:
            return previous
    path.parent.mkdir(parents=True, exist_ok=True)
    response = None
    error = None
    for attempt in range(3):
        try:
            response = requests.get(url, timeout=90)
            if response.status_code not in {429,500,502,503,504}:
                break
        except requests.RequestException as exc:
            error = str(exc)
        time.sleep(attempt+1)
    status = response.status_code if response is not None else 0
    if status==200:
        path.write_bytes(response.content)
    receipt = dict(url=url, path=str(path), status_code=status,
                   retrieved_utc=datetime.now(timezone.utc).isoformat(),
                   sha256=hashlib.sha256(response.content).hexdigest() if response is not None else None,
                   bytes=len(response.content) if response is not None else 0, error=error)
    meta_path.write_text(json.dumps(receipt,indent=2))
    return receipt


def run(root: Path, elevation_screen=False):
    root = Path(root)
    cache = root/'data/external_weather_validation_20261022/noaa'
    out = root/'outputs/weather_validation_20261022/noaa'
    if elevation_screen:
        out=out/'elevation_300m'
    cache.mkdir(parents=True,exist_ok=True); out.mkdir(parents=True,exist_ok=True)
    receipts = [fetch_source(README_URL,cache/'readme.txt'),fetch_source(INVENTORY_URL,cache/'isd-history.csv')]
    if any(r['status_code']!=200 for r in receipts):
        raise RuntimeError('NOAA documentation/inventory unavailable')
    # Assert source semantics before accepting 24-hour precipitation flags.
    doc=(cache/'readme.txt').read_bytes().decode('latin1')
    for phrase in ['D = Summation of 4 reports of 6-hour','F = Summation of 2 reports of 12-hour','G = 1 report of 24-hour']:
        if phrase not in doc:
            raise ValueError(f'GSOD documentation requires review: {phrase}')
    inventory=pd.read_csv(cache/'isd-history.csv',dtype=str)
    fields=pd.read_csv(root/'data/final_study/evidence/disease_joint_forecast/inputs/fields.csv')
    sites=fields[['site_id','latitude','longitude']].drop_duplicates('site_id').sort_values('site_id').reset_index(drop=True)
    receipts.append(fetch_source('https://open-meteo.com/en/docs/elevation-api',cache/'elevation'/'documentation.html'))
    elevations=[]
    for i in range(0,len(sites),100):
        batch=sites.iloc[i:i+100]
        url='https://api.open-meteo.com/v1/elevation?'+urlencode({
            'latitude':','.join(batch.latitude.astype(str)), 'longitude':','.join(batch.longitude.astype(str))})
        path=cache/'elevation'/f'batch_{i//100}.json'
        receipt=fetch_source(url,path);receipts.append(receipt)
        payload=json.loads(path.read_text()) if receipt['status_code']==200 else {}
        elevations.append(elevation_table(batch,payload))
    elevations=pd.concat(elevations,ignore_index=True)
    elevations.to_csv(root/'outputs/weather_validation_20261022/noaa/field_elevation.csv',index=False)
    fields=fields.merge(elevations[['site_id','target_dem_elevation_m','elevation_status']],on='site_id',validate='many_to_one')
    windows=eligible_windows(fields)
    weather=pd.read_parquet(root/'data/final_study/inputs/paper_study/field_weather_completed/daily_weather.parquet')
    site_weather={site:select_site_weather(weather,windows,site) for site in windows.site_id.unique()}
    candidates={}
    for site,g in windows.groupby('site_id'):
        candidates[site]=rank_stations(inventory,g.latitude.iloc[0],g.longitude.iloc[0],
                                      g.first_forcing_date.min(),g.last_forcing_date.max(),
                                      target_elevation_m=g.target_dem_elevation_m.iloc[0] if elevation_screen else None)
    attempted={}; station_frames={}; matched={}; failures={}; ranks={site:0 for site in candidates}
    pending=set(candidates)
    while pending:
        round_sites={s:candidates[s].iloc[ranks[s]] for s in pending if ranks[s]<len(candidates[s])}
        if not round_sites:
            break
        tasks={}
        for site,station in round_sites.items():
            for year in sorted(site_weather[site].date.dt.year.unique()):
                key=(station.station_id,int(year))
                if key not in attempted:
                    tasks[key]=(GSOD_URL.format(year=year,station=station.station_id),cache/'gsod'/str(year)/f'{station.station_id}.csv')
        with ThreadPoolExecutor(max_workers=5) as pool:
            results=list(pool.map(lambda kv:(kv[0],fetch_source(*kv[1])),tasks.items()))
        for key,receipt in results:
            attempted[key]=receipt; receipts.append(receipt)
        for site,station in round_sites.items():
            sid=station.station_id
            raw=[]
            for year in sorted(site_weather[site].date.dt.year.unique()):
                key=(sid,int(year))
                if key not in station_frames and attempted[key]['status_code']==200:
                    station_frames[key]=parse_gsod(pd.read_csv(attempted[key]['path'],dtype={'STATION':str}))
                if key in station_frames:
                    raw.append(station_frames[key])
            obs=pd.concat(raw,ignore_index=True).drop_duplicates('date') if raw else pd.DataFrame({'date':pd.Series(dtype='datetime64[ns]')})
            expected=site_weather[site].date
            date_coverage=float(expected.isin(obs.date).mean()) if len(expected) else 0
            if date_coverage>=.8:
                matched[site]=(station,obs,date_coverage); pending.remove(site)
            else:
                failures[site]=f'nearest attempted station {sid}: daily record coverage {date_coverage:.3f}'
                ranks[site]+=1
                if ranks[site]>=len(candidates[site]):
                    pending.remove(site)
        print(json.dumps({'matched_sites':len(matched),'pending_sites':len(pending),'station_year_requests':len(attempted)}),flush=True)
    matches=[]; paired=[]; metrics=[]
    for site,g in windows.groupby('site_id'):
        base=dict(site_id=site,country=g.country.iloc[0],latitude=g.latitude.iloc[0],longitude=g.longitude.iloc[0],
                  target_dem_elevation_m=g.target_dem_elevation_m.iloc[0],elevation_status=g.elevation_status.iloc[0],
                  n_fields=len(g),expected_days=len(site_weather[site]),weather_start=str(site_weather[site].date.min().date()),
                  weather_end=str(site_weather[site].date.max().date()),candidate_count_50km=len(candidates[site]))
        if site not in matched:
            base.update(matched=False,unmatched_reason=failures.get(site,'no inventory-period-covering station within 50 km'))
            matches.append(base); continue
        station,obs,coverage=matched[site]
        base.update(matched=True,station_id=station.station_id,station_name=station['STATION NAME'],
                    station_latitude=station.LAT,station_longitude=station.LON,distance_km=station.distance_km,
                    station_elevation_m=station.station_elevation_m,high_elevation_caution=bool(station.high_elevation_caution),
                    elevation_difference_m=abs(station.station_elevation_m-base['target_dem_elevation_m']),
                    within_25km=bool(station.within_25km),inventory_begin=station.BEGIN,inventory_end=station.END,
                    station_record_coverage=coverage,station_rank=int(ranks[site])+1)
        matches.append(base)
        w=site_weather[site].rename(columns={'location_id':'site_id'})
        p=w.merge(obs,on='date',how='left',suffixes=('_forcing','_station'))
        p['station_id']=station.station_id; p['distance_km']=station.distance_km;p['country']=base['country']
        p['station_elevation_m']=station.station_elevation_m
        p['target_dem_elevation_m']=base['target_dem_elevation_m']
        p['high_elevation_caution']=bool(station.high_elevation_caution)
        paired.append(p)
        for v in VARIABLES:
            m=pair_metrics(pd.DataFrame({'forcing':p[v+'_forcing'],'station':p[v+'_station']}),len(p),v=='precipitation_mm')
            metrics.append(dict(site_id=site,station_id=station.station_id,country=base['country'],distance_km=station.distance_km,variable=v,**m))
    matches=pd.DataFrame(matches); matches.to_csv(out/'station_matches.csv',index=False)
    windows.merge(matches,on='site_id',suffixes=('','_station_match')).to_csv(out/'field_station_matches.csv',index=False)
    pairs=pd.concat(paired,ignore_index=True) if paired else pd.DataFrame()
    pairs.to_parquet(out/'paired_daily.parquet',index=False);pairs.to_csv(out/'paired_daily.csv',index=False)
    metrics=pd.DataFrame(metrics);metrics.to_csv(out/'site_variable_metrics.csv',index=False)
    pooled=[]
    for group_kind,group_col in [('all',None),('country','country'),('station','station_id'),('elevation_sensitivity','high_elevation_caution')]:
        grouped=[('all',pairs)] if group_col is None else pairs.groupby(group_col)
        for group,p in grouped:
            for v in VARIABLES:
                m=pair_metrics(pd.DataFrame({'forcing':p[v+'_forcing'],'station':p[v+'_station']}),len(p),v=='precipitation_mm')
                pooled.append(dict(group_kind=group_kind,group=str(group),variable=v,**m))
    pd.DataFrame(pooled).to_csv(out/'pooled_variable_metrics.csv',index=False)
    summary=dict(eligible_fields=len(windows),eligible_sites=windows.site_id.nunique(),
                 matched_fields=int(matches.loc[matches.matched,'n_fields'].sum()),matched_sites=len(matched),
                 matched_sites_within_25km=int(matches.within_25km.fillna(False).sum()),
                 paired_site_days=len(pairs),date_start=str(pairs.date.min().date()),date_end=str(pairs.date.max().date()),
                 station_year_requests=len(attempted),successful_station_years=sum(r['status_code']==200 for r in attempted.values()),
                 stations=int(pairs.station_id.nunique()),
                 high_elevation_station_sites=int(matches.high_elevation_caution.fillna(False).sum()),
                 selection='Nearest station within 50 km whose inventory period covers the site windows and whose available GSOD daily dates cover >=80% of required forcing days; no disease outcomes used.'+(' Station-target DEM elevation difference must be <=300 m; unknown DEM excluded.' if elevation_screen else ''),
                 elevation_screen=elevation_screen,
                 elevation_source='Open-Meteo elevation API; Copernicus DEM 2021 GLO-90, 90 m resolution. Rounded field coordinates make target DEM approximate.',
                 temperature_quality='Daily temperature and dewpoint means require >=18 reports; extremes retain valid source values with reporting-day caveat.',
                 precipitation_quality='Only full-period D/F/G totals; H/incomplete/unknown and 99.99 missing, never replaced by zero.',
                 adequate_metric_coverage='At least 30 paired days and >=70% of expected days.',
                 humidity='No observed mean RH is supplied by GSOD; dewpoint is compared directly. No RH derived from daily means.',
                 limitations=['NOAA-hosted GSOD is ISD-derived from synoptic/hourly records exchanged under WMO programmes, not WMO-hosted observations.',
                              'Daily summaries use GMT and operating hours; daily extrema and precipitation may include part of the preceding day.',
                              'Station and gridded field forcing differ in spatial support and siting; coordinates come from inventory metadata and do not resolve relocations.',
                              'Station elevations >=1500 m are flagged for representativeness caution and reported separately; field DEM uses rounded coordinates and cannot establish actual field elevation. No lapse-rate correction was applied.',
                              'Full-period precipitation is often unavailable; low-coverage metrics do not establish rainfall accuracy.',
                              'Station observations may have contributed to reanalysis assimilation; this is an external product check rather than guaranteed independent observation assimilation.',
                              'No weather correction was fitted or selected using disease test outcomes.'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    if elevation_screen:
        nearest=pd.read_csv(out.parent/'station_matches.csv',dtype={'station_id':str})
        comparison=matches[['site_id','matched','station_id']].merge(nearest[['site_id','matched','station_id']],on='site_id',suffixes=('_elevation_screen','_nearest'))
        comparison['changed']=comparison.station_id_elevation_screen.fillna('unmatched').ne(comparison.station_id_nearest.fillna('unmatched'))
        comparison.to_csv(out/'selection_changes.csv',index=False)
        summary['changed_station_or_match_sites']=int(comparison.changed.sum())
        (out/'summary.json').write_text(json.dumps(summary,indent=2))
    (out/'source_receipt.json').write_text(json.dumps(receipts,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--elevation-screen',action='store_true')
    args=parser.parse_args();run(args.root,args.elevation_screen)
