"""Qualify official French hourly observations without disease-based selection."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .dwd_station import join_station_daily, pair_metrics
from .evaluation import distances_km
from .inputs import WEATHER, sha

ROOT = Path(__file__).resolve().parents[2]
CACHE = 'data/external_weather_validation_20261022/meteo_france'
OUTPUT = 'outputs/weather_validation_20261022/meteo_france'


def parse_hourly(raw):
    frame = raw.copy()
    frame['station_id'] = frame.NUM_POSTE.astype(str).str.zfill(8)
    frame['time'] = pd.to_datetime(frame.AAAAMMJJHH.astype(str), format='%Y%m%d%H', errors='raise')
    if frame.duplicated(['station_id','time']).any():
        raise ValueError('Duplicate station/hour observations')
    for name, lower, upper in [('RR1',0,500),('T',-80,60),('U',0,100),
                               ('DHUMEC',0,60),('DRR1',0,60),('WW',0,99)]:
        if name not in frame:
            frame[name] = np.nan
        value = pd.to_numeric(frame[name], errors='coerce')
        quality = pd.to_numeric(frame.get('Q'+name, pd.Series(np.nan,index=frame.index)), errors='coerce')
        frame[name] = value.where(value.between(lower,upper)&quality.isin([0,1]))
    return frame.sort_values(['station_id','time']).reset_index(drop=True)


def daily_aggregate(hourly):
    if hourly.station_id.nunique() != 1:
        raise ValueError('One station per daily aggregation required')
    instant = hourly.assign(date=hourly.time.dt.normalize())
    accum = hourly.assign(date=(hourly.time-pd.Timedelta(hours=1)).dt.normalize())
    g = instant.groupby('date')
    out = g.agg(tmean_c=('T','mean'),tmin_c=('T','min'),tmax_c=('T','max'),
        rh_mean_pct=('U','mean'),temperature_hours=('T','count'),humidity_hours=('U','count'))
    humid = instant.assign(humid=instant.U.ge(90).where(instant.U.notna()))
    out['rh_hours_ge_90pct'] = pd.to_numeric(humid.groupby('date').humid.sum(min_count=24))
    out.loc[out.temperature_hours.ne(24),['tmean_c','tmin_c','tmax_c']] = np.nan
    out.loc[out.humidity_hours.ne(24),['rh_mean_pct','rh_hours_ge_90pct']] = np.nan
    rain = accum.groupby('date').agg(precipitation_mm=('RR1',lambda a:a.sum(min_count=24)),
                                     rain_hours_available=('RR1','count'))
    wet = accum.assign(wet=accum.RR1.gt(.1).where(accum.RR1.notna()))
    rain['station_precip_hours_gt_0_1mm'] = pd.to_numeric(wet.groupby('date').wet.sum(min_count=24))
    rain.loc[rain.rain_hours_available.ne(24),['precipitation_mm','station_precip_hours_gt_0_1mm']] = np.nan
    # WMO present-weather codes with frozen/mixed precipitation. Missing codes
    # cannot establish that solid precipitation was absent.
    solid = accum.WW.isin([23,24,26,27,56,57,66,67,68,69,70,71,72,73,74,75,76,77,78,79,83,84,85,86,87,88,89,90])
    rain['solid_precipitation_observed'] = solid.groupby(accum.date).any()
    rain['present_weather_hours'] = accum.groupby('date').WW.count()
    out = out.join(rain,how='outer').reset_index()
    out['station_id'] = hourly.station_id.iloc[0]
    return out


def candidates(root):
    root = Path(root)
    metadata = json.loads((root/CACHE/'station_metadata.geojson').read_text())
    fields = pd.read_csv(root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv')
    sites = fields.loc[fields.country.eq('FRANCE'),['site_id','latitude','longitude']].drop_duplicates()
    elevation = pd.read_csv(root/'outputs/weather_validation_20261022/noaa/field_elevation.csv')
    sites = sites.merge(elevation[['site_id','target_dem_elevation_m']],on='site_id',validate='one_to_one')
    forcing = pd.read_parquet(root/WEATHER)
    forcing['date'] = pd.to_datetime(forcing.date)
    rows = []
    for site in sites.itertuples(index=False):
        dates = forcing.loc[forcing.location_id.eq(site.site_id),'date']
        start,end = str(dates.min()),str(dates.max())
        eligible = []
        for station in metadata:
            def covers(item):
                return item['dateDebut'] <= start and (not item['dateFin'] or item['dateFin'] >= end)
            if not any(x['nom']=='HAUTEUR DE PRECIPITATIONS HORAIRE' and covers(x) for x in station.get('parametres',[])):
                continue
            positions = [x for x in station.get('positions',[]) if covers(x)]
            if len(positions) != 1 or not np.isfinite(site.target_dem_elevation_m):
                continue
            position = positions[0]
            distance = float(distances_km([[site.latitude,site.longitude]],[[position['latitude'],position['longitude']]])[0,0])
            delta = abs(position['altitude']-site.target_dem_elevation_m)
            if distance > 50 or delta > 300:
                continue
            eligible.append(dict(site_id=site.site_id,station_id=station['id'],department=station['id'][:2],
                station_name=station['nom'],latitude=position['latitude'],longitude=position['longitude'],
                station_elevation_m=position['altitude'],target_dem_elevation_m=site.target_dem_elevation_m,
                distance_km=distance,elevation_difference_m=delta,start=start,end=end))
        rows.extend(sorted(eligible,key=lambda a:(a['distance_km'],a['station_id']))[:3])
    return pd.DataFrame(rows), forcing


def run(root=ROOT,output=None):
    root = Path(root)
    cache,out = root/CACHE,Path(output or root/OUTPUT)
    if (out/'receipt.json').exists():
        raise FileExistsError('Use a new output directory to preserve the archived collection')
    out.mkdir(parents=True,exist_ok=True)
    matches,forcing = candidates(root)
    matches.to_csv(out/'station_candidates.csv',index=False)
    selected_ids = set(matches.station_id)
    resources = json.loads((cache/'selected_resources.json').read_text())
    tables,quality,receipts = [],[],[]
    for resource in resources:
        path = cache/(resource['title']+'.csv.gz')
        receipt = json.loads(path.with_suffix(path.suffix+'.receipt.json').read_text())
        if receipt['sha256'] != sha(path):
            raise ValueError('Source hash mismatch: '+str(path))
        receipts.append(receipt)
        chunks = []
        columns = ['NUM_POSTE','AAAAMMJJHH','RR1','QRR1','T','QT','U','QU','WW','QWW','DHUMEC','QDHUMEC','DRR1','QDRR1']
        for block in pd.read_csv(path,sep=';',usecols=lambda c:c in columns,
                                 dtype={'NUM_POSTE':str,'AAAAMMJJHH':str},chunksize=150000):
            use = block.NUM_POSTE.isin(selected_ids)&block.AAAAMMJJHH.between('2013010100','2019123123')
            if use.any():
                chunks.append(block.loc[use])
        if not chunks:
            continue
        raw = pd.concat(chunks,ignore_index=True)
        for station,part in raw.groupby('NUM_POSTE'):
            for name in ['RR1','T','U','DHUMEC','DRR1']:
                if name not in part:
                    continue
                counts = part['Q'+name].value_counts(dropna=False)
                for code,count in counts.items():
                    quality.append(dict(station_id=station,variable=name,quality_code=code,n=int(count)))
            hourly = parse_hourly(part)
            tables.append(daily_aggregate(hourly))
            hourly.to_parquet(out/f'{station}_hourly.parquet',index=False)
        print(resource['title'],len(raw),'selected hourly rows',flush=True)
    daily = pd.concat(tables,ignore_index=True)
    daily.to_parquet(out/'station_daily.parquet',index=False)
    pd.DataFrame(quality).to_csv(out/'quality_counts.csv',index=False)
    coverage = []
    accepted,pairs = [],[]
    for site,options in matches.groupby('site_id',sort=False):
        weather = forcing.loc[forcing.location_id.eq(site)].copy()
        chosen = None
        for option in options.itertuples(index=False):
            observed = daily.loc[daily.station_id.eq(option.station_id)].drop(columns='station_id')
            pair = join_station_daily(weather,observed)
            complete = int(pair.precipitation_mm_station.notna().sum())
            fraction = complete/len(pair)
            coverage.append(dict(site_id=site,station_id=option.station_id,complete_rain_days=complete,
                                 required_days=len(pair),coverage=fraction,distance_km=option.distance_km))
            if chosen is None and fraction >= .8:
                chosen = option
                pair['site_id'] = site
                pair['station_id_RR'] = option.station_id
                pair['distance_km_RR'] = option.distance_km
                pairs.append(pair)
        if chosen is not None:
            accepted.append(chosen._asdict())
    paired = pd.concat(pairs,ignore_index=True)
    paired.to_parquet(out/'paired_daily.parquet',index=False)
    pd.DataFrame(accepted).to_csv(out/'station_matches.csv',index=False)
    pd.DataFrame(coverage).to_csv(out/'candidate_coverage.csv',index=False)
    metrics = pair_metrics(paired)
    metrics.to_csv(out/'station_metrics.csv',index=False)
    aggregate = paired.copy();aggregate['site_id'] = 'ALL'
    pair_metrics(aggregate).to_csv(out/'aggregate_metrics.csv',index=False)
    inputs = [cache/'catalog.json',cache/'station_catalog.json',cache/'station_metadata.geojson',
              cache/'documentation.pdf',cache/'field_definitions.csv',cache/'selected_resources.json',root/WEATHER,
              root/'outputs/disease_improvement/quality_weather_20261021_v2/field_metadata_ledger.csv',
              root/'outputs/weather_validation_20261022/noaa/field_elevation.csv']
    receipt = dict(retrieved_utc=datetime.now(timezone.utc).isoformat(),source='Météo-France validated historical hourly archive',
        source_sha256={str(p.relative_to(root)):sha(p) for p in inputs},archives=receipts,code_sha256=sha(Path(__file__)),
        valid_quality_codes=[0,1],excluded_quality_codes=[2,9],matched_sites=len(accepted),sites_considered=24,
        complete_precipitation_pairs=int(paired.precipitation_mm_station.notna().sum()),
        station_selection='Nearest of at most three eligible historical stations with >=80% complete rain-day coverage; <=50km and <=300m DEM difference; no disease outcomes',
        rainfall_timing='RR1 interpreted as hour ending at timestamp; independent provider daily-total reconciliation required before substitution',
        rainfall_approved_for_substitution=False,real_time_latency_verified=False)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    print(json.dumps(run(output=args.output),indent=2))
