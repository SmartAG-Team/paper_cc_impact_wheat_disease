"""Read-only Earth Engine retrieval of regional MODIS canopy history."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ROOT/'data/disease_improvement/station_preferred_ie_20261022/fields.csv'
OBS = ROOT/'data/final_monitoring/cohort/training_observation_ledger.csv'
COLLECTION = 'MODIS/061/MCD15A3H'
CROPLAND = 'ESA/GLOBCOVER_L4_200901_200912_V2_3'


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquisition_months(dates):
    days = [datetime.fromisoformat(day) for day in dates]
    if not days:
        raise ValueError('Assessment dates required')
    current = (min(days)-timedelta(days=100)).replace(day=1)
    final = max(days).replace(day=1)
    months = []
    while current <= final:
        nxt = current.replace(year=current.year+1,month=1) if current.month==12 else current.replace(month=current.month+1)
        months.append((current.date().isoformat(),nxt.date().isoformat())); current=nxt
    return months


def gather(output=None):
    import ee
    from importlib.metadata import version
    out = Path(output or ROOT/'data/disease_improvement/canopy_regional_v2_20261022').resolve()
    if out.exists():
        raise FileExistsError('Fresh retrieval destination required')
    out.mkdir(parents=True); (out/'raw').mkdir()
    with FIELDS.open() as handle:
        fields = list(csv.DictReader(handle))
    sites = {row['site_id']:(float(row['longitude']),float(row['latitude'])) for row in fields}
    with OBS.open() as handle:
        months = acquisition_months([row['date'] for row in csv.DictReader(handle)])
    codes = [Path(__file__),Path(__file__).with_name('canopy_inputs.py')]
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        collection=COLLECTION,cropland=CROPLAND,earthengine_api=version('earthengine-api'),
        source_sha256={str(p.relative_to(ROOT)):_sha(p) for p in [FIELDS,OBS]},
        code_sha256={str(p.relative_to(ROOT)):_sha(p) for p in codes},
        sites=len(sites),months=months,buffer_metres=10000,scale_metres=500,
        cropland_classes=[11,14],cropland_interpretation='Static2009irrigated/rainfedcropland, releasedDec2010; regionalproxy, not current wheat/field/cultivar identity',
        quality='MODLANDgood, no dead detector, cloudstateclear, mainSCF0or1;extraQAland/no snow,aerosol,cirrus,internalcloud,shadow;allLai/Fpar/retrievalSDrawvalues0..100',
        feature_requirement='Complete system:time_end before query, plus fixed7dayassumedavailabilitylag; no future composites/smoothing; operational latency unverified',
        information='Only public site geometries and date extent submitted; no disease severities, histories or other target labels uploaded',
        source_urls=['https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MCD15A3H',
            'https://ladsweb.modaps.eosdis.nasa.gov/filespec/MODIS/61/MCD15A3H_c61.fs',
            'https://developers.google.com/earth-engine/datasets/catalog/ESA_GLOBCOVER_L4_200901_200912_V2_3',
            'https://www.esa.int/Applications/Observing_the_Earth/Envisat/ESA_unveils_latest_map_of_world_s_land_cover'],
        attribution='NASA LP DAAC MODIS; ESA GlobCover2009 Project and Universite catholique de Louvain',
        disease_model_fit=False)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for code in codes:
        dest=out/'code_snapshot'/code.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code,dest)
    ee.Initialize(project='ee-gangzhaomodel')
    collection = ee.ImageCollection(COLLECTION)
    projection = ee.Image(collection.first()).select('Lai').projection()
    crop = ee.Image(CROPLAND).select('landcover').remap([11,14],[1,1],0).rename('crop').reproject(projection)
    regions = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(list(coordinate)).buffer(10000),
        dict(site_id=site)) for site,coordinate in sorted(sites.items())])
    statics = crop.reduceRegions(regions,ee.Reducer.mean().combine(ee.Reducer.sum(),sharedInputs=True),
        scale=500,crs=projection,tileScale=4).map(lambda f:f.setGeometry(None)).getInfo()
    (out/'raw/crop_2009.json').write_text(json.dumps(statics,sort_keys=True)+'\n')
    crop_by_site = {f['properties']['site_id']:f['properties'] for f in statics['features']}
    assert len(crop_by_site)==len(sites)

    def reduce_image(item):
        image = ee.Image(item); qc=image.select('FparLai_QC');extra=image.select('FparExtra_QC')
        good = qc.bitwiseAnd(1).eq(0).And(qc.bitwiseAnd(4).eq(0))
        good = good.And(qc.rightShift(3).bitwiseAnd(3).eq(0)).And(qc.rightShift(5).bitwiseAnd(7).lte(1))
        good = good.And(extra.bitwiseAnd(3).eq(0)).And(extra.bitwiseAnd(124).eq(0))
        raw = image.select(['Lai','Fpar','LaiStdDev','FparStdDev'])
        good = good.And(raw.gte(0).And(raw.lte(100)).reduce(ee.Reducer.min()))
        values = raw.multiply(ee.Image.constant([.1,.01,.1,.01])).rename(
            ['lai','fpar','lai_uncertainty','fpar_uncertainty']).updateMask(good).updateMask(crop)
        values = values.addBands(good.unmask(0).toFloat().rename('valid').updateMask(crop))
        reducer = ee.Reducer.mean().combine(ee.Reducer.stdDev(),sharedInputs=True).combine(ee.Reducer.count(),sharedInputs=True)
        reduced = values.reduceRegions(regions,reducer,scale=500,crs=projection,tileScale=4)
        metadata = dict(image_index=image.get('system:index'),period_start=image.get('system:time_start'),
            period_end=image.get('system:time_end'))
        return reduced.map(lambda f:f.set(metadata).setGeometry(None))

    def fetch(interval):
        start,end=interval;name=start[:7].replace('-','_')
        selected=collection.filterDate(start,end)
        feature_collection=ee.FeatureCollection(selected.toList(10).map(reduce_image)).flatten()
        for attempt in range(3):
            try:
                payload=feature_collection.getInfo();break
            except Exception:
                if attempt==2:raise
                time.sleep(5*(attempt+1))
        (out/'raw'/f'{name}.json').write_text(json.dumps(payload,sort_keys=True)+'\n')
        rows=[]
        for feature in payload['features']:
            row=feature['properties'];site=row['site_id']
            row['crop_fraction']=crop_by_site[site].get('mean',crop_by_site[site].get('crop_mean'))
            row['crop_pixel_equivalent_count']=crop_by_site[site].get('sum',crop_by_site[site].get('crop_sum'))
            assert row.get('period_end') is not None
            rows.append(row)
        print('retrieved',name,'rows',len(rows),flush=True)
        return rows

    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for future in as_completed([pool.submit(fetch,interval) for interval in months]):
            rows.extend(future.result())
    assert len({(r['site_id'],r['image_index']) for r in rows})==len(rows)
    rows.sort(key=lambda r:(r['site_id'],r['period_end']))
    keys=sorted({k for row in rows for k in row})
    with (out/'regional_canopy.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    verification=dict(passed=True,rows=len(rows),sites=len(sites),months=len(months),
        disease_model_fit=False,source_unchanged=all(_sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        artifacts_sha256={str(p.relative_to(out)):_sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print('retrieval_complete',len(rows),len(sites),'sites',flush=True)
    return out


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    gather(args.output)
