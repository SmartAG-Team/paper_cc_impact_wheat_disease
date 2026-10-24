"""Read public MODIS night skin temperatures; upload only site/date geometry."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import csv
from datetime import datetime,timedelta,timezone
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[2]
FIELDS=ROOT/'data/disease_improvement/station_preferred_ie_20261022/fields.csv'
QUERIES=ROOT/'data/disease_improvement/phenology_alignment_20261022/query_provenance.csv'
CROPLAND='ESA/GLOBCOVER_L4_200901_200912_V2_3'
COLLECTIONS=dict(terra='MODIS/061/MOD11A1',aqua='MODIS/061/MYD11A1')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def acquisition_plan(queries,sites_by_field):
    plan={}
    for row in queries:
        now=datetime.fromisoformat(row['query_date']);current=(now-timedelta(days=100)).replace(day=1)
        final=now.replace(day=1)
        while current<=final:
            key=current.date().isoformat();plan.setdefault(key,set()).add(sites_by_field[row['field_id']])
            current=current.replace(year=current.year+1,month=1) if current.month==12 else current.replace(month=current.month+1)
    return {key:sorted(value) for key,value in sorted(plan.items())}


def gather():
    import ee
    from importlib.metadata import version
    out=ROOT/'data/disease_improvement/night_lst_regional_20261022'
    if out.exists():raise FileExistsError('Fresh night-temperature retrieval destination required')
    out.mkdir(parents=True);(out/'raw').mkdir()
    with FIELDS.open() as handle:fields=list(csv.DictReader(handle))
    with QUERIES.open() as handle:queries=list(csv.DictReader(handle))
    sites={r['site_id']:[float(r['longitude']),float(r['latitude'])] for r in fields}
    plan=acquisition_plan(queries,{r['field_id']:r['site_id'] for r in fields})
    codes=[Path(__file__),Path(__file__).with_name('night_lst_inputs.py')]
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),collections=COLLECTIONS,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [FIELDS,QUERIES]},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        earthengine_api=version('earthengine-api'),acquisition_plan=plan,
        regions='10km public rounded-coordinate buffers; nearest-cell2009GlobCover11/14crop mask on MODIS native projection; reduction1000m; regional land-surface mixture, not identified wheat fields',
        quality='MandatoryQA0OR1 with detailed dataquality0,emissivityerrorclass<=1(<=.02),LSTerrorclass<=1(<=2K); rawtemperature7500..65535,viewtime0..240,viewangle0..130; shared qualification for all retrieval channels',
        correction='Initial strict mandatoryQA0 feasibility gave zero accepted pixels; independent raw QC65 and official detailed-QA definition support QA1 only with above detailed gates. No disease outcomes inspected to choose QA.',
        time='Explicit complete daily system:time_start/end retained; planned features use end+2days assumed availability; historical real publication latency unverified, product is retrospectively reprocessed',
        information='Only public site geometry and month ranges submitted; query metadata file contains no disease severity. No target values, current-season disease histories or fitted models uploaded.',
        interpretation='Clear-sky skin temperature, neither leaf wetness nor proof condensation; cloud-selected availability and spatial mixtures explicit',
        source_urls=['https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MOD11A1',
                     'https://developers.google.com/earth-engine/datasets/catalog/MODIS_061_MYD11A1',
                     'https://www.esa.int/Applications/Observing_the_Earth/Envisat/ESA_unveils_latest_map_of_world_s_land_cover'],
        attribution='NASA LP DAAC MODIS; ESA GlobCover2009 Project and Universite catholique de Louvain',
        disease_model_fit=False)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for code in codes:
        dest=out/'code_snapshot'/code.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code,dest)
    ee.Initialize(project='ee-gangzhaomodel')
    projection=ee.Image(ee.ImageCollection(COLLECTIONS['terra']).first()).select('LST_Night_1km').projection()
    crop=ee.Image(CROPLAND).select('landcover').remap([11,14],[1,1],0).rename('crop').reproject(projection)
    all_regions=ee.FeatureCollection([ee.Feature(ee.Geometry.Point(coordinate).buffer(10000),dict(site_id=site))
                                      for site,coordinate in sorted(sites.items())])
    static=crop.reduceRegions(all_regions,ee.Reducer.mean().combine(ee.Reducer.sum(),sharedInputs=True),
        scale=1000,crs=projection,tileScale=4).map(lambda f:f.setGeometry(None)).getInfo()
    (out/'raw/cropland2009.json').write_text(json.dumps(static,sort_keys=True)+'\n')

    def fetch(sensor,month,active):
        regions=all_regions.filter(ee.Filter.inList('site_id',active))
        start=datetime.fromisoformat(month)
        end=start.replace(year=start.year+1,month=1) if start.month==12 else start.replace(month=start.month+1)
        selected=ee.ImageCollection(COLLECTIONS[sensor]).filterDate(month,end.date().isoformat())

        def reduce_image(item):
            image=ee.Image(item);q=image.select('QC_Night');raw=image.select('LST_Night_1km')
            view=image.select('Night_view_time');angle=image.select('Night_view_angle')
            good=q.bitwiseAnd(3).lte(1).And(q.rightShift(2).bitwiseAnd(3).eq(0))
            good=good.And(q.rightShift(4).bitwiseAnd(3).lte(1)).And(q.rightShift(6).bitwiseAnd(3).lte(1))
            good=good.And(raw.gte(7500)).And(raw.lte(65535)).And(view.gte(0)).And(view.lte(240))
            good=good.And(angle.gte(0)).And(angle.lte(130))
            values=raw.multiply(.02).subtract(273.15).rename('night_lst_c')
            values=values.addBands(view.multiply(.1).rename('view_hour'))
            values=values.addBands(angle.subtract(65).abs().rename('view_angle_abs')).updateMask(good).updateMask(crop)
            values=values.addBands(good.unmask(0).toFloat().rename('valid').updateMask(crop))
            reducer=ee.Reducer.mean().combine(ee.Reducer.stdDev(),sharedInputs=True).combine(ee.Reducer.count(),sharedInputs=True)
            reduced=values.reduceRegions(regions,reducer,scale=1000,crs=projection,tileScale=4)
            metadata=dict(sensor=sensor,image_index=image.get('system:index'),period_start=image.get('system:time_start'),period_end=image.get('system:time_end'))
            return reduced.map(lambda f:f.set(metadata).setGeometry(None))

        collection=ee.FeatureCollection(selected.toList(32).map(reduce_image)).flatten()
        for attempt in range(3):
            try:payload=collection.getInfo();break
            except Exception:
                if attempt==2:raise
                time.sleep(5*(attempt+1))
        filename=f'{sensor}_{month[:7].replace("-","_")}.json'
        (out/'raw'/filename).write_text(json.dumps(payload,sort_keys=True)+'\n')
        rows=[f['properties'] for f in payload['features']]
        assert all(r.get('period_end') is not None and r['period_end']-r['period_start']==86400000 for r in rows)
        print('retrieved',sensor,month,'sites',len(active),'rows',len(rows),flush=True)
        return rows

    rows=[]
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(fetch,sensor,month,active) for sensor in COLLECTIONS for month,active in plan.items()]
        for future in as_completed(futures):rows.extend(future.result())
    assert len({(r['site_id'],r['sensor'],r['image_index']) for r in rows})==len(rows)
    rows.sort(key=lambda r:(r['site_id'],r['sensor'],r['period_start']))
    keys=sorted({k for r in rows for k in r})
    with (out/'regional_night_lst.csv').open('w') as handle:
        writer=csv.DictWriter(handle,fieldnames=keys);writer.writeheader();writer.writerows(rows)
    result=dict(passed=True,rows=len(rows),sites=len(sites),sensor_months=2*len(plan),
        source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        disease_model_fit=False,artifacts_sha256={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print('retrieval_complete',len(rows),'rows',flush=True)


if __name__=='__main__':gather()
