"""Offline verification of original SPAM files and current full-grid lineage."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .exposure import METRIC, PRODUCTION, require, _same

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):
            h.update(chunk)
    return h.hexdigest()


def load_verified_inputs(root=ROOT):
    from PIL import Image
    root = Path(root)
    prod = root/'data/paper_study/wheat_production'
    area = root/'data/paper_study/wheat_area'
    current = root/'analysis/paper_study/full_grid_climate_20261008/results'
    aggregation = json.loads((prod/'aggregation_receipt.json').read_text())
    download = json.loads((prod/'download_receipt.json').read_text())
    completion = json.loads((current/'completion_receipt.json').read_text())
    hashes = {}
    def verify(path, expected=None):
        actual = digest(path)
        if expected is not None:
            require(actual==expected,f'Checksum mismatch: {path}')
        hashes[str(path.relative_to(root))] = actual
    require(completion['status']=='complete' and completion['all_planned_periods_complete'], 'Incomplete full grid')
    for name in ['full_grid_paired_changes_by_gcm.parquet','full_grid_ensemble_paired_changes.parquet','full_landuse_cell_registry.csv']:
        verify(current/name,completion['outputs_sha256'][name])
    verify(current.parent/'summarize.py',completion['summary_code_sha256'])
    verify(current/'completion_receipt.json')
    verify(prod/'aggregation_receipt.json')
    verify(prod/'download_receipt.json',aggregation['production_source_receipt_sha256'])
    for path,expected in aggregation['output_sha256'].items():
        verify(root/path,expected)
    verify(area/'spam2020_dataverse.json',download['metadata_sha256'])
    metadata = json.loads((area/'spam2020_dataverse.json').read_text())
    archive = download['source_archives'][0]
    archive_path = prod/archive['file']
    entry = next(x['dataFile'] for x in metadata['data']['latestVersion']['files']
                 if x['dataFile']['filename']==archive['file'])
    require(archive_path.stat().st_size==entry['filesize'], 'SPAM archive size differs from Dataverse')
    require(digest(archive_path,'md5')==entry['md5']==archive['md5'],'SPAM archive MD5 mismatch')
    verify(archive_path,archive['sha256'])
    source = pd.read_parquet(prod/'europe_source_wheat_production.parquet')
    raster_checks = []
    for col,tech in zip(PRODUCTION,['A','I','R']):
        path = prod/f'spam2020_V2r2_global_P_WHEA_{tech}.tif'
        verify(path,aggregation['source_raster_sha256'][str(path.relative_to(root))])
        with Image.open(path) as im:
            require(im.size==(4320,2160),'Unexpected raster dimensions')
            scale = im.tag_v2[33550]
            origin = im.tag_v2[33922]
            _same(scale[:2],[1/12,1/12],'Unexpected native resolution',atol=1e-10)
            _same(origin[3:5],[-180,90],'Unexpected native raster origin',atol=1e-10)
            nodata = float(im.tag_v2[42113])
            values = np.asarray(im)[source.source_row.to_numpy(),source.source_col.to_numpy()].astype(float)
        valid = np.isfinite(values)&(values!=nodata)
        require(np.array_equal(valid,source[col+'_source_raster_valid']),'Raster validity differs from native table')
        normalized = np.where(valid,values,0.)
        _same(normalized,source[col],f'Raw native production mismatch: {tech}',atol=0.)
        raster_checks.append(dict(technology=tech,native_pixels=len(source),valid_pixels=int(valid.sum()),
                                  max_abs_raw_difference=float(np.max(np.abs(normalized-source[col])))))
    inputs = dict(registry=pd.read_csv(current/'full_landuse_cell_registry.csv'),
                  production=pd.read_parquet(prod/'europe_wheat_production_025.parquet'),source=source,
                  country=pd.read_parquet(prod/'europe_cell_country_wheat_production.parquet'),
                  paired=pd.read_parquet(current/'full_grid_paired_changes_by_gcm.parquet',filters=[('metric','==',METRIC)]),
                  ensemble=pd.read_parquet(current/'full_grid_ensemble_paired_changes.parquet',filters=[('metric','==',METRIC)]))
    receipt = dict(source_hashes=hashes,raw_raster_checks=raster_checks,
                   provider='IFPRI SPAM2020 V2r2; Harvard Dataverse version 6',
                   dataset_doi='https://doi.org/10.7910/DVN/SWPENT',reference_year=2020,
                   unit='metric tonnes',unit_definition_url='https://www.mapspam.info/methodology/',
                   unit_definition_verified_utc_date='2026-10-09',
                   unit_evidence='Provider methodology, Production (P): production is measured in metric tons.',
                   country_assignment='Native source ADM0_NAME/FIPS0 production multiplied once by Europe fraction, then summed within current cells; no dominant-country assignment.',
                   europe_scope='Fixed operational European wheat-area grid. Border-country values describe the included European portion, not necessarily national totals.',
                   baseline_scope='SPAM all wheat in the fixed 2020 land-use support; winter/spring production unresolved.',
                   provenance_limit='Native pixel support and Europe mask inherited from the archived positive-area registry; production outside that support is not extrapolated.')
    return inputs, receipt
