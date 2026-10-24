"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/winter_wheat_population_20261010')
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from analysis.methods.winter_crop_bounds.sources import parse_eurostat, national_margins, native_country_totals
HERE = evidence_dir('analysis/paper_study/winter_wheat_population_20261010')
ROOT = PROJECT_ROOT

def sha(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def raw_native_audit(root=ROOT, destination=HERE / 'inputs'):
    """Requires the separately downloadable, provider-verified native inputs."""
    identity = root / 'data/final_study/inputs/paper_study/wheat_area/spam2020V2r2_global_harvested_area.csv.zip'
    metadata = root / 'data/final_study/inputs/paper_study/wheat_area/spam2020_dataverse.json'
    production = root / 'data/final_study/inputs/paper_study/wheat_production/spam2020_V2r2_global_P_WHEA_A.tif'
    files = json.loads(metadata.read_text())['data']['latestVersion']['files']
    entry, = [x['dataFile'] for x in files if x['dataFile']['filename'] == identity.name]
    if sha(identity, 'md5') != entry['checksum']['value']:
        raise ValueError('Native country-identity provider hash mismatch')
    expected = '718c230babcd872bfeba1d5cec72b8e5944ec2fe90103b890fff2d33d095fac7'
    if sha(production) != expected:
        raise ValueError('Native production provider-verified hash mismatch')
    member = 'spam2020V2r2_global_harvested_area/spam2020V2r2_global_H_TA.csv'
    with zipfile.ZipFile(identity) as z:
        with z.open(member) as stream:
            native = pd.read_csv(stream, usecols=['grid_code', 'x', 'y', 'FIPS0', 'ADM0_NAME'])
    indices = np.floor((90 - native.y) * 12).astype(int) * 4320 + np.floor((native.x + 180) * 12).astype(int)
    if not np.array_equal(indices, native.grid_code):
        raise ValueError('Native cell codes disagree with raster coordinates')
    with Image.open(production) as image:
        tags = image.tag_v2
        spacing = tags[33550]
        tie = tags[33922]
        if image.size != (4320, 2160) or not np.allclose(spacing[:2], [1 / 12, 1 / 12]):
            raise ValueError('Unexpected native production raster')
        if not np.allclose(tie, (0, 0, 0, -180, 90, 0)):
            raise ValueError('Unexpected raster origin')
        totals, audit = native_country_totals(native, np.asarray(image), float(tags[42113]))
    destination.mkdir(parents=True, exist_ok=True)
    totals.to_csv(destination / 'native_country_totals.csv', index=False)
    audit.update(dict(source_doi='10.7910/DVN/SWPENT', identity_zip_sha256=sha(identity), identity_zip_provider_md5=entry['checksum']['value'], identity_member=member, identity_metadata_sha256=sha(metadata), production_raster_sha256=sha(production), table_sha256=sha(destination / 'native_country_totals.csv'), archive_scope='Country totals reproduce downstream bounds; raw national aggregation requires original provider inputs.', allowance_rule='All globally unassigned positive production is available outside each country; a conservative relaxation, not a jointly sharp global allocation.'))
    (destination / 'native_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    return (totals, audit)

def prepare(directory=HERE):
    inputs = directory / 'inputs'
    provenance = json.loads((inputs / 'retrieval.json').read_text())
    records = []
    for source in provenance:
        path = inputs / source['file']
        if sha(path) != source['sha256']:
            raise ValueError('Eurostat snapshot hash mismatch')
        records.append(parse_eurostat(json.loads(path.read_text())))
    records = pd.concat(records, ignore_index=True)
    protocol = json.loads((directory / 'protocol.json').read_text())
    crosswalk = pd.read_csv(inputs / 'country_crosswalk.csv', keep_default_na=False)
    margins = national_margins(records, crosswalk, years=tuple(protocol['reference_years']), tolerance_thousand_tonnes=protocol['annual_identity_tolerance_thousand_tonnes'])
    audit = json.loads((inputs / 'native_audit.json').read_text())
    if sha(inputs / 'native_country_totals.csv') != audit['table_sha256']:
        raise ValueError('Native country table hash mismatch')
    totals = pd.read_csv(inputs / 'native_country_totals.csv')
    table = margins.merge(totals, on=['country', 'country_fips'], how='left', validate='one_to_one')
    if table.mapped_national_total_tonnes.isna().any():
        raise ValueError('Missing full-country native production')
    table['unassigned_outside_allowance_tonnes'] = audit['unassigned_positive_production_tonnes']
    table['total_tonnes'] = table.mapped_national_total_tonnes + table.unassigned_outside_allowance_tonnes
    table['winter_lower_tonnes'] = table.total_tonnes * table.fraction_lower
    table['winter_upper_tonnes'] = table.total_tonnes * table.fraction_upper
    out = directory / 'results'
    out.mkdir(exist_ok=True)
    records.to_csv(out / 'eurostat_wheat_records.csv', index=False)
    table.to_csv(out / 'national_margins.csv', index=False)
    return table
