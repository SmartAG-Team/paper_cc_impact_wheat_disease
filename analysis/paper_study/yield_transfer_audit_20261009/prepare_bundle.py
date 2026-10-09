"""Optional bundle refresh from the local acquisition; audit.py uses only the ZIP.

All writes are confined to this directory. No download, source mutation, or model fit.
"""
from pathlib import Path
import hashlib
import io
import json
import zipfile

import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NEW = ROOT / 'data/STB_paper_followup_20261009'
PARALLEL = ROOT / 'data/public_septoria/paper-followup-20261009'


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    members, inventory = {}, []

    def add(path, member, role, doi='', license_id='CC-BY-4.0'):
        data = path.read_bytes()
        members[member] = data
        inventory.append(dict(original_path=str(path.relative_to(ROOT)), bundle_member=member,
                              role=role, bytes=len(data), sha256=digest(data),
                              source_doi=doi, license=license_id, included=True))
        return data

    srcnames = ['swiss-mixtures/README.txt', 'swiss-mixtures/blendit_full_pub.csv',
                'swiss-mixtures-v1/blendit_full.csv', 'zenodo_17432866.json',
                'zenodo_10209176.json', 'zenodo_5393959.json', 'acquisition.json']
    srcnames += ['montazeaud2022/'+n+'.csv' for n in
                 ['STB_symptoms','STB_RAW_RYT','GY_RAW_RYT','raw_yield_variables','Traits_monocultures']]
    for name in srcnames:
        doi = ('10.5281/zenodo.5393959' if 'montazeaud' in name or '5393959' in name else
               '10.5281/zenodo.10209176' if '-v1/' in name or '10209176' in name else
               '10.5281/zenodo.17432866')
        add(NEW/'sources'/name, 'sources/'+name, 'original_public_source', doi)
        alias = PARALLEL / name
        if alias.exists():
            data = alias.read_bytes()
            inventory.append(dict(original_path=str(alias.relative_to(ROOT)),
                                  bundle_member='sources/'+name, role='parallel_copy',
                                  bytes=len(data), sha256=digest(data), source_doi=doi,
                                  license='CC-BY-4.0', included=False))
    for path in sorted((NEW/'analysis_ready').glob('*.csv')):
        add(path, 'analysis_ready/'+path.name, 'local_derived_extract',
            license_id='Derived from CC-BY-4.0 public data; original attribution retained')
    for name in ['START_HERE.txt','new_data_dictionary.csv','verification.json','build_followup.py']:
        add(NEW/name, 'context/'+name, 'local_provenance', license_id='Local provenance; no new licence assigned')

    archive = PARALLEL/'montazeaud2022/Montazeaud_et_al_allelic_mixtures.zip'
    data = archive.read_bytes()
    acquisition = json.loads(members['sources/acquisition.json'])
    expected = next(x for x in acquisition if x['file'].endswith('.zip'))
    if digest(data) != expected['sha256']:
        raise ValueError('French source archive differs from acquisition SHA-256')
    meta = json.loads(members['sources/zenodo_5393959.json'])
    md5 = meta['files'][0]['checksum'].removeprefix('md5:')
    if hashlib.md5(data).hexdigest() != md5:
        raise ValueError('French archive differs from retained Zenodo MD5')
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if z.testzip() is not None:
            raise ValueError('French archive integrity failed')
        names = [n for n in z.namelist() if n.endswith((
            'Spatial_analyses_yield_variables.R','Allelic_richness_phenotypic_file_prep.R'))]
        for n in names:
            members['source_code/'+Path(n).name] = z.read(n)
        for name in [n for n in srcnames if n.startswith('montazeaud')]:
            found = [n for n in z.namelist() if Path(n).name == Path(name).name]
            if len(found) != 1 or z.read(found[0]) != members['sources/'+name]:
                raise ValueError('French compact source differs from ZIP member: '+name)
    members['context/french_archive_verification.json'] = json.dumps({
        'source_doi': '10.5281/zenodo.5393959', 'original_archive_path': str(archive.relative_to(ROOT)),
        'sha256': digest(data), 'bytes':len(data), 'repository_md5':md5,
        'retained_repository_checksum_matches':True, 'zip_crc_passed':True,
        'phenotype_members_byte_identical':True, 'selected_r_scripts':names,
        'scope':'Local verification against retained acquisition/Zenodo metadata; no fresh remote check'
    },indent=2).encode()

    # Previous benchmarks remain frozen: grain diagnostics plus the existing receipts.
    prior_dir = ROOT/'data/stb_collection_20261009/analysis_ready'
    assessments = pd.read_csv(prior_dir/'combined_severity_yield_assessments.csv')
    units = pd.read_csv(prior_dir/'combined_yield_units.csv')
    prior = {'source_hashes':{}, 'grain':{}, 'benchmark_receipts':{}}
    for name in ['combined_severity_yield_assessments.csv','combined_yield_units.csv']:
        p = prior_dir/name
        prior['source_hashes'][str(p.relative_to(ROOT))] = digest(p.read_bytes())
    for name, group in assessments.groupby('dataset_id'):
        u = units[units.dataset_id.eq(name)]
        prior['grain'][name] = dict(assessment_rows=len(group), harvest_units=group.yield_unit_id.nunique(),
            repeated_harvest_representations=len(group)-group.yield_unit_id.nunique(),
            conflicting_yields=int(group.groupby('yield_unit_id').yield_t_ha.nunique().gt(1).sum()),
            source_flagged_alias_units=int(u.source_duplicate_control.sum()),
            environments=group.environment_id.nunique())
    for name in ['crop_damage_validation_20261009/analysis_receipt.json',
                 'crop_damage_validation_20261009/briwecs_receipt.json',
                 'nordic_yield_validation_20261009/analysis_receipt.json']:
        p = ROOT/'analysis/paper_study'/name
        prior['benchmark_receipts'][name] = json.loads(p.read_text())
        prior['source_hashes'][str(p.relative_to(ROOT))] = digest(p.read_bytes())
    p = ROOT/'analysis/paper_study/crop_damage_validation_20261009/source_readiness.csv'
    prior['source_readiness_snapshot'] = pd.read_csv(p).fillna('').to_dict('records')
    prior['source_hashes'][str(p.relative_to(ROOT))] = digest(p.read_bytes())
    prior['scope'] = 'Frozen prior evidence; no benchmark rerun. Original combined tables are already tracked in the repository.'
    members['context/existing_evidence.json'] = json.dumps(prior, indent=2).encode()

    members['context/source_inventory.csv'] = pd.DataFrame(inventory).to_csv(index=False).encode()
    members['ATTRIBUTION.txt'] = (
        'Swiss observations: Stefan, Fossati, Camp, Pellet, Foiada and Levy Häner. '
        'Zenodo 10.5281/zenodo.17432866; identifier release 10.5281/zenodo.10209176.\n'
        'French observations: Montazeaud and coauthors. Zenodo 10.5281/zenodo.5393959. '
        'Full creator lists and source metadata are retained in sources/zenodo_*.json.\n'
        'The two original public datasets and selected original R scripts are licensed CC BY 4.0 '
        '(https://creativecommons.org/licenses/by/4.0/).\n'
        'Original bytes are retained. Analysis-ready tables are local transformations; '
        'their existing code and dictionary are included for provenance. No scores are converted to percent.\n'
        'context/existing_evidence.json preserves local benchmark receipts and grain summaries, '
        'not new observations. No new licence is assigned to pre-existing local analysis code or receipts.\n'
        'No excluded-journal paper or article PDF is included. Dataset attribution is not a journal endorsement.\n'
    ).encode()
    members['CHECKSUMS.json'] = json.dumps({n:digest(b) for n,b in sorted(members.items())},indent=2).encode()
    with zipfile.ZipFile(HERE/'public_source_subset.zip','w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for name, data in sorted(members.items()):
            info = zipfile.ZipInfo(name, date_time=(2026,10,9,0,0,0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            z.writestr(info,data,compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    print(json.dumps({'bundle':'public_source_subset.zip','members':len(members),
                      'bytes':(HERE/'public_source_subset.zip').stat().st_size,
                      'sha256':digest((HERE/'public_source_subset.zip').read_bytes())},indent=2))


if __name__ == '__main__':
    main()
