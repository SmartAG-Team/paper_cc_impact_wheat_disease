"""Refresh the compact source archive and verify every archived byte digest."""
from pathlib import Path
import hashlib, json, zipfile

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'publication/european_wheat_stb'
REVIEW=ROOT/'analysis/paper_study/manuscript_review_20261006'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    # Preserve the established compact membership; extend it with reviewed outputs.
    previous=json.loads((DEST/'reproducibility_manifest.json').read_text())
    paths={ROOT/r['path'] for r in previous['files']}
    paths.update((ROOT/'analysis/paper_study/manuscript').glob('*.py'))
    paths.update((ROOT/'analysis/paper_study/manuscript').glob('*.js'))
    paths.update((ROOT/'analysis/paper_study/manuscript/csl_vendor').glob('*'))
    paths.update(DEST/name for name in ['references.csl.json','references.bib','references_author_date.txt',
        'citation_input.json','citation_rendered.json'])
    paths.add(ROOT/'analysis/paper_study/citation_author_date_20261006/citation_validation.json')
    paths.add(ROOT/'analysis/paper_study/manuscript/climate_figure_contract.json')
    paths.update((ROOT/'data/paper_study/publication').glob('*.csv'))
    paths.update((ROOT/'data/paper_study/projection_uncertainty_reporting/parameter_periods').rglob('*.npz'))
    paths.update((ROOT/'data/paper_study/projection_uncertainty_reporting/parameter_periods').rglob('*.json'))
    paths.update(p for p in REVIEW.glob('*.json') if p.name!='archive_validation.json')
    paths.update(REVIEW.glob('*.npz'))
    paths.update((DEST/'figures').glob('*.png'));paths.update((DEST/'figures').glob('*.pdf'))
    paths.update((DEST/'figures').glob('*.json'))
    records=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    manifest=dict(included_files=len(records),included_bytes=sum(r['bytes'] for r in records),files=records,
        continental_forcing_and_annual_grids_embedded=False,parameter_period_arrays_embedded=True,
        review_date='2026-10-06',source_code_sha256=sha(Path(__file__)))
    (DEST/'reproducibility_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    archive=DEST/'Reproducibility.zip'
    temporary=archive.with_suffix('.tmp.zip')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in records:z.write(ROOT/row['path'],row['path'])
        z.write(DEST/'reproducibility_manifest.json','publication/european_wheat_stb/reproducibility_manifest.json')
    temporary.replace(archive)
    with zipfile.ZipFile(archive) as source:
        assert source.testzip() is None
        for row in records:
            assert hashlib.sha256(source.read(row['path'])).hexdigest()==row['sha256']
        # Greedy balance by actual compressed size, preserving valid independent ZIPs.
        buckets=[[],[]];sizes=[0,0]
        for info in sorted(source.infolist(),key=lambda x:x.compress_size,reverse=True):
            i=min(range(2),key=lambda j:sizes[j]);buckets[i].append(info.filename);sizes[i]+=info.compress_size
        partition_paths=[]
        for i,bucket in enumerate(buckets,1):
            path=DEST/f'Reproducibility_part_{i}.zip'
            with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
                for name in sorted(bucket):z.writestr(name,source.read(name))
            assert path.stat().st_size<30_000_000
            partition_paths.append(path)
        # Two extracts together reproduce every full-archive member exactly.
        restored={}
        for path in partition_paths:
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None
                for name in z.namelist():
                    assert name not in restored
                    restored[name]=hashlib.sha256(z.read(name)).hexdigest()
        assert set(restored)==set(source.namelist())
        for name,digest in restored.items():assert digest==hashlib.sha256(source.read(name)).hexdigest()
    deliverables=['Manuscript.docx','Manuscript.pdf','Manuscript.txt','Supplementary_Information.docx',
        'Supplementary_Information.pdf','Source_Data.xlsx','Reproducibility.zip',
        'Reproducibility_part_1.zip','Reproducibility_part_2.zip','reproducibility_manifest.json','final_validation.json',
        'references.csl.json','references.bib','references_author_date.txt']
    hashes={name:dict(bytes=(DEST/name).stat().st_size,sha256=sha(DEST/name)) for name in deliverables}
    (DEST/'DELIVERY_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    receipt=dict(status='passed',archive_members=len(restored),archived_content_hashes_checked=len(records),
        split_archive_members_checked=len(restored),split_bytes=[p.stat().st_size for p in partition_paths],
        all_split_files_below_30_MB=True,source_code_sha256=sha(Path(__file__)))
    (REVIEW/'archive_validation.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
