#!/usr/bin/env python3
"""Refresh wide climate metadata without rebuilding measurement partitions."""
from pathlib import Path
from datetime import datetime, timezone
import json, shutil, subprocess, sys
import pandas as pd
import build_package as package

ROOT=package.ROOT
HERE=package.HERE
OUT=package.OUT


def main():
    original=json.loads((OUT/'manifest.json').read_text())
    protected={p:package.sha(OUT/p) for p in original['common_schema_partitions']}
    # Retain the first integration's dated provenance and verification receipt.
    for path in [OUT/'manifest.json', OUT/'metadata/climate_coverage_snapshot.json',
                 OUT/'continental_forcing_partition_registry.csv', OUT/'continental_partition_integrity.csv',
                 HERE/'independent_validation.json']:
        dest=path.with_name(path.stem+'_at_initial_integration'+path.suffix)
        if not dest.exists():shutil.copyfile(path,dest)
    status=package.climate_registry(pd.read_parquet(OUT/'metadata/europe_wheat_cells_025.parquet'))
    deep_source=ROOT/'analysis/paper_study/climate/actual_coverage_validation_era5.json'
    deep_bytes=deep_source.read_bytes()
    deep=json.loads(deep_bytes)
    s=deep['summary']['era5']
    assert deep['deep'] and deep['coverage_status']=='Complete'
    assert s['expected']==s['valid']==372 and s['missing']==s['invalid']==0
    assert deep['registry_cells']==14941
    (OUT/'metadata/climate_era5_deep_validation_snapshot.json').write_bytes(deep_bytes)
    registry=pd.read_csv(OUT/'continental_forcing_partition_registry.csv')
    era_paths=set(registry.loc[registry.source.eq('era5'),'path'])
    assert era_paths=={r['path'] for r in deep['files']['era5'] if r['status']=='valid'}
    registry.loc[registry.source.eq('era5'),'download_status']='available_complete_baseline'
    registry.to_csv(OUT/'continental_forcing_partition_registry.csv',index=False)
    status['per_source_status']={
        'era5':dict(status='complete_deep_validated',full_download_and_validation_complete=True,
                    audit_as_of=deep['checked_utc'],expected_partitions=372,valid_partitions=372,
                    missing_partitions=0,invalid_partitions=0,registry_cells=14941,
                    start_date='1990-01-01',end_date='2020-12-31',
                    deep_validation_snapshot='metadata/climate_era5_deep_validation_snapshot.json',
                    deep_validation_sha256=package.sha(deep_source)),
        'nasa':dict(status='partial_download_or_pending_deep_coverage_validation',
                    full_download_and_validation_complete=False,**status['summary']['nasa'])}
    nasa_source=ROOT/'analysis/paper_study/climate/actual_coverage_validation_nasa.json'
    nasa_deep=json.loads(nasa_source.read_text())
    ns=nasa_deep['summary']['nasa']
    assert nasa_deep['deep'] and nasa_deep['coverage_status']=='Complete'
    assert ns['expected']==ns['valid']==687 and ns['missing']==ns['invalid']==0
    assert nasa_deep['registry_cells']==14941
    nasa_snapshot='metadata/climate_nasa_deep_validation_snapshot.json'
    (OUT/nasa_snapshot).write_bytes(nasa_source.read_bytes())
    nasa_paths=set(registry.loc[registry.source.eq('nasa'),'path'])
    assert nasa_paths=={r['path'] for r in nasa_deep['files']['nasa'] if r['status']=='valid'}
    registry.loc[registry.source.eq('nasa'),'download_status']='available_complete_ensemble'
    registry.to_csv(OUT/'continental_forcing_partition_registry.csv',index=False)
    status['per_source_status']['nasa']=dict(status='complete_deep_validated',
        full_download_and_validation_complete=True,audit_as_of=nasa_deep['checked_utc'],
        expected_partitions=687,valid_partitions=687,missing_partitions=0,invalid_partitions=0,
        registry_cells=14941,deep_validation_snapshot=nasa_snapshot,
        deep_validation_sha256=package.sha(nasa_source))
    status['status']='complete_deep_validated'
    status['full_download_and_validation_complete']=True
    status['source_hashes_are_provenance_reported']=True
    original['continental_climate_status']=status
    original['climate_registry_refreshed_utc']=datetime.now(timezone.utc).isoformat()
    (OUT/'manifest.json').write_text(json.dumps(original,indent=2)+'\n')
    # Independent verifier recomputes complete-byte hashes of every registered partition.
    subprocess.run([sys.executable,'-B',str(HERE/'verify_package.py')],check=True,cwd=ROOT)
    validation=json.loads((HERE/'independent_validation.json').read_text())
    assert validation['failed_checks']==0
    assert validation['continental_partitions_byte_checksum_verified']==len(registry)
    assert all(package.sha(OUT/path)==expected for path,expected in protected.items())
    registry['checksum_verified_against_bytes']=True
    registry['validation_scope']='source_coverage_audit; package_footer_recheck; independent_complete_byte_checksum'
    registry.to_csv(OUT/'continental_forcing_partition_registry.csv',index=False)
    status['all_registered_partition_byte_checksums_verified']=True
    old='The dated coverage snapshot contains229/372 ERA5 monthly partitions and17/687 NASA merged annual partitions, with zero invalid partitions at that audit. All246 registered partition byte checksums pass independent integrity verification. The continental download and full coverage-validation status remains partial; this snapshot does not describe later download progress.'
    nasa=status['summary']['nasa']
    new=(f"The refreshed coverage snapshot dated {status['audit_as_of']} contains372/372 ERA5 monthly partitions "
         f"and{nasa['valid']}/{nasa['expected']} NASA merged annual partitions, with zero invalid partitions. "
         f"All{len(registry)} registered partition byte checksums pass independent integrity verification. "
         "The separate deep ERA5 receipt confirms all14,941 registry cells, daily dates and numerical coverage for1990–2020. "
         "The separate deep NASA receipt confirms all687 model-scenario-year partitions and their complete daily grid coverage. "
         "ERA5 and NASA are complete and deeply validated. Initial integration snapshots and receipts remain archived with the suffix _at_initial_integration.")
    readme=OUT/'README.txt'
    text=readme.read_text()
    if old in text:text=text.replace(old,new)
    else:
        marker='The refreshed coverage snapshot dated '
        if marker in text:
            start=text.index(marker);end=text.index(' The live source audit',start)
            text=text[:start]+new+text[end:]
        else:raise AssertionError('README climate status paragraph unavailable')
    readme.write_text(text)
    if 'harmonization/refresh_climate_registry.py' not in text:
        text=text.replace('.venv/bin/python -B analysis/paper_study/harmonization/verify_package.py',
             '.venv/bin/python -B analysis/paper_study/harmonization/refresh_climate_registry.py\n'
             '.venv/bin/python -B analysis/paper_study/harmonization/verify_package.py')
        readme.write_text(text)
    receipt=dict(checked_utc=datetime.now(timezone.utc).isoformat(),status='passed',
                 scientific_measurement_rows=original['common_records'],measurement_partition_hashes_unchanged=True,
                 all_original_source_hashes_reverified=True,
                 climate_status=status,independent_checks=validation['check_count'],
                 source_snapshots_preserved=True,model_imported_or_fitted=False,
                 Corteva_model_predictions_accessed=False,executable_sha256=package.sha(Path(__file__)))
    (HERE/'climate_registry_refresh_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    original['independent_validation']=dict(path=str((HERE/'independent_validation.json').relative_to(ROOT)),
        sha256=package.sha(HERE/'independent_validation.json'),check_count=validation['check_count'],failed_checks=0,
        continental_partition_byte_checksums_verified=len(registry))
    original['harmonization_executable_sha256']['refresh_climate_registry.py']=package.sha(Path(__file__))
    original['harmonization_executable_sha256']['verify_package.py']=package.sha(HERE/'verify_package.py')
    original['files']={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=package.sha(p))
                       for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    original['climate_registry_refresh_receipt']=dict(path=str((HERE/'climate_registry_refresh_receipt.json').relative_to(ROOT)),
        sha256=package.sha(HERE/'climate_registry_refresh_receipt.json'))
    (OUT/'manifest.json').write_text(json.dumps(original,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k!='climate_status'},indent=2))


if __name__=='__main__':main()
