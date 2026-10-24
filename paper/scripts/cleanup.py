"""Externalize large working caches while preserving every compatibility path."""
from datetime import datetime, timezone
import json
from pathlib import Path
import os

PROTECTED = ['.git', '.venv', 'data/final_climate', 'data/final_models', 'data/final_study',
    'data/final_monitoring/folds',
    'data/final_monitoring/cohort',
    'data/final_monitoring/evaluation']


def inventory(directory):
    result = []
    for parent, dirs, files in os.walk(directory, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(parent)/name).is_symlink()]
        for name in files:
            p = Path(parent)/name
            if not p.is_symlink():
                s = p.stat()
                result.append(dict(path=str(p.relative_to(directory)), bytes=s.st_size, inode=s.st_ino, device=s.st_dev))
    return result


def externalize(root, relative, archive_root):
    root, archive_root = Path(root).resolve(), Path(archive_root).resolve()
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts or relative == Path('.'):
        raise ValueError('A safe relative cache path is required')
    if relative.parts[0] not in ['data', 'outputs'] or any(
            relative == Path(p) or relative.is_relative_to(p) or Path(p).is_relative_to(relative) for p in PROTECTED):
        raise ValueError('Protected scientific/Git/active evidence cannot be externalized')
    if archive_root.is_relative_to(root): raise ValueError('Archive must be outside the repository')
    source, target = root/relative, archive_root/relative
    if source.is_symlink():
        if source.resolve() != target.resolve(): raise ValueError('Unexpected existing external path')
        return dict(path=str(relative), archive=str(target), status='already_external')
    if not source.is_dir(): raise ValueError('A real working-cache directory is required')
    if target.exists() or target.is_symlink(): raise FileExistsError(target)
    before = inventory(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    try:
        source.symlink_to(target, target_is_directory=True)
    except BaseException:
        target.rename(source)
        raise
    after = inventory(source)
    if before != after: raise RuntimeError('Cache identity changed during externalization')
    return dict(path=str(relative), archive=str(target), status='externalized', files=len(before),
        bytes=sum(row['bytes'] for row in before), inode_and_size_unchanged=True, inventory=before)


def run():
    root = Path(__file__).resolve().parents[2]
    archive = root.parent/'paper_cc_impact_wheat_disease_intermediates_20261023'
    output = root/'outputs/repository_cleanup_20261023'; output.mkdir(exist_ok=False)
    candidates = ['data/paper_study/climate/nasa_v2/daily',
        'data/disease_improvement/tabpfn_weights_20261022',
        'data/disease_improvement/canopy_trait_raw_20261022', 'data/external_weather_validation_20261022']
    for p in sorted((root/'outputs/disease_improvement').iterdir()):
        relative = str(p.relative_to(root))
        if p.is_dir() and not p.is_symlink() and relative not in PROTECTED:
            if sum(row['bytes'] for row in inventory(p)) >= 8*1024**2: candidates.append(relative)
    receipt = dict(started_utc=datetime.now(timezone.utc).isoformat(), archive=str(archive),
        method='Same-filesystem atomic moves with compatibility symlinks; original inode/size identities checked',
        protected=PROTECTED, moved=[], original_data_deleted=False)
    (output/'plan.json').write_text(json.dumps(dict(candidates=candidates, **receipt),indent=2)+'\n')
    for path in candidates:
        record = externalize(root, path, archive); receipt['moved'].append(record)
        (output/'archive_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(record['status'], path, round(record.get('bytes',0)/1024**3,3), 'GiB',flush=True)
    receipt['bytes_externalized'] = sum(row.get('bytes',0) for row in receipt['moved'])
    receipt['passed'] = all((root/row['path']).is_symlink() and (root/row['path']).resolve()==Path(row['archive'])
        for row in receipt['moved'])
    receipt['completed_utc'] = datetime.now(timezone.utc).isoformat()
    (output/'archive_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    (archive/'archive_manifest.json').write_text(json.dumps(receipt,indent=2)+'\n')
    summary = {key:value for key,value in receipt.items() if key != 'moved'}
    summary['paths'] = [{k:v for k,v in row.items() if k != 'inventory'} for row in receipt['moved']]
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('completed',round(receipt['bytes_externalized']/1024**3,3),'GiB',flush=True)
    return output


if __name__ == '__main__': run()
