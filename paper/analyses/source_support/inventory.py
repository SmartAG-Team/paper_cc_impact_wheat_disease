"""Create the current repository inventory, reusing authenticated forcing hashes."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
EXCLUDE={'.git','.venv','__pycache__','.pytest_cache','.mypy_cache','.ruff_cache'}


def build():
    files={}
    for parent,dirs,names in os.walk(ROOT,followlinks=False):
        dirs[:]=[name for name in dirs if name not in EXCLUDE and not (Path(parent)/name).is_symlink()]
        for name in names:
            path=Path(parent)/name;relative=path.relative_to(ROOT).as_posix()
            if path.is_symlink() or name in {'repository_manifest.json','.DS_Store'} or path.suffix in {'.pyc','.pyo','.nbc','.nbi','.tmp','.log'}:
                continue
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            files[relative]=dict(path=relative,bytes=path.stat().st_size,sha256=digest)
    forcing=json.loads((ROOT/'data/final_climate/scientific_manifest.json').read_text())['forcing']
    for row in forcing:
        path=ROOT/row['path']
        if not path.is_file() or path.stat().st_size!=row['bytes']:
            raise ValueError('Missing or changed climate forcing: '+row['path'])
        files[row['path']]={key:row[key] for key in ['path','bytes','sha256']}
    return dict(schema_version=2,created_utc=datetime.now(timezone.utc).isoformat(),
        scope='Current manuscript, source data, fitted models, scientific code and final numerical results',
        canonical_manuscript='paper/Manuscript.docx',
        canonical_supplement='paper/Supplementary_Information.docx',
        infrastructure_excluded=sorted(EXCLUDE),inventory_self_hash_excluded=True,
        large_forcing_hashes='Frozen authenticated source hashes; run final_climate --check-inputs --verify-forcing-hashes to reread content',
        files=[files[name] for name in sorted(files)],final_files_excluding_inventory=len(files),
        final_bytes_excluding_inventory=sum(row['bytes'] for row in files.values()))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true')
    args=parser.parse_args()
    if args.write:
        result=build();(ROOT/'repository_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(status='written',files=len(result['files'])),indent=2))
    else:
        from .verify import verify_inventory
        print(json.dumps(verify_inventory(),indent=2))
