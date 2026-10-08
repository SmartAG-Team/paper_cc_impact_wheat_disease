"""Resolve byte-identical numerical source snapshots and record live-code drift."""
from pathlib import Path
import hashlib, json, sys

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
SNAPSHOT=HERE/'source_snapshot'


def manifest():
    return json.loads((HERE/'retained_source_snapshot.json').read_text())


def resolve(path):
    path=Path(path)
    relative=str(path.relative_to(ROOT)) if path.is_absolute() else str(path)
    saved={row['original_path']:ROOT/row['snapshot_path'] for row in manifest()['rows']}
    return saved.get(relative,ROOT/relative)


def activate():
    for row in manifest()['rows']:
        source=ROOT/row['snapshot_path']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError(f'Retained numerical source changed: {source}')
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(ROOT))
    sys.path.insert(0,str(SNAPSHOT))


def drift():
    changed=[]
    for row in manifest()['rows']:
        live=ROOT/row['original_path']
        actual=hashlib.sha256(live.read_bytes()).hexdigest() if live.exists() else None
        if actual!=row['sha256']:
            changed.append({'path':row['original_path'],'frozen_sha256':row['sha256'],'live_sha256':actual})
    return changed
