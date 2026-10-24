"""Authenticated inputs and non-destructive output/provenance utilities."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


@lru_cache(maxsize=8)
def _records(root):
    records = {}
    for relative in ['data/final_study/path_manifest.json', 'data/final_climate/scientific_manifest.json']:
        path = Path(root) / relative
        if not path.is_file():
            continue
        content = json.loads(path.read_text())
        rows = list(content.get('files', {}).values())
        for key in ['inputs', 'forcing', 'core_sources', 'implementation_sources',
                    'reference_monthly_statistics', 'reference_source_receipts']:
            rows.extend(content.get(key, []))
        for row in rows:
            if 'path' not in row or 'sha256' not in row:
                continue
            if row['path'] in records and records[row['path']]['sha256'] != row['sha256']:
                raise ValueError('Conflicting registered input identity: ' + row['path'])
            records[row['path']] = row
    return records


def auth_source(root, relative):
    root = Path(root).resolve()
    relative = Path(relative).as_posix()
    row = _records(str(root)).get(relative)
    if row is None:
        raise ValueError('Input is not registered: ' + relative)
    path = root / relative
    if not path.is_file() or ('bytes' in row and path.stat().st_size != row['bytes']):
        raise ValueError('Registered input missing or changed size: ' + relative)
    actual = digest(path)
    if actual != row['sha256']:
        raise ValueError('Registered input checksum changed: ' + relative)
    return {'path': relative, 'sha256': actual, 'bytes': path.stat().st_size}


def prepare_destination(destination, root=ROOT):
    destination, root = Path(destination).resolve(), Path(root).resolve()
    reproductions=root/'data/reproductions'
    if reproductions in destination.parents:
        destination.mkdir(parents=True,exist_ok=False)
        return destination
    for name in ['data', 'paper', 'models', 'tests', 'analysis']:
        frozen = root / name
        if destination == frozen or frozen in destination.parents:
            raise ValueError('Output overlaps a frozen/source directory: ' + str(destination))
    if destination == root:
        raise ValueError('Output overlaps the frozen repository root')
    allowed = root / 'outputs'
    if destination == allowed or allowed not in destination.parents:
        raise ValueError('New results require a distinct directory within repository outputs/ or data/reproductions/')
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _finite(value):
    if isinstance(value, dict):
        return {str(key): _finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return _finite(value.tolist())
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_json(path, value):
    Path(path).write_text(json.dumps(_finite(value), indent=2, allow_nan=False) + '\n')


def output_hashes(output):
    output = Path(output)
    return {path.relative_to(output).as_posix(): digest(path)
            for path in sorted(output.rglob('*')) if path.is_file()
            and path.name != 'receipt.json'}
