"""Explicit relocation of retained scientific inputs; no historical snapshots."""
from functools import lru_cache
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
MANIFEST = ROOT / 'data/final_study/path_manifest.json'


@lru_cache(maxsize=1)
def manifest():
    return json.loads(MANIFEST.read_text())


def original_key(value):
    path = Path(value)
    if path.is_absolute():
        try: return path.relative_to(ROOT).as_posix()
        except ValueError: return path.as_posix()
    return path.as_posix().removeprefix('./')


def current_path(key):
    layout=json.loads((ROOT/'data/repository_layout.json').read_text())
    for old,new in sorted(layout.get('consolidation_path_mapping',{}).items(),key=lambda item:-len(item[0])):
        if key==old or key.startswith(old+'/'):return new+key[len(old):]
    return key


def relocated(value):
    key = current_path(original_key(value))
    if key.startswith('data/paper_study/climate/nasa_v2/daily'):
        return ROOT / key
    if (ROOT/key).exists():return ROOT/key
    parts = Path(key).parts
    if parts and parts[0] in {'model','process_model','calibration','configurations'}:
        return ROOT / key
    if parts[:2] == ('analysis','paper_study') and len(parts) >= 3:
        scope = re.sub(r'_20\d{6}$', '', parts[2])
        return ROOT.joinpath('data/final_study/evidence',scope,*parts[3:])
    if parts and parts[0] == 'analysis':
        return ROOT.joinpath('data/final_study/evidence/analysis',*parts[1:])
    if parts and parts[0] == 'data':
        return ROOT.joinpath('data/final_study/inputs',*parts[1:])
    if parts[:2] == ('publication','european_wheat_stb'):
        return ROOT.joinpath('data/final_study/evidence/publication_source',*parts[2:])
    raise FileNotFoundError(f'Unmapped final-study source: {key}')


def resolve(value):
    key = original_key(value)
    record = manifest()['files'].get(key)
    if record is not None: return ROOT / record['path']
    if key.startswith(('analysis/final_', 'models/methods/', 'data/final_', 'models/', 'models/phenology/', 'models/calibration/')):
        return ROOT / key
    key=current_path(key)
    if key.startswith(('models/','paper/')):return ROOT/key
    return relocated(key)


def evidence_dir(original_directory):
    """Plain directory path preserving original relative scientific filenames."""
    return relocated(original_directory)
