"""One immutable station partition shared by calibration and all model runs."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_station_split(path, station_ids=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Frozen station split is required: {path}")
    if path.suffix == '.json':
        groups = json.loads(path.read_text())
        if set(groups) != {'train', 'val', 'test'}:
            raise ValueError('Station split must contain exactly train, val, test')
        frame = pd.DataFrame([(site, group) for group, sites in groups.items()
                              for site in sites], columns=['PEP_ID', 'split'])
    else:
        frame = pd.read_csv(path)
    if set(frame.columns) != {'PEP_ID', 'split'} or frame.isna().any().any():
        raise ValueError('Station split requires non-null PEP_ID and split columns')
    ids = pd.to_numeric(frame.PEP_ID, errors='raise')
    if not np.isfinite(ids).all() or (ids % 1 != 0).any():
        raise ValueError('Station IDs must be finite integers')
    frame['PEP_ID'] = ids.astype('int64')
    if frame.PEP_ID.duplicated().any():
        raise ValueError('Duplicate/overlapping station IDs in split')
    if set(frame.split) != {'train', 'val', 'test'}:
        raise ValueError('Station split requires nonempty train, val, test groups')
    if station_ids is not None:
        unknown = set(station_ids) - set(frame.PEP_ID)
        if unknown:
            raise ValueError(f'Stations missing from frozen split: {sorted(unknown)[:10]}')
    return frame.sort_values('PEP_ID').reset_index(drop=True)


def split_identity(frame):
    groups = {group: sorted(frame.loc[frame.split == group, 'PEP_ID'].astype(int).tolist())
              for group in ['train', 'val', 'test']}
    payload = json.dumps(groups, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def assert_training_sites(station_ids, frame):
    forbidden = set(station_ids) - set(frame.loc[frame.split == 'train', 'PEP_ID'])
    if forbidden:
        raise ValueError(f'Fitting includes non-training stations: {sorted(forbidden)[:10]}')


def evaluation_identity(document):
    payload = json.dumps(document, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(payload).hexdigest()


def load_evaluation_sites(path, split, seq_len=None, pred_len=None):
    """Validate a scoring subset without changing any reserved-site assignment."""
    document = json.loads(Path(path).read_text())
    if document.get('schema_version') != 1 or document.get('split_sha256') != split_identity(split):
        raise ValueError('Evaluation manifest uses an unsupported schema or different station split')
    reserved = set(split.loc[split.split == 'test', 'PEP_ID'])
    groups = []
    for key in ['station_ids', 'excluded_station_ids']:
        values = document.get(key)
        if not isinstance(values, list) or any(type(v) is not int for v in values):
            raise ValueError('Evaluation and excluded station IDs must be integer lists')
        if len(set(values)) != len(values):
            raise ValueError('Duplicate evaluation/excluded station IDs')
        groups.append(set(values))
    selected, excluded = groups
    if not selected or selected & excluded or selected | excluded != reserved:
        raise ValueError('Evaluation and exclusions must partition exactly the reserved test stations')
    if excluded and not document.get('exclusion_reason'):
        raise ValueError('Excluded test stations require a documented reason')
    for key, requested in [('seq_len', seq_len), ('pred_len', pred_len)]:
        if type(document.get(key)) is not int or document[key] <= 0:
            raise ValueError('Evaluation manifest requires positive window lengths')
        if requested is not None and requested != document[key]:
            raise ValueError('Evaluation window lengths differ from the approved protocol')
    return document
