"""Small original numerical and local-output identity utilities."""
import hashlib
import json
from pathlib import Path
import numpy as np
from analysis.final_study.paths import ROOT


def json_write(path, value):
    def clean(x):
        if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)): return [clean(v) for v in x]
        if isinstance(x, np.integer): return int(x)
        if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
        if isinstance(x, np.bool_): return bool(x)
        return x
    Path(path).write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')


def file_record(path):
    path = Path(path).resolve()
    return dict(path=str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                sha256=digest(path))


def verify_records(records):
    for record in records:
        path = Path(record['path'])
        if not path.is_absolute(): path = ROOT / path
        if not path.is_file() or digest(path) != record['sha256']:
            raise ValueError('Frozen input changed: ' + str(path))


def crps_values(cdf, targets):
    cdf, targets = np.asarray(cdf), np.asarray(targets)
    truth = np.arange(cdf.shape[2])[None, None, :] >= targets[:, :, None]
    values = np.square(cdf.astype(np.float64) - truth).sum(2)
    return np.where(targets >= 0, values, np.nan)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()

