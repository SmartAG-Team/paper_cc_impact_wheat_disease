"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/review_response_20261010/attribution')
from pathlib import Path
import hashlib
import json
import numpy as np

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()

def pair_years(offset):
    if not isinstance(offset, int) or not 0 <= offset < 30:
        raise ValueError('Offset must be an integer in [0,30).')
    reference = np.arange(1991, 2021)
    return (reference, 2071 + (np.arange(30) + offset) % 30)

def components(y00, y10, y01, y11, supported=None):
    a, b, c, d = np.broadcast_arrays(*[np.asarray(x, float) for x in [y00, y10, y01, y11]])
    common = np.isfinite(a) & np.isfinite(b) & np.isfinite(c) & np.isfinite(d)
    if supported is not None:
        common &= np.asarray(supported, bool)
    interaction = d - b - c + a
    terms = dict(net=d - a, weather_alone=b - a, host_alone=c - a, interaction=interaction, weather=0.5 * (b - a + (d - c)), host=0.5 * (c - a + (d - b)))
    return dict(common=common, **{k: np.where(common, v, np.nan) for k, v in terms.items()})

def offset_percent(weather, host):
    return -100 * host / weather if np.isfinite(weather) and np.isfinite(host) and (abs(weather) > 1e-12) else np.nan

def validate_checkpoint(output, receipt, registration_sha256):
    output, receipt = (Path(output), Path(receipt))
    if not output.exists() or not receipt.exists():
        raise ValueError('Partial output checkpoint.')
    record = json.loads(receipt.read_text())
    if record['registration_sha256'] != registration_sha256:
        raise ValueError('Checkpoint registration mismatch.')
    if record['output_sha256'] != sha(output):
        raise ValueError('Checkpoint output checksum mismatch.')
    return record

def domain_ratio(values, common, weights, indicator):
    """Design-weighted area-time ratio; zeros outside the domain retain design strata."""
    values, common = (np.asarray(values, float), np.asarray(common, bool))
    weights, indicator = (np.asarray(weights, float), np.asarray(indicator, float))
    x = np.where(common, values, 0.0).mean(axis=1) * indicator
    z = common.mean(axis=1) * indicator
    coverage = float(weights @ z)
    if coverage <= 0:
        return (np.nan, np.full(len(weights), np.nan), coverage)
    mean = float(weights @ x) / coverage
    return (mean, (x - mean * z) / coverage, coverage)
