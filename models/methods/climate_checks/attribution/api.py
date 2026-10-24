"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/review_response_20261010/attribution')
from pathlib import Path
import hashlib
import json
ROOT = PROJECT_ROOT
RELATIVE = Path('data/final_study/evidence/review_response/attribution')

def verify_results(root=ROOT):
    root = Path(root)
    manifest = json.loads((root / RELATIVE / 'portable_manifest.json').read_text())
    for relative, expected in manifest['required_sha256'].items():
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(f'Required packaged result/source: {relative}')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise ValueError(f'Packaged checksum mismatch: {relative}')
    return dict(status='verified', required_files_verified=len(manifest['required_sha256']), raw_inputs_required=False)

def read(root=ROOT):
    """Return verified source-data tables; no simulation or raw-input access occurs."""
    import pandas as pd
    verify_results(root)
    here = Path(root) / RELATIVE
    result = {name: pd.read_csv(here / file) for name, file in {'ensemble': 'ensemble_decomposition.csv', 'by_gcm': 'decomposition_by_gcm.csv', 'headlines': 'headline_sensitivity.csv', 'draws': 'sample_region_membership.csv'}.items()}
    result['protocol'] = json.loads((here / 'prospective_protocol.json').read_text())
    return result
