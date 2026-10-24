"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/food_security_exposure_20261009')
import json
from pathlib import Path
import pandas as pd
from analysis.methods.production_exposure.provenance import HERE, digest
from analysis.methods.production_exposure.exposure import require

def read(table=None, verify=True, directory=HERE):
    """Load all result DataFrames, or one table; verify receipt and output hashes.

    ``domain_summary`` contains all geography levels and both analysis modes.
    Filter ``analysis_mode == 'primary_any_paired_year'`` for the primary figure.
    Verification checks internal integrity against receipt hashes, not digital
    authenticity or a fresh download of upstream data.
    """
    directory = Path(directory)
    receipt = json.loads((directory / 'receipt.json').read_text())
    require(receipt['status'] == 'passed' and receipt['schema_version'] == 1, 'Unverified or incompatible receipt')
    if verify:
        for name, expected in receipt['output_sha256'].items():
            path = (directory / name).resolve()
            require(path.is_relative_to(directory.resolve()), 'Receipt path escapes output directory')
            require(digest(path) == expected, f'Output checksum mismatch: {name}')
    names = [table] if table is not None else list(receipt['tables'])
    results = {}
    for name in names:
        path = directory / receipt['tables'][name]
        frame = pd.read_parquet(path) if path.suffix == '.parquet' else pd.read_csv(path, float_precision='round_trip')
        require(len(frame) == receipt['table_rows'][name], f'Row count mismatch: {name}')
        results[name] = frame
    return results[table] if table is not None else results
