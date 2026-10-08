"""Numerically compare the original and copied T-P-V runtime in fresh processes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def verify(source_root, output):
    source_root = Path(source_root).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    provenance = json.loads((ROOT / 'process_model/SOURCE_PROVENANCE.json').read_text())
    checks = []
    for item in provenance['files']:
        source = source_root / item['source_path']
        destination = ROOT / item['destination_path']
        source_bytes = source.read_bytes()
        destination_bytes = destination.read_bytes()
        assert hashlib.sha256(source_bytes).hexdigest() == item['source_sha256'], source
        assert hashlib.sha256(destination_bytes).hexdigest() == item['destination_sha256'], destination
        functions_equal = None
        if item.get('function_bodies_unchanged'):
            original = {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(source_bytes).body if isinstance(n, ast.FunctionDef)}
            copied = {n.name: ast.dump(n, include_attributes=False)
                for n in ast.parse(destination_bytes).body if isinstance(n, ast.FunctionDef)}
            assert original == copied, destination
            functions_equal = True
        checks.append(dict(destination=item['destination_path'], source_hash_matches=True,
            destination_hash_matches=True, function_AST_equal=functions_equal))
    example = ROOT / 'analysis/phenology/example_source_station4'
    source_script = r'''
import sys
sys.dont_write_bytecode = True
from pathlib import Path
import json
import pandas as pd
source, example, output = map(Path, sys.argv[1:])
sys.path.insert(0, str(source))
from process_model.calibrate import transform, simulate_frame
from process_model.sowing import attach_sowing
from process_model.evaluate_event_dates import event_rows
raw = pd.read_csv(example / 'weather.csv')
records = pd.read_csv(example / 'sowing_records.csv')
parameters = json.loads((source / 'dataset/calibrated_causal/calibration.json').read_text())
features = transform(attach_sowing(raw, records), parameters)
stages = simulate_frame(features, parameters)
events = event_rows(features.assign(CODE=-1), parameters)
output.mkdir()
features.to_csv(output / 'daily_features.csv', index=False)
stages.to_csv(output / 'daily_stages.csv', index=False)
events.to_csv(output / 'event_dates.csv', index=False)
'''
    with tempfile.TemporaryDirectory(prefix='tpv_original_comparison_') as tmp:
        original = Path(tmp) / 'original'
        copied = Path(tmp) / 'copied'
        subprocess.run([sys.executable, '-B', '-c', source_script, str(source_root),
            str(example), str(original)], check=True, cwd=source_root)
        subprocess.run([sys.executable, '-B', '-m', 'process_model.run_tpv',
            '--weather', str(example / 'weather.csv'),
            '--sowing', str(example / 'sowing_records.csv'),
            '--output', str(copied)], check=True, cwd=ROOT, capture_output=True, text=True)
        comparisons = []
        for name in ['daily_features.csv', 'daily_stages.csv', 'event_dates.csv']:
            first = pd.read_csv(original / name)
            second = pd.read_csv(copied / name)
            pd.testing.assert_frame_equal(first, second, check_exact=True)
            numeric = first.select_dtypes(include='number').columns
            difference = (first[numeric] - second[numeric]).abs().to_numpy()
            comparisons.append(dict(file=name, rows=len(first), columns=len(first.columns),
                dataframes_exactly_equal=True,
                bytes_identical=(original / name).read_bytes() == (copied / name).read_bytes(),
                max_abs_numeric_difference=float(np.nanmax(difference)) if difference.size else 0.))
        frame = pd.read_csv(copied / 'event_dates.csv').astype(object)
        event_dates = frame.where(pd.notna(frame), None).to_dict('records')
    report = dict(status='complete', python=sys.version, numpy=np.__version__, pandas=pd.__version__,
        source_root=str(source_root),
        example='Existing archived AGC-Transformer station 4, 1991-10-05 to 1992-09-30, 362 daily rows.',
        purpose='Numerical implementation reproduction; not field validation on BASF trials.',
        source_files=checks, numerical_comparisons=comparisons,
        copied_event_dates=event_dates)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('/Users/gangzhao/Documents/workspace/AGC-Transformer'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.source, args.output)
    print(json.dumps(result['numerical_comparisons'], indent=2))
