"""Verify standalone reproduction inside this owned directory, then remove scratch files."""
from pathlib import Path
import json
import shutil
import subprocess
import sys
import tempfile

from audit import sha256

HERE = Path(__file__).resolve().parent


def run(args,cwd,expected=0):
    result = subprocess.run([sys.executable,'-B',*args],cwd=cwd,text=True,capture_output=True)
    if result.returncode != expected:
        raise RuntimeError(result.stdout+'\n'+result.stderr)
    return result


def main():
    for script in ['audit.py','prediction.py']:
        run([script],HERE)
    test = run(['-m','pytest','-q','-p','no:cacheprovider','test_audit.py'],HERE)
    receipts = {name:json.loads((HERE/name).read_text()) for name in ['audit_receipt.json','prediction_receipt.json']}
    tracked = set()
    for receipt in receipts.values():
        for name,digest in receipt['output_hashes'].items():
            if sha256((HERE/name).read_bytes()) != digest:
                raise ValueError('Output hash mismatch: '+name)
            tracked.add(name)
    required = ['audit.py','prediction.py','validate_contract.py','test_audit.py','protocol.json',
                'input_contract.json','journal_provenance.json','public_source_subset.zip']
    comparisons = []
    with tempfile.TemporaryDirectory(prefix='.verify-',dir=HERE) as temporary:
        isolated = Path(temporary)
        for name in required:
            shutil.copyfile(HERE/name,isolated/name)
        for script in ['audit.py','prediction.py']:
            run([script],isolated)
        isolated_test = run(['-m','pytest','-q','-p','no:cacheprovider','test_audit.py'],isolated)
        for name in sorted(tracked | set(receipts)):
            a,b = sha256((HERE/name).read_bytes()),sha256((isolated/name).read_bytes())
            if a != b:
                raise ValueError('Standalone reproduction differs: '+name)
            comparisons.append(dict(file=name,sha256=a,standalone_identical=True))
        # Real current extracts are not silently adapted into a physiological contract.
        fail_closed = run(['validate_contract.py',str(isolated)],isolated,expected=2)
        validation = json.loads(fail_closed.stdout)
        if validation['input_ready'] or validation['mechanistic_validation_established']:
            raise ValueError('Incomplete contract accepted')
    report = dict(status='passed',tests=test.stdout.strip(),isolated_tests=isolated_test.stdout.strip(),
        standalone_reproduction_files=len(comparisons),comparisons=comparisons,
        no_original_data_paths_required=True,contract_missing_input_exit_code=2,
        contract_missing_input_errors=validation['errors'],
        verification_scope='Only the new audit, prediction and validator; existing external benchmarks and model suite not rerun',
        source_bundle_sha256=sha256((HERE/'public_source_subset.zip').read_bytes()),
        code_hashes={p.name:sha256(p.read_bytes()) for p in sorted(HERE.glob('*.py'))})
    (HERE/'verification_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['status','tests','standalone_reproduction_files','no_original_data_paths_required']},indent=2))


if __name__ == '__main__':
    main()
