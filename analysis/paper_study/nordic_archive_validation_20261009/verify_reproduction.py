"""Execute tests and reproduce in an isolated copy, recording commands and checksums."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    receipts = HERE / 'receipts'
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    runs = []
    commands = [
        ('run', [sys.executable, 'run_analysis.py']),
        ('tests', [sys.executable, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
                   '--basetemp', str(receipts/'test_tmp'), 'test_analysis.py']),
    ]
    for name, command in commands:
        started = datetime.now(timezone.utc).isoformat()
        result = subprocess.run(command, cwd=HERE, env=env, capture_output=True, text=True)
        (receipts/(name+'.log')).write_text(result.stdout + result.stderr)
        runs.append(dict(name=name, command=command, started_utc=started,
                         finished_utc=datetime.now(timezone.utc).isoformat(),
                         return_code=result.returncode, log=name+'.log'))
        if result.returncode:
            print(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
    # The isolated copy contains exactly the script, fixed protocol and source bundle.
    # Its audit hook rejects original-data reads and network connections.
    guard = '''import os, runpy, sys
def reject_external_data(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes, os.PathLike)):
        name = os.fsdecode(args[0])
        if 'STB_consolidated_collection_20261009.zip' in name or 'STB_research_sources_20261009.zip' in name:
            raise RuntimeError('Forbidden original archive read: ' + name)
        if '/paper_cc_impact_wheat_disease/data/' in os.path.abspath(name):
            raise RuntimeError('Forbidden original data read: ' + name)
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('Network access prohibited during reproduction')
sys.addaudithook(reject_external_data)
sys.argv = ['run_analysis.py']
runpy.run_path('run_analysis.py', run_name='__main__')
'''
    with tempfile.TemporaryDirectory(prefix='standalone_', dir=receipts) as temp:
        root = Path(temp)
        for folder in ('source_bundle', 'receipts'):
            (root/folder).mkdir()
        for name in ('run_analysis.py', 'PROTOCOL.md', 'receipts/protocol_fixed.json',
                     'source_bundle/nordic_public_sources.zip', 'source_bundle/manifest.json'):
            shutil.copy2(HERE/name, root/name)
        result = subprocess.run([sys.executable, '-c', guard], cwd=root, env=env,
                                capture_output=True, text=True)
        (receipts/'standalone.log').write_text(result.stdout + result.stderr)
        if result.returncode:
            print(result.stdout + result.stderr)
            raise SystemExit(result.returncode)
        compared = []
        for expected in sorted((HERE/'outputs').iterdir()):
            actual = root/'outputs'/expected.name
            if expected.name == 'analysis_receipt.json':
                e, a = json.loads(expected.read_text()), json.loads(actual.read_text())
                e.pop('executed_utc'); a.pop('executed_utc')
                if e != a: raise AssertionError('Analysis receipt differs beyond execution timestamp')
            elif expected.read_bytes() != actual.read_bytes():
                raise AssertionError('Standalone output differs: ' + expected.name)
            compared.append(expected.name)
    # Temporary test artifacts belong to this module and are not part of its evidence release.
    shutil.rmtree(receipts/'test_tmp', ignore_errors=True)
    receipt = dict(verified_utc=datetime.now(timezone.utc).isoformat(), runs=runs,
        standalone_return_code=0, standalone_original_data_and_network_access='blocked by Python audit hook',
        standalone_compared_outputs=compared, comparison='Byte identity except analysis execution timestamp',
        tested_files_sha256={n: digest(HERE/n) for n in ('run_analysis.py', 'test_analysis.py',
            'verify_reproduction.py', 'PROTOCOL.md', 'requirements.txt')},
        source_bundle_sha256=digest(HERE/'source_bundle/nordic_public_sources.zip'),
        test_scope='All tests in the standalone owned module; other parallel-worker modules are outside this run.')
    (receipts/'verification.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print((receipts/'tests.log').read_text().strip())
    print(f'Standalone reproduction matched {len(compared)} outputs; original data and network access blocked.')


if __name__ == '__main__':
    main()
