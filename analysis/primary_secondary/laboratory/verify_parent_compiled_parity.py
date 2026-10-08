"""Reconcile Numba and Python execution of the retained parent-model snapshot."""
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
os.environ['NUMBA_DISABLE_JIT'] = '0'
os.environ['NUMBA_CACHE_DIR'] = str(HERE / '.numba_cache')
sys.dont_write_bytecode = True
sys.path.insert(0, str(HERE))
from parent_model_audit_source import core


def verify():
    compiled = core._integrate
    rng = np.random.default_rng(334)
    temperature = rng.uniform(-4, 35, (2, 25))
    humidity = rng.uniform(0, 24, (2, 25))
    rain = rng.uniform(0, 20, (2, 25))
    maximum = 0.
    for stages in [1, 3, 5]:
        parameters = core.Parameters(alpha=.15, beta=5., latent_stages=stages)
        arguments = dict(initial_latent=.5, leaf_ranks=[1, 2, 7],
            leaf_active=[[True, True, False], [True, False, True]])
        actual = core.simulate([[.1, .3, 0], [.7, 0, .1]], temperature, humidity, rain, parameters, **arguments)
        core._integrate = compiled.py_func
        expected = core.simulate([[.1, .3, 0], [.7, 0, .1]], temperature, humidity, rain, parameters, **arguments)
        core._integrate = compiled
        for attribute in ['state', 'damage', 'pycnidia', 'infectious', 'origin_total', 'primary_flux', 'secondary_flux']:
            difference = float(np.max(abs(getattr(actual, attribute) - getattr(expected, attribute))))
            assert difference < 1e-12, (stages, attribute, difference)
            maximum = max(maximum, difference)
    return dict(status='verified', tested_stage_counts=[1, 3, 5],
        maximum_compiled_python_difference=maximum, trajectory_arrays_checked=21,
        parent_model_mutated=False,
        audited_core_sha256=hashlib.sha256((HERE / 'parent_model_audit_source/core.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    report = verify()
    (HERE / 'parent_model_compiled_parity.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
