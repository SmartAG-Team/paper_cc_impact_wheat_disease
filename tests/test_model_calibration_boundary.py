"""The simulator namespace must not own fitting or observation evaluation."""
from pathlib import Path
import ast
import importlib
import numpy as np

ROOT=Path(__file__).resolve().parents[1]


def test_legacy_model_namespaces_have_no_fitting_or_evaluation_dependencies():
    problems=[]
    for family in ('primary_secondary','seasonal_septoria'):
        for path in (ROOT/'model'/family).glob('*.py'):
            tree=ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node,ast.ImportFrom):
                    name=node.module or ''
                    if name.startswith(('calibration','analysis','sklearn','scipy.optimize')) or (node.level and name in {'calibrate','field_data','refinement','empirical','endpoints','spatial_estimation'}):
                        problems.append(f'{path.name}:{node.lineno}:{name}')
                elif isinstance(node,ast.Import):
                    for alias in node.names:
                        if alias.name.startswith(('calibration','analysis','sklearn','scipy.optimize')):
                            problems.append(f'{path.name}:{node.lineno}:{alias.name}')
                elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('fit_'):
                    problems.append(f'{path.name}:{node.lineno}:{node.name}')
    assert not problems, '\n'.join(problems)


def test_legacy_model_namespaces_do_not_store_fitted_configuration_json():
    files=[str(path.relative_to(ROOT)) for family in ('primary_secondary','seasonal_septoria')
           for path in (ROOT/'model'/family).glob('*.json')]
    assert not files, files


def test_observation_adapters_and_fitters_are_available_outside_model():
    modules=('primary_secondary.calibrate','primary_secondary.field_data','primary_secondary.uncertainty',
             'primary_secondary.crop_protection_metrics','seasonal_septoria.calibrate',
             'seasonal_septoria.field_data','seasonal_septoria.refinement','seasonal_septoria.structural',
             'seasonal_septoria.infection_events','seasonal_septoria.empirical','seasonal_septoria.endpoints',
             'seasonal_septoria.spatial_estimation','seasonal_septoria.climate_alignment',
             'seasonal_septoria.publication_model','seasonal_septoria.infection_priority')
    for name in modules:
        try: module=importlib.import_module('calibration.'+name)
        except ModuleNotFoundError: module=None
        assert module is not None, name


def test_runtime_event_and_canopy_simulation_remain_directly_available():
    from model.seasonal_septoria.infection_events import EventParameters, simulate_events
    from model.seasonal_septoria.structural import CanopyParameters, simulate_canopy
    t=np.full((1,4),18.);active=np.ones((1,4,1),bool)
    events=simulate_events(t,np.ones_like(t),active,EventParameters(.5,1,2.))
    assert events.infection_day.tolist()==[[1]]
    assert events.symptom_day.tolist()==[[3]]
    result=simulate_canopy(t,np.full_like(t,90.),np.full_like(t,2.),active,
                          np.zeros_like(active,float),CanopyParameters())
    np.testing.assert_allclose(result.state.sum(axis=-1),1.,atol=1e-12)
