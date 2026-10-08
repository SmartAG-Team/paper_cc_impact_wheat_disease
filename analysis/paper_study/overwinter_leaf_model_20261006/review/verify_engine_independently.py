"""Independent controlled checks; no model, fit or archival source mutations."""
from pathlib import Path
from dataclasses import replace
from datetime import date, timedelta
import ast
import hashlib
import json
import math
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from model.wheat_stb import EngineConfig, WeatherDay, WeatherProvider, WheatSTBSimulation
from model.wheat_stb.config import TPVParameters
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.overwinter import OverwinterModel, OverwinterParameters, simulate_overwinter
from model.seasonal_septoria.regional import tpv_accumulation

DEST = Path(__file__).resolve().parent


def run_checks():
    source_paths = list((ROOT/'model/wheat_stb').glob('*.py')) + [ROOT/'calibration/wheat_stb/configuration.py', ROOT/'model/seasonal_septoria/overwinter.py', ROOT/'model/seasonal_septoria/leaf_phenology.py']
    initial_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    base = EngineConfig.from_json(ROOT/'examples/wheat_stb/configuration.json')
    sowing = date(2026, 10, 1)
    first = sowing-timedelta(days=5)
    rows = tuple(WeatherDay(first+timedelta(days=i), [18., -3., 12., 24.][i % 4], [22., 1., 16., 28.][i % 4], 85., 2. if i % 3 == 0 else 0., imported_pressure=0. if i % 7 == 0 else None, exposure_override=0. if i % 11 == 0 else 1.) for i in range(230))
    provider = WeatherProvider(rows)
    cases = []
    for stages in (1, 3, 6):
        for dt in (.1, .25, .3, 1.):
            for juvenile in ('handover_31_39', 'persistent'):
                config = replace(base, sowing_date=sowing, crop_end_date=sowing+timedelta(days=195),
                    phenology=TPVParameters(100000., 200000., 100000., 200000.), vernalization_required=False,
                    leaf=replace(base.leaf, juvenile_policy=juvenile),
                    disease=replace(base.disease, latent_stages=stages), time_step=dt, detection_fraction=.003)
                t=np.array([[r.tmean_c for r in provider]]); tx=np.array([[r.tmax_c for r in provider]])
                rain=np.array([[r.precipitation_mm for r in provider]])
                exposure=np.array([[r.exposure_override for r in provider]])
                imported=np.array([[config.background_imported_pressure if r.imported_pressure is None else r.imported_pressure for r in provider]])
                valid=np.array([[config.sowing_date <= r.date <= config.crop_end_date for r in provider]])
                tpv=tpv_accumulation(t,tx,[config.latitude],np.array([r.date.timetuple().tm_yday for r in provider]),valid,config.to_dict()['phenology'],vernalization_required=False)
                host=leaf_host(tpv,t,dict(config.leaf.stage_thresholds),rank_spacing_units=config.leaf.rank_spacing_units,forcing_mask=valid,juvenile_policy=juvenile)
                batch=simulate_overwinter(t,exposure,rain,host.active,host.renewal,host.area,valid,config.disease,
                    initial_local_source=config.initial_local_source,imported_pressure=imported,time_step=dt,detection_fraction=config.detection_fraction)
                engine=WheatSTBSimulation(config,provider)
                states=[]
                for _ in provider:
                    engine.step(); states.append(engine.disease.state)
                np.testing.assert_array_equal(np.array(states),batch.state[0,1:])
                np.testing.assert_array_equal(np.array([o.affected_fraction for o in engine.outputs]),batch.state[0,1:,:,1:].sum(axis=-1))
                np.testing.assert_array_equal(np.array([o.latent_fraction for o in engine.outputs]),batch.state[0,1:,:,1:-3].sum(axis=-1))
                np.testing.assert_array_equal(np.array([o.symptomatic_fraction for o in engine.outputs]),batch.damage[0,1:])
                for route in ('local_flow','imported_flow','splash_flow','contact_flow'):
                    np.testing.assert_array_equal(np.array([getattr(o,route) for o in engine.outputs]),getattr(batch,route)[0])
                summary=engine.finalize().summary
                for i,label in enumerate([f'F{k}' for k in range(1,8)]+['juvenile']):
                    for kind,batch_days in [('infection',batch.infection_day),('symptom',batch.symptom_day)]:
                        index=int(batch_days[0,i]); expected=None if index<0 else provider[index-1].date.isoformat()
                        assert summary[f'first_{kind}_date_by_leaf'][label] == expected
                for stage,index in host.stage_day_index.items():
                    expected=None if index[0]<0 else provider[int(index[0])].date.isoformat()
                    assert summary['stage_dates'][str(stage)] == expected
                prefix=WheatSTBSimulation(config,WeatherProvider(rows[:121]));prefix.run()
                restarted=WheatSTBSimulation(config,provider).restore(json.loads(json.dumps(prefix.snapshot())))
                restarted.run(); assert restarted.outputs == engine.outputs
                cases.append({'latent_stages':stages,'requested_step':dt,'juvenile_policy':juvenile,'batch_state_and_flow_equal':True,'event_dates_equal':True,'restart_daily_equal':True})
    # Source-free wet weather cannot create disease without an infectious donor.
    p=OverwinterParameters(primary_scale=1.,secondary_scale=100.)
    source_free=OverwinterModel(p,initial_local_source=0.)
    for _ in range(20): source_free.integrate(source_free.calc_rates(18.,1.,20.,np.ones(8,bool),np.zeros(8),np.ones(8),0.))
    assert source_free.state[:,1:].sum() == 0.
    # Cold-day disease progression and residue aging freeze under the zero-base scenario.
    cold=OverwinterModel(p)
    state=cold.snapshot(); tissue=np.asarray(state['tissue_state']); tissue[:,0]=.5; tissue[:,1]=.5; state['tissue_state']=tissue.tolist();cold.restore(state)
    before=cold.snapshot();cold.integrate(cold.calc_rates(-5.,0.,0.,np.ones(8,bool),np.zeros(8),np.ones(8),0.))
    np.testing.assert_array_equal(cold.state,before['tissue_state']);np.testing.assert_array_equal(cold.residue,before['residue_state'])
    # Numerical Erlang progression approaches the continuous three-phase CDF.
    exact=1-math.exp(-3.)*(1+3+3**2/2)
    erlang=[]
    for dt in (1.,.25,.05,.01):
        m=OverwinterModel(OverwinterParameters(primary_scale=0.,secondary_scale=0.,latent_stages=3,latent_reference_days=20.),leaf_count=1,initial_local_source=0.,time_step=dt)
        snap=m.snapshot(); snap['tissue_state']=[[0.,1.,0.,0.,0.,0.,0.]];m.restore(snap)
        for _ in range(20):m.integrate(m.calc_rates(18.,0.,0.,np.ones(1,bool),np.zeros(1),np.ones(1),0.))
        observed=float(m.damage[0]);erlang.append({'requested_step':dt,'symptomatic_at_20_days':observed,'continuous_erlang_cdf':exact,'absolute_difference':abs(observed-exact)})
    assert all(a['absolute_difference']>b['absolute_difference'] for a,b in zip(erlang,erlang[1:]))
    findings=[]
    for path in (ROOT/'model').rglob('*.py'):
        tree=ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom) and (node.module or '').startswith(('calibration','analysis','sklearn','scipy.optimize')): findings.append(str(path.relative_to(ROOT))+':'+str(node.lineno))
            elif isinstance(node,ast.Import):
                for alias in node.names:
                    if alias.name.startswith(('calibration','analysis','sklearn','scipy.optimize')):findings.append(str(path.relative_to(ROOT))+':'+str(node.lineno))
            elif isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name.startswith('fit_'):findings.append(str(path.relative_to(ROOT))+':'+str(node.lineno))
    assert not findings
    final_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}
    assert initial_hashes == final_hashes, 'Runtime changed during independent review; rerun on stable source.'
    result={'status':'passed','controlled_batch_restart_cases':cases,'source_free_secondary_infection_zero':True,'cold_zero_base_progression_frozen':True,'erlang_numerical_convergence':erlang,'remaining_model_AST_findings':findings,'reviewed_source_sha256':final_hashes}
    (DEST/'engine_independent_checks.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':'passed','controlled_cases':len(cases),'erlang_numerical_convergence':erlang}))


if __name__=='__main__':run_checks()
