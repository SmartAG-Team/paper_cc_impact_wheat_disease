"""Symmetric weather–phenology decomposition with fixed spatial/bootstrap draws."""

from datetime import datetime,timezone
import argparse,json
from pathlib import Path
import sys,time

import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.paper_study.regional_parameter_uncertainty import inputs_for_year,register_sample,DATA,SAMPLES,MODELS,SCENARIOS,sha
from model.seasonal_septoria.core import Parameters
from model.seasonal_septoria.counterfactual import aligned_crop_inputs,decompose_weather_host

DEST=ROOT/'analysis/paper_study/climate_counterfactuals'
OUT=DATA/'climate_counterfactuals'


def prepared(model,scenario,year,draws,phenology):
    path=OUT/'prepared'/model/scenario/f'{year}.npz'
    receipt_path=path.with_suffix('.json')
    if path.exists() and receipt_path.exists():
        receipt=json.loads(receipt_path.read_text())
        if sha(path)!=receipt['output_sha256']:raise ValueError('Prepared crop inputs changed.')
        return dict(np.load(path))
    dates,weather,sow,forcing,eligible,complete,endpoint,sources,coefficient=inputs_for_year(model,scenario,year,draws,phenology)
    active=np.zeros((len(draws),len(dates),8),bool);renewal=np.zeros_like(active,dtype=float)
    active[eligible],renewal[eligible]=forcing[-2:]
    sow_indices=dates.get_indexer(pd.to_datetime(sow))
    result=dict(**weather,active=active,renewal=renewal,eligible=eligible,
        complete=complete,endpoint=endpoint,sow_indices=sow_indices)
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix('.tmp.npz');np.savez_compressed(temporary,**result);temporary.replace(path)
    receipt=dict(model=model,scenario=scenario,year=year,forcing_sources=sources,
        coefficient_sha256=coefficient,spatial_draw_sha256=sha(SAMPLES/'spatial_draws.csv'),
        phenology_sha256=sha(ROOT/'process_model/parameters/calibration.json'),output_sha256=sha(path),
        source_code_sha256=sha(Path(__file__)),preparation_source_sha256=sha(ROOT/'analysis/paper_study/regional_parameter_uncertainty.py'))
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return result


def run_pair(model,scenario,future_year,draws,parameters,phenology):
    output=OUT/'annual'/model/scenario/f'{future_year}.npz';receipt_path=output.with_suffix('.json')
    if output.exists() and receipt_path.exists():
        if sha(output)!=json.loads(receipt_path.read_text())['output_sha256']:raise ValueError('Counterfactual output changed.')
        return 'existing'
    base_year=future_year-(40 if future_year<2070 else 80)
    reference=[]
    for year in [base_year,future_year]:
        path=SAMPLES/'annual'/model/scenario/f'{year}.npz'
        if not path.exists() or not path.with_suffix('.json').exists():raise FileNotFoundError(path)
        if sha(path)!=json.loads(path.with_suffix('.json').read_text())['output_sha256']:raise ValueError('Frozen ensemble reference changed.')
        reference.append(path)
    base,future=[prepared(model,scenario,year,draws,phenology) for year in [base_year,future_year]]
    duration0=base['endpoint']-base['sow_indices']+1
    duration1=future['endpoint']-future['sow_indices']+1
    required=np.maximum(duration0,duration1)
    common=base['eligible']&future['eligible']&base['complete']&future['complete']
    for state in [base,future]:
        common&=(state['sow_indices']>=0)&(state['sow_indices']+required<=state['active'].shape[1])
    if not common.any():raise ValueError('No paired complete sampled seasons.')
    def align(state):
        weather={key:state[key][common] for key in ['tmean_c','rh_mean_pct','precipitation_mm']}
        return aligned_crop_inputs(weather,state['active'][common],state['renewal'][common],
            state['sow_indices'][common],state['endpoint'][common],required[common])
    result=decompose_weather_host(align(base),align(future),parameters)
    for key,path in zip(['y00','y11'],reference):
        expected=np.load(path)['cell_damage_percent'][:,common]
        np.testing.assert_allclose(result[key],expected,rtol=0,atol=1e-10)
    arrays={key:np.full((len(parameters),len(draws)),np.nan) for key in
        ['y00','y10','y01','y11','weather_effect','host_effect','total_effect']}
    for key in arrays:arrays[key][:,common]=result[key]
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp.npz');np.savez_compressed(temporary,**arrays,common_valid=common);temporary.replace(output)
    receipt=dict(model=model,scenario=scenario,future_year=future_year,paired_baseline_year=base_year,
        common_valid_spatial_draws=int(common.sum()),spatial_draws=len(draws),bootstrap_draws=len(parameters)-1,
        maximum_mass_error=result['mass_error'],full_trajectory_reference_reconciliation_all_parameter_draws=True,
        reference_outputs={str(path.relative_to(ROOT)):sha(path) for path in reference},
        prepared_inputs={str((OUT/'prepared'/model/scenario/f'{year}.npz').relative_to(ROOT)):
            sha(OUT/'prepared'/model/scenario/f'{year}.npz') for year in [base_year,future_year]},
        output_sha256=sha(output),source_code_sha256=sha(Path(__file__)),
        decomposition_source_sha256=sha(ROOT/'model/seasonal_septoria/counterfactual.py'))
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return 'simulated'


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--watch',action='store_true');args=parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    draws=register_sample()
    frozen=ROOT/'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json'
    parameters=[Parameters(**json.loads(frozen.read_text())['parameters'])]
    parameter_sources=[dict(path=str(frozen.relative_to(ROOT)),sha256=sha(frozen))]
    for path in sorted((DATA/'seasonal_statistics/bootstrap_fits').glob('fit_*.json')):
        fit=json.loads(path.read_text())
        if not fit['calibration_only'] or fit['validation_or_external_targets_used']:raise ValueError('Unfrozen calibration draw.')
        parameters.append(Parameters(**fit['fit']['parameters']));parameter_sources.append(dict(path=str(path.relative_to(ROOT)),sha256=sha(path)))
    if len(parameters)!=101:raise ValueError('Exactly100 calibration bootstrap draws required.')
    config=dict(registered_utc=datetime.now(timezone.utc).isoformat(),parameter_sources=parameter_sources,
        decomposition='two-factor symmetric Shapley decomposition of disease-weather and host-development pathways',
        paired_years='2031–2060 to1991–2020;2071–2100 to1991–2020, ordinal pairs',
        alignment='integer days since fixed-calendar sowing; leap-year differences retained in native weather and host trajectories',
        host_pathway='joint crop development, cohort emergence and renewal, and predicted soft-dough endpoint',
        disease_weather_pathway='temperature/moisture establishment, thermal progression and rain-mediated transmission',
        unsupported_or_incomplete='all four outcomes restricted to common complete paired seasons with actual cross-weather coverage',
        spatial_sample_sha256=sha(SAMPLES/'spatial_draws.csv'),empirical_causal_effect=False,
        field_residual_uncertainty_included=False,phenology_parameter_uncertainty_included=False,
        source_code_sha256=sha(Path(__file__)))
    config_path=DEST/'configuration_before_counterfactual_results.json'
    if not config_path.exists():config_path.write_text(json.dumps(config,indent=2)+'\n')
    else:
        previous=json.loads(config_path.read_text())
        for key in ['parameter_sources','decomposition','spatial_sample_sha256','source_code_sha256']:
            if previous[key]!=config[key]:raise ValueError('Counterfactual configuration changed.')
    phenology=json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    pending={(model,scenario,year) for model in MODELS for scenario in SCENARIOS
        for year in list(range(2031,2061))+list(range(2071,2101))}
    while pending:
        for model,scenario,year in sorted(pending):
            try:status=run_pair(model,scenario,year,draws,parameters,phenology)
            except FileNotFoundError:continue
            pending.remove((model,scenario,year))
            print(json.dumps(dict(status=status,model=model,scenario=scenario,year=year,pending=len(pending))),flush=True)
        if pending and not args.watch:raise RuntimeError(f'{len(pending)} paired inputs incomplete.')
        if pending:time.sleep(30)
    (DEST/'receipt.json').write_text(json.dumps(dict(status='complete',paired_seasons=540,
        counterfactual_cases=4,bootstrap_draws=100,spatial_draws=len(draws),parameters_refitted=False,
        spatial_sampling_error_requires_separate_reporting=True),indent=2)+'\n')


if __name__=='__main__':main()
