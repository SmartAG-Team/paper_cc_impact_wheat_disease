"""Propagate frozen calibration bootstraps with an explicit spatial sample."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional import DATA,DEST as REGIONAL,FIELDS,MODELS,SCENARIOS,paths_for,sha
from model.seasonal_septoria.core import Parameters,simulate_season
from model.seasonal_septoria.host import cohort_inputs,sowing_date_from_calendar
from model.seasonal_septoria.regional import phenology_window,tpv_accumulation,simulate_grid_seasons
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment

DEST = ROOT/'analysis/paper_study/regional_parameter_uncertainty'
SAMPLES = DATA/'regional_parameter_uncertainty'


def register_sample():
    SAMPLES.mkdir(parents=True,exist_ok=True)
    path = SAMPLES/'spatial_draws.csv'
    if path.exists():
        return pd.read_csv(path)
    cells = pd.read_parquet(DATA/'wheat_area/europe_wheat_cells_025.parquet')
    cal = pd.read_parquet(DATA/'wheat_area/europe_wheat_calendar_scenarios.parquet')
    cal = cal[cal.crop_season.eq('winter_wheat')&cal.water_system.eq('rainfed')&cal.calendar_valid.eq(True)]
    cells = cells.merge(cal[['cell_id','planting_doy','maturity_doy']],on='cell_id',validate='one_to_one')
    cells['lat_band'] = pd.cut(cells.latitude,[34,43,51,59,68],labels=False)
    cells['lon_band'] = pd.cut(cells.longitude,[-10,8,25,42,60],labels=False)
    if cells[['lat_band','lon_band']].isna().any().any():
        raise ValueError('A wheat cell is outside declared sampling strata.')
    cells['stratum'] = cells.lat_band.astype(str)+'|'+cells.lon_band.astype(str)
    rng = np.random.default_rng(20261005)
    total = float(cells.harvested_total_ha.sum())
    frames = []
    for stratum,group in cells.groupby('stratum',sort=True):
        area = float(group.harvested_total_ha.sum())
        probability = group.harvested_total_ha.to_numpy()/area
        indices = rng.choice(len(group),size=4,replace=True,p=probability)
        selected = group.iloc[indices].copy()
        selected['draw_in_stratum'] = np.arange(4)
        selected['stratum_ha'],selected['area_mean_weight'] = area,area/total/4
        frames.append(selected)
    draws = pd.concat(frames,ignore_index=True)
    draws['spatial_draw_id'] = np.arange(len(draws))
    draws.to_csv(path,index=False)
    config = dict(seed=20261005,sampling='four independent area-proportional draws with replacement per nonempty geographical stratum',
        latitude_breaks=[34,43,51,59,68],longitude_breaks=[-10,8,25,42,60],
        draws=len(draws),unique_cells=draws.cell_id.nunique(),strata=draws.stratum.nunique(),
        eligible_harvested_ha=total,source_area_sha256=sha(DATA/'wheat_area/europe_wheat_cells_025.parquet'),
        calendar_sha256=sha(DATA/'wheat_area/europe_wheat_calendar_scenarios.parquet'),
        genetic_wheat_type_known=False,spatial_sampling_error_requires_separate_reporting=True,
        full_region_point_maps_from_sample=False,draw_sha256=sha(path))
    (SAMPLES/'sampling_receipt.json').write_text(json.dumps(config,indent=2)+'\n')
    return draws


def load_sample_window(paths,start,end,draws):
    dates = pd.date_range(start,end)
    ids = draws.cell_id.drop_duplicates().tolist()
    lookup = {cell:i for i,cell in enumerate(ids)}
    unique_arrays = {key:np.full((len(ids),len(dates)),np.nan) for key in FIELDS}
    sources = []
    for path in paths:
        if not path.exists() or not path.with_suffix('.json').exists():
            raise FileNotFoundError(path)
        receipt = json.loads(path.with_suffix('.json').read_text())
        if receipt['parquet_sha256']!=sha(path):
            raise ValueError('Sample forcing checksum mismatch.')
        table = pq.read_table(path,columns=['date','cell_id',*FIELDS])
        table = table.filter(pc.is_in(table['cell_id'],value_set=pa.array(ids)))
        frame = table.to_pandas()
        frame = frame.loc[frame.date.between(str(start.date()),str(end.date()))]
        rr = frame.cell_id.map(lookup).to_numpy(int)
        dd = dates.get_indexer(pd.to_datetime(frame.date))
        if np.any(dd<0) or frame.duplicated(['date','cell_id']).any():
            raise ValueError('Sample forcing keys conflict.')
        for key in FIELDS:
            unique_arrays[key][rr,dd] = frame[key].to_numpy(float)
        sources.append(dict(path=str(path.relative_to(ROOT)),sha256=receipt['parquet_sha256']))
    if not all(np.isfinite(value).all() for value in unique_arrays.values()):
        raise ValueError('Sample forcing has missing dates/cells.')
    indices = draws.cell_id.map(lookup).to_numpy(int)
    return dates,{key:value[indices] for key,value in unique_arrays.items()},sources


def inputs_for_year(model,scenario,year,draws,phenology):
    start,end = pd.Timestamp(year-1,1,1),pd.Timestamp(year,12,31)
    paths = paths_for('nasa',model,scenario,start,end)
    dates,weather,sources = load_sample_window(paths,start,end,draws)
    coefficient_path = REGIONAL/'bias_alignment'/model/scenario/'coefficients.parquet'
    if not coefficient_path.exists():
        raise FileNotFoundError(coefficient_path)
    coefficients = pd.read_parquet(coefficient_path).set_index(['cell_id','month']).reindex(
        pd.MultiIndex.from_product([draws.cell_id,range(1,13)],names=['cell_id','month']))
    coefficients = {key:coefficients[key].to_numpy().reshape(len(draws),12) for key in
                    ['temperature_offset','humidity_logit_offset','precipitation_ratio']}
    weather = apply_monthly_alignment(weather,dates.month,coefficients)
    sow = np.array([sowing_date_from_calendar(year,row.planting_doy,row.maturity_doy).to_datetime64().astype('datetime64[D]')
                    for row in draws.itertuples()],dtype='datetime64[D]')
    included = dates.to_numpy(dtype='datetime64[D]')[None,:]>=sow[:,None]
    thresholds = {row['BBCH']:row['Cumulative_t_pp_v_GDD'] for row in phenology['thresholds']}
    mask,complete,endpoint,unsupported = phenology_window(weather['tmean_c'],weather['tmax_c'],draws.latitude,
        dates.dayofyear,included,phenology,thresholds[85],True)
    eligible = ~unsupported
    accumulated = tpv_accumulation(weather['tmean_c'][eligible],weather['tmax_c'][eligible],draws.latitude.to_numpy()[eligible],
        dates.dayofyear,mask[eligible],phenology,True)
    active,renewal = cohort_inputs(accumulated,weather['tmean_c'][eligible],thresholds)
    active &= mask[eligible,:,None];renewal[~active] = 0.
    forcing = (weather['tmean_c'][eligible],weather['rh_mean_pct'][eligible],weather['precipitation_mm'][eligible],active,renewal)
    return dates,weather,sow,forcing,eligible,complete,endpoint,sources,sha(coefficient_path)


def run_year(model,scenario,year,draws,parameters,parameter_sources,phenology):
    output = SAMPLES/'annual'/model/scenario/f'{year}.npz'
    receipt_path = output.with_suffix('.json')
    if output.exists() and receipt_path.exists():
        if sha(output)!=json.loads(receipt_path.read_text())['output_sha256']:
            raise ValueError('Archived uncertainty result changed.')
        return 'existing'
    dates,weather,sow,forcing,eligible,complete,endpoint,sources,coefficient_hash = inputs_for_year(model,scenario,year,draws,phenology)
    forecasts = np.full((len(parameters),len(draws)),np.nan)
    row = np.arange(int(eligible.sum()))
    end = endpoint[eligible]+1
    valid = complete[eligible]
    mass_error = 0.
    for i,parameter in enumerate(parameters):
        trajectory = simulate_season(*forcing,parameter)
        result = trajectory.damage[row,end,:3].mean(axis=1)*100
        result[~valid] = np.nan
        forecasts[i,eligible] = result
        mass_error = max(mass_error,float(np.max(abs(trajectory.state.sum(axis=-1)-1))))
    # Draw zero is the frozen point fit, and is independently reconciled with
    # the standard regional endpoint operator on the same sampled inputs.
    point_check = simulate_grid_seasons(dates,weather['tmean_c'][eligible],weather['tmax_c'][eligible],
        weather['rh_mean_pct'][eligible],weather['precipitation_mm'][eligible],draws.latitude.to_numpy()[eligible],
        sow[eligible],parameters[0],phenology)
    expected = point_check['final_damage_percent'][:,:3].mean(axis=1)
    np.testing.assert_allclose(forecasts[0,eligible],expected,rtol=0,atol=1e-10,equal_nan=True)
    weights = draws.area_mean_weight.to_numpy(float)
    denominator = np.sum(np.isfinite(forecasts)*weights[None,:],axis=1)
    means = np.divide(np.nansum(forecasts*weights[None,:],axis=1),denominator,
                      out=np.full(len(parameters),np.nan),where=denominator>0)
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,cell_damage_percent=forecasts,area_mean_percent=means,
                        eligible=eligible,complete_season=complete,unsupported=~eligible,
                        valid_area_fraction=denominator);temporary.replace(output)
    receipt = dict(model=model,scenario=scenario,harvest_year=year,spatial_draws=len(draws),
        bootstrap_draws=len(parameters)-1,frozen_point_row=0,parameter_draws=parameter_sources,
        forcing_sources=sources,coefficient_sha256=coefficient_hash,spatial_sample_sha256=sha(SAMPLES/'spatial_draws.csv'),
        output_sha256=sha(output),maximum_mass_error=mass_error,point_operator_reconciliation=True,
        observation_residual_error_included=False,phenology_parameter_uncertainty_included=False,
        source_code_sha256=sha(Path(__file__)))
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return 'simulated'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    draws = register_sample()
    frozen = ROOT/'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json'
    parameters = [Parameters(**json.loads(frozen.read_text())['parameters'])]
    parameter_sources = [dict(draw='point',path=str(frozen.relative_to(ROOT)),sha256=sha(frozen))]
    for path in sorted((DATA/'seasonal_statistics/bootstrap_fits').glob('fit_*.json')):
        fit = json.loads(path.read_text())
        if not fit['calibration_only'] or fit['validation_or_external_targets_used']:
            raise ValueError('Calibration-only bootstrap required.')
        parameters.append(Parameters(**fit['fit']['parameters']))
        parameter_sources.append(dict(draw=fit['bootstrap_id'],path=str(path.relative_to(ROOT)),sha256=sha(path)))
    if len(parameters)!=101:
        raise ValueError('Exactly100 frozen calibration bootstraps are required.')
    phenology = json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    years = list(range(1991,2021))+list(range(2031,2061))+list(range(2071,2101))
    pending = {(model,scenario,year) for model in MODELS for scenario in SCENARIOS for year in years}
    while pending:
        for model,scenario,year in sorted(pending):
            paths = paths_for('nasa',model,scenario,pd.Timestamp(year-1,1,1),pd.Timestamp(year,12,31))
            if not all(path.exists() and path.with_suffix('.json').exists() for path in paths):
                continue
            status = run_year(model,scenario,year,draws,parameters,parameter_sources,phenology)
            pending.remove((model,scenario,year))
            print(json.dumps(dict(status=status,model=model,scenario=scenario,year=year,pending=len(pending))),flush=True)
        if pending and not args.watch:
            raise RuntimeError(f'Incomplete climate forcing for{len(pending)} uncertainty seasons.')
        if pending:
            time.sleep(30)
    (DEST/'receipt.json').write_text(json.dumps(dict(status='complete',annual_outputs=810,bootstrap_draws=100,
        spatial_draws=len(draws),frozen_parameters_modified=False,publication_ready=False),indent=2)+'\n')


if __name__=='__main__':
    main()
