"""Run resumable whole-domain crop seasons from verified daily forcing."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from model.seasonal_septoria.core import Parameters
from model.seasonal_septoria.host import sowing_date_from_calendar
from model.seasonal_septoria.regional import simulate_grid_seasons,phenology_window
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment

DATA = ROOT/'data/paper_study'
DEST = ROOT/'analysis/paper_study/regional_v1'
FIELDS = ['tmean_c','tmax_c','rh_mean_pct','precipitation_mm']
MODELS = ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
SCENARIOS = ['ssp126','ssp245','ssp585']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def paths_for(provider,model,scenario,start,end):
    if provider=='era5':
        return [DATA/'climate/era5_daily'/f'{month:%Y-%m-%d}_{month+pd.offsets.MonthBegin():%Y-%m-%d}.parquet'
                for month in pd.date_range(start.replace(day=1),end.replace(day=1),freq='MS')]
    return [DATA/'climate/nasa_v2/daily'/model/('historical' if year<=2014 else scenario)/f'{year}.parquet'
            for year in range(start.year,end.year+1)]


def load_window(paths,start,end,cells):
    dates = pd.date_range(start,end)
    arrays = {key:np.empty((len(cells),len(dates)),np.float32) for key in FIELDS}
    sources = []
    seen = np.zeros(len(dates),bool)
    for path in paths:
        receipt_path = path.with_suffix('.json')
        if not path.exists() or not receipt_path.exists():
            raise FileNotFoundError(path)
        receipt = json.loads(receipt_path.read_text())
        if receipt.get('parquet_sha256')!=sha(path):
            raise ValueError(f'Forcing checksum changed: {path}')
        frame = pq.read_table(path,columns=['date','cell_id',*FIELDS]).to_pandas()
        frame = frame.loc[frame.date.between(str(start.date()),str(end.date()))]
        if frame.empty:
            continue
        unique_dates = pd.DatetimeIndex(pd.to_datetime(frame.date.unique()))
        if len(frame)!=len(cells)*len(unique_dates):
            raise ValueError('Forcing has incomplete daily cell coverage.')
        ids = frame.cell_id.to_numpy().reshape(len(unique_dates),len(cells))
        if not np.all(ids==cells.cell_id.to_numpy()[None,:]):
            raise ValueError('Forcing cell order or membership differs from the area registry.')
        positions = dates.get_indexer(unique_dates)
        if np.any(positions<0) or seen[positions].any():
            raise ValueError('Forcing dates conflict.')
        seen[positions] = True
        for field in FIELDS:
            arrays[field][:,positions] = frame[field].to_numpy().reshape(len(unique_dates),len(cells)).T
        sources.append(dict(path=str(path.relative_to(ROOT)),sha256=receipt['parquet_sha256']))
    if not seen.all():
        raise ValueError('Daily forcing window is incomplete.')
    return dates,arrays,sources


def configure():
    DEST.mkdir(parents=True,exist_ok=True)
    config = dict(schema_version=1,baseline_harvest_years=list(range(1991,2021)),
        future_harvest_years=list(range(2031,2061))+list(range(2071,2101)),
        models=MODELS,scenarios=SCENARIOS,reference_area_year=2020,
        endpoint='predicted_BBCH85_soft_dough',severity='normalized_untreated_leaf_damage_proxy_percent',
        primary_exposure='first cohort whose accumulated primary flow reaches0.0001',
        cutoffs_percent=[.1,1.,5.],cutoffs_are_fungicide_action_thresholds=False,
        main_calendar='winter_wheat_rainfed_all_wheat_area_scenario',
        calendar_sensitivities=['winter_wheat_irrigated','spring_wheat_rainfed','spring_wheat_irrigated'],
        sensitivity_scenarios_are_additive_area_partitions=False,
        spring_phenology='photoperiod process with vernalization demand disabled; frozen winter thresholds, unvalidated transfer sensitivity',
        supplemental_sowing_offsets_days=[-14,14],supplemental_endpoint_stage=75,
        missing_soft_dough_is_zero_disease=False,grid_resolution_degrees=.25,
        fitted_parameter_sha256=sha(ROOT/'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json'),
        area_sha256=sha(DATA/'wheat_area/europe_wheat_cells_025.parquet'),
        calendar_sha256=sha(DATA/'wheat_area/europe_wheat_calendar_scenarios.parquet'),
        operational_forecast=False,future_land_use_prediction=False)
    path = DEST/'configuration_before_regional_results.json'
    if path.exists():
        if json.loads(path.read_text())!=config:
            raise ValueError('Regional configuration changed; use a new analysis version.')
    else:
        path.write_text(json.dumps(config,indent=2)+'\n')
    return config


def run_season(provider,model,scenario,year,calendar_season='winter_wheat',water='rainfed',
               adjusted=False,sowing_offset=0,end_stage=85,chunk_size=128):
    config = configure()
    code = f'{calendar_season}_{water}_sow{sowing_offset:+d}_stage{end_stage}'
    output = DEST/'annual'/provider/model/scenario/('adjusted' if adjusted else 'original')/code/f'{year}.parquet'
    receipt_path = output.with_suffix('.json')
    if output.exists() and receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if receipt['parquet_sha256']!=sha(output):
            raise ValueError('Archived regional output changed.')
        return {'status':'existing','year':year,'model':model,'scenario':scenario,'calendar':code}
    cells = pd.read_parquet(DATA/'wheat_area/europe_wheat_cells_025.parquet')
    calendars = pd.read_parquet(DATA/'wheat_area/europe_wheat_calendar_scenarios.parquet')
    calendar = calendars.loc[calendars.crop_season.eq(calendar_season)&calendars.water_system.eq(water)]
    cells = cells.merge(calendar[['cell_id','planting_doy','maturity_doy','calendar_valid']],
                        on='cell_id',how='left',validate='one_to_one',sort=False)
    # January antecedent permits actual scenario planting and both sowing offsets.
    start,end = pd.Timestamp(year-1,1,1),pd.Timestamp(year,12,31)
    paths = paths_for(provider,model,scenario,start,end)
    dates,arrays,sources = load_window(paths,start,end,cells)
    coefficients = None
    coefficient_path = DEST/'bias_alignment'/model/scenario/'coefficients.parquet'
    if adjusted:
        if provider!='nasa' or not coefficient_path.exists():
            raise FileNotFoundError('Full historical-only bias coefficients are required.')
        coefficients = pd.read_parquet(coefficient_path)
        if coefficients.duplicated(['cell_id','month']).any():
            raise ValueError('Duplicate climate coefficient keys.')
        coefficients = coefficients.set_index(['cell_id','month']).reindex(
            pd.MultiIndex.from_product([cells.cell_id,range(1,13)],names=['cell_id','month']))
    frozen_path = ROOT/'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json'
    if sha(frozen_path)!=config['fitted_parameter_sha256']:
        raise ValueError('Frozen field parameters changed.')
    parameters = Parameters(**json.loads(frozen_path.read_text())['parameters'])
    phenology_path = ROOT/'process_model/parameters/calibration.json'
    phenology = json.loads(phenology_path.read_text())
    eligible = cells.calendar_valid.fillna(False).to_numpy(bool)
    sowing = np.full(len(cells),np.datetime64('NaT','D'))
    for i in np.flatnonzero(eligible):
        sowing[i] = sowing_date_from_calendar(year,cells.planting_doy.iloc[i],cells.maturity_doy.iloc[i]).to_datetime64().astype('datetime64[D]')+np.timedelta64(sowing_offset,'D')
    rows,numerical = [],[]
    for first in range(0,len(cells),chunk_size):
        indices = np.arange(first,min(len(cells),first+chunk_size))
        metadata = cells.iloc[indices][['cell_id','harvested_total_ha','physical_total_ha','latitude','longitude','dominant_source_country']].copy()
        metadata['harvest_year'],metadata['calendar_scenario'] = year,calendar_season+'_'+water
        metadata['sowing_date'],metadata['sowing_offset_days'] = sowing[indices],sowing_offset
        metadata['status'] = np.where(eligible[indices],'pending','missing_calendar')
        weather = {key:arrays[key][indices].astype(float) for key in FIELDS}
        if adjusted:
            co = {key:coefficients[key].to_numpy().reshape(len(cells),12)[indices]
                  for key in ['temperature_offset','humidity_logit_offset','precipitation_ratio']}
            weather = apply_monthly_alignment(weather,dates.month,co)
        valid = eligible[indices].copy()
        for key in FIELDS:
            valid &= np.isfinite(weather[key]).all(axis=1)
        in_crop = dates.to_numpy(dtype='datetime64[D]')[None,:]>=sowing[indices,None]
        unsupported = np.zeros(len(indices),bool)
        if valid.any():
            thresholds = {row['BBCH']:row['Cumulative_t_pp_v_GDD'] for row in phenology['thresholds']}
            terminal = thresholds[85] if end_stage==85 else (thresholds[51]+thresholds[85])/2
            _,_,_,unsupported[valid] = phenology_window(weather['tmean_c'][valid],weather['tmax_c'][valid],
                metadata.latitude.to_numpy()[valid],dates.dayofyear,in_crop[valid],phenology,
                terminal,calendar_season=='winter_wheat')
        metadata.loc[eligible[indices]&unsupported,'status'] = 'unsupported_mean_temperature_above40C'
        valid &= ~unsupported
        if valid.any():
            result = simulate_grid_seasons(dates,weather['tmean_c'][valid],weather['tmax_c'][valid],
                weather['rh_mean_pct'][valid],weather['precipitation_mm'][valid],
                metadata.latitude.to_numpy()[valid],sowing[indices][valid],parameters,phenology,
                end_stage=end_stage,vernalization_required=calendar_season=='winter_wheat')
            metadata.loc[valid,'status'] = np.where(result['complete_season'],'complete','soft_dough_not_reached')
            metadata.loc[valid,'endpoint_date'] = result['endpoint_dates']
            metadata.loc[valid,'primary_exposure_date'] = result['primary_exposure_dates']
            for rank in range(7):
                metadata.loc[valid,f'leaf{rank+1}_final_damage_percent'] = result['final_damage_percent'][:,rank]
                for j,cutoff in enumerate(result['cutoff_percent']):
                    metadata.loc[valid,f'leaf{rank+1}_onset_{cutoff:g}pct_date'] = result['first_visible_dates'][:,rank,j]
            metadata.loc[valid,'upper3_final_damage_percent'] = result['final_damage_percent'][:,:3].mean(axis=1)
            for stage,event_dates in result['stage_dates'].items():
                metadata.loc[valid,f'bbch{stage}_date'] = event_dates
            numerical.append(dict(first_cell_index=first,mass_error=result['mass_error'],minimum_state=result['minimum_state']))
        metadata.loc[eligible[indices]&~valid&~unsupported,'status'] = 'missing_weather'
        rows.append(metadata)
    frame = pd.concat(rows,ignore_index=True)
    output.parent.mkdir(parents=True,exist_ok=True)
    tmp = output.with_suffix('.tmp.parquet')
    frame.to_parquet(tmp,index=False,compression='zstd');tmp.replace(output)
    receipt = dict(provider=provider,model=model,scenario=scenario,harvest_year=year,
        calendar=code,bias_adjusted=adjusted,configuration_sha256=sha(DEST/'configuration_before_regional_results.json'),
        phenology_sha256=sha(phenology_path),fitted_parameter_sha256=sha(frozen_path),
        execution_source_sha256={name:sha(ROOT/'model/seasonal_septoria'/name) for name in
                                 ['core.py','host.py','regional.py','climate_alignment.py']},
        execution_driver_sha256=sha(Path(__file__)),
        source_forcing=sources,coefficient_sha256=sha(coefficient_path) if adjusted else None,
        status_counts=frame.status.value_counts().to_dict(),total_harvested_ha=float(frame.harvested_total_ha.sum()),
        complete_harvested_ha=float(frame.loc[frame.status.eq('complete'),'harvested_total_ha'].sum()),
        numerical=numerical,parquet_sha256=sha(output),publication_ready=False)
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    return {'status':'simulated',**{key:receipt[key] for key in ['harvest_year','model','scenario','calendar','status_counts','complete_harvested_ha']}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provider',choices=['era5','nasa'],default='era5')
    parser.add_argument('--model',default='ERA5')
    parser.add_argument('--scenario',default='baseline')
    parser.add_argument('--years',type=int,nargs='+',default=list(range(1991,2021)))
    parser.add_argument('--calendar-season',choices=['winter_wheat','spring_wheat'],default='winter_wheat')
    parser.add_argument('--water',choices=['rainfed','irrigated'],default='rainfed')
    parser.add_argument('--adjusted',action='store_true')
    parser.add_argument('--sowing-offset',type=int,default=0)
    parser.add_argument('--end-stage',type=int,default=85)
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    pending = set(args.years)
    while pending:
        for year in sorted(pending):
            try:
                result = run_season(args.provider,args.model,args.scenario,year,args.calendar_season,args.water,
                                    args.adjusted,args.sowing_offset,args.end_stage)
                pending.remove(year)
                print(json.dumps(result),flush=True)
            except FileNotFoundError:
                pass
        if pending and not args.watch:
            raise RuntimeError(f'Incomplete forcing for harvest years{sorted(pending)}')
        if pending:
            print(json.dumps({'waiting_for_forcing_years':sorted(pending)}),flush=True)
            time.sleep(30)


if __name__=='__main__':
    main()
