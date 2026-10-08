"""Resumable full-European replay of the unchanged frozen crop/STB kernel."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,time
import numpy as np
import pandas as pd
import pyarrow as pa
from analysis.paper_study.overwinter_leaf_model_20261006.climate import run as frozen
from analysis.paper_study.run_regional import load_window,paths_for
from model.seasonal_septoria.host import sowing_date_from_calendar
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment

HERE=Path(__file__).resolve().parent
ROOT=frozen.ROOT
FIELDS=frozen.cache.FIELDS
AREA=ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet'
CALENDAR=ROOT/'data/paper_study/wheat_area/europe_wheat_calendar_scenarios.parquet'


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def merge_registry(cells,calendar):
    if not cells.cell_id.is_unique or not calendar.cell_id.is_unique:raise ValueError('Unique cell identifiers required.')
    result=cells.merge(calendar[['cell_id','planting_doy','maturity_doy','calendar_valid']],
        on='cell_id',how='left',sort=False,validate='one_to_one')
    result['calendar_valid']=result.calendar_valid.fillna(False).astype(bool)
    result['calendar_valid'] &= result.planting_doy.notna()&result.maturity_doy.notna()
    if result.cell_id.tolist()!=cells.cell_id.tolist():raise ValueError('Cell registry order changed.')
    return result


def alignment_arrays(coefficients,cell_ids):
    keys=['temperature_offset','humidity_logit_offset','precipitation_ratio']
    if coefficients.duplicated(['cell_id','month']).any():raise ValueError('Duplicate cell/month alignment.')
    joined=coefficients.set_index(['cell_id','month']).reindex(
        pd.MultiIndex.from_product([cell_ids,range(1,13)],names=['cell_id','month']))
    if not np.isfinite(joined[keys].to_numpy()).all():raise ValueError('Missing or invalid cell/month alignment.')
    return {key:joined[key].to_numpy().reshape(len(cell_ids),12) for key in keys}


def simulate_all_cells(dates,weather,cells,phenology,fit,thresholds,*,harvest_year,chunk_size=256,
                       simulator=frozen.simulate_draw_seasons,progress=False):
    if chunk_size<1:raise ValueError('Positive chunk size required.')
    dates=pd.DatetimeIndex(dates)
    if any(np.shape(weather[key])!=(len(cells),len(dates)) for key in FIELDS):raise ValueError('Weather/cell/day axes differ.')
    eligible=np.flatnonzero(cells.calendar_valid.to_numpy(bool))
    outputs=[];maximum_mass_error=0.
    for first in range(0,len(eligible),chunk_size):
        ids=eligible[first:first+chunk_size];part=cells.iloc[ids]
        sow=np.array([sowing_date_from_calendar(harvest_year,r.planting_doy,r.maturity_doy).to_datetime64().astype('datetime64[D]') for r in part.itertuples()])
        result,checks=simulator(dates,{k:weather[k][ids] for k in FIELDS},sow,part.latitude.to_numpy(),phenology,fit,thresholds)
        result['cell_id']=part.cell_id.to_numpy();outputs.append(result)
        maximum_mass_error=max(maximum_mass_error,checks['mass_error'])
        if progress and (first//chunk_size+1)%10==0:print(f'processed {min(first+chunk_size,len(eligible))}/{len(eligible)} eligible cells',flush=True)
    if not outputs:raise ValueError('No eligible crop calendars.')
    metrics=pd.concat(outputs,ignore_index=True)
    columns=['cell_id','latitude','longitude','harvested_total_ha']
    if 'dominant_source_country' in cells:columns.append('dominant_source_country')
    output=cells[columns].merge(metrics,on='cell_id',how='left',sort=False,validate='one_to_one')
    output['status']=output.status.fillna('missing_calendar')
    output['valid_complete_season']=output.valid_complete_season.fillna(False).astype(bool)
    assert output.cell_id.tolist()==cells.cell_id.tolist()
    return output,dict(maximum_mass_error=maximum_mass_error,eligible_cells=len(eligible),all_registered_cells=len(cells))


def configuration():
    inputs=[AREA,CALENDAR,frozen.FIT,frozen.STAGES,ROOT/'process_model/parameters/calibration.json',Path(__file__)]
    inputs += sorted((ROOT/'model').rglob('*.py'))
    return dict(schema_version=1,study='Full-domain frozen-current-model comparison',
        grid_resolution_degrees=.25,reference_crop='SPAM2020 all-wheat, imposed winter/rainfed calendar',
        model_parameters_changed=False,source_model='original28-field seasonal STB fit',
        retrospective_disease_validation=True,absolute_field_yield_forecast=False,
        conditional_yield_output='Parker top-three transfer per nominal upper-three LAI; separate empirical response remains unlinked',
        domain_cells=14941,expected_eligible_calendar_cells=14932,
        core_sources={str(p.relative_to(ROOT)):sha(p) for p in inputs})


def run_year(year,model='ACCESS-CM2',scenario='ssp585',provider='nasa',*,chunk_size=256):
    pa.set_cpu_count(2)
    if provider=='era5':model,scenario='ERA5','baseline'
    output=HERE/'annual_outputs'/provider/model/scenario/f'{year}.parquet'
    receipt_path=output.with_suffix('.json');config=configuration()
    config_path=HERE/'configuration_before_results.json'
    if config_path.exists():
        if json.loads(config_path.read_text())!=config:raise ValueError('Configuration changed; new output version required.')
    else:config_path.write_text(json.dumps(config,indent=2)+'\n')
    if output.exists() or receipt_path.exists():
        if not output.exists() or not receipt_path.exists():raise ValueError('Incomplete checkpoint.')
        prior=json.loads(receipt_path.read_text())
        if prior['output_sha256']!=sha(output) or prior['configuration_sha256']!=sha(config_path):raise ValueError('Checkpoint changed.')
        return prior
    started=time.perf_counter();cells=pd.read_parquet(AREA)
    calendars=pd.read_parquet(CALENDAR);calendar=calendars.loc[calendars.crop_season.eq('winter_wheat')&calendars.water_system.eq('rainfed')]
    cells=merge_registry(cells,calendar)
    assert len(cells)==14941 and cells.calendar_valid.sum()==14932
    start,end=pd.Timestamp(year-1,1,1),pd.Timestamp(year,12,31)
    paths=paths_for(provider,model,scenario,start,end)
    dates,weather,sources=load_window(paths,start,end,cells)
    coefficient_path=None
    if provider=='nasa':
        coefficient_path=frozen.coefficient_path(model,scenario)
        coefficients=alignment_arrays(pd.read_parquet(coefficient_path),cells.cell_id.tolist())
        weather=apply_monthly_alignment(weather,dates.month,coefficients)
    phenology=json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    fit=json.loads(frozen.FIT.read_text())['fitted']
    thresholds={int(k):v for k,v in json.loads(frozen.STAGES.read_text())['all_stage_thresholds'].items()}
    result,checks=simulate_all_cells(dates,weather,cells,phenology,fit,thresholds,
        harvest_year=year,chunk_size=chunk_size,progress=True)
    result['provider']=provider;result['model']=model;result['scenario']=scenario;result['harvest_year']=year
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix('.tmp.parquet');result.to_parquet(temporary,index=False,compression='zstd');temporary.replace(output)
    receipt=dict(status='complete',provider=provider,model=model,scenario=scenario,harvest_year=year,
        all_registered_cells=len(result),eligible_calendar_cells=checks['eligible_cells'],
        full_domain_census=True,chunk_size=chunk_size,configuration_sha256=sha(config_path),
        source_forcing=sources,alignment_sha256=sha(coefficient_path) if coefficient_path else None,
        output_sha256=sha(output),status_counts=result.status.value_counts().to_dict(),
        maximum_mass_error=checks['maximum_mass_error'],wall_seconds=time.perf_counter()-started,
        crop_area_ha=float(result.harvested_total_ha.sum()),
        complete_crop_area_ha=float(result.loc[result.valid_complete_season,'harvested_total_ha'].sum()),
        inference_scope='unchanged frozen-model spatial extension; not new epidemic or yield validation',
        completed_utc=datetime.now(timezone.utc).isoformat())
    receipt_path.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
    return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year',type=int,required=True);parser.add_argument('--provider',choices=['era5','nasa'],default='nasa')
    parser.add_argument('--model',default='ACCESS-CM2');parser.add_argument('--scenario',default='ssp585')
    parser.add_argument('--chunk-size',type=int,default=256)
    args=parser.parse_args();run_year(args.year,args.model,args.scenario,args.provider,chunk_size=args.chunk_size)
