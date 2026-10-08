"""Checkpointed census of every land-use wheat cell using the frozen runtime."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import argparse
import contextlib
import hashlib
import json
import os
import time

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MODELS=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
SCENARIOS=['ssp126','ssp245','ssp585']
YEARS=list(range(1991,2021))+list(range(2031,2061))+list(range(2071,2101))


def _sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def _write(path, record):
    temporary=Path(path).with_suffix('.tmp.json')
    temporary.write_text(json.dumps(record,indent=2)+'\n')
    temporary.replace(path)


def _worker(job,chunk_size):
    from analysis.paper_study.nature_food_revision_20261007.continental_replay import driver
    model,scenario,year=job
    log=HERE/'logs'/f'{model}__{scenario}__{year}.log'
    with log.open('a',buffering=1) as stream,contextlib.redirect_stdout(stream),contextlib.redirect_stderr(stream):
        return driver.run_year(year,model,scenario,chunk_size=chunk_size)


def run(workers=4,chunk_size=512):
    from analysis.paper_study.nature_food_revision_20261007.continental_replay import driver
    from analysis.paper_study.overwinter_leaf_model_20261006.climate import run as frozen
    (HERE/'logs').mkdir(exist_ok=True)
    configuration=driver.configuration()
    frozen_contract=driver.HERE/'configuration_before_results.json'
    if json.loads(frozen_contract.read_text())!=configuration:
        raise ValueError('Full-grid runtime or source configuration changed; retain the prior output version.')
    priority=[2001,2045,2085]+[year for year in YEARS if year not in {2001,2045,2085}]
    jobs=[(model,scenario,year) for year in priority for model in MODELS for scenario in SCENARIOS]
    assert len(jobs)==810 and len(set(jobs))==810
    forcing_paths={frozen.cache.source_path(model,scenario,y) for model,scenario,year in jobs for y in [year-1,year]}
    missing=[str(path) for path in forcing_paths if not path.is_file() or not path.with_suffix('.json').is_file()]
    if missing:raise FileNotFoundError('Required forcing files missing: '+', '.join(missing[:5]))
    contract=dict(study='Full-grid wheat land-use census',registered_utc=datetime.now(timezone.utc).isoformat(),
        landuse_mask='SPAM2020 positive wheat harvested-area cells at 0.25 degrees',
        crop_management='Existing GGCMI winter-wheat rainfed calendar imposed on the fixed all-wheat mask',
        domain_cells=14941,calendar_eligible_cells=14932,expected_grid_seasons=14932*810,
        models=MODELS,scenarios=SCENARIOS,periods=['1991-2020','2031-2060','2071-2100'],
        annual_jobs=len(jobs),unique_source_forcing_files=len(forcing_paths),
        scenario_specific_alignment=True,baseline_scenarios_not_aliased=True,
        model_parameters_changed=False,runtime_configuration_sha256=_sha(frozen_contract),
        runner_sha256=_sha(Path(__file__)),jobs=[dict(model=m,scenario=s,year=y) for m,s,y in jobs])
    contract_path=HERE/'run_configuration.json'
    if contract_path.exists():
        prior=json.loads(contract_path.read_text());contract['registered_utc']=prior['registered_utc']
        if contract!=prior:raise ValueError('Batch configuration changed; use a new version.')
    else:_write(contract_path,contract)
    completed=[];pending=[]
    for job in jobs:
        model,scenario,year=job
        parquet=driver.HERE/'annual_outputs/nasa'/model/scenario/f'{year}.parquet'
        receipt=parquet.with_suffix('.json')
        if parquet.exists() or receipt.exists():
            if not parquet.exists() or not receipt.exists():raise ValueError('Incomplete checkpoint: '+str(parquet))
            record=json.loads(receipt.read_text())
            if record['configuration_sha256']!=contract['runtime_configuration_sha256'] or record['output_sha256']!=_sha(parquet):
                raise ValueError('Checkpoint consistency failed: '+str(parquet))
            if record['all_registered_cells']!=14941 or record['eligible_calendar_cells']!=14932:
                raise ValueError('Checkpoint does not contain the entire land-use mask.')
            completed.append(job)
        else:pending.append(job)
    reused=len(completed);started=time.monotonic();active={};errors=[]
    def progress(status):
        elapsed=time.monotonic()-started;new=len(completed)-reused
        record=dict(status=status,pid=os.getpid(),started_utc=contract['registered_utc'],updated_utc=datetime.now(timezone.utc).isoformat(),
            workers=workers,chunk_size=chunk_size,total_annual_jobs=810,completed_annual_jobs=len(completed),
            reused_verified_jobs=reused,new_jobs_this_run=new,remaining_annual_jobs=810-len(completed),
            landuse_mask_cells=14941,calendar_eligible_cells=14932,
            completed_eligible_grid_seasons=14932*len(completed),elapsed_seconds=elapsed,
            estimated_remaining_hours=(elapsed/new*(810-len(completed))/3600 if new else None),
            active_jobs=[dict(model=m,scenario=s,year=y) for m,s,y in active.values()],errors=errors,
            output_root=str((driver.HERE/'annual_outputs/nasa').relative_to(ROOT)),
            completed_results_require_period_aggregation=len(completed)==810)
        _write(HERE/'status.json',record)
    progress('running')
    iterator=iter(pending)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in range(min(workers,len(pending))):
            job=next(iterator);active[pool.submit(_worker,job,chunk_size)]=job
        while active:
            done,_=wait(active,timeout=20,return_when=FIRST_COMPLETED)
            for future in done:
                job=active.pop(future)
                try:
                    record=future.result()
                    if record['status']!='complete':raise ValueError('Annual job did not finish.')
                    completed.append(job)
                    print(f'Full-grid annual jobs {len(completed)}/810: {job[0]}/{job[1]}/{job[2]}',flush=True)
                except Exception as error:
                    errors.append(dict(model=job[0],scenario=job[1],year=job[2],error=repr(error)))
                    progress('failed')
                    raise
                next_job=next(iterator,None)
                if next_job is not None:active[pool.submit(_worker,next_job,chunk_size)]=next_job
            progress('running')
    progress('simulations_complete')
    from .summarize import summarize
    summarize()
    progress('complete')
    return json.loads((HERE/'status.json').read_text())


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--chunk-size',type=int,default=512)
    args=parser.parse_args()
    if not 1<=args.workers<=8 or args.chunk_size<1:parser.error('Positive chunk size and 1–8 workers required.')
    print(json.dumps(run(args.workers,args.chunk_size),indent=2))
