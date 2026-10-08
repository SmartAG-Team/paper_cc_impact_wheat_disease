"""Bounded workers for complete verified annual European scenario simulations."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import sys
import time

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional_fast import DEST,MODELS,SCENARIOS,configure,paths_for,run_season,sha


def execute(job):
    model,scenario,year,season = job
    return run_season('nasa',model,scenario,year,season,'rainfed',adjusted=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--calendar-season',choices=['winter_wheat','spring_wheat'],default='winter_wheat')
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    if not 1<=args.workers<=6:
        raise ValueError('Workers must be1–6.')
    configure()
    years = list(range(1991,2021))+list(range(2031,2061))+list(range(2071,2101))
    pending = {(model,scenario,year,args.calendar_season) for model in MODELS for scenario in SCENARIOS for year in years}
    status_path = DEST/f'supervision_{args.calendar_season}.json'
    while pending:
        ready = []
        for model,scenario,year,season in sorted(pending):
            output = DEST/'annual/nasa'/model/scenario/'adjusted'/f'{season}_rainfed_sow+0_stage85'/f'{year}.parquet'
            if output.exists() and output.with_suffix('.json').exists():
                receipt=json.loads(output.with_suffix('.json').read_text())
                if sha(output)!=receipt['parquet_sha256']:
                    raise ValueError('Completed regional archive checksum differs.')
                pending.remove((model,scenario,year,season));continue
            if not (DEST/'bias_alignment'/model/scenario/'coefficients.parquet').exists():
                continue
            paths = paths_for('nasa',model,scenario,pd.Timestamp(year-1,1,1),pd.Timestamp(year,12,31))
            if all(path.exists() and path.with_suffix('.json').exists() for path in paths):
                ready.append((model,scenario,year,season))
        status = dict(updated_utc=pd.Timestamp.now(tz='UTC').isoformat(),pending_seasons=len(pending),
                      ready_seasons=len(ready),expected_seasons=810,workers=args.workers,
                      calendar_season=args.calendar_season,publication_ready=False)
        status_path.write_text(json.dumps(status,indent=2)+'\n')
        if ready:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures = {pool.submit(execute,job):job for job in ready}
                for future in as_completed(futures):
                    result = future.result()
                    pending.remove(futures[future])
                    status.update(updated_utc=pd.Timestamp.now(tz='UTC').isoformat(),
                        pending_seasons=len(pending),completed_seasons=810-len(pending))
                    status_path.write_text(json.dumps(status,indent=2)+'\n')
                    print(json.dumps(result),flush=True)
        if pending and not args.watch:
            raise RuntimeError(f'Forcing not yet complete for{len(pending)} crop seasons.')
        if pending:
            print(json.dumps({'pending_seasons':len(pending),'calendar_season':args.calendar_season}),flush=True)
            time.sleep(30)
    status_path.write_text(json.dumps(dict(status='complete',expected_seasons=810,completed_seasons=810,
                                         calendar_season=args.calendar_season,publication_ready=False),indent=2)+'\n')


if __name__=='__main__':
    main()
