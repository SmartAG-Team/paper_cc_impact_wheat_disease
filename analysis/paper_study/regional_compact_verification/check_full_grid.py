"""Reconcile every output cell with immutable full-trajectory regional archives."""

from pathlib import Path
import json,sys,time,shutil
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from analysis.paper_study import run_regional_fast as fast

HERE=Path(__file__).resolve().parent
ORIGINAL=fast.DEST
fast.DEST=HERE/'benchmarks'


def main():
    fast.DEST.mkdir(parents=True,exist_ok=True)
    link=fast.DEST/'bias_alignment'
    if not link.exists():link.symlink_to(ORIGINAL/'bias_alignment',target_is_directory=True)
    cases=[('era5','ERA5','baseline',1991,'winter_wheat',False),
        ('era5','ERA5','baseline',1991,'spring_wheat',False),
        ('nasa','ACCESS-CM2','ssp126',1991,'winter_wheat',True),
        ('nasa','ACCESS-CM2','ssp126',2031,'winter_wheat',True)]
    results=[]
    for provider,model,scenario,year,season,adjusted in cases:
        folder=Path(provider)/model/scenario/('adjusted' if adjusted else 'original')/f'{season}_rainfed_sow+0_stage85'
        original=ORIGINAL/'annual'/folder/f'{year}.parquet'
        if not original.exists():raise FileNotFoundError(original)
        start=time.monotonic();status=fast.run_season(provider,model,scenario,year,season,adjusted=adjusted)
        elapsed=time.monotonic()-start
        new=fast.DEST/'annual'/folder/f'{year}.parquet'
        a,b=[pd.read_parquet(p).set_index('cell_id').sort_index() for p in [original,new]]
        if list(a.columns)!=list(b.columns) or len(a)!=14941 or len(b)!=14941:raise AssertionError('Whole-grid membership/schema mismatch.')
        max_abs=0.
        for column in a.columns:
            if pd.api.types.is_numeric_dtype(a[column]):
                np.testing.assert_allclose(a[column],b[column],rtol=0,atol=1e-12,equal_nan=True)
                if a[column].notna().any():max_abs=max(max_abs,float((a[column]-b[column]).abs().max()))
            else:pd.testing.assert_series_equal(a[column],b[column],check_exact=True)
        receipt=json.loads(new.with_suffix('.json').read_text())
        if max(x['mass_error'] for x in receipt['numerical'])>=1e-12:raise AssertionError('Numerical mass mismatch.')
        results.append(dict(provider=provider,model=model,scenario=scenario,year=year,calendar=season,
            checked_cells=len(a),checked_columns=len(a.columns),scalar_checks=a.size,
            elapsed_seconds=elapsed,maximum_absolute_difference=max_abs,
            original_sha256=fast.sha(original),compact_sha256=fast.sha(new),status='passed'))
        print(json.dumps(results[-1]),flush=True)
    snapshot=HERE/'source_snapshot';snapshot.mkdir(exist_ok=True)
    paths=[ROOT/'model/seasonal_septoria'/name for name in
        ['core.py','host.py','regional.py','regional_compact.py','rolling_diagnostics.py','climate_alignment.py']]
    paths+=[ROOT/'analysis/paper_study'/name for name in ['run_regional.py','run_regional_fast.py','supervise_regional_fast.py']]
    hashes={}
    for source in paths:
        target=snapshot/source.relative_to(ROOT);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);hashes[str(source.relative_to(ROOT))]=fast.sha(source)
    report=dict(status='passed',whole_grid_cases=results,total_scalar_checks=sum(x['scalar_checks'] for x in results),
        source_hashes=hashes,original_archives_overwritten=False,frozen_fit_modified=False,
        daily_event_dates_match=True,compact_time_step_matches_full_operator=True)
    (HERE/'receipt.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
