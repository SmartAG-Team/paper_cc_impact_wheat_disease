"""Wait for the full comparison, build and verify before replacing publication."""
from pathlib import Path
from datetime import datetime,timezone
import argparse
import json
import os
import shutil
import time
from .data import HERE,ROOT,GRID,grid_ready


def write(value):
    value['updated_utc']=datetime.now(timezone.utc).isoformat();value['pid']=os.getpid()
    temporary=HERE/'publication_status.tmp.json';temporary.write_text(json.dumps(value,indent=2)+'\n')
    temporary.replace(HERE/'publication_status.json')


def finalize(wait=False):
    write(dict(status='waiting_for_full_grid',required_annual_jobs=810))
    while not grid_ready():
        if not wait:raise FileNotFoundError('Full-grid comparison has not finished.')
        simulation_status=json.loads((ROOT/'analysis/paper_study/full_grid_climate_20261008/status.json').read_text())
        if simulation_status['status']=='failed':raise RuntimeError('Climate batch failed; publication remains unchanged.')
        write(dict(status='waiting_for_full_grid',completed_annual_jobs=simulation_status['completed_annual_jobs'],
                   required_annual_jobs=810,estimated_simulation_hours=simulation_status['estimated_remaining_hours']))
        time.sleep(20)
    from .build import build
    from .verify import verify
    destination=ROOT/'publication/european_wheat_stb'
    stamp=datetime.now().strftime('%Y%m%d_%H%M%S')
    staging=HERE/f'publication_staging_{stamp}'
    write(dict(status='building',staging=str(staging)))
    build(staging,evidence_archive=destination/'Software_and_Evidence.zip')
    write(dict(status='verifying',staging=str(staging)))
    verify(staging)
    # Rendered pages are saved alongside the source for visual review.
    import pymupdf
    render=HERE/'render_check';render.mkdir(exist_ok=True)
    for name in ['Manuscript','Supplementary_Information']:
        pdf=pymupdf.open(staging/f'{name}.pdf')
        selected=list(range(len(pdf))) if name=='Manuscript' else [0,len(pdf)-1]
        for i in selected:pdf[i].get_pixmap(matrix=pymupdf.Matrix(1.4,1.4)).save(render/f'{name}_{i+1:02d}.png')
    backup=ROOT/f'publication/european_wheat_stb_before_climate_impacts_{stamp}'
    destination.rename(backup)
    try:shutil.move(str(staging),str(destination))
    except BaseException:
        if not destination.exists():backup.rename(destination)
        raise
    write(dict(status='complete',publication=str(destination),backup=str(backup),
        verified_reporting=True,visual_review_pending=True,all_planned_climate_periods_complete=True))
    return destination


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true')
    try:print(finalize(parser.parse_args().wait))
    except BaseException as error:
        write(dict(status='failed',error=repr(error)));raise
