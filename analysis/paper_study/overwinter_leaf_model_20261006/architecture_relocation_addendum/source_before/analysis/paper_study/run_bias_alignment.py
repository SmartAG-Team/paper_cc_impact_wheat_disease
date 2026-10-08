"""Accumulate exact full-baseline monthly forcing means without future data."""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional import DATA,DEST,FIELDS,MODELS,SCENARIOS,sha
from model.seasonal_septoria.climate_alignment import alignment_from_monthly_means,RH_EPSILON


def partial_statistics(path,cells):
    receipt_path = path.with_suffix('.json')
    if not path.exists() or not receipt_path.exists():
        raise FileNotFoundError(path)
    receipt = json.loads(receipt_path.read_text())
    digest = sha(path)
    if digest!=receipt.get('parquet_sha256'):
        raise ValueError('Historical forcing hash mismatch.')
    cache = DEST/'bias_alignment/partial_statistics'/digest[:2]/f'{digest}.npz'
    if cache.exists():
        with np.load(cache) as archive:
            return {key:archive[key] for key in archive.files},digest
    frame = pq.read_table(path,columns=['date','cell_id',*FIELDS]).to_pandas()
    dates = pd.DatetimeIndex(pd.to_datetime(frame.date.unique()))
    if (len(frame)!=len(cells)*len(dates) or not dates.equals(pd.date_range(dates[0],dates[-1]))
            or not np.all(frame.cell_id.to_numpy().reshape(len(dates),len(cells))==cells.cell_id.to_numpy()[None,:])):
        raise ValueError('Historical dates/cells are incomplete or misordered.')
    included = (dates.year>=1991)&(dates.year<=2020)
    months = dates.month.to_numpy()
    values = {key:frame[key].to_numpy(float).reshape(len(dates),len(cells)).T for key in FIELDS}
    if not all(np.isfinite(x).all() for x in values.values()):
        raise ValueError('Historical forcing contains missing values.')
    values['rh_logit'] = logit(np.clip(values['rh_mean_pct']/100,RH_EPSILON,1-RH_EPSILON))
    statistics = {'count':np.array([np.sum(included&(months==month)) for month in range(1,13)],int)}
    for key in ['tmean_c','rh_logit','precipitation_mm']:
        statistics[key] = np.column_stack([values[key][:,included&(months==month)].sum(axis=1)
                                           for month in range(1,13)])
    cache.parent.mkdir(parents=True,exist_ok=True)
    temporary = cache.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**statistics);temporary.replace(cache)
    return statistics,digest


def full_means(paths,cells):
    sums = {key:np.zeros((len(cells),12)) for key in ['tmean_c','rh_logit','precipitation_mm']}
    count = np.zeros(12,int)
    sources = []
    for path in paths:
        statistics,digest = partial_statistics(path,cells)
        count += statistics['count']
        for key in sums:
            sums[key] += statistics[key]
        sources.append(dict(path=str(path.relative_to(ROOT)),sha256=digest))
    dates = pd.date_range('1991-01-01','2020-12-31')
    expected = np.array([(dates.month==m).sum() for m in range(1,13)],int)
    if not np.array_equal(count,expected):
        raise ValueError('Monthly baseline day counts differ from complete1991–2020.')
    return {key:value/count[None,:] for key,value in sums.items()},sources,count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--watch',action='store_true')
    args = parser.parse_args()
    cells = pd.read_parquet(DATA/'wheat_area/europe_wheat_cells_025.parquet')
    months = pd.date_range('1991-01-01','2020-12-01',freq='MS')
    reference_paths = [DATA/'climate/era5_daily'/f'{m:%Y-%m-%d}_{m+pd.offsets.MonthBegin():%Y-%m-%d}.parquet' for m in months]
    reference = None
    pending = {(model,scenario) for model in MODELS for scenario in SCENARIOS}
    while pending:
        if reference is None:
            try:
                reference,reference_sources,count = full_means(reference_paths,cells)
                print(json.dumps({'reference_baseline':'complete','daily_count':int(count.sum())}),flush=True)
            except FileNotFoundError:
                pass
        if reference is not None:
            for model,scenario in sorted(pending):
                dest = DEST/'bias_alignment'/model/scenario
                output = dest/'coefficients.parquet'
                receipt_path = dest/'receipt.json'
                if output.exists() and receipt_path.exists():
                    if json.loads(receipt_path.read_text())['coefficient_sha256']!=sha(output):
                        raise ValueError('Archived coefficients changed.')
                    pending.remove((model,scenario));continue
                model_paths = [DATA/'climate/nasa_v2/daily'/model/('historical' if y<=2014 else scenario)/f'{y}.parquet' for y in range(1991,2021)]
                try:
                    historical,model_sources,model_count = full_means(model_paths,cells)
                except FileNotFoundError:
                    continue
                coefficients = alignment_from_monthly_means(reference,historical)
                table = pd.DataFrame({'cell_id':np.repeat(cells.cell_id,12),'month':np.tile(np.arange(1,13),len(cells))})
                for key,value in coefficients.items():
                    table[key] = value.ravel()
                dest.mkdir(parents=True,exist_ok=True)
                table.to_parquet(output,index=False,compression='zstd')
                receipt = dict(model=model,scenario=scenario,baseline_start='1991-01-01',baseline_end='2020-12-31',
                    baseline_days=int(count.sum()),month_day_counts=count.tolist(),cells=len(cells),future_forcing_used=False,
                    reference_forcing=reference_sources,model_forcing=model_sources,
                    coefficient_sha256=sha(output),algorithm_sha256=sha(ROOT/'model/seasonal_septoria/climate_alignment.py'),
                    humidity_logit_bound_percent=[.1,99.9],temperature_correction_applies_to_mean_and_maximum=True,
                    wet_day_frequency_corrected=False,distribution_tails_corrected=False,
                    coefficient_ranges={key:[float(value.min()),float(value.max())] for key,value in coefficients.items()})
                receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
                pending.remove((model,scenario))
                print(json.dumps({'alignment':'complete','model':model,'scenario':scenario}),flush=True)
        if pending and not args.watch:
            raise RuntimeError(f'Incomplete baseline forcing for{sorted(pending)}')
        if pending:
            print(json.dumps({'pending_alignment':sorted(pending)}),flush=True)
            time.sleep(30)


if __name__=='__main__':
    main()
