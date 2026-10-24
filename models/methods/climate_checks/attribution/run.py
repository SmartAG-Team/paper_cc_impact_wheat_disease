"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/review_response_20261010/attribution')
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from analysis.methods.retained_validation.climate.decomposition import run as original
from analysis.methods.climate_checks.attribution.utilities import pair_years, sha, validate_checkpoint
HERE = evidence_dir('analysis/paper_study/review_response_20261010/attribution')
ROOT = PROJECT_ROOT
REGISTRATION = HERE / 'prospective_protocol.json'
METRICS = original.METRICS

def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def resolve_source(relative, expected, roots=None):
    if roots is None:
        path = resolve(relative)
        if path.is_file() and sha(path) == expected:
            return path
        raise FileNotFoundError(relative)
    found = []
    for root in roots:
        path = Path(root) / relative
        if path.is_file():
            if sha(path) == expected:
                return path
            found.append(str(path))
    if found:
        raise ValueError(f'No source matches registered checksum: {relative}')
    raise FileNotFoundError(relative)

def select_diagonal(frame, profile, model, year, draw_ids, metrics=METRICS):
    rows = frame.loc[frame.setting.eq(profile) & frame.model.eq(model) & frame.harvest_year.eq(year)].sort_values('spatial_draw_id')
    if not np.array_equal(rows.spatial_draw_id.to_numpy(), draw_ids):
        raise ValueError('Diagonal profile/year has incomplete or duplicate draw support.')
    return rows[['status', 'valid_complete_season', *metrics]].reset_index(drop=True)

def verify_sources(registration):
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def forcing(model, year, draws, coeff, reg):
    frames = []
    dates = []
    for y in [year - 1, year]:
        path = Path(reg['raw_cache_paths'][model][str(y)])
        if sha(path) != reg['source_sha256'][str(path)]:
            raise ValueError('Raw cache checksum changed.')
        with np.load(path) as cache:
            index = {cell: i for i, cell in enumerate(cache['cell_ids'].tolist())}
            selected = draws.cell_id.map(index).to_numpy(int)
            frames.append({k: cache[k][selected].copy() for k in original.FIELDS})
            dates.append(cache['dates'].copy())
    dates = pd.DatetimeIndex(np.concatenate(dates))
    weather = original.apply_monthly_alignment({k: np.concatenate([x[k] for x in frames], axis=1) for k in original.FIELDS}, dates.month, coeff)
    sow = np.array([original.sowing_date_from_calendar(year, r.planting_doy, r.maturity_doy).to_datetime64().astype('datetime64[D]') for r in draws.itertuples()])
    return dict(dates=dates, weather=weather, sowing=sow)

def run_model(model, reg):
    draws, phenology, _, _ = original.inputs()
    fit = reg['fitted']
    diagonal = pd.read_parquet(reg['diagonal_path'])
    diagonal = diagonal.loc[diagonal.model.eq(model)]
    c = pd.read_parquet(reg['coefficient_paths'][model]).set_index(['cell_id', 'month']).reindex(pd.MultiIndex.from_product([draws.cell_id, range(1, 13)], names=['cell_id', 'month']))
    coeff = {k: c[k].to_numpy().reshape(64, 12) for k in ['temperature_offset', 'humidity_logit_offset', 'precipitation_ratio']}
    regsha = sha(REGISTRATION)
    reports = []
    for name, setting in reg['settings'].items():
        thresholds = {int(k): v for k, v in setting['thresholds'].items()}
        reference, future = pair_years(setting['future_cyclic_shift'])
        for i, (yr, yf) in enumerate(zip(reference, future)):
            path = HERE / 'paired_outputs' / name / model / f'{yr}.parquet'
            receipt = path.with_suffix('.json')
            if path.exists() or receipt.exists():
                reports.append(validate_checkpoint(path, receipt, regsha))
                continue
            started = time.perf_counter()
            ref = forcing(model, int(yr), draws, coeff, reg)
            fut = forcing(model, int(yf), draws, coeff, reg)
            rh = original.host_inputs(ref, draws, phenology, fit, thresholds)
            fh = original.host_inputs(fut, draws, phenology, fit, thresholds)
            corners = {}
            checks = {}
            hot = {}
            for corner, year in [('Wreference_Hreference', yr), ('Wfuture_Hfuture', yf)]:
                corners[corner] = select_diagonal(diagonal, setting['profile'], model, year, draws.spatial_draw_id.to_numpy())
            for corner, w, h, hs in [('Wfuture_Hreference', fut, ref, rh), ('Wreference_Hfuture', ref, fut, fh)]:
                corners[corner], checks[corner] = original.hybrid(w, h, hs, draws, phenology, fit, thresholds)
                aligned, _ = original.align_crop_day(w, h, hs['mask'])
                hot[corner] = ((aligned['tmean_c'] > 40) & hs['mask']).sum(axis=1)
            excluded = hot['Wfuture_Hreference'] + hot['Wreference_Hfuture'] > 0
            frames = []
            for corner, frame in corners.items():
                frame = frame.copy()
                frame['corner'] = corner
                frame['model'] = model
                frame['setting'] = name
                frame['profile'] = setting['profile']
                frame['future_cyclic_shift'] = setting['future_cyclic_shift']
                frame['reference_year'] = int(yr)
                frame['future_year'] = int(yf)
                frame['excluded_unsupported_hybrid_forcing'] = excluded
                for k, v in hot.items():
                    frame[f'{k}_meanT_above40_days'] = v
                for k in ['spatial_draw_id', 'cell_id', 'stratum', 'area_mean_weight']:
                    frame[k] = draws[k].to_numpy()
                frames.append(frame)
            output = pd.concat(frames, ignore_index=True)
            path.parent.mkdir(parents=True, exist_ok=True)
            output.to_parquet(path, index=False)
            record = dict(model=model, setting=name, reference_year=int(yr), future_year=int(yf), rows=len(output), seconds=time.perf_counter() - started, checks=checks, output_sha256=sha(path), registration_sha256=regsha)
            write_json(receipt, record)
            reports.append(record)
        print(model, name, '30/30 complete', flush=True)
    return reports
