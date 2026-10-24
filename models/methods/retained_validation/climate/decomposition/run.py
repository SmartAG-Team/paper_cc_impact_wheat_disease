"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/nature_food_fix_20261007/climate/decomposition')
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
from analysis.methods.retained_validation.climate import run as robustness
from model.seasonal_septoria.host import sowing_date_from_calendar
from model.seasonal_septoria.regional import phenology_window, tpv_accumulation
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.wetness import duration_exposure
from model.seasonal_septoria.overwinter import OverwinterParameters, simulate_overwinter
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment
HERE = evidence_dir('analysis/paper_study/nature_food_fix_20261007/climate/decomposition')
ROOT = PROJECT_ROOT
CLIMATE = HERE.parent
ORIGINAL = robustness.original
MODELS = ORIGINAL.MODELS
YEARS = list(range(1991, 2021))
FIELDS = ORIGINAL.cache.FIELDS
METRICS = [f'F{leaf}_{event}_relative_anthesis_days' for leaf in (1, 2, 3) for event in ('infection', 'symptom')] + ['GS65_85_lost_had3', 'GS31_85_lost_had3', 'GS65_85_functional_lost_fraction', 'GS65_85_reference_had3', 'grain_fill_days']
COMPONENTS = ['total_change', 'weather_alone', 'host_alone', 'interaction', 'weather_shapley', 'host_shapley']

def sha(path):
    return robustness.sha(path)

def write_json(path, value):
    return robustness.write_json(path, value)

def coefficients(model, draws):
    frame = pd.read_parquet(ORIGINAL.coefficient_path(model, 'ssp585'))
    assert not frame.duplicated(['cell_id', 'month']).any()
    frame = frame.set_index(['cell_id', 'month']).reindex(pd.MultiIndex.from_product([draws.cell_id, range(1, 13)], names=['cell_id', 'month']))
    return {k: frame[k].to_numpy().reshape(64, 12) for k in ['temperature_offset', 'humidity_logit_offset', 'precipitation_ratio']}

def forcing(model, year, draws, coeff):
    prev, pw = ORIGINAL.load_cache_year(model, 'ssp585', year - 1, draws)
    cur, cw = ORIGINAL.load_cache_year(model, 'ssp585', year, draws)
    dates = pd.DatetimeIndex(np.concatenate([prev, cur]))
    weather = apply_monthly_alignment({k: np.concatenate([pw[k], cw[k]], axis=1) for k in FIELDS}, dates.month, coeff)
    sow = np.array([sowing_date_from_calendar(year, r.planting_doy, r.maturity_doy).to_datetime64().astype('datetime64[D]') for r in draws.itertuples()])
    return dict(dates=dates, weather=weather, sowing=sow)

def host_inputs(f, draws, phenology, fit, thresholds):
    dates = f['dates']
    raw = f['weather']
    sow = f['sowing']
    days = len(dates)
    included = dates.to_numpy(dtype='datetime64[D]')[None, :] >= sow[:, None]
    finite = np.logical_and.reduce([np.isfinite(raw[k]) for k in FIELDS])
    finite &= (raw['tmax_c'] >= raw['tmean_c'] - 0.001) & (raw['rh_mean_pct'] >= 0) & (raw['rh_mean_pct'] <= 100) & (raw['precipitation_mm'] >= 0)
    gap = np.where((included & ~finite).any(axis=1), (included & ~finite).argmax(axis=1), days)
    prefix = included & (np.arange(days)[None, :] < gap[:, None])
    safe = {k: np.where(finite, raw[k], 0.0) for k in FIELDS}
    mask, complete, endpoint, unsupported = phenology_window(safe['tmean_c'], safe['tmax_c'], draws.latitude.to_numpy(), dates.dayofyear, prefix, phenology, float(thresholds[85]), True)
    valid = complete & ~unsupported
    mask &= valid[:, None]
    t = np.where(mask, safe['tmean_c'], 0.0)
    tx = np.where(mask, safe['tmax_c'], 0.0)
    a = tpv_accumulation(t, tx, draws.latitude.to_numpy(), dates.dayofyear, mask, phenology, True)
    host = leaf_host(a, t, thresholds, rank_spacing_units=fit['rank_spacing_units'], forcing_mask=mask, juvenile_policy=fit['juvenile_policy'])
    return dict(host=host, mask=mask, valid=valid, accumulation=a)

def align_crop_day(weather_source, host_target, target_mask):
    """One-to-one real-day mapping by days since each crop-calendar sowing.

    Only target crop-window values are material. No leap-day interpolation,
    repeated weather, padding of active forcing or calendar-date assumptions.
    """
    target_dates = host_target['dates'].to_numpy(dtype='datetime64[D]')
    source_dates = weather_source['dates'].to_numpy(dtype='datetime64[D]')
    elapsed = (target_dates[None, :] - host_target['sowing'][:, None]).astype('timedelta64[D]').astype(int)
    source_index = (weather_source['sowing'][:, None] - source_dates[0]).astype('timedelta64[D]').astype(int) + elapsed
    in_range = (source_index >= 0) & (source_index < len(source_dates))
    unavailable = target_mask & ~in_range
    if unavailable.any():
        raise ValueError(f'Active crop days lack actual source forcing: {int(unavailable.sum())}.')
    field = np.arange(len(source_index))[:, None]
    clipped = np.clip(source_index, 0, len(source_dates) - 1)
    aligned = {k: np.where(target_mask, np.asarray(weather_source['weather'][k])[field, clipped], 0.0) for k in FIELDS}
    assert all((np.isfinite(v).all() for v in aligned.values()))
    assert all((np.all(np.diff(source_index[i, target_mask[i]]) == 1) for i in range(len(source_index))))
    return (aligned, dict(active_target_days=int(target_mask.sum()), unavailable_active_weather_days=int(unavailable.sum()), alignment='elapsed calendar days from actual sowing; each active source day consumed once', source_date_origin=str(pd.Timestamp(source_dates[0]).date()), target_date_origin=str(pd.Timestamp(target_dates[0]).date()), source_forcing_days=len(source_dates), target_forcing_days=len(target_dates)))

def hybrid(weather_source, host_target, host_state, draws, phenology, fit, thresholds):
    w, alignment = align_crop_day(weather_source, host_target, host_state['mask'])
    exposure = duration_exposure(w['tmean_c'], w['tmax_c'], w['rh_mean_pct'], w['precipitation_mm'], fit['weather_preprocessing']['rain_rate_mm_hour'])['exposure']
    host = host_state['host']
    mask = host_state['mask']
    trajectory = simulate_overwinter(w['tmean_c'], exposure, w['precipitation_mm'], host.active, host.renewal, host.area, mask, OverwinterParameters(**fit['parameters']), initial_local_source=1.0, imported_pressure=np.full_like(w['tmean_c'], fit['constant_imported_pressure']), detection_fraction=0.001)
    rows, checks = robustness.simulation.simulate_draw_seasons(host_target['dates'], host_target['weather'], host_target['sowing'], draws.latitude.to_numpy(), phenology, fit, thresholds, trajectory_override=trajectory)
    assert np.array_equal(rows.valid_complete_season.to_numpy(), host_state['valid'])
    return (rows[['status', 'valid_complete_season', *METRICS]], dict(**alignment, mass_error=float(checks['mass_error']), disease_weather_meanT_above40_active_days=int(((w['tmean_c'] > 40) & mask).sum())))

def inputs():
    draws = pd.read_csv(ORIGINAL.cache.DRAW_PATH)
    phenology = json.loads((ROOT / 'process_model/parameters/calibration.json').read_text())
    fit = json.loads(ORIGINAL.FIT.read_text())['fitted']
    thresholds = {int(k): v for k, v in json.loads(ORIGINAL.STAGES.read_text())['all_stage_thresholds'].items()}
    return (draws, phenology, fit, thresholds)

def pilot():
    draws, phenology, fit, thresholds = inputs()
    coeff = coefficients(MODELS[0], draws)
    records = []
    for year in (1991, 2020):
        r = forcing(MODELS[0], year, draws, coeff)
        f = forcing(MODELS[0], year + 80, draws, coeff)
        rh = host_inputs(r, draws, phenology, fit, thresholds)
        fh = host_inputs(f, draws, phenology, fit, thresholds)
        for label, w, h, hs in [('Wfuture_Hreference', f, r, rh), ('Wreference_Hfuture', r, f, fh)]:
            t = time.perf_counter()
            rows, details = hybrid(w, h, hs, draws, phenology, fit, thresholds)
            records.append(dict(reference_year=year, label=label, seconds=time.perf_counter() - t, complete=int(rows.valid_complete_season.sum()), **details))
            print(records[-1], flush=True)
    write_json(HERE / 'pilot_cost.json', dict(pilot=records, planned_hybrid_batches=180, estimated_serial_hybrid_seconds=float(np.median([r['seconds'] for r in records]) * 180), outcomes_not_used_to_select_or_modify_decomposition=True))

def run_model(model, reg):
    draws, phenology, fit, thresholds = inputs()
    coeff = coefficients(model, draws)
    baseline = pd.read_parquet(CLIMATE / 'draw_season_outputs.parquet')
    baseline = baseline.loc[baseline.setting.eq('baseline') & baseline.model.eq(model)]
    reports = []
    for i, year in enumerate(YEARS):
        path = HERE / 'paired_outputs' / model / f'{year}.parquet'
        receipt = path.with_suffix('.json')
        if path.exists() or receipt.exists():
            assert path.exists() and receipt.exists()
            r = json.loads(receipt.read_text())
            assert sha(path) == r['output_sha256']
            reports.append(r)
            continue
        started = time.perf_counter()
        ref = forcing(model, year, draws, coeff)
        future = forcing(model, year + 80, draws, coeff)
        rh = host_inputs(ref, draws, phenology, fit, thresholds)
        fh = host_inputs(future, draws, phenology, fit, thresholds)
        corners = {}
        checks = {}
        for name, y in [('Wreference_Hreference', year), ('Wfuture_Hfuture', year + 80)]:
            corners[name] = baseline.loc[baseline.harvest_year.eq(y)].sort_values('spatial_draw_id')[['status', 'valid_complete_season', *METRICS]].reset_index(drop=True)
        for name, w, h, hs in [('Wfuture_Hreference', future, ref, rh), ('Wreference_Hfuture', ref, future, fh)]:
            corners[name], checks[name] = hybrid(w, h, hs, draws, phenology, fit, thresholds)
        frames = []
        for name, frame in corners.items():
            frame = frame.copy()
            frame['corner'] = name
            frame['model'] = model
            frame['reference_year'] = year
            frame['future_year'] = year + 80
            for k in ['spatial_draw_id', 'cell_id', 'stratum', 'area_mean_weight']:
                frame[k] = draws[k].to_numpy()
            frames.append(frame)
        output = pd.concat(frames, ignore_index=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        output.to_parquet(path, index=False)
        r = dict(model=model, reference_year=year, rows=len(output), seconds=time.perf_counter() - started, checks=checks, output_sha256=sha(path), registration_sha256=sha(HERE / 'registration_before_decomposition_results.json'))
        write_json(receipt, r)
        reports.append(r)
        if (i + 1) % 10 == 0:
            print(model, f'{i + 1}/30 pairs', r['seconds'], flush=True)
    return reports

def aggregate(corners, draws):
    summaries = []
    ensemble = []
    draw_records = []
    identities = []
    for model in MODELS:
        arrays = {}
        for name, g in corners.loc[corners.model.eq(model)].groupby('corner'):
            arrays[name] = g.sort_values(['spatial_draw_id', 'reference_year'])[METRICS].to_numpy(float).reshape(64, 30, len(METRICS))
        y00 = arrays['Wreference_Hreference']
        y10 = arrays['Wfuture_Hreference']
        y01 = arrays['Wreference_Hfuture']
        y11 = arrays['Wfuture_Hfuture']
        common = np.logical_and.reduce([np.isfinite(y) for y in [y00, y10, y01, y11]])
        values = {'total_change': y11 - y00, 'weather_alone': y10 - y00, 'host_alone': y01 - y00, 'interaction': y11 - y10 - y01 + y00, 'weather_shapley': 0.5 * (y10 - y00 + (y11 - y01)), 'host_shapley': 0.5 * (y01 - y00 + (y11 - y10))}
        error = np.where(common, values['weather_shapley'] + values['host_shapley'] - values['total_change'], 0.0)
        identities.append(float(np.max(np.abs(error))))
        for j, metric in enumerate(METRICS):
            z = common[:, :, j].mean(axis=1)
            for component in COMPONENTS:
                x = np.where(common[:, :, j], values[component][:, :, j], 0.0).mean(axis=1)
                s = ORIGINAL.ratio_summary(x, z, draws)
                summaries.append(dict(model=model, metric=metric, component=component, mean=s['mean'], spatial_mcse=s['spatial_mcse'], common_four_corner_area_time_coverage=s['coverage'], common_draw_year_pairs=int(common[:, :, j].sum()), draws_complete_all30=int(common[:, :, j].all(axis=1).sum())))
                for k, draw in draws.iterrows():
                    draw_records.append(dict(model=model, metric=metric, component=component, spatial_draw_id=int(draw.spatial_draw_id), numerator=float(x[k]), denominator=float(z[k]), area_mean_weight=draw.area_mean_weight, stratum=draw.stratum))
    summary = pd.DataFrame(summaries)
    mom = pd.DataFrame(draw_records)
    indexed = {(m, metric, c): g.sort_values('spatial_draw_id') for (m, metric, c), g in mom.groupby(['model', 'metric', 'component'])}
    for (metric, component), g in summary.groupby(['metric', 'component']):
        influences = []
        for row in g.itertuples():
            p = indexed[row.model, metric, component]
            influences.append((p.numerator.to_numpy() - row.mean * p.denominator.to_numpy()) / row.common_four_corner_area_time_coverage)
        ensemble.append(dict(metric=metric, component=component, gcm_mean=float(g['mean'].mean()), gcm_min=float(g['mean'].min()), gcm_max=float(g['mean'].max()), spatial_mcse_gcm_mean=robustness.shared_mcse(influences, draws), minimum_gcm_common_coverage=float(g.common_four_corner_area_time_coverage.min()), maximum_gcm_common_coverage=float(g.common_four_corner_area_time_coverage.max()), minimum_gcm_pairs=int(g.common_draw_year_pairs.min()), maximum_gcm_pairs=int(g.common_draw_year_pairs.max())))
    return (summary, pd.DataFrame(ensemble), mom, max(identities))
