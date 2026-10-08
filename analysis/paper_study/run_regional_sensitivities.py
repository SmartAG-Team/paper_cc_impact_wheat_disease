"""Frozen point-parameter sensitivities on the registered spatial sample."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.paper_study.regional_parameter_uncertainty import register_sample, load_sample_window, SAMPLES
from analysis.paper_study.run_regional import DATA, DEST as REGIONAL, MODELS, SCENARIOS, paths_for, sha
from model.seasonal_septoria.climate_alignment import apply_monthly_alignment
from model.seasonal_septoria.core import Parameters
from model.seasonal_septoria.host import cohort_inputs, sowing_date_from_calendar
from model.seasonal_septoria.regional import phenology_window, tpv_accumulation
from model.seasonal_septoria.terminal_ensemble import simulate_terminal_ensemble

DEST = ROOT / 'analysis/paper_study/regional_sensitivities'
OUT = DATA / 'regional_sensitivities'
CASES = ['main', 'sowing_minus14', 'sowing_plus14', 'endpoint75', 'unaligned_climate']


def endpoint_prediction(dates, weather, sow, draws, parameter, phenology, stage=85):
    thresholds = {row['BBCH']: row['Cumulative_t_pp_v_GDD'] for row in phenology['thresholds']}
    threshold = thresholds[85] if stage == 85 else (thresholds[51] + thresholds[85]) / 2
    included = dates.to_numpy(dtype='datetime64[D]')[None, :] >= sow[:, None]
    mask, complete, endpoint, unsupported = phenology_window(
        weather['tmean_c'], weather['tmax_c'], draws.latitude, dates.dayofyear,
        included, phenology, threshold, True)
    eligible = ~unsupported
    damage = np.full(len(draws), np.nan)
    mass_error = 0.
    if eligible.any():
        accumulated = tpv_accumulation(weather['tmean_c'][eligible], weather['tmax_c'][eligible],
            draws.latitude.to_numpy()[eligible], dates.dayofyear, mask[eligible], phenology, True)
        active, renewal = cohort_inputs(accumulated, weather['tmean_c'][eligible], thresholds)
        active &= mask[eligible, :, None]
        renewal[~active] = 0.
        result = simulate_terminal_ensemble(weather['tmean_c'][eligible], weather['rh_mean_pct'][eligible],
            weather['precipitation_mm'][eligible], active, renewal, endpoint[eligible], [parameter])
        values = result['damage'][0, :, :3].mean(axis=1) * 100
        values[~complete[eligible]] = np.nan
        damage[eligible] = values
        mass_error = result['mass_error']
    duration = endpoint - dates.get_indexer(pd.to_datetime(sow)) + 1
    duration = np.where(complete & eligible, duration, np.nan)
    return dict(damage=damage, complete=complete, unsupported=unsupported,
        crop_duration_days=duration, mass_error=mass_error)


def run_year(provider, model, scenario, year, draws, parameter, phenology):
    output = OUT / 'annual' / provider / model / scenario / f'{year}.npz'
    receipt_path = output.with_suffix('.json')
    if output.exists() and receipt_path.exists():
        if sha(output) != json.loads(receipt_path.read_text())['output_sha256']:
            raise ValueError('Frozen sensitivity archive changed.')
        return 'existing'
    start, end = pd.Timestamp(year - 1, 1, 1), pd.Timestamp(year, 12, 31)
    dates, raw, sources = load_sample_window(paths_for(provider, model, scenario, start, end), start, end, draws)
    coefficient_hash = None
    if provider == 'nasa':
        coefficient_path = REGIONAL / 'bias_alignment' / model / scenario / 'coefficients.parquet'
        coefficients = pd.read_parquet(coefficient_path).set_index(['cell_id', 'month']).reindex(
            pd.MultiIndex.from_product([draws.cell_id, range(1, 13)], names=['cell_id', 'month']))
        coefficients = {key: coefficients[key].to_numpy().reshape(len(draws), 12) for key in
            ['temperature_offset', 'humidity_logit_offset', 'precipitation_ratio']}
        aligned = apply_monthly_alignment(raw, dates.month, coefficients)
        coefficient_hash = sha(coefficient_path)
    else:
        aligned = raw
    sow = np.array([sowing_date_from_calendar(year, row.planting_doy, row.maturity_doy)
        .to_datetime64().astype('datetime64[D]') for row in draws.itertuples()], dtype='datetime64[D]')
    results = []
    for case in CASES:
        offset = -14 if case == 'sowing_minus14' else 14 if case == 'sowing_plus14' else 0
        weather = raw if case == 'unaligned_climate' else aligned
        results.append(endpoint_prediction(dates, weather, sow + np.timedelta64(offset, 'D'),
            draws, parameter, phenology, 75 if case == 'endpoint75' else 85))
    if provider == 'nasa':
        reference = SAMPLES / 'annual' / model / scenario / f'{year}.npz'
        if sha(reference) != json.loads(reference.with_suffix('.json').read_text())['output_sha256']:
            raise ValueError('Point reference archive changed.')
        expected = np.load(reference)['cell_damage_percent'][0]
    else:
        reference = REGIONAL / 'annual/era5/ERA5/baseline/original/winter_wheat_rainfed_sow+0_stage85' / f'{year}.parquet'
        if sha(reference) != json.loads(reference.with_suffix('.json').read_text())['parquet_sha256']:
            raise ValueError('Full-grid baseline reference changed.')
        frame = pd.read_parquet(reference).set_index('cell_id').reindex(draws.cell_id)
        expected = frame[[f'leaf{rank}_final_damage_percent' for rank in [1, 2, 3]]].mean(axis=1, skipna=False).to_numpy()
    np.testing.assert_allclose(results[0]['damage'], expected, rtol=0, atol=1e-10, equal_nan=True)
    if provider == 'era5':
        np.testing.assert_array_equal(results[0]['damage'], results[-1]['damage'])
    arrays = {key: np.stack([result[key] for result in results]) for key in
        ['damage', 'complete', 'unsupported', 'crop_duration_days']}
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.npz')
    np.savez_compressed(temporary, **arrays)
    temporary.replace(output)
    receipt = dict(provider=provider, model=model, scenario=scenario, harvest_year=year, cases=CASES,
        forcing_sources=sources, coefficient_sha256=coefficient_hash,
        spatial_draw_sha256=sha(SAMPLES / 'spatial_draws.csv'), reference_path=str(reference.relative_to(ROOT)),
        reference_sha256=sha(reference), frozen_point_reference_reconciled=True,
        maximum_mass_error=max(result['mass_error'] for result in results),
        output_sha256=sha(output), source_code_sha256=sha(Path(__file__)),
        parameters_refitted=False, spatial_sampling_error_requires_reporting=True)
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    return 'simulated'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    DEST.mkdir(parents=True, exist_ok=True)
    draws = register_sample()
    fit_path = ROOT / 'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json'
    parameter = Parameters(**json.loads(fit_path.read_text())['parameters'])
    phenology = json.loads((ROOT / 'process_model/parameters/calibration.json').read_text())
    configuration = dict(cases=CASES, sowing_offsets_days=[-14, 14],
        endpoint75='midpoint of transferred BBCH51 and BBCH85 developmental thresholds; unvalidated interpolation',
        unaligned_climate='original NEX physical fields before historical monthly ERA5 alignment',
        main_calendar='winter_wheat_rainfed_all_wheat_area', frozen_fit_sha256=sha(fit_path),
        spatial_draw_sha256=sha(SAMPLES / 'spatial_draws.csv'), source_code_sha256=sha(Path(__file__)),
        calibration_or_external_outcome_selection=False, parameter_refitting=False,
        spatial_sample_is_full_grid=False, registered_utc=datetime.now(timezone.utc).isoformat())
    path = DEST / 'configuration_before_sensitivity_results.json'
    if path.exists():
        previous = json.loads(path.read_text())
        for key in configuration:
            if key != 'registered_utc' and previous[key] != configuration[key]:
                raise ValueError('Registered sensitivity design changed.')
    else:
        path.write_text(json.dumps(configuration, indent=2) + '\n')
    jobs = [('era5', 'ERA5', 'baseline', year) for year in range(1991, 2021)]
    jobs += [('nasa', model, scenario, year) for model in MODELS for scenario in SCENARIOS
        for year in list(range(1991, 2021)) + list(range(2031, 2061)) + list(range(2071, 2101))]
    for index, job in enumerate(jobs[:args.limit] if args.limit else jobs):
        status = run_year(*job, draws, parameter, phenology)
        print(json.dumps(dict(status=status, job=job, processed=index + 1, expected_jobs=840)), flush=True)
    if not args.limit:
        (DEST / 'receipt.json').write_text(json.dumps(dict(status='complete', annual_outputs=840,
            cases=CASES, spatial_draws=len(draws), point_parameter_only=True, parameters_refitted=False), indent=2) + '\n')


if __name__ == '__main__':
    main()
