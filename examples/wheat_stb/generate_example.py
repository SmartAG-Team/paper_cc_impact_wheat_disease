"""Generate a declared synthetic forcing case from explicit frozen fit files."""

import argparse
from dataclasses import replace
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calibration.wheat_stb.configuration import make_configuration
from model.wheat_stb import YieldParameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tpv-fit', type=Path, required=True)
    parser.add_argument('--stage-fit', type=Path, required=True)
    parser.add_argument('--disease-fit', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    names = ['configuration.json', 'configuration_standardized_had.json', 'weather.csv', 'inputs.json']
    if any((args.output/name).exists() for name in names):
        raise FileExistsError('Example input files must be new.')
    args.output.mkdir(parents=True, exist_ok=True)
    config = make_configuration(tpv_fit=args.tpv_fit, stage_fit=args.stage_fit,
        disease_fit=args.disease_fit, latitude=51.5, sowing_date='2026-10-01',
        crop_end_date='2027-09-30', field_id='synthetic_example')
    cases = {'configuration.json': config,
        'configuration_standardized_had.json': replace(config,
            yield_model=YieldParameters(mode='standardized_model_proxy'))}
    for filename, case in cases.items():
        (args.output/filename).write_text(json.dumps(case.to_dict(), indent=2, allow_nan=False)+'\n')
    first, last = date(2026, 9, 20), date(2027, 9, 30)
    records = []
    for index in range((last-first).days+1):
        current = first+timedelta(days=index)
        phase = 2*np.pi*(current.timetuple().tm_yday-105)/365.
        mean = 10.+8.*np.sin(phase)
        records.append(dict(date=current.isoformat(), tmean_c=float(mean),
            tmax_c=float(mean+5.), rh_mean_pct=float(82.+10.*np.cos(phase)),
            precipitation_mm=3. if index % 8 in [0, 1] else 0.,
            imported_pressure=config.background_imported_pressure))
    pd.DataFrame(records).to_csv(args.output/'weather.csv', index=False)
    metadata = dict(forcing_status='synthetic seasonal weather; no field observations',
        source_fit_status=config.parameter_status, stage_status=config.leaf.threshold_status,
        synthetic_weather_formula=dict(tmean_c='10 + 8 sin(2 pi (day_of_year - 105)/365)',
            tmax_c='tmean_c + 5', rh_mean_pct='82 + 10 cos(2 pi (day_of_year - 105)/365)',
            precipitation_mm='3 on example indices mod 8 in {0,1}; zero otherwise'),
        imported_pressure='explicit selected constant; relative source units, not spore count',
        detection_fraction=config.detection_fraction,
        detection_fraction_status='fixed observation assumption matching the registered primary 0.001 operator; not estimated',
        no_field_yield_observations=True,
        source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [args.tpv_fit, args.stage_fit, args.disease_fit]},
        input_sha256={name: hashlib.sha256((args.output/name).read_bytes()).hexdigest()
            for name in names[:-1]})
    (args.output/'inputs.json').write_text(json.dumps(metadata, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
