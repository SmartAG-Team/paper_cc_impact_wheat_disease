"""Run the copied frozen T-P-V model with explicit daily forcing and sowing."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from process_model.calibrate import transform, simulate_frame, sha256
from process_model.evaluate_event_dates import event_rows
from process_model.sowing import attach_sowing, validate_records

DEFAULT_CALIBRATION = Path(__file__).resolve().parent / 'parameters/calibration.json'


def run(weather, sowing, output, calibration=DEFAULT_CALIBRATION,
        gdd_convention='supplied'):
    weather = Path(weather).resolve()
    sowing = Path(sowing).resolve()
    calibration = Path(calibration).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'Output must be new: {output}')
    parameters = json.loads(calibration.read_text())
    if parameters.get('schema_version') != 2 or parameters.get('observed_stage_conditioned') is not False:
        raise ValueError('Frozen schema-2 calibration with label-free transformation is required')
    raw = pd.read_csv(weather)
    if raw.empty:
        raise ValueError('Daily weather is empty')
    if gdd_convention == 'supplied':
        if 'GDD' not in raw:
            raise ValueError('GDD is required; select an explicit derivation convention to derive it')
    elif gdd_convention == 'clipped_mean_0_20':
        if 't_mean' not in raw:
            raise ValueError('t_mean in degrees Celsius is required to derive GDD')
        raw['GDD'] = np.clip(raw.t_mean, 0., 20.)
    elif gdd_convention == 'archived_temperature_response':
        if 't_mean' not in raw:
            raise ValueError('t_mean in degrees Celsius is required to derive GDD')
        if (raw.t_mean > 40.).any():
            raise ValueError('Reconstructed response above 40 degrees Celsius is unsupported; supply GDD explicitly')
        raw['GDD'] = np.where(raw.t_mean > 30., 20. - 2. * (raw.t_mean - 30.),
            np.clip(raw.t_mean, 0., 20.))
    else:
        raise ValueError(f'Unknown GDD convention: {gdd_convention}')
    # Observed stage labels do not contribute to simulation or event extraction.
    raw = raw.drop(columns=['CODE', 'CODE_new', 'BBCH'], errors='ignore')
    records = validate_records(pd.read_csv(sowing))
    raw = attach_sowing(raw, records)
    features = transform(raw, parameters)
    stages = simulate_frame(features, parameters)
    anchored = features.loc[features.SOWING_DATE.notna()].copy()
    if anchored.empty:
        raise ValueError('No supplied sowing record has occurred within the weather series')
    events = event_rows(anchored.assign(CODE=-1), parameters)
    cycles = events[['PEP_ID', 'SOWING_DATE', 'cycle_complete']].drop_duplicates()
    output.mkdir(parents=True)
    features.to_csv(output / 'daily_features.csv', index=False)
    stages.to_csv(output / 'daily_stages.csv', index=False)
    events.to_csv(output / 'event_dates.csv', index=False)
    (output / 'calibration.json').write_bytes(calibration.read_bytes())
    report = dict(schema_version=1, status='complete', model='AGC-Transformer schema-2 T-P-V',
        rows=len(features), stations=int(features.PEP_ID.nunique()), cycles=len(cycles),
        incomplete_cycles=int((~cycles.cycle_complete).sum()),
        unanchored_daily_rows=int(features.SOWING_DATE.isna().sum()),
        gdd_convention=gdd_convention, temperature_units='degrees Celsius',
        thermal_preprocessing_status=('Supplied archived GDD remains authoritative; reconstructed forcing matches all source rows. '
            'Its decreasing branch is empirically observed only through 30.86704152398937 degrees Celsius; higher temperatures extrapolate that branch.'),
        observed_validation=False, cultivar_specific_calibration=False,
        interpretation='Frozen-parameter, supplied-sowing weather-driven simulation; no field calibration or observed-event validation implied.',
        event_rule='First daily threshold crossing; incomplete daily cycles retain missing event dates.',
        inputs={name: dict(path=str(path), sha256=sha256(path))
                for name, path in [('weather', weather), ('sowing', sowing), ('calibration', calibration)]},
        artifacts={name: sha256(output / name) for name in
            ['daily_features.csv', 'daily_stages.csv', 'event_dates.csv', 'calibration.json']})
    (output / 'run_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weather', required=True, type=Path)
    parser.add_argument('--sowing', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--calibration', type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument('--gdd-convention', choices=['supplied', 'clipped_mean_0_20',
        'archived_temperature_response'], default='supplied')
    args = parser.parse_args()
    print(json.dumps(run(args.weather, args.sowing, args.output,
        args.calibration, args.gdd_convention), indent=2))


if __name__ == '__main__':
    main()
