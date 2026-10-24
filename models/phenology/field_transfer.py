"""Run frozen T-P-V transfers using documented sowing and linked ERA5 forcing."""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from process_model.run_tpv import run as run_model
from process_model.calibrate import sha256


def run(weather, links, sowing_cases, output):
    weather = Path(weather).resolve()
    links = Path(links).resolve()
    sowing_cases = Path(sowing_cases).resolve()
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(output)
    daily = pd.read_csv(weather, parse_dates=['date'])
    linked = pd.read_csv(links)
    cases = pd.read_csv(sowing_cases, parse_dates=['SOWING_DATE', 'SOWING_KNOWN_AT'])
    pieces = []
    metadata = []
    for row in cases.itertuples(index=False):
        match = linked.loc[(linked.dataset_id == row.weather_link_dataset_id)
            & (linked.site_id == row.site_id) & (linked.season_year == row.season_year)]
        if len(match) != 1:
            raise ValueError(f'Exactly one verified site/season weather link required: {row.case_id}')
        site = match.iloc[0]
        end = pd.Timestamp(year=row.season_year, month=8, day=31)
        frame = daily.loc[(daily.location_id == site.location_id)
            & daily.date.ge(row.SOWING_DATE) & daily.date.le(end)].copy()
        if frame.empty:
            raise ValueError(f'Linked daily weather missing: {row.case_id}')
        frame = frame.rename(columns={'date': 'DATE', 'tmean_c': 't_mean',
            'tmax_c': 't_max', 'tmin_c': 't_min'})
        frame['PEP_ID'] = row.PEP_ID
        frame['LAT'] = site.requested_latitude
        frame['LON'] = site.requested_longitude
        frame['case_id'] = row.case_id
        pieces.append(frame)
        metadata.append(dict(PEP_ID=row.PEP_ID, case_id=row.case_id,
            dataset_id=row.dataset_id, site_id=row.site_id, season_year=row.season_year,
            sowing_date=str(row.SOWING_DATE.date()), location_id=site.location_id,
            latitude=site.requested_latitude, longitude=site.requested_longitude,
            coordinate_source=site.coordinate_source, sowing_source=row.sowing_source,
            wheat_type=row.wheat_type, transfer_status=row.transfer_status,
            available_first_date=str(frame.DATE.min().date()),
            available_last_date=str(frame.DATE.max().date()), weather_rows=len(frame)))
    forcing = pd.concat(pieces, ignore_index=True)
    output.mkdir(parents=True)
    forcing.to_csv(output / 'canonical_weather.csv', index=False)
    cases[['PEP_ID', 'SOWING_DATE', 'SOWING_KNOWN_AT']].to_csv(output / 'sowing_records.csv', index=False)
    pd.DataFrame(metadata).to_csv(output / 'case_metadata.csv', index=False)
    model_report = run_model(output / 'canonical_weather.csv', output / 'sowing_records.csv',
        output / 'simulation', gdd_convention='archived_temperature_response')
    events = pd.read_csv(output / 'simulation/event_dates.csv')
    events = events.merge(pd.DataFrame(metadata), on='PEP_ID', validate='many_to_one')
    events.to_csv(output / 'transfer_event_dates.csv', index=False)
    report = dict(status='complete', cases=len(cases), weather_rows=len(forcing),
        observed_validation=False, recalibrated=False,
        sowing_availability_assumption='Published dates are supplied retrospectively; availability on the sowing date is assumed for the copied model input contract.',
        interpretation='Illustrative transfer of frozen German winter-wheat process parameters to documented external field sowing cases; no cultivar, wheat-type or regional validation.',
        gdd_convention='Explicit reconstructed archived response: linear 0-20, plateau through 30, decline 20-2*(Tmean-30) above 30 degrees Celsius. All 48 high-temperature source rows fit the declining branch; original generator unavailable.',
        high_temperature_extrapolation='Source declining-branch observations end at 30.86704152398937 degrees Celsius; warmer field forcing extrapolates the observed slope. Mean temperatures above 40 degrees Celsius require supplied GDD.',
        coordinate_limit='Thiverval-Grignon uses a station-representative laboratory/greenhouse coordinate; exact experimental plot coordinates are unavailable.',
        model_run_manifest_sha256=sha256(output / 'simulation/run_manifest.json'),
        inputs={name: dict(path=str(path), sha256=sha256(path)) for name, path in
            [('weather', weather), ('weather_links', links), ('sowing_cases', sowing_cases)]},
        artifacts={name: sha256(output / name) for name in
            ['canonical_weather.csv', 'sowing_records.csv', 'case_metadata.csv', 'transfer_event_dates.csv']},
        incomplete_cycles=model_report['incomplete_cycles'])
    (output / 'transfer_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weather', type=Path, default=ROOT / 'data/era5/daily_weather.csv')
    parser.add_argument('--links', type=Path, default=ROOT / 'data/era5/external_site_weather_links.csv')
    parser.add_argument('--sowing', type=Path, default=ROOT / 'analysis/phenology/field_transfer_sowing_cases.csv')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.weather, args.links, args.sowing, args.output), indent=2))
