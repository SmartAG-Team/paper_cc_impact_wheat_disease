"""Known-sowing process event-date hindcasts on the frozen common test sites.

Predicted events are first threshold crossings, independent of observed dates.
Headline metrics share the same matched events across all three process models.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from process_model.calibrate import INDICES
from process_model._support.data_contract import file_hash
from process_model._support.station_split import load_station_split, load_evaluation_sites, split_identity, evaluation_identity

MODELS = ['GDD', 'T-P', 'T-P-V']
STAGES = [10, 31, 51, 85]
KEYS = ['PEP_ID', 'SOWING_DATE']


def event_rows(frame, parameters):
    """One row per site/sowing/stage, retaining absent truths and predictions."""
    frame = frame.copy()
    for col in ['DATE', 'SOWING_DATE']:
        frame[col] = pd.to_datetime(frame[col])
    if frame.duplicated(['PEP_ID', 'DATE']).any():
        raise ValueError('Duplicate station/date rows')
    if frame.SOWING_DATE.isna().any() or (frame.SOWING_DATE > frame.DATE).any():
        raise ValueError('Event evaluation requires known, nonfuture sowing dates')
    frame = frame.sort_values(KEYS + ['DATE'])
    cycles = frame.groupby(KEYS).DATE.agg(['min', 'max', 'count']).reset_index()
    cycles['cycle_complete'] = (cycles['min'].eq(cycles.SOWING_DATE)
        & cycles['count'].eq((cycles['max'] - cycles['min']).dt.days + 1))
    observed = frame.loc[frame.CODE.isin([0, 2, 4, 6, 8]), KEYS + ['DATE', 'CODE']].copy()
    observed['BBCH'] = observed.CODE.map({0: 0, 2: 10, 4: 31, 6: 51, 8: 85})
    event_keys = KEYS + ['BBCH']
    if observed.duplicated(event_keys).any():
        raise ValueError('Duplicate observed stage within a sowing cycle')
    out = cycles[KEYS + ['cycle_complete']].merge(pd.DataFrame({'BBCH': [0] + STAGES}), how='cross')
    out = out.merge(observed[event_keys + ['DATE']].rename(columns={'DATE': 'true_date'}),
                    on=event_keys, how='left', validate='one_to_one')
    thresholds = {row['BBCH']: row for row in parameters['thresholds']}
    for model, column in zip(MODELS, INDICES):
        predictions = []
        for stage in STAGES:
            crossing = (frame.loc[frame[column] >= thresholds[stage][column]]
                        .groupby(KEYS).DATE.min().reset_index())
            crossing['BBCH'] = stage
            predictions.append(crossing)
        predicted = pd.concat(predictions, ignore_index=True).rename(columns={'DATE': model + '_date'})
        out = out.merge(predicted, on=event_keys, how='left', validate='one_to_one')
        out.loc[~out.cycle_complete, model + '_date'] = pd.NaT
        # Supplied management anchor is kept for auditing, excluded by default.
        out.loc[out.BBCH == 0, model + '_date'] = out.loc[out.BBCH == 0, 'SOWING_DATE']
    return out.sort_values(event_keys).reset_index(drop=True)


def summarize(events, include_sowing=False):
    scored = events.loc[events.BBCH.isin(([0] if include_sowing else []) + STAGES)].copy()
    observed = scored.true_date.notna()
    common = observed & scored[[m + '_date' for m in MODELS]].notna().all(axis=1)
    result = dict(stages=([0] if include_sowing else []) + STAGES,
        observed_events=int(observed.sum()), common_matched_events=int(common.sum()),
        observed_events_outside_common=int((observed & ~common).sum()),
        scored_station_ids=sorted(scored.loc[common, 'PEP_ID'].astype(int).unique().tolist()),
        incomplete_cycles=int(scored.loc[~scored.cycle_complete, KEYS].drop_duplicates().shape[0]),
        models={})
    for model in MODELS:
        predicted = scored[model + '_date'].notna()
        error = (scored.loc[common, model + '_date'] - scored.loc[common, 'true_date']).dt.days
        result['models'][model] = dict(MAE_days=float(error.abs().mean()) if len(error) else None,
            RMSE_days=float(np.sqrt((error.astype(float)**2).mean())) if len(error) else None,
            matched_events=int(common.sum()),
            missing_predictions=int((observed & ~predicted).sum()),
            predictions_without_observation=int((~observed & predicted).sum()))
    return result


def evaluate(data, selection, output):
    data = Path(data).resolve(); output = Path(output).resolve()
    split = load_station_split(data / 'station_split.json')
    protocol = load_evaluation_sites(selection, split)
    manifest = json.loads((data / 'manifest.json').read_text())
    if (manifest.get('status') != 'complete' or manifest.get('split_sha256') != split_identity(split)
            or protocol.get('data_manifest_sha256') != file_hash(data / 'manifest.json')):
        raise ValueError('Completed calibrated inputs and matching site protocol are required')
    for name in ['data_cleaned.csv', 'calibration.json']:
        if file_hash(data / name) != manifest['artifacts'].get(name):
            raise ValueError(f'Calibrated artifact changed: {name}')
    parameters = json.loads((data / 'calibration.json').read_text())
    if (parameters.get('schema_version') != 2 or parameters.get('observed_stage_conditioned') is not False
            or parameters['split_sha256'] != split_identity(split)
            or parameters['fit_station_ids'] != sorted(split.loc[split.split == 'train', 'PEP_ID'].astype(int))):
        raise ValueError('Frozen training-only causal calibration required')
    if output.exists():
        raise FileExistsError(f'Use a new event-evaluation directory: {output}')
    selected = set(protocol['station_ids'])
    columns = ['PEP_ID', 'DATE', 'SOWING_DATE', 'CODE'] + INDICES
    carry = None; parts = []; seen = set()
    for chunk in pd.read_csv(data / 'data_cleaned.csv', usecols=columns, chunksize=200000):
        chunk = chunk.loc[chunk.PEP_ID.isin(selected)]
        if chunk.empty:
            continue
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk.PEP_ID.iloc[-1]
        carry = chunk.loc[chunk.PEP_ID == last].copy()
        ready = chunk.loc[chunk.PEP_ID != last]
        ids = set(ready.PEP_ID)
        if ids & seen:
            raise ValueError('Source stations must be contiguous')
        seen |= ids
        if not ready.empty:
            parts.append(event_rows(ready, parameters))
    if carry is not None:
        if set(carry.PEP_ID) & seen:
            raise ValueError('Source stations must be contiguous')
        seen.update(carry.PEP_ID)
        parts.append(event_rows(carry, parameters))
    if seen != selected:
        raise ValueError('Event evaluation lacks approved sites')
    events = pd.concat(parts, ignore_index=True)
    summary = summarize(events)
    if not summary['common_matched_events']:
        raise ValueError('No events can be scored by every process model')
    report = dict(schema_version=1, status='complete', units='days',
        split_sha256=split_identity(split), evaluation_sha256=evaluation_identity(protocol),
        test_station_ids=sorted(selected), excluded_test_station_ids=protocol['excluded_station_ids'],
        calibration_manifest_sha256=file_hash(data / 'manifest.json'),
        observed_stage_conditioned=False, known_sowing_conditioned=True,
        weather_policy=parameters['weather_policy'],
        event_rule='First daily threshold crossing per station, known sowing cycle and stage; no interpolation',
        matching_rule='Exact station/sowing/stage keys; no truth-conditioned nearest-date selection or calendar-year restriction',
        denominator='Observed physiological events with a predicted date from all three models',
        missing_policy='Unreached thresholds or incomplete daily cycles retained as missing, never zero-filled',
        limitations=['Matched-event MAE is conditional on shared event availability; coverage is reported separately.',
                     'Predictions without observations are unverified events, not established false positives.',
                     'Realized-weather hindcast; not issue-time fixed-horizon forecasting.'],
        summary=summary,
        by_stage={str(stage): summarize(events.loc[events.BBCH == stage]) for stage in STAGES})
    output.mkdir(parents=True)
    events.to_csv(output / 'event_dates.csv', index=False)
    (output / 'event_metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    audit = dict(status='complete', calibration_manifest_sha256=report['calibration_manifest_sha256'],
        evaluation_sha256=report['evaluation_sha256'],
        artifacts={name: file_hash(output / name) for name in ['event_dates.csv', 'event_metrics.json']})
    (output / 'manifest.json').write_text(json.dumps(audit, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT / 'dataset/calibrated_causal')
    parser.add_argument('--evaluation', type=Path, default=ROOT / 'configs/evaluation_sites.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'dataset/calibrated_causal/common_event_evaluation')
    args = parser.parse_args()
    report = evaluate(args.data, args.evaluation, args.output)
    summary = {k: v for k, v in report['summary'].items() if k != 'scored_station_ids'}
    print(json.dumps(summary, indent=2))
