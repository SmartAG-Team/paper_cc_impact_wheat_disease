"""Training-only process calibration with causal simulated-stage gates.

Sowing dates are supplied as management records known when sowing occurs.
Transformations require no observed phenological labels. Realized daily weather
supports weather-driven hindcasts, not fixed-horizon weather forecasts.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'dl_model'))
sys.path.insert(0, str(ROOT))
from process_model.sowing import attach_sowing, validate_records
from utils.station_split import load_station_split, split_identity, assert_training_sites

CODES = {0: 0, 2: 10, 4: 31, 6: 51, 8: 85}
INDICES = ['Cumulative_GDD', 'Cumulative_t_pp_GDD', 'Cumulative_t_pp_v_GDD']
REQUIRED = ['PEP_ID', 'DATE', 'GDD', 'LAT', 't_mean', 't_max', 'SOWING_DATE', 'SOWING_KNOWN_AT']


def prepare(frame):
    missing = set(REQUIRED) - set(frame)
    if missing:
        raise ValueError(f'Missing process inputs: {sorted(missing)}')
    frame = frame.sort_values(['PEP_ID', 'DATE']).reset_index(drop=True).copy()
    frame['DATE'] = pd.to_datetime(frame.DATE)
    if frame.duplicated(['PEP_ID', 'DATE']).any():
        raise ValueError('Duplicate station/date process input')
    if not np.isfinite(frame[['PEP_ID', 'GDD', 'LAT', 't_mean', 't_max']]).all().all():
        raise ValueError('Nonfinite process weather or station inputs')
    if (frame.LAT.abs() > 90).any():
        raise ValueError('Latitude outside [-90, 90]')
    lat = np.radians(frame.LAT.to_numpy())
    decl = .4093 * np.sin(2 * np.pi * (frame.DATE.dt.dayofyear.to_numpy() - 81) / 365)
    frame['daylength'] = 24 / np.pi * np.arccos(np.clip(-np.tan(lat)*np.tan(decl), -1, 1))
    frame['SOWING_DATE'] = pd.to_datetime(frame.SOWING_DATE)
    frame['SOWING_KNOWN_AT'] = pd.to_datetime(frame.SOWING_KNOWN_AT)
    if (frame.SOWING_DATE.isna() != frame.SOWING_KNOWN_AT.isna()).any():
        raise ValueError('Sowing dates and availability must be jointly present')
    if ((frame.SOWING_DATE > frame.DATE) | (frame.SOWING_KNOWN_AT > frame.SOWING_DATE)).any():
        raise ValueError('Sowing management records must already be known, with no future sowing applied')
    frame['_cycle'] = frame.SOWING_DATE
    frame['Cumulative_GDD'] = frame.GDD.groupby([frame.PEP_ID, frame._cycle]).cumsum()
    return frame


def photoperiod(frame, onset, stop_gdd):
    frame = frame.copy()
    groups = [frame.PEP_ID, frame._cycle]
    prior_gdd = frame.Cumulative_GDD.groupby(groups).shift(fill_value=0.)
    active = prior_gdd.ge(onset) & prior_gdd.lt(stop_gdd) & frame._cycle.notna()
    factor = np.maximum(1 - .09 * (16 - frame.daylength), 0).clip(upper=1)
    frame['ppfun'] = np.where(active, factor, 1.)
    frame.loc[frame._cycle.isna(), 'ppfun'] = np.nan
    frame['t_pp_GDD'] = frame.GDD * frame.ppfun
    frame['Cumulative_t_pp_GDD'] = frame.t_pp_GDD.groupby(groups).cumsum()
    return frame


def vernalization(frame, onset, stop_tpp):
    frame = frame.copy()
    n = len(frame)
    verday = np.zeros(n); cumver = np.zeros(n); verfun = np.ones(n)
    groups = [frame.PEP_ID, frame._cycle]
    prior_tpp = frame.Cumulative_t_pp_GDD.groupby(groups).shift(fill_value=0.)
    active = (prior_tpp.ge(onset) & frame._cycle.notna()).to_numpy()
    sensitive = prior_tpp.lt(stop_tpp).to_numpy()
    tmax = frame.t_max.to_numpy()
    response = np.interp(frame.t_mean, [-4, 0, 10, 16], [0., 1., 1., 0.])
    resets = (frame.PEP_ID.ne(frame.PEP_ID.shift()) | frame._cycle.ne(frame._cycle.shift())).to_numpy()
    cum = 0.
    for i in range(n):
        if resets[i]:
            cum = 0.
        if active[i]:
            verday[i] = response[i] if sensitive[i] else 0.
            updated = cum + verday[i]
            if updated < 10 and tmax[i] > 30:
                updated = max(0., cum - .5 * (tmax[i] - 30))
            cum = updated
            if sensitive[i]:
                verfun[i] = .3 + .7 * min(cum / 40., 1.)
        cumver[i] = cum
    unanchored = frame._cycle.isna().to_numpy()
    verday[unanchored] = np.nan; cumver[unanchored] = np.nan; verfun[unanchored] = np.nan
    frame['VERDAY'] = verday; frame['CUMVER'] = cumver; frame['verfun'] = verfun
    frame['t_pp_v_GDD'] = frame.t_pp_GDD * verfun
    frame['Cumulative_t_pp_v_GDD'] = frame.t_pp_v_GDD.groupby(groups).cumsum()
    return frame.drop(columns='_cycle')


def event_median(frame, code, column):
    values = frame.loc[frame.CODE == code, column].dropna()
    if values.empty or not np.isfinite(values).all():
        raise ValueError(f'Missing/nonfinite training stage CODE={code}: {column}')
    return round(float(values.median()), 2)


def fit_calibration(raw, split):
    unknown = set(raw.PEP_ID) - set(split.PEP_ID)
    if unknown:
        raise ValueError('Stations missing from frozen split')
    train_ids = set(split.loc[split.split == 'train', 'PEP_ID'])
    train = raw.loc[raw.PEP_ID.isin(train_ids)].copy()
    assert_training_sites(train.PEP_ID, split)
    if set(train.PEP_ID) != train_ids:
        raise ValueError('Training stations missing from calibration inputs')
    train = prepare(train)
    onset_gdd = event_median(train, 2, 'Cumulative_GDD')
    stop_gdd = event_median(train, 6, 'Cumulative_GDD')
    train = photoperiod(train, onset_gdd, stop_gdd)
    onset_tpp = event_median(train, 2, 'Cumulative_t_pp_GDD')
    stop_tpp = event_median(train, 4, 'Cumulative_t_pp_GDD')
    train = vernalization(train, onset_tpp, stop_tpp)
    thresholds = [dict(BBCH=bbch, **{col: event_median(train, code, col) for col in INDICES})
                  for code, bbch in CODES.items()]
    for col in INDICES:
        if np.any(np.diff([row[col] for row in thresholds]) <= 0):
            raise ValueError(f'Non-increasing training stage thresholds: {col}')
    calibration = dict(schema_version=2, split_sha256=split_identity(split),
        fit_station_ids=sorted(int(v) for v in train_ids),
        photoperiod_onset_gdd=onset_gdd, vernalization_onset_tpp=onset_tpp,
        photoperiod_stop_gdd=stop_gdd, vernalization_stop_tpp=stop_tpp,
        thresholds=thresholds, observed_stage_conditioned=False,
        process_state_policy='Previous-day GDD state gates photoperiod; previous-day T-P state gates vernalization',
        sowing_policy='Management record known by sowing date; no future sowing applied',
        neural_history_policy='Simulated T-P-V stage replaces historical observed target input',
        weather_policy='Only weather through each feature date; realized-weather process hindcast',
        limitations=['Fixed-horizon process forecasts require issue-time weather forecasts.',
                     'Historical canopy preprocessing and observation availability remain unverified.'])
    return calibration, train


def transform(raw, calibration):
    return vernalization(photoperiod(prepare(raw), calibration['photoperiod_onset_gdd'],
                                    calibration['photoperiod_stop_gdd']),
                         calibration['vernalization_onset_tpp'], calibration['vernalization_stop_tpp'])


def simulate_frame(frame, calibration):
    out = frame[['PEP_ID', 'DATE'] + INDICES].copy()
    for col, name in zip(INDICES, ['gdd_simulate', 'tpp_simulate', 'tppv_simulate']):
        out[name] = np.where(out[col].notna(), 0., np.nan)
        for row in calibration['thresholds']:
            out.loc[out[col] >= row[col], name] = row['BBCH']
    return out


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def complete_station_batches(path, chunksize=200000):
    carry = None; seen = set()
    for chunk in pd.read_csv(path, chunksize=chunksize):
        chunk = pd.concat([carry, chunk], ignore_index=True) if carry is not None else chunk
        last = chunk.PEP_ID.iloc[-1]
        carry = chunk.loc[chunk.PEP_ID == last].copy()
        ready = chunk.loc[chunk.PEP_ID != last]
        ids = set(ready.PEP_ID)
        if ids & seen:
            raise ValueError('Input CSV must have contiguous rows per station')
        seen |= ids
        if not ready.empty:
            yield ready
    if carry is not None:
        if set(carry.PEP_ID) & seen:
            raise ValueError('Input CSV must have contiguous rows per station')
        yield carry


def run(data_path, split_path, output, sowing_path=None):
    data_path = Path(data_path).resolve(); output = Path(output).resolve()
    split = load_station_split(split_path)
    sowing = validate_records(pd.read_csv(sowing_path)) if sowing_path is not None else None
    if output.exists():
        raise FileExistsError(f'Output must be new to preserve archived artifacts: {output}')
    train_ids = set(split.loc[split.split == 'train', 'PEP_ID'])
    print('Reading training rows for calibration', flush=True)
    parts = []; observed = set()
    for chunk in pd.read_csv(data_path, chunksize=200000):
        observed.update(chunk.PEP_ID.unique())
        training_rows = chunk.loc[chunk.PEP_ID.isin(train_ids)]
        if sowing is not None:
            training_rows = attach_sowing(training_rows, sowing.loc[sowing.PEP_ID.isin(train_ids)])
        parts.append(training_rows)
    if observed != set(split.PEP_ID):
        raise ValueError('Frozen station split and source station inventory differ')
    calibration, _ = fit_calibration(pd.concat(parts, ignore_index=True), split)
    del parts
    output.mkdir(parents=True)
    if sowing is not None:
        sowing.to_csv(output/'sowing_records.csv', index=False)
    groups = {g: sorted(split.loc[split.split == g, 'PEP_ID'].astype(int).tolist())
              for g in ['train', 'val', 'test']}
    (output / 'station_split.json').write_text(json.dumps(groups, indent=2) + '\n')
    split.to_csv(output / 'data_split_updated.csv', index=False)
    pd.DataFrame(calibration['thresholds']).to_csv(output / 'gdd_config.csv', index=False)
    (output / 'calibration.json').write_text(json.dumps(calibration, indent=2) + '\n')
    header = True; test_header = True; rows = 0; sums = np.zeros(3); counts = np.zeros(3, dtype=int)
    test_ids = set(groups['test']); seen_test = set()
    print('Regenerating all splits with frozen training parameters', flush=True)
    for raw in complete_station_batches(data_path):
        if sowing is not None:
            raw = attach_sowing(raw, sowing)
        features = transform(raw, calibration); simulations = simulate_frame(features, calibration)
        features.to_csv(output / 'data_cleaned.csv', index=False, mode='w' if header else 'a', header=header)
        simulations.to_csv(output / 'simulation_all.csv', index=False, mode='w' if header else 'a', header=header)
        test = simulations.loc[simulations.PEP_ID.isin(test_ids)].copy()
        seen_test.update(test.PEP_ID)
        if not test.empty:
            test.to_csv(output / 'simulation_test.csv', index=False,
                        mode='w' if test_header else 'a', header=test_header)
            test_header = False
            truth = features.loc[test.index, 'CODE_new']
            for j, name in enumerate(['gdd_simulate', 'tpp_simulate', 'tppv_simulate']):
                prediction = test[name].map({0: 0, 10: 1, 31: 2, 51: 3, 85: 4})
                valid = truth.notna() & prediction.notna()
                sums[j] += (truth[valid] - prediction[valid]).abs().sum(); counts[j] += valid.sum()
        header = False; rows += len(features)
        print(f'Regenerated {rows:,} rows', flush=True)
    if seen_test != test_ids:
        raise ValueError('Missing designated test sites in process evaluation')
    metrics = {name: dict(MAE_stage_code=float(sums[j]/counts[j]), rows=int(counts[j]))
               for j, name in enumerate(['GDD', 'T-P', 'T-P-V'])}
    (output / 'process_test_metrics.json').write_text(json.dumps(dict(
        split_sha256=calibration['split_sha256'], test_station_ids=groups['test'],
        observed_stage_conditioned=False, known_sowing_conditioned=True,
        weather_policy=calibration['weather_policy'], metrics=metrics), indent=2) + '\n')
    manifest = dict(status='complete', source=str(data_path), source_sha256=sha256(data_path),
        split_sha256=calibration['split_sha256'], rows=rows,
        artifacts={p.name: sha256(p) for p in output.iterdir() if p.is_file()})
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(metrics, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'dataset/data_cleaned.csv')
    parser.add_argument('--split', type=Path, default=ROOT/'configs/station_split.json')
    parser.add_argument('--output', type=Path, default=ROOT/'dataset/calibrated_causal')
    parser.add_argument('--sowing', type=Path, default=ROOT/'dataset/sowing_records.csv')
    args = parser.parse_args()
    run(args.data, args.split, args.output, args.sowing)


if __name__ == '__main__':
    main()
