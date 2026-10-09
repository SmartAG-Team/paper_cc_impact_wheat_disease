"""Freeze wheat-only weather/target arrays from the causal donor and station split."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
STAGES = [10, 31, 51, 85]
HORIZON = 366
FEATURES = ['t_mean', 't_max', 't_min', 'daylength', 'CUMVER',
            'Cumulative_GDD', 'Cumulative_t_pp_v_GDD', 'elapsed_fraction']
COLUMNS = ['PEP_ID', 'DATE', 'SOWING_DATE', *FEATURES[:-1], 't_pp_v_GDD']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def pack_cycle(frame, sowing, horizon=HORIZON):
    frame = frame.sort_values('DATE').copy()
    frame['DATE'] = pd.to_datetime(frame.DATE)
    if frame.DATE.duplicated().any():
        raise ValueError('Duplicate weather date within sowing cycle')
    x = np.zeros((horizon, len(FEATURES)), dtype=np.float32)
    g = np.zeros(horizon, dtype=np.float32)
    dates = pd.DatetimeIndex(frame.DATE)
    expected = pd.date_range(sowing, periods=min(len(frame), horizon))
    finite = np.isfinite(frame[FEATURES[:-1] + ['t_pp_v_GDD']].to_numpy()).all(1)
    valid = (dates[:len(expected)] == expected) & finite[:len(expected)]
    valid &= frame.t_pp_v_GDD.to_numpy()[:len(expected)] >= 0
    n = int(np.flatnonzero(~valid)[0]) if (~valid).any() else len(expected)
    if n:
        x[:n, :-1] = frame[FEATURES[:-1]].iloc[:n].to_numpy(dtype=np.float32)
        x[:n, -1] = np.arange(1, n+1) / HORIZON
        g[:n] = frame.t_pp_v_GDD.iloc[:n].to_numpy(dtype=np.float32)
    return x, g, n


def fit_normalizer(x, lengths, train_ids):
    count = 0
    sums = np.zeros(x.shape[2], dtype=np.float64)
    squares = sums.copy()
    for i in train_ids:
        valid = x[i, :lengths[i]].astype(np.float64)
        sums += valid.sum(0)
        squares += np.square(valid).sum(0)
        count += len(valid)
    if not count:
        raise ValueError('No calibration weather')
    mean = sums / count
    std = np.sqrt(np.maximum(squares / count - mean**2, 0))
    return mean, np.maximum(std, 1e-6)


def prepare(source, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    reference = ROOT / 'analysis/paper_study/full_validation_20261007'
    frames = []
    sources = {}
    for cohort in ['calibration', 'validation', 'testing']:
        p = reference / f'german_{cohort}_tpv_events.parquet'
        f = pd.read_parquet(p)
        assert f.cohort.eq(cohort).all() and not f.duplicated(['PEP_ID', 'SOWING_DATE', 'BBCH']).any()
        frames.append(f)
        sources[str(p.relative_to(ROOT))] = sha(p)
    events = pd.concat(frames, ignore_index=True)
    events['SOWING_DATE'] = pd.to_datetime(events.SOWING_DATE)
    events['true_date'] = pd.to_datetime(events.true_date)
    events['predicted_date'] = pd.to_datetime(events.predicted_date)
    env = events[['PEP_ID', 'SOWING_DATE', 'cohort']].drop_duplicates().sort_values(['PEP_ID', 'SOWING_DATE']).reset_index(drop=True)
    if env.duplicated(['PEP_ID', 'SOWING_DATE']).any():
        raise ValueError('A sowing cycle appears in more than one split')
    split = json.loads((ROOT / 'process_model/parameters/station_split.json').read_text())
    mapping = {'calibration': 'train', 'validation': 'val', 'testing': 'test'}
    for cohort, key in mapping.items():
        assert set(env.loc[env.cohort.eq(cohort), 'PEP_ID']).issubset(set(split[key]))
    assert not set(split['train']) & (set(split['val']) | set(split['test']))
    assert not set(split['val']) & set(split['test'])
    keys = {(int(r.PEP_ID), str(r.SOWING_DATE.date())): i for i, r in enumerate(env.itertuples())}
    n = len(env)
    x = np.zeros((n, HORIZON, len(FEATURES)), np.float32)
    g = np.zeros((n, HORIZON), np.float32)
    lengths = np.zeros(n, np.int16)
    seen = set()
    def consume(frame):
        frame = frame.loc[frame.SOWING_DATE.notna()]
        for (station, sowing), group in frame.groupby(['PEP_ID', 'SOWING_DATE'], sort=False):
            key = (int(station), str(sowing))
            if key not in keys:
                continue
            i = keys[key]
            if i in seen:
                raise ValueError('Noncontiguous source cycle: ' + str(key))
            seen.add(i)
            x[i], g[i], lengths[i] = pack_cycle(group, pd.Timestamp(sowing))
    carry = None
    for k, chunk in enumerate(pd.read_csv(source, usecols=COLUMNS, chunksize=300000)):
        if carry is not None:
            chunk = pd.concat([carry, chunk], ignore_index=True)
        last = chunk.PEP_ID.iloc[-1]
        carry = chunk.loc[chunk.PEP_ID.eq(last)].copy()
        consume(chunk.loc[chunk.PEP_ID.ne(last)])
        if k % 10 == 0:
            print(json.dumps({'chunks_read': k+1, 'cycles_read': len(seen)}), flush=True)
    if carry is not None:
        consume(carry)
    if len(seen) != n:
        raise ValueError(f'Missing source weather for {n-len(seen)} archived cycles')
    y = np.zeros((n, 4), np.int16)
    baseline = np.full((n, 4), np.nan, np.float32)
    source_observed = np.zeros((n, 4), np.int16)
    exclusions = []
    for r in events.itertuples():
        i = keys[(int(r.PEP_ID), str(r.SOWING_DATE.date()))]
        j = STAGES.index(int(r.BBCH))
        day = None if pd.isna(r.true_date) else int((r.true_date-r.SOWING_DATE).days)+1
        if day is not None:
            source_observed[i, j] = day
            reason = None
            if not r.eligible: reason = 'archived_cycle_ineligible'
            elif day <= 0: reason = 'event_before_sowing'
            elif day > lengths[i]: reason = 'outside_contiguous_weather_prefix'
            if reason:
                exclusions.append(dict(PEP_ID=int(r.PEP_ID), SOWING_DATE=str(r.SOWING_DATE.date()), BBCH=r.BBCH, cohort=r.cohort, reason=reason))
            else: y[i, j] = day
        if r.eligible and pd.notna(r.predicted_date):
            pred = int((r.predicted_date-r.SOWING_DATE).days)+1
            if 1 <= pred <= lengths[i]: baseline[i, j] = pred
    train = np.flatnonzero(env.cohort.eq('calibration').to_numpy() & (y > 0).any(1))
    mean, std = fit_normalizer(x, lengths, train)
    parameters = json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    thresholds = [next(r['Cumulative_t_pp_v_GDD'] for r in parameters['thresholds'] if r['BBCH']==s) for s in STAGES]
    # The existing deterministic comparator stays exactly as archived, including missing predictions.
    recomputed = np.full_like(baseline, np.nan)
    cumulative = np.cumsum(g.astype(np.float64), 1)
    for j, t in enumerate(thresholds):
        crossing = (cumulative >= t) & (np.arange(HORIZON)[None, :] < lengths[:, None])
        recomputed[:, j] = np.where(crossing.any(1), crossing.argmax(1)+1, np.nan)
    common = np.isfinite(baseline) & np.isfinite(recomputed)
    baseline_differences = (recomputed-baseline)[common]
    env['weather_days'] = lengths
    env['observed_targets'] = (y > 0).sum(1)
    env.to_csv(output/'environments.csv', index=False)
    pd.DataFrame(exclusions, columns=['PEP_ID','SOWING_DATE','BBCH','cohort','reason']).to_csv(output/'exclusions.csv', index=False)
    np.savez_compressed(output/'arrays.npz', X=x, G=g, Y=y, baseline=baseline,
                        lengths=lengths, source_observed=source_observed)
    sources['external_causal_daily_weather'] = sha(source)
    sources['process_model/parameters/calibration.json'] = sha(ROOT/'process_model/parameters/calibration.json')
    sources['process_model/parameters/station_split.json'] = sha(ROOT/'process_model/parameters/station_split.json')
    architecture = json.loads((HERE/'architecture_provenance.json').read_text())
    metadata = dict(status='prepared', stages=STAGES, features=FEATURES, horizon=HORIZON,
        model_source=architecture['source_files'],
        maize_weights_loaded=False, mean=mean.tolist(), std=std.tolist(), thermal_scale=thresholds[-1],
        thresholds=(np.array(thresholds)/thresholds[-1]).tolist(), source_hashes=sources,
        protocol_sha256=sha(HERE/'protocol.json'), source_daily_path=str(Path(source).resolve()),
        source_daily_bytes=Path(source).stat().st_size, input_arrays_sha256=sha(output/'arrays.npz'),
        environments_sha256=sha(output/'environments.csv'), date_index='Position 1 is sowing day; elapsed days after sowing = position - 1',
        counts={c:dict(environments=int(env.cohort.eq(c).sum()),observed_targets=int((y[env.cohort.eq(c)]>0).sum()),
                      stage_targets=(y[env.cohort.eq(c)]>0).sum(0).tolist()) for c in mapping},
        baseline_float32_check=dict(compared=int(common.sum()),different=int((baseline_differences!=0).sum()),
            maximum_difference_days=float(np.abs(baseline_differences).max())),
        limits=['Station holdout, not forward-year validation', 'No German GS39 or GS65 truth',
                'Weather is realized, not issue-time forecast', 'Training weather source remains external'])
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata['counts']),flush=True)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();prepare(a.source,a.output)
