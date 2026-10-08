"""One known-weather sample per complete recorded sowing cycle.

Eligibility depends on the declared calendar and finite input values. A missing
calendar day excludes the complete cycle; observed stage dates never truncate it.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.data_contract import file_hash
from utils.station_split import load_station_split, split_identity, load_evaluation_sites, evaluation_identity


WEATHER = ['t_mean', 't_min', 't_max', 'p_sum']
ACCUMULATION = ['Cumulative_GDD', 'Cumulative_t_pp_GDD', 'Cumulative_t_pp_v_GDD']
INCREMENTS = ['GDD', 't_pp_GDD', 't_pp_v_GDD']
STAGES = [10, 31, 51, 85]
DAY_NAT = np.iinfo(np.int64).min


def _fit(values):
    count = np.isfinite(values).sum(axis=0)
    median = np.array([np.median(col[np.isfinite(col)]) if n else 0.
                       for col, n in zip(values.T, count)])
    filled = np.where(np.isfinite(values), values, median)
    mean, scale = filled.mean(axis=0), filled.std(axis=0)
    scale = np.where(scale > 1e-8, scale, 1.)
    return dict(imputation=median.tolist(), mean=mean.tolist(), scale=scale.tolist(),
                observed_count=count.tolist())


def _transform(values, fit):
    filled = np.where(np.isfinite(values), values, fit['imputation'])
    return ((filled - fit['mean']) / fit['scale']).astype(np.float32)


def _identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


POLICY = dict(name='whole_season_sowing_onward_v2', reference='beginning_of_sowing_day',
              first_lead='sowing_day', endpoint='next_recorded_sowing_minus_one_else_station_record_end',
              eligibility='complete_daily_calendar_and_finite_weather_process_coordinates',
              missing_calendar='whole_cycle_ineligible_with_observed_events_retained',
              future_weather='observed_full_season_weather', length_cap=None,
              missing_event='no_negative_evidence', cycle_weight='one_per_sowing_cycle')
PROCESS_MODELS = dict(zip(['GDD', 'T-P', 'T-P-V'], ACCUMULATION))


def _day_string(day):
    return str(np.datetime64(int(day), 'D'))


def _days(values):
    return pd.to_datetime(values).to_numpy(dtype='datetime64[D]').astype(np.int64)


def _sources(args):
    root = Path(args.root_path).expanduser().resolve()
    data = root/getattr(args, 'data_path', 'data_cleaned.csv')
    split_path = Path(getattr(args, 'split_path', None) or root/'station_split.json').resolve()
    evaluation_path = Path(getattr(args, 'evaluation_sites', None) or
                           getattr(args, 'evaluation_path', None) or root/'evaluation_sites.json').resolve()
    paths = dict(data=data, split=split_path, evaluation=evaluation_path,
                 manifest=root/'manifest.json', calibration=root/'calibration.json',
                 sowing=root/'sowing_records.csv')
    for path in paths.values():
        if not path.is_file():
            raise ValueError(f'Verified seasonal input is required: {path}')
    return paths


def _verify_sources(paths):
    split = load_station_split(paths['split'])
    split_hash = split_identity(split)
    evaluation = load_evaluation_sites(paths['evaluation'], split)
    parameters = json.loads(paths['calibration'].read_text())
    manifest = json.loads(paths['manifest'].read_text())
    hashes = {name: file_hash(path) for name, path in paths.items()}
    groups = {group: sorted(split.loc[split.split == group, 'PEP_ID'].astype(int).tolist())
              for group in ['train', 'val', 'test']}
    if (parameters.get('schema_version') != 2 or parameters.get('observed_stage_conditioned') is not False
            or parameters.get('split_sha256') != split_hash
            or parameters.get('fit_station_ids') != groups['train']):
        raise ValueError('Causal train-only calibration and the frozen split must match')
    if manifest.get('status') != 'complete' or manifest.get('split_sha256') != split_hash:
        raise ValueError('Calibration manifest is incomplete or differs from frozen split')
    if evaluation.get('data_manifest_sha256') != hashes['manifest']:
        raise ValueError('Evaluation sites were approved against a different calibrated manifest')
    for name in ['data', 'calibration', 'sowing']:
        if manifest.get('artifacts', {}).get(paths[name].name) != hashes[name]:
            raise ValueError(f'Calibrated artifact changed: {paths[name].name}')
    thresholds = {model: {int(row['BBCH']): float(row[column]) for row in parameters['thresholds']}
                  for model, column in PROCESS_MODELS.items()}
    for model, stages in thresholds.items():
        try:
            values = np.asarray([stages[stage] for stage in [0] + STAGES])
        except KeyError as error:
            raise ValueError('Calibration lacks complete stage thresholds') from error
        if not np.isfinite(values).all() or (np.diff(values) <= 0).any():
            raise ValueError(f'Positive calibrated threshold gaps required: {model}')
    base = dict(schema_version=2, split_sha256=split_hash, evaluation_sha256=evaluation_identity(evaluation),
                evaluated_station_ids=evaluation['station_ids'],
                excluded_test_station_ids=evaluation['excluded_station_ids'],
                **{f'{group}_station_ids': sites for group, sites in groups.items()},
                data_sha256={paths[name].name: hashes[name] for name in ['data', 'calibration', 'sowing']},
                source_sha256=hashes, seasonal_policy=POLICY)
    return base, parameters, thresholds, groups, hashes


def prepare_seasonal_source(args):
    """Verify and read once; explicitly share the result between split constructors."""
    paths = _sources(args)
    base, parameters, thresholds, groups, hashes = _verify_sources(paths)
    columns = ['PEP_ID', 'DATE', 'CODE', 'SOWING_DATE', 'SOWING_KNOWN_AT', 'LON', 'LAT'] + WEATHER + ACCUMULATION + INCREMENTS
    header = pd.read_csv(paths['data'], nrows=0).columns
    required = set(columns) - set(WEATHER)
    if not required.issubset(header):
        raise ValueError(f'Seasonal source lacks required columns: {sorted(required-set(header))}')
    pieces = []
    for frame in pd.read_csv(paths['data'], usecols=lambda c: c in columns, chunksize=500000):
        for column in ['DATE', 'SOWING_DATE', 'SOWING_KNOWN_AT']:
            frame[column] = _days(frame[column])
        frame = frame.reindex(columns=columns)
        if not np.isfinite(frame.PEP_ID).all() or (frame.PEP_ID % 1 != 0).any():
            raise ValueError('Station IDs must be finite integers')
        frame.PEP_ID = frame.PEP_ID.astype(np.int64)
        pieces.append(frame)
    source = pd.concat(pieces, ignore_index=True).sort_values(['PEP_ID', 'DATE']).reset_index(drop=True)
    if (source.DATE == DAY_NAT).any() or source.duplicated(['PEP_ID', 'DATE']).any():
        raise ValueError('Missing or duplicate station/date rows in seasonal source')
    unknown = set(source.PEP_ID)-set(sum(groups.values(), []))
    if unknown:
        raise ValueError('Source has stations absent from the frozen split')
    sowing = pd.read_csv(paths['sowing'])
    for column in ['SOWING_DATE', 'SOWING_KNOWN_AT']:
        sowing[column] = _days(sowing[column])
    if (sowing.PEP_ID.isna().any() or (sowing.PEP_ID % 1 != 0).any()
            or (sowing.SOWING_DATE == DAY_NAT).any()
            or sowing.duplicated(['PEP_ID', 'SOWING_DATE']).any()):
        raise ValueError('Invalid or duplicate known sowing records')
    sowing.PEP_ID = sowing.PEP_ID.astype(np.int64)
    sowing = sowing.sort_values(['PEP_ID', 'SOWING_DATE']).reset_index(drop=True)
    if set(sowing.PEP_ID)-set(sum(groups.values(), [])):
        raise ValueError('Sowing records have stations absent from the frozen split')
    # Array views are immutable; sharing cannot change a later split's predictors.
    arrays = {column: source[column].to_numpy() for column in columns}
    arrays['weather'] = source[WEATHER].to_numpy(dtype=float)
    arrays['process'] = source[ACCUMULATION+INCREMENTS].to_numpy(dtype=float)
    arrays['coordinates'] = source[['LON', 'LAT']].to_numpy(dtype=float)
    for values in arrays.values():
        values.flags.writeable = False
    station_bounds = {int(site): (int(indices[0]), int(indices[-1])+1)
                      for site, indices in source.groupby('PEP_ID', sort=False).indices.items()}
    sowing_groups = {int(site): frame.to_dict('records') for site, frame in sowing.groupby('PEP_ID', sort=False)}
    return SimpleNamespace(paths=paths, source_hashes=hashes, base=base, parameters=parameters,
                           thresholds=thresholds, groups=groups, arrays=arrays,
                           station_bounds=station_bounds, sowing_groups=sowing_groups)


class SeasonalDataset(Dataset):
    feature_dims = dict(static_dim=2, future_dim=12)

    def __init__(self, args, flag, preprocessing=None, prepared=None):
        if flag not in ['train', 'val', 'test']:
            raise ValueError('flag must be train, val or test')
        for argument in ['seq_len', 'history_length']:
            if hasattr(args, argument):
                raise ValueError(f'{argument} is unsupported for sowing-onward seasonal inputs')
        if flag != 'train' and preprocessing is None:
            raise ValueError('Held-out datasets require training-fitted preprocessing')
        self.flag = flag
        if prepared is None:
            prepared = prepare_seasonal_source(args)
        else:
            paths = _sources(args)
            if paths != prepared.paths or any(file_hash(path) != prepared.source_hashes[name]
                                              for name, path in paths.items()):
                raise ValueError('Shared prepared source changed or belongs to different inputs')
        self.prepared = prepared
        self._a = prepared.arrays
        self.requested_station_ids = (prepared.base['evaluated_station_ids'] if flag == 'test'
                                      else prepared.groups[flag])
        gaps = np.diff([prepared.thresholds['T-P-V'][stage] for stage in [0]+STAGES])
        provenance = dict(schema_version=2, fit_station_ids=prepared.groups['train'],
                          split_sha256=prepared.base['split_sha256'],
                          source_sha256=prepared.source_hashes, threshold_gaps=gaps.tolist(),
                          seasonal_policy=POLICY)
        if preprocessing is None:
            training = np.isin(self._a['PEP_ID'], prepared.groups['train'])
            if not training.any():
                raise ValueError('Training source is empty')
            increments = self._a['process'][training, -1]
            positive = increments[np.isfinite(increments)&(increments > 0)]
            if not len(positive):
                raise ValueError('Training source lacks positive process increments for priors')
            floor = .1*float(np.quantile(positive, .1))/gaps
            finite = increments[np.isfinite(increments)]
            prior = np.maximum(float(np.maximum(finite, 0).mean())/gaps, floor)
            self.preprocessing = dict(**provenance, weather=_fit(self._a['weather'][training]),
                coordinates=_fit(self._a['coordinates'][training]), rate_floor=floor.tolist(),
                prior_rate=prior.tolist(), prior_policy='training mean nonnegative T-P-V increment / threshold gap at sowing')
        else:
            unsupported = {'history', 'process_history', 'history_valid', 'history_length',
                           'history_policy', 'seq_len'}.intersection(preprocessing)
            if unsupported:
                raise ValueError(f'Training preprocessing provenance contains pre-sowing fields: {sorted(unsupported)}')
            for key, expected in provenance.items():
                if preprocessing.get(key) != expected:
                    raise ValueError(f'Training preprocessing provenance differs: {key}')
            self.preprocessing = copy.deepcopy(preprocessing)
        self._prior = np.asarray(self.preprocessing['prior_rate'], dtype=np.float32)
        if self._prior.shape != (4,) or not np.isfinite(self._prior).all() or (self._prior <= 0).any():
            raise ValueError('Training prior rates must be finite and positive')
        self.records, self.coverage = [], []
        self._build_records()
        self.lengths = np.asarray([record['length'] for record in self.records], dtype=np.int64)
        self.contract = dict(prepared.base, feature_dims=self.feature_dims,
                             feature_order=dict(static=['LON', 'LAT'],
                                future=['annual_sin','annual_cos','daylength_fraction','lead_years']+WEATHER+
                                       [f'{name}_observed' for name in WEATHER]),
                             preprocessing=self.preprocessing, preprocessing_sha256=_identity(self.preprocessing))

    def _build_records(self):
        a = self._a
        for site in self.requested_station_ids:
            bounds = self.prepared.station_bounds.get(site)
            dates = a['DATE'][slice(*bounds)] if bounds else np.array([], dtype=np.int64)
            sows = self.prepared.sowing_groups.get(site, [])
            if not sows:
                # A site remains in the cohort even when it has no cycle key.
                continue
            for number, sow in enumerate(sows):
                start = int(sow['SOWING_DATE'])
                terminal = number == len(sows)-1
                end = int(dates[-1]) if terminal and len(dates) else (
                    int(sows[number+1]['SOWING_DATE'])-1 if not terminal else start-1)
                lo = int(np.searchsorted(dates, start))+(bounds[0] if bounds else 0)
                hi = int(np.searchsorted(dates, end, side='right'))+(bounds[0] if bounds else 0)
                actual = a['DATE'][lo:hi] if bounds else np.array([], dtype=np.int64)
                length = max(0, end-start+1)
                reasons = []
                if sow['SOWING_KNOWN_AT'] == DAY_NAT or sow['SOWING_KNOWN_AT'] > start:
                    reasons.append('sowing_not_known_at_reference')
                if (not len(actual) or len(actual) != length or actual[0] != start
                        or actual[-1] != end or (np.diff(actual) != 1).any()):
                    reasons.append('incomplete_daily_calendar')
                if len(actual):
                    if not np.isfinite(a['weather'][lo:hi]).all():
                        reasons.append('missing_weather')
                    if not np.isfinite(a['process'][lo:hi]).all():
                        reasons.append('missing_process')
                    if not np.isfinite(a['coordinates'][lo:hi]).all():
                        reasons.append('missing_coordinates')
                    if (a['SOWING_DATE'][lo:hi] != start).any():
                        reasons.append('inconsistent_sowing_assignment')
                    known = a['SOWING_KNOWN_AT'][lo:hi]
                    if ((known == DAY_NAT)|(known > start)).any():
                        reasons.append('row_sowing_not_known_at_reference')
                truth = []
                for code in [2,4,6,8]:
                    found = actual[a['CODE'][lo:hi] == code] if bounds else np.array([])
                    if len(found) > 1:
                        raise ValueError(f'Ambiguous duplicate recorded event for station {site}, sowing {_day_string(start)}, CODE {code}')
                    truth.append(_day_string(found[0]) if len(found) else None)
                ordered_truth = [value for value in truth if value is not None]
                order_conflict = any(first > second for first, second in zip(ordered_truth[:-1], ordered_truth[1:]))
                process_dates = {model: [None]*4 for model in PROCESS_MODELS}
                if not reasons:
                    for model, column in PROCESS_MODELS.items():
                        values = a[column][lo:hi]
                        for i, stage in enumerate(STAGES):
                            hits = np.flatnonzero(values >= self.prepared.thresholds[model][stage])
                            if len(hits):
                                process_dates[model][i] = _day_string(actual[hits[0]])
                record = dict(PEP_ID=int(site), SOWING_DATE=_day_string(start), END_DATE=_day_string(max(start,end)),
                    length=length, observed_rows=len(actual), terminal=terminal,
                    endpoint_reason=('no_observed_rows' if terminal and end < start else
                                     'station_record_end' if terminal else 'next_recorded_sowing'),
                    true_dates=truth, process_dates=process_dates, eligible=not reasons,
                    observation_order_conflict=order_conflict,
                    reason=';'.join(reasons), row_start=lo, row_stop=hi)
                self.coverage.append(record)
                if not reasons:
                    self.records.append(record)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[int(index)]
        lo, hi = record['row_start'], record['row_stop']
        a, n = self._a, record['length']
        sowing = int(a['DATE'][lo])
        dates = a['DATE'][lo:hi]
        doy = pd.DatetimeIndex(dates.astype('datetime64[D]')).dayofyear.to_numpy()
        phase = 2*np.pi*(doy-1)/365.25
        decl = .4093*np.sin(2*np.pi*(doy-81)/365)
        latitude = a['LAT'][lo]
        daylength = np.arccos(np.clip(-np.tan(np.radians(latitude))*np.tan(decl),-1,1))/np.pi
        future = np.column_stack([np.sin(phase),np.cos(phase),daylength,np.arange(1,n+1)/365.25,
                    _transform(a['weather'][lo:hi],self.preprocessing['weather']),
                    np.isfinite(a['weather'][lo:hi]).astype(np.float32)]).astype(np.float32)
        static = _transform(a['coordinates'][lo:lo+1], self.preprocessing['coordinates'])[0]
        lower, upper, kind = (np.zeros(4,dtype=np.int64) for _ in range(3))
        for i, value in enumerate(record['true_dates']):
            if value is not None:
                lead = int(np.datetime64(value,'D').astype(np.int64)-sowing+1)
                kind[i], lower[i], upper[i] = 1, lead-1, lead
        observed = np.full(n+1,-1,dtype=np.int64)
        for code, state in [(0,0),(2,1),(4,2),(6,3),(8,4)]:
            observed[1:][a['CODE'][lo:hi] == code] = state
        return dict(static=static,known_future=future,future_valid=np.ones(n,dtype=bool),
                    rate_prior=self._prior.copy(),event_lower=lower,event_upper=upper,event_kind=kind,
                    event_limit=np.full(4,n,dtype=np.int64),cycle_weight=np.float32(1),
                    observed_state=observed,sample_index=np.int64(index))


def seasonal_collate(samples):
    """Pad known weather to the batch horizon and keep supervision off padding."""
    if not samples:
        raise ValueError('Cannot collate an empty seasonal batch')
    longest = max(len(sample['known_future']) for sample in samples)
    output = {}
    for key in samples[0]:
        values = []
        for sample in samples:
            value = np.asarray(sample[key])
            if key in ['known_future','future_valid','observed_state']:
                length = longest+1 if key == 'observed_state' else longest
                padded = np.full((length,)+value.shape[1:], -1 if key == 'observed_state' else 0, dtype=value.dtype)
                padded[:len(value)] = value
                value = padded
            values.append(value)
        output[key] = torch.as_tensor(np.stack(values))
    return output
