"""Bounded transfer of frozen wheat thresholds, with no fitting or selection.

Preparation requires wheat Python (pandas/pyarrow), evaluation requires torch
but no parquet reader. Commands: ``python -B -m <this module> prepare --output
PATH`` and ``python -B -m <this module> evaluate --prepared PATH --checkpoints
SEED1.pt SEED2.pt SEED3.pt --output PATH``. The caller must select the encoder
and its three seeds using German validation only before calling evaluation.

The original observed forcing horizons and calendar-scenario sowing dates are
retained. GS39/65 are auxiliary threshold transfers, not German stage truths.
Caller-selected output directories must not already exist; sources are read only.
Checkpoints are loaded with PyTorch's weights-only loader.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from analysis.paper_study.wheat_dl_phenology_20261009.prepare import FEATURES, pack_cycle

ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = Path('/tmp/wheat-dl-20261009/field')
STAGES = [31, 39, 65, 85]
THERMAL_SCALE = 1760.39
FROZEN_THRESHOLDS = {31: 542.3, 39: 807.799125, 65: 1165.925558, 85: 1760.39}
FIELD_COUNTS = {'calibration': 28, 'reused_development_2019': 45, 'reused_external_strict': 143}
KEYS = ['partition', 'field_id', 'event']
PHENOLOGY = 'analysis/paper_study/overwinter_leaf_model_20261006/phenology'
EVIDENCE = 'analysis/paper_study/full_validation_20261007'
SOURCE_PATHS = dict(
    membership=f'{PHENOLOGY}/original_input_membership.csv',
    basf_registry='analysis/paper_study/seasonal_calibration_v1/field_season_membership.csv',
    external_registry='analysis/paper_study/corteva_external_seasonal_v1/external_input_membership.csv',
    constraints=f'{EVIDENCE}/field_stage_constraints_all.csv',
    predictions=f'{PHENOLOGY}/predicted_stage_dates.csv',
    archived_scores=f'{EVIDENCE}/field_stage_scores_all.csv',
    thresholds=f'{PHENOLOGY}/calibrated_stage_thresholds.json',
    weather='data/paper_study/field_weather_completed/daily_weather.parquet',
    process_parameters='process_model/parameters/calibration.json',
    process_transform='process_model/calibrate.py',
    packing='analysis/paper_study/wheat_dl_phenology_20261009/prepare.py',
    network='model/wheat_phenology_dl/network.py')
LIMITS = [
    'Retrospective realized-weather transfer with calendar-scenario sowing dates, not observed sowing dates.',
    'Original observed forcing horizons only; no extension to full season or forced event dates.',
    'GS39/65 are existing auxiliary thresholds transferred to neural progress; no German GS39/65 truths.',
    'Calibration, reused BASF validation and location-disjoint external fields remain separate.',
    'Strict external denotes the original location split, not untouched stage evidence; prior analyses used these records.',
    'Interval distances are constraint violations, not exact event-date errors; missing events supply only lower bounds.',
    'The caller selects three seeds of one encoder on German validation only; field results must not select checkpoints.',
    'No normalizer, threshold, sowing date, or model parameter is fitted to fields.',
    'Original horizons beyond 366 days are retained and flagged as sequence-length extrapolation.'
]


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def new_output(path):
    """Resolve a caller-selected destination without replacing an existing path."""
    if Path(path).is_symlink():
        raise FileExistsError(path)
    path = Path(path).resolve()
    if path.exists():
        raise FileExistsError(path)
    return path


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def unique(frame, keys, label):
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError(f'Duplicate or missing {label} identity')


def align_sources(members, constraints, baseline):
    """Left-align all original field/stage identities, including no-observation rows."""
    unique(members, ['field_id'], 'field')
    unique(constraints, KEYS, 'constraint')
    unique(baseline, KEYS, 'prediction')
    grid = members.merge(pd.DataFrame({'event': STAGES}), how='cross')
    for label, frame in [('constraint', constraints), ('prediction', baseline)]:
        check = frame[KEYS].merge(grid[KEYS], on=KEYS, how='left', indicator=True)
        if not check._merge.eq('both').all():
            raise ValueError(f'Unknown field, partition or stage in {label} source')
    identity_columns = [c for c in ['site_id', 'latitude', 'longitude', 'season_year', 'dataset_id']
                        if c in constraints and c in members]
    if identity_columns:
        check = constraints.merge(members[['field_id', 'partition', *identity_columns]],
                                  on=['field_id', 'partition'], suffixes=('', '_registry'), validate='many_to_one')
        for column in identity_columns:
            if not check[column].eq(check[column + '_registry']).all():
                raise ValueError(f'Constraint source identity changed: {column}')
    overlaps = (set(grid) & set(constraints)) - set(KEYS)
    aligned = grid.merge(constraints.drop(columns=sorted(overlaps)), on=KEYS,
                         how='left', validate='one_to_one', indicator=True)
    absent = aligned._merge.eq('left_only')
    aligned['constraint_record_present'] = ~absent
    aligned.loc[absent, 'reason'] = 'no_archived_stage_constraint'
    for column in ['eligible_constraint', 'genuinely_bracketed']:
        aligned[column] = aligned[column].eq(True)
    aligned.loc[absent, 'censoring'] = 'none'
    aligned = aligned.drop(columns='_merge')
    aligned = aligned.merge(baseline[KEYS + ['predicted_date', 'forcing_end']].rename(
        columns={'predicted_date': 'tpv_date'}), on=KEYS, how='left', validate='one_to_one', indicator=True)
    if not aligned._merge.eq('both').all():
        raise ValueError('Missing archived prediction/horizon membership')
    aligned = aligned.drop(columns='_merge')
    for column in ['sowing_date', 'last_forcing_date', 'tpv_date', 'forcing_end',
                   'lower_exclusive', 'upper_inclusive']:
        aligned[column] = pd.to_datetime(aligned[column])
    if not aligned.forcing_end.eq(aligned.last_forcing_date).all():
        raise ValueError('Archived prediction forcing horizon differs from field registry')
    if ((aligned.tpv_date < aligned.sowing_date) | (aligned.tpv_date > aligned.forcing_end)).any():
        raise ValueError('Archived date outside original forcing horizon')
    bracketed = aligned.genuinely_bracketed
    widths = (aligned.upper_inclusive - aligned.lower_exclusive).dt.days
    if not aligned.loc[bracketed, 'bracket_width_days'].eq(widths[bracketed]).all():
        raise ValueError('Archived interval width differs from its native date bounds')
    return aligned.reset_index(drop=True)


def interval_score(predicted, lower, upper, forcing_end):
    """Integer-day distance to (lower, upper]; missing date gives only a bound."""
    predicted, lower, upper, end = [pd.Timestamp(x) if pd.notna(x) else pd.NaT
                                    for x in [predicted, lower, upper, forcing_end]]
    if pd.isna(lower) and pd.isna(upper):
        raise ValueError('At least one native date bound is required')
    if pd.notna(lower) and pd.notna(upper) and lower >= upper:
        raise ValueError('Inconsistent date bounds')
    missing = pd.isna(predicted)
    if missing:
        if pd.isna(end):
            raise ValueError('Missing crossing requires a known forcing horizon')
        signed = float(max(0, (end + pd.Timedelta(days=1) - upper).days)) if pd.notna(upper) else 0.
    elif pd.notna(lower) and predicted <= lower:
        signed = float((predicted - lower).days - 1)
    elif pd.notna(upper) and predicted > upper:
        signed = float((predicted - upper).days)
    else:
        signed = 0.
    return dict(signed_distance_days=signed, distance_days=abs(signed),
                distance_is_lower_bound=missing, compatible=pd.NA if missing else signed == 0,
                compatible_or_not_excluded=signed == 0,
                prediction_status='not_reached_by_forcing_end' if missing else 'reached')


def transform_weather(weather, member, parameters):
    """Apply the frozen wheat transform using the actual daily tmean column."""
    from process_model.calibrate import transform
    frame = weather.rename(columns={'date': 'DATE', 'tmean_c': 't_mean',
                                     'tmax_c': 't_max', 'tmin_c': 't_min'}).copy()
    values = frame[['t_mean', 't_max', 't_min']].to_numpy(float)
    if not np.isfinite(values).all() or (frame.t_mean > 40).any():
        raise ValueError('Unsupported or missing temperature input')
    frame['PEP_ID'] = 0  # Each call contains exactly one identified field cycle.
    frame['LAT'] = float(member['latitude'])
    frame['SOWING_DATE'] = member['sowing_date']
    frame['SOWING_KNOWN_AT'] = member['sowing_date']  # Existing calendar-availability assumption.
    frame['GDD'] = np.where(frame.t_mean > 30, 20 - 2 * (frame.t_mean - 30),
                            np.clip(frame.t_mean, 0, 20))
    return transform(frame, parameters)


def prepare(output, root=ROOT):
    """Create immutable, unnormalized field arrays and exact archived comparisons."""
    output, root = new_output(output), Path(root)
    paths = {name: root / relative for name, relative in SOURCE_PATHS.items()}
    hashes = {name: sha(path) for name, path in paths.items()}
    members = pd.read_csv(paths['membership'], dtype={'source_unit': str})
    if members.groupby('partition').size().to_dict() != FIELD_COUNTS:
        raise ValueError('Original 28/45/143 field membership changed')
    unique(members, ['field_id'], 'field')
    # Reconcile the archived clock registry with the original independent registries.
    basf = pd.read_csv(paths['basf_registry'], dtype={'source_unit': str})
    basf['partition'] = basf.partition.replace({'validation': 'reused_development_2019'})
    external = pd.read_csv(paths['external_registry'], dtype={'source_unit': str})
    external = external.loc[external.strict_location_disjoint.eq(True)].copy()
    external['partition'] = 'reused_external_strict'
    registry = pd.concat([basf, external], ignore_index=True)
    unique(registry, ['field_id'], 'original registry')
    identity = ['partition', 'site_id', 'season_year', 'source_unit', 'latitude', 'longitude',
                'sowing_date', 'sowing_basis', 'first_forcing_date', 'last_forcing_date',
                'forcing_days', 'calendar_known_at_sowing_assumed']
    paired = members.merge(registry[['field_id', *identity]], on='field_id',
                            how='outer', validate='one_to_one', suffixes=('', '_original'), indicator=True)
    if not paired._merge.eq('both').all():
        raise ValueError('Original registry field identities changed')
    for column in identity:
        if not paired[column].eq(paired[column + '_original']).all():
            raise ValueError(f'Original field identity changed: {column}')
    thresholds = json.loads(paths['thresholds'].read_text())['all_stage_thresholds']
    if {stage: thresholds[str(stage)] for stage in STAGES} != FROZEN_THRESHOLDS:
        raise ValueError('Existing frozen auxiliary thresholds changed')
    parameters = json.loads(paths['process_parameters'].read_text())
    native = {r['BBCH']: r['Cumulative_t_pp_v_GDD'] for r in parameters['thresholds']}
    if [native[s] for s in [10, 31, 51, 85]] != [142.01, 542.3, 986.65, 1760.39]:
        raise ValueError('Frozen wheat native thresholds changed')
    constraints = pd.read_csv(paths['constraints'])
    constraints = constraints.loc[constraints.partition.isin(FIELD_COUNTS) & constraints.event.isin(STAGES)]
    baseline = pd.read_csv(paths['predictions'])
    baseline = baseline.loc[baseline.partition.isin(FIELD_COUNTS) & baseline.event.isin(STAGES)]
    if not baseline.q.eq(1.).all():
        raise ValueError('Archived T-P-V comparator must use frozen q=1')
    records = align_sources(members, constraints, baseline)
    weather = pd.read_parquet(paths['weather'], columns=['location_id', 'date', 'tmean_c', 'tmax_c', 'tmin_c'])
    weather['date'] = pd.to_datetime(weather.date)
    unique(weather, ['location_id', 'date'], 'weather')
    sites = {site: frame.set_index('date') for site, frame in weather.groupby('location_id')}
    horizon = max(366, int(members.forcing_days.max()))
    x = np.zeros((len(members), horizon, len(FEATURES)), dtype=np.float32)
    g = np.zeros((len(members), horizon), dtype=np.float32)
    lengths = np.zeros(len(members), dtype=np.int16)
    checks = []
    for i, member in enumerate(members.to_dict('records')):
        dates = pd.date_range(member['sowing_date'], member['last_forcing_date'])
        if (len(dates) != member['forcing_days'] or str(dates[0].date()) != member['first_forcing_date']
                or member['site_id'] not in sites):
            raise ValueError(f'Invalid original forcing identity: {member["field_id"]}')
        frame = sites[member['site_id']].reindex(dates).rename_axis('date').reset_index()
        transformed = transform_weather(frame, member, parameters)
        x[i], g[i], lengths[i] = pack_cycle(transformed, dates[0], horizon)
        if lengths[i] != member['forcing_days']:
            raise ValueError(f'Incomplete original weather prefix: {member["field_id"]}')
        cumulative = transformed.Cumulative_t_pp_v_GDD.to_numpy(float)
        for stage in STAGES:
            hits = np.flatnonzero(cumulative >= FROZEN_THRESHOLDS[stage])
            checks.append(dict(partition=member['partition'], field_id=member['field_id'], event=stage,
                               reconstructed_date=dates[hits[0]] if len(hits) else pd.NaT))
    audit = records[KEYS + ['tpv_date']].merge(pd.DataFrame(checks), on=KEYS, validate='one_to_one')
    equal = audit.tpv_date.eq(audit.reconstructed_date) | (audit.tpv_date.isna() & audit.reconstructed_date.isna())
    if not equal.all():
        raise ValueError(f'Frozen transform disagrees with {int((~equal).sum())} archived T-P-V dates')
    members['array_index'] = np.arange(len(members))
    members['weather_days'] = lengths
    members['beyond_training_horizon'] = lengths > 366
    archived = pd.read_csv(paths['archived_scores'])
    archived = archived.loc[archived.partition.isin(FIELD_COUNTS) & archived.event.isin(STAGES)
                            & archived.forcing_scope.eq('observed_horizon')]
    unique(archived, KEYS, 'archived score')
    # Recompute only the archival comparison, never an observed point date.
    self_comparison = compare_predictions(records, records[KEYS + ['tpv_date']].rename(
        columns={'tpv_date': 'predicted_date'}))
    scored = self_comparison.loc[self_comparison.eligible_constraint]
    reconciled = scored.merge(archived[KEYS + ['distance_days', 'signed_distance_days', 'compatible',
                                                'distance_is_lower_bound']], on=KEYS,
                               how='outer', validate='one_to_one', indicator=True)
    if not reconciled._merge.eq('both').all():
        raise ValueError('Archived score support differs from native constraints')
    for column in ['distance_days', 'signed_distance_days', 'distance_is_lower_bound']:
        if not reconciled['tpv_' + column].eq(reconciled[column]).all():
            raise ValueError(f'Archived scoring convention mismatch: {column}')
    if not reconciled.tpv_compatible_or_not_excluded.eq(reconciled.compatible).all():
        raise ValueError('Archived compatibility convention mismatch')
    for name, path in paths.items():
        if sha(path) != hashes[name]:
            raise ValueError(f'Source changed during preparation: {path}')
    output.mkdir(parents=True)
    np.savez_compressed(output / 'arrays.npz', X=x, G=g, lengths=lengths)
    members.to_csv(output / 'fields.csv', index=False)
    records.to_csv(output / 'constraints_and_tpv.csv', index=False)
    audit.to_csv(output / 'tpv_reconciliation.csv', index=False)
    field_counts = members.groupby('partition').size().to_dict()
    counts = []
    for (partition, stage), part in records.groupby(['partition', 'event']):
        valid = part.loc[part.eligible_constraint]
        counts.append(dict(partition=partition, stage=int(stage), original_fields=len(part),
            constrained_fields=len(valid), two_sided=int(valid.genuinely_bracketed.sum()),
            left_only=int(valid.censoring.eq('left').sum()), right_only=int(valid.censoring.eq('right').sum()),
            no_eligible_constraint=int((~part.eligible_constraint).sum()),
            archived_missing_predictions=int(part.tpv_date.isna().sum())))
    metadata = dict(status='prepared', stages=STAGES, features=FEATURES, horizon=horizon,
        thermal_scale=THERMAL_SCALE, frozen_thresholds=FROZEN_THRESHOLDS, normalized=False,
        normalizer_source='Each selected checkpoint metadata; never field weather',
        forcing_scope='observed_horizon', field_counts=field_counts, counts=counts,
        weather_rows_available=len(weather), weather_days_used=int(lengths.sum()),
        fields_beyond_training_horizon=int((lengths > 366).sum()),
        field_stage_rows=len(records), native_constraint_rows=len(constraints),
        eligible_constraints=int(records.eligible_constraint.sum()),
        temperature_definition='t_mean = recorded tmean_c; not (tmin_c+tmax_c)/2',
        gdd_definition='0 below 0; tmean through 20; 20 through 30; 20-2*(tmean-30) above 30; reject above 40',
        date_index='Position 1 is sowing day; elapsed_fraction = position/366 including horizons >366',
        archived_tpv_reconciliation=dict(rows=len(audit), different=int((~equal).sum())),
        archived_score_reconciliation=dict(rows=len(reconciled), different=0),
        sources={name: dict(path=str(path), sha256=hashes[name]) for name, path in paths.items()},
        artifacts={name: sha(output / name) for name in ['arrays.npz', 'fields.csv',
                   'constraints_and_tpv.csv', 'tpv_reconciliation.csv']},
        preparation_code_sha256=sha(__file__), limits=LIMITS)
    write_json(output / 'metadata.json', metadata)
    return metadata


def load_prepared(path):
    """No parquet or torch import: the prepared bundle crosses Python environments."""
    path = Path(path)
    metadata = json.loads((path / 'metadata.json').read_text())
    for name, digest in metadata['artifacts'].items():
        if sha(path / name) != digest:
            raise ValueError(f'Prepared artifact hash mismatch: {name}')
    if (metadata['features'] != FEATURES or metadata['stages'] != STAGES or metadata['normalized']
            or metadata['thermal_scale'] != THERMAL_SCALE
            or {int(k): v for k, v in metadata['frozen_thresholds'].items()} != FROZEN_THRESHOLDS):
        raise ValueError('Prepared feature/threshold contract changed')
    with np.load(path / 'arrays.npz', allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    fields = pd.read_csv(path / 'fields.csv', dtype={'source_unit': str})
    records = pd.read_csv(path / 'constraints_and_tpv.csv', parse_dates=[
        'sowing_date', 'last_forcing_date', 'lower_exclusive', 'upper_inclusive', 'tpv_date', 'forcing_end'])
    unique(fields, ['field_id'], 'prepared field')
    unique(records, KEYS, 'prepared comparison')
    expected = (len(fields), metadata['horizon'])
    if (arrays['X'].shape != (*expected, len(FEATURES)) or arrays['G'].shape != expected
            or arrays['lengths'].shape != (len(fields),)
            or not np.array_equal(fields.array_index, np.arange(len(fields)))
            or not np.array_equal(arrays['lengths'], fields.weather_days)
            or not np.isfinite(arrays['X']).all() or not np.isfinite(arrays['G']).all()
            or (arrays['G'] < 0).any() or (arrays['lengths'] <= 0).any()
            or (arrays['lengths'] > metadata['horizon']).any()):
        raise ValueError('Prepared arrays and field identities differ')
    if fields.groupby('partition').size().to_dict() != FIELD_COUNTS or len(records) != len(fields) * len(STAGES):
        raise ValueError('Prepared field coverage changed')
    return arrays, fields, records, metadata


def validate_checkpoint_metadata(metadata):
    native = np.array([142.01, 542.3, 986.65, 1760.39]) / THERMAL_SCALE
    if (metadata['features'] != FEATURES or metadata['stages'] != [10, 31, 51, 85]
            or metadata['thermal_scale'] != THERMAL_SCALE or metadata['horizon'] != 366
            or np.shape(metadata['thresholds']) != (4,)
            or not np.allclose(metadata['thresholds'], native, rtol=0, atol=1e-8)):
        raise ValueError('Checkpoint wheat feature/stage/scale contract differs')
    mean, std = np.asarray(metadata['mean']), np.asarray(metadata['std'])
    if (mean.shape != (len(FEATURES),) or std.shape != mean.shape
            or not np.isfinite(mean).all() or not np.isfinite(std).all() or (std <= 0).any()):
        raise ValueError('Invalid checkpoint input normalizer')


def threshold_cdf(state, raw_sigma, normalized_thresholds):
    """Same log-state Gaussian CDF as network.py, at frozen auxiliary thresholds."""
    import torch
    thresholds = torch.as_tensor(normalized_thresholds, dtype=state.dtype, device=state.device)
    if thresholds.ndim != 1 or not torch.isfinite(thresholds).all() or (thresholds <= 0).any():
        raise ValueError('Positive finite threshold vector required')
    sigma = .02 + .78 * torch.sigmoid(raw_sigma)
    z = (state.clamp_min(1e-12).log()[:, None, :] - thresholds.log()[None, :, None]) / sigma
    return torch.where(state[:, None, :] > 0, .5 * (1 + torch.erf(z / math.sqrt(2))), torch.zeros_like(z))


def ensemble_median_days(cdfs, lengths):
    """Average exactly three seed CDFs, then find the first available 0.5 crossing."""
    cdfs, lengths = np.asarray(cdfs), np.asarray(lengths)
    if cdfs.ndim != 4 or cdfs.shape[0] != 3:
        raise ValueError('Exactly three seed CDF arrays required')
    if (lengths.shape != (cdfs.shape[1],) or not np.isfinite(cdfs).all()
            or (cdfs < 0).any() or (cdfs > 1).any()
            or (lengths < 0).any() or (lengths > cdfs.shape[-1]).any()
            or (np.diff(cdfs, axis=-1) < -1e-6).any()):
        raise ValueError('Invalid CDFs or forcing lengths')
    mean = cdfs.mean(axis=0, dtype=np.float64)
    crossed = (mean >= .5) & (np.arange(mean.shape[-1])[None, None, :] < lengths[:, None, None])
    return np.where(crossed.any(axis=-1), crossed.argmax(axis=-1) + 1, np.nan)


def compare_predictions(records, predictions):
    """Preserve the full field/stage grid and score the same eligible constraints."""
    unique(predictions, KEYS, 'neural prediction')
    joined = records.merge(predictions[KEYS + ['predicted_date']].rename(
        columns={'predicted_date': 'neural_date'}), on=KEYS, how='outer', validate='one_to_one', indicator=True)
    if not joined._merge.eq('both').all():
        raise ValueError('Neural predictions must cover exactly the original field/stage grid')
    joined = joined.drop(columns='_merge')
    joined['neural_date'] = pd.to_datetime(joined.neural_date)
    if ((joined.neural_date < joined.sowing_date) | (joined.neural_date > joined.forcing_end)).any():
        raise ValueError('Neural prediction outside original weather horizon')
    for model in ['tpv', 'neural']:
        rows = []
        for row in joined.to_dict('records'):
            rows.append(interval_score(row[model + '_date'], row['lower_exclusive'],
                         row['upper_inclusive'], row['forcing_end']) if row['eligible_constraint'] else {})
        distances = pd.DataFrame(rows, index=joined.index).add_prefix(model + '_')
        joined = pd.concat([joined, distances], axis=1)
    return joined


def summarize(scored):
    """Report reached-date errors and censoring lower bounds separately, with pairs."""
    summaries = []
    for (partition, stage), group in scored.groupby(['partition', 'event'], sort=True):
        eligible = group.loc[group.eligible_constraint]
        for scope, selected in [('all_constraints', eligible),
                                ('genuine_two_sided', eligible.loc[eligible.genuinely_bracketed]),
                                ('left_only', eligible.loc[eligible.censoring.eq('left')]),
                                ('right_only', eligible.loc[eligible.censoring.eq('right')])]:
            widths = selected.bracket_width_days.dropna()
            row = dict(partition=partition, event=int(stage), scope=scope,
                       registered_fields=len(group), constraints=len(selected),
                       unavailable_constraints=int((~group.eligible_constraint).sum()),
                       median_interval_width_days=widths.median(),
                       minimum_interval_width_days=widths.min(), maximum_interval_width_days=widths.max())
            for model in ['tpv', 'neural']:
                reached = selected.loc[selected[model + '_date'].notna()]
                row.update({model + '_reached': len(reached), model + '_missing': len(selected) - len(reached),
                    model + '_prediction_coverage': len(reached) / len(selected) if len(selected) else np.nan,
                    model + '_all_fields_missing': int(group[model + '_date'].isna().sum()),
                    model + '_mean_distance_days': reached[model + '_distance_days'].mean(),
                    model + '_mean_signed_distance_days': reached[model + '_signed_distance_days'].mean(),
                    model + '_compatible_fraction_reached': reached[model + '_compatible'].mean(),
                    model + '_mean_distance_lower_bound_days': selected[model + '_distance_days'].mean(),
                    model + '_missing_lower_bounds': int(selected[model + '_distance_is_lower_bound'].eq(True).sum())})
            paired = selected.loc[selected.tpv_date.notna() & selected.neural_date.notna()]
            row['paired_reached'] = len(paired)
            row['paired_tpv_mean_distance_days'] = paired.tpv_distance_days.mean()
            row['paired_neural_mean_distance_days'] = paired.neural_distance_days.mean()
            row['paired_neural_minus_tpv_distance_days'] = (paired.neural_distance_days - paired.tpv_distance_days).mean()
            summaries.append(row)
    return pd.DataFrame(summaries)


def evaluate(prepared, checkpoints, output, *, device='cpu', batch_size=64):
    """Evaluate three caller-selected checkpoints; never rank or choose candidates."""
    import torch
    from model.wheat_phenology_dl.network import WheatDevelopmentModel
    output = new_output(output)
    if len(checkpoints) != 3 or batch_size < 1:
        raise ValueError('Exactly three selected seed checkpoints and positive batch_size required')
    arrays, fields, records, metadata = load_prepared(prepared)
    cdfs, checkpoint_records, seeds, encoders, training_inputs, protocols = [], [], [], [], [], []
    network_sha = sha(ROOT / SOURCE_PATHS['network'])
    normalized_thresholds = np.array([FROZEN_THRESHOLDS[stage] for stage in STAGES]) / THERMAL_SCALE
    for checkpoint in map(Path, checkpoints):
        digest = sha(checkpoint)
        payload = torch.load(checkpoint, map_location='cpu', weights_only=True)
        m, config = payload['metadata'], payload['config']
        validate_checkpoint_metadata(m)
        if payload['network_sha256'] != network_sha:
            raise ValueError('Checkpoint network source hash differs from current wheat network')
        if not np.isfinite(payload['validation_crps']):
            raise ValueError('Checkpoint lacks finite German validation selection evidence')
        model = WheatDevelopmentModel(m['thresholds'], len(m['features']),
                                       encoder=config['encoder'], width=config['width'])
        model.load_state_dict(payload['state_dict'])
        if not np.allclose(model.thresholds.detach().numpy(), m['thresholds'], rtol=0, atol=1e-7):
            raise ValueError('Checkpoint native threshold buffer differs from metadata')
        model.to(device).eval()
        mean, std = np.asarray(m['mean'], np.float32), np.asarray(m['std'], np.float32)
        seed_cdf = np.empty((len(fields), len(STAGES), metadata['horizon']), np.float32)
        with torch.inference_mode():
            for start in range(0, len(fields), batch_size):
                stop = min(start + batch_size, len(fields))
                x = ((arrays['X'][start:stop] - mean) / std).astype(np.float32)
                state, _ = model(torch.as_tensor(x, device=device),
                                 torch.as_tensor(arrays['G'][start:stop], device=device), THERMAL_SCALE)
                seed_cdf[start:stop] = threshold_cdf(state, model.raw_sigma, normalized_thresholds).cpu().numpy()
        if sha(checkpoint) != digest:
            raise ValueError('Checkpoint changed during evaluation; wait for completed fitting')
        cdfs.append(seed_cdf)
        seeds.append(config['seed']); encoders.append(config['encoder'])
        training_inputs.append(payload['input_arrays_sha256']); protocols.append(payload['protocol_sha256'])
        checkpoint_records.append(dict(path=str(checkpoint.resolve()), sha256=digest, config=config,
            best_epoch=payload['best_epoch'], validation_crps=payload['validation_crps'],
            metadata_sha256=hashlib.sha256(json.dumps(m, sort_keys=True).encode()).hexdigest(),
            sigma=float((.02 + .78 * torch.sigmoid(model.raw_sigma)).detach().cpu())))
    if (len(set(seeds)) != 3 or len(set(encoders)) != 1 or len(set(training_inputs)) != 1
            or len(set(protocols)) != 1 or len({r['sha256'] for r in checkpoint_records}) != 3):
        raise ValueError('Use three distinct seeds of one encoder from the same frozen training inputs/protocol')
    days = ensemble_median_days(np.stack(cdfs), arrays['lengths'])
    rows = []
    for i, field in enumerate(fields.to_dict('records')):
        for j, stage in enumerate(STAGES):
            day = days[i, j]
            rows.append(dict(partition=field['partition'], field_id=field['field_id'], event=stage,
                median_day=day, predicted_date=pd.NaT if np.isnan(day) else
                pd.Timestamp(field['sowing_date']) + pd.Timedelta(days=int(day) - 1)))
    predictions = pd.DataFrame(rows)
    scored = compare_predictions(records, predictions)
    metrics = summarize(scored)
    # Revalidate prepared file hashes after evaluation, before publishing results.
    load_prepared(prepared)
    output.mkdir(parents=True)
    predictions.to_csv(output / 'neural_predictions.csv', index=False)
    scored.to_csv(output / 'field_stage_comparison.csv', index=False)
    metrics.to_csv(output / 'interval_metrics.csv', index=False)
    report = dict(status='evaluated', encoder=encoders[0], seeds=seeds,
        caller_selection_contract='Encoder and three checkpoints selected on German validation only; no selection implemented here',
        selection_independence_verified=False, field_results_used_for_selection=False,
        selection_independence_limit='External parent selection history cannot be established from checkpoint payloads alone',
        checkpoints=checkpoint_records, field_counts=metadata['field_counts'],
        eligible_constraints=int(scored.eligible_constraint.sum()), comparison_rows=len(scored),
        forcing_scope=metadata['forcing_scope'], frozen_thresholds=FROZEN_THRESHOLDS,
        ensemble='Mean of three per-seed log-state Gaussian CDFs, then first median crossing within original forcing',
        normalizer='Each checkpoint metadata mean/std; raw thermal increments remain unnormalized',
        fields_beyond_training_horizon=metadata['fields_beyond_training_horizon'],
        prepared_metadata_sha256=sha(Path(prepared) / 'metadata.json'),
        evaluation_code_sha256=sha(__file__), network_sha256=network_sha,
        artifacts={name: sha(output / name) for name in ['neural_predictions.csv',
                   'field_stage_comparison.csv', 'interval_metrics.csv']}, limits=LIMITS)
    write_json(output / 'evaluation.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--output', type=Path, default=OUTPUT_ROOT / 'prepared')
    evaluation = commands.add_parser('evaluate')
    evaluation.add_argument('--prepared', type=Path, default=OUTPUT_ROOT / 'prepared')
    evaluation.add_argument('--checkpoints', type=Path, nargs=3, required=True)
    evaluation.add_argument('--output', type=Path, required=True)
    evaluation.add_argument('--device', default='cpu')
    evaluation.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    if args.command == 'prepare':
        result = prepare(args.output)
    else:
        result = evaluate(args.prepared, args.checkpoints, args.output,
                          device=args.device, batch_size=args.batch_size)
    print(json.dumps({k: result[k] for k in ['status', 'field_counts', 'eligible_constraints']}, indent=2))


if __name__ == '__main__':
    main()
