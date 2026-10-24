"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/field_stage_extension_20261010')
import hashlib
import io
import json
from pathlib import Path
import numpy as np
import pandas as pd
EVENTS = (37, 39, 65)
MEMBER = 'stb_collection_20261009/sources/other_field_sources/corteva-2014-2018/corteva-wheat-diseases.txt'
MEMBER_SHA = 'd58a4a66639a4cdee2560ca53c38288cc4958b7b5d030370dbb403a91e4465f8'
ROW_COLUMNS = ['field_id', 'date', 'stage_from', 'stage_to', 'source_keys', 'latitude', 'longitude']
IDENTITY = ['field_id', 'source', 'partition', 'season_year', 'site_id', 'sowing_date', 'horizon_end']

def load_sources(root):
    root = Path(root)
    base = root / 'data/final_study/evidence'
    basf = pd.read_csv(base / 'seasonal_calibration_v1/field_season_membership.csv', dtype={'source_unit': str})
    basf['source'] = 'BASF'
    if not basf.season_year.isin([2017, 2018, 2019]).all():
        raise ValueError('Unexpected BASF registry years')
    if not basf.partition.eq('calibration').equals(basf.season_year.isin([2017, 2018])):
        raise ValueError('Original BASF calibration membership changed')
    basf['partition'] = np.where(basf.partition.eq('calibration'), 'calibration', 'reused_development_2019')
    external = pd.read_csv(base / 'corteva_external_seasonal_v1/external_input_membership.csv', dtype={'source_unit': str})
    external = external.loc[external.strict_location_disjoint.astype(str).str.lower().eq('true')].copy()
    external['source'], external['partition'] = ('Corteva', 'reused_external_strict')
    registry = pd.concat([basf, external], ignore_index=True)
    registry['sowing_date'] = pd.to_datetime(registry.sowing_date)
    registry['horizon_end'] = registry.sowing_date + pd.Timedelta(days=329)
    registry = registry.sort_values('field_id').reset_index(drop=True)
    if registry.field_id.duplicated().any() or len(registry) != 216:
        raise ValueError('The frozen 216-field registry has changed')
    registry['prediction_index'] = np.arange(len(registry))
    source = pd.read_csv(base / 'seasonal_calibration_v1/basf_source_assessments_snapshot.csv', dtype={'physical_unit': str})
    source['field_id'] = source.dataset_id + '|' + source.physical_unit + '|' + source.season_year.astype(str)
    source = source.rename(columns={'sourcekeys': 'source_keys'})[ROW_COLUMNS]
    content = (root / 'data/final_study/inputs/stb_collection_20261009/sources/other_field_sources/corteva-2014-2018/corteva-wheat-diseases.txt').read_bytes()
    if hashlib.sha256(content).hexdigest() != MEMBER_SHA:
        raise ValueError('Original Corteva source hash mismatch')
    raw = pd.read_csv(io.BytesIO(content), sep='\t')
    raw['source_keys'] = [f'Corteva:line:{i}' for i in range(2, len(raw) + 2)]
    raw = raw.loc[raw.Treatment.eq('Untreated') & raw.NumSprays.eq(0) & (raw.GsFrom.notna() | raw.GsTo.notna())].copy()
    raw['field_id'] = 'corteva-2014-2018-external|' + raw.Field + '|' + raw.Field.str.split('-').str[0]
    raw = raw.rename(columns={'Date': 'date', 'GsFrom': 'stage_from', 'GsTo': 'stage_to', 'Latitude': 'latitude', 'Longitude': 'longitude'})[ROW_COLUMNS]
    legacy = pd.read_csv(root / 'data/final_study/inputs/paper_study/observations/corteva_external_assessments.csv', dtype={'physical_unit': str})
    legacy['field_id'] = legacy.dataset_id + '|' + legacy.physical_unit + '|' + legacy.season_year.astype(str)
    legacy = legacy.rename(columns={'stage_sourcekeys': 'source_keys'})[ROW_COLUMNS]
    return (registry, pd.concat([source, raw], ignore_index=True), pd.concat([source, legacy], ignore_index=True))

def qualify(rows, registry):
    if {'value', 'Value'} & set(rows):
        raise ValueError('Numeric disease values are forbidden at the stage boundary')
    if registry.field_id.duplicated().any():
        raise ValueError('Duplicate field registry')
    frame = rows[ROW_COLUMNS].copy().reset_index(drop=True)
    frame['date'] = pd.to_datetime(frame.date).dt.normalize()
    if frame.date.isna().any():
        raise ValueError('A stage-bearing source row has no recorded date')
    for col in ('stage_from', 'stage_to'):
        frame[col] = pd.to_numeric(frame[col], errors='raise')
    meta = registry[IDENTITY + ['latitude', 'longitude']].rename(columns={'latitude': 'registry_latitude', 'longitude': 'registry_longitude'})
    frame = frame.merge(meta, on='field_id', how='left', validate='many_to_one')
    frame['sowing_date'] = pd.to_datetime(frame.sowing_date)
    frame['horizon_end'] = pd.to_datetime(frame.horizon_end)
    known = frame.source.notna()
    for col in ('latitude', 'longitude'):
        if not np.isclose(frame.loc[known, col], frame.loc[known, 'registry_' + col], rtol=0, atol=1e-08).all():
            raise ValueError('Source coordinates do not match the original field registry')
    frame['reason'] = ''
    frame.loc[~known, 'reason'] = 'outside_original_field_membership'

    def reject(mask, reason):
        frame.loc[mask & frame.reason.eq(''), 'reason'] = reason
    reject(~frame.date.between(frame.sowing_date, frame.horizon_end), 'outside_fixed_weather_horizon')
    reject(frame.stage_from.isna() & frame.stage_to.isna(), 'missing_both_stage_endpoints')
    reject(frame.stage_from.notna() & ~frame.stage_from.between(0, 99) | frame.stage_to.notna() & ~frame.stage_to.between(0, 99), 'stage_outside_supported_range')
    reject(frame.stage_from.gt(frame.stage_to), 'reversed_stage_interval')
    rejected = [frame.loc[frame.reason.ne('')].copy()]
    accepted = []
    valid = frame.loc[frame.reason.eq('')]
    for (field, date), group in valid.groupby(['field_id', 'date'], sort=True):
        lower, upper = (group.stage_from.max(), group.stage_to.min())
        if pd.notna(lower) and pd.notna(upper) and (lower > upper):
            rejected.append(group.assign(reason='conflicting_date_stage_ranges'))
            continue
        accepted.append(dict(field_id=field, date=date, stage_from=lower, stage_to=upper, source_keys=json.dumps(sorted(set(group.source_keys.astype(str)))), source_row_count=len(group), unique_source_intervals=len(group[['stage_from', 'stage_to']].drop_duplicates())))
    columns = ['field_id', 'date', 'stage_from', 'stage_to', 'source_keys', 'source_row_count', 'unique_source_intervals']
    normalized = pd.DataFrame(accepted, columns=columns)
    bad_fields = set()
    for field, group in normalized.groupby('field_id', sort=True):
        prior_lower = -np.inf
        for row in group.sort_values('date').itertuples(index=False):
            if pd.notna(row.stage_to) and row.stage_to < prior_lower:
                bad_fields.add(field)
            if pd.notna(row.stage_from):
                prior_lower = max(prior_lower, row.stage_from)
    if bad_fields:
        bad_dates = normalized.loc[normalized.field_id.isin(bad_fields), ['field_id', 'date']]
        rejected.append(valid.merge(bad_dates, on=['field_id', 'date'], validate='many_to_one').assign(reason='temporally_inconsistent_field_sequence'))
        normalized = normalized.loc[~normalized.field_id.isin(bad_fields)]
    return (normalized.reset_index(drop=True), pd.concat(rejected, ignore_index=True))

def constraints(normalized, registry):
    result = []
    for meta in registry.itertuples(index=False):
        group = normalized.loc[normalized.field_id.eq(meta.field_id)]
        sowing, end = (pd.Timestamp(meta.sowing_date), pd.Timestamp(meta.horizon_end))
        for event in EVENTS:
            lower = group.loc[group.stage_to.lt(event), 'date'].max()
            upper = group.loc[group.stage_from.ge(event), 'date'].min()
            lower, upper = (pd.Timestamp(lower), pd.Timestamp(upper))
            available = pd.notna(lower) or pd.notna(upper)
            bracket = pd.notna(lower) and pd.notna(upper)
            inconsistent = bracket and lower >= upper
            kind = 'interval' if bracket else 'left' if pd.notna(upper) else 'right' if pd.notna(lower) else 'none'
            result.append(dict(field_id=meta.field_id, source=meta.source, partition=meta.partition, site_id=meta.site_id, season_year=meta.season_year, event=event, sowing_date=sowing, horizon_end=end, horizon_index=(end - sowing).days, lower_exclusive=lower, upper_inclusive=upper, lower_index=(lower - sowing).days if pd.notna(lower) else np.nan, upper_index=(upper - sowing).days if pd.notna(upper) else np.nan, censoring=kind, eligible_constraint=bool(available and (not inconsistent)), genuinely_bracketed=bool(bracket and (not inconsistent)), bracket_width_days=(upper - lower).days if bracket else np.nan, reason='temporally_inconsistent_event_bounds' if inconsistent else '' if available else 'no_event_bound', n_stage_dates=len(group)))
    return pd.DataFrame(result)
