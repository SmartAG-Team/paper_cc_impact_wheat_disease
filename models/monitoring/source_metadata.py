"""Recover source metadata without accepting severity values or inventing plot IDs.

Assessment-date stages are oracle diagnostics. A prospective consumer must select
calendar rows with usable=True and date strictly earlier than its query date.
"""
from pathlib import Path
import hashlib
import json
import argparse

import numpy as np
import pandas as pd

KEYS = ['global_target_index', 'field_id', 'endpoint_series', 'date']


def normalize_source(frame, source, source_file, source_sha256):
    """Normalize raw wheat-table metadata; source_row_id is physical TSV line."""
    d = frame.copy()
    basf = source == 'BASF'
    unit = d['TrialId'].astype(str) if basf else d['Field'].astype(str)
    year = pd.to_datetime(d.Date).dt.year.astype(str) if basf else unit.str[:4]
    prefix = 'basf-wheat-diseases' if basf else 'corteva-2014-2018-external'
    organ = d['PlantPart' if basf else 'SamplingUnit'].fillna('unspecified')
    out = pd.DataFrame(dict(field_id=prefix+'|'+unit+'|'+year,
        endpoint_series=unit+'|'+organ, date=pd.to_datetime(d.Date).dt.strftime('%Y-%m-%d'),
        source=source, source_trial_id=unit, source_organ=organ,
        country=d.Country, latitude=d.Latitude, longitude=d.Longitude,
        treatment=d.Treatment, num_sprays=pd.to_numeric(d.NumSprays, errors='coerce'),
        treatment_code=d.get('TreatmentCode', pd.Series(pd.NA, index=d.index)),
        applications=d.get('Applications', pd.Series(pd.NA, index=d.index)),
        pest_code=d['Organism' if basf else 'PestCode'],
        evaluation_type=d['Parameter' if basf else 'EvaluationType'],
        stage_from=pd.to_numeric(d.GsFrom, errors='coerce'),
        stage_to=pd.to_numeric(d.GsTo, errors='coerce'),
        source_row_id=np.arange(len(d))+2, source_file=str(source_file),
        source_sha256=source_sha256))
    lo, hi = out.stage_from, out.stage_to
    complete = lo.notna() & hi.notna()
    bounded = lo.between(0, 99) & hi.between(0, 99) & lo.mod(1).eq(0) & hi.mod(1).eq(0)
    out['stage_status'] = np.select([~complete, ~bounded, lo.gt(hi)],
        ['incomplete', 'out_of_range_or_noninteger', 'reversed_range'], default='valid')
    return out


def _joined(values):
    return ';'.join(map(str, sorted(values.dropna().unique())))


def stage_calendar(raw):
    """Keep all dates; disagreeing ranges remain explicit and unusable."""
    rows = []
    untreated = raw[raw.treatment.eq('Untreated') & raw.num_sprays.eq(0)]
    for (field, date), group in untreated.groupby(['field_id', 'date'], sort=True):
        valid = group[group.stage_status.eq('valid')]
        ranges = valid[['stage_from', 'stage_to']].drop_duplicates()
        usable = len(ranges) == 1
        lo, hi = ranges.iloc[0] if usable else (np.nan, np.nan)
        rows.append(dict(field_id=field, date=date, stage_from=lo, stage_to=hi,
            stage_midpoint=(lo+hi)/2, usable=usable,
            stage_status='valid' if usable else ('conflicting_ranges' if len(ranges) else 'unavailable'),
            source=_joined(group.source), source_rows=_joined(valid.source_row_id),
            source_file=_joined(group.source_file), source_sha256=_joined(group.source_sha256),
            pest_codes=_joined(valid.pest_code), evaluation_types=_joined(valid.evaluation_type),
            stage_ranges=';'.join(f'{a:g}:{b:g}' for a,b in ranges.itertuples(index=False, name=None)),
            n_source_rows=len(valid), n_invalid_stage_rows=len(group)-len(valid)))
    return pd.DataFrame(rows)


def recover_metadata(target_keys, raw):
    """Return one row per unchanged target identity and a source field calendar."""
    if set(target_keys.columns) - set(KEYS):
        raise ValueError('Only alignment keys accepted; target values are forbidden')
    if target_keys.global_target_index.duplicated().any():
        raise ValueError('Duplicate global target index')
    raw = raw[raw.field_id.isin(target_keys.field_id)].copy()
    untreated = raw[raw.treatment.eq('Untreated') & raw.num_sprays.eq(0)]
    stb = untreated[untreated.pest_code.eq('SEPTTR') & untreated.evaluation_type.eq('INFECT')]
    groups = {key: value for key, value in stb.groupby(['field_id', 'endpoint_series', 'date'])}
    rows = []
    for target in target_keys.itertuples(index=False):
        key = (target.field_id, target.endpoint_series, target.date)
        if key not in groups:
            raise ValueError(f'No untreated zero-spray STB source match: {key}')
        group = groups[key]
        valid = group[group.stage_status.eq('valid')]
        ranges = valid[['stage_from', 'stage_to']].drop_duplicates()
        lo, hi = ranges.iloc[0] if len(ranges) == 1 else (np.nan, np.nan)
        row = {name: getattr(target, name) for name in KEYS}
        row.update(source=_joined(group.source), source_trial_id=_joined(group.source_trial_id),
            treatment='Untreated', num_sprays=0, treatment_codes=_joined(group.treatment_code),
            applications=_joined(group.applications), source_organ=_joined(group.source_organ),
            source_row_ids=_joined(group.source_row_id), n_source_rows=len(group),
            source_file=_joined(group.source_file), source_sha256=_joined(group.source_sha256),
            own_stage_from=lo, own_stage_to=hi,
            own_stage_status='valid' if len(ranges)==1 else ('conflicting_ranges' if len(ranges) else 'unavailable'),
            anatomical_leaf_status='explicit_flag_leaf' if group.source_organ.eq('LEAF, 1ST / FLAG LEAF').all()
                else 'ordinal_reference_unverified',
            plot_id=pd.NA, replicate_id=pd.NA, cultivar=pd.NA, resistance_class=pd.NA,
            observed_sowing_date=pd.NA, replicate_identity_status='not_in_raw_source',
            assessment_stage_use='oracle_diagnostic_only')
        rows.append(row)
    calendar = stage_calendar(raw)
    ledger = pd.DataFrame(rows).merge(calendar[['field_id','date','stage_from','stage_to',
        'stage_midpoint','usable','stage_status','source_rows']], on=['field_id','date'],
        how='left', validate='many_to_one')
    return ledger, calendar


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(root, output, basf_path):
    root, output = Path(root), Path(output)
    obs_path = root/'outputs/disease_improvement/station_repairs_20261022_v4/observation_ledger.csv'
    corteva_path = root/'data/final_study/inputs/stb_collection_20261009/sources/other_field_sources/corteva-2014-2018/corteva-wheat-diseases.txt'
    observations = pd.read_csv(obs_path)
    paths = {'BASF':Path(basf_path), 'Corteva':corteva_path}
    raw = pd.concat([normalize_source(pd.read_csv(path, sep='\t'), source, path, _sha(path))
                     for source,path in paths.items()], ignore_index=True)
    raw = raw[raw.field_id.isin(observations.field_id)].copy()
    # Confirm field matching independently of severity, dates and source line order.
    coordinates = observations[['field_id','latitude','longitude']].drop_duplicates()
    aligned = raw.merge(coordinates, on='field_id', suffixes=('_raw','_target'), validate='many_to_one')
    if not (np.isclose(aligned.latitude_raw,aligned.latitude_target) &
            np.isclose(aligned.longitude_raw,aligned.longitude_target)).all():
        raise ValueError('Source and retained field coordinates disagree')
    ledger, calendar = recover_metadata(observations[KEYS],raw)
    if not ledger.global_target_index.tolist() == observations.global_target_index.tolist():
        raise ValueError('Target identity/order changed')
    prior = []
    for target in observations.itertuples(index=False):
        history = calendar[calendar.field_id.eq(target.field_id) & calendar.usable & calendar.date.lt(target.date)]
        last = history.iloc[-1] if len(history) else None
        prior.append(dict(global_target_index=target.global_target_index,source=target.source,
            field_id=target.field_id,date=target.date,strictly_prior_stage_available=last is not None,
            prior_stage_date=last.date if last is not None else None,
            prior_stage_midpoint=last.stage_midpoint if last is not None else np.nan,
            prior_stage_age_days=(pd.Timestamp(target.date)-pd.Timestamp(last.date)).days if last is not None else np.nan))
    prior = pd.DataFrame(prior)
    field_ledger = ledger.groupby(['field_id','source'],as_index=False).agg(
        treatment=('treatment','first'),num_sprays=('num_sprays','first'),
        n_qualified_targets=('global_target_index','size'),source_file=('source_file','first'),
        source_sha256=('source_sha256','first'))
    output.mkdir(parents=True,exist_ok=True)
    for name,frame in [('metadata_ledger',ledger),('stage_calendar',calendar),('source_metadata_rows',raw),
                       ('field_treatment_ledger',field_ledger),('strictly_prior_stage_coverage',prior)]:
        frame.to_csv(output/f'{name}.csv',index=False)
    report = dict(n_targets=len(ledger),n_fields=ledger.field_id.nunique(),
        n_treatment_verified=int(ledger.treatment.eq('Untreated').sum()),
        n_zero_sprays_verified=int(ledger.num_sprays.eq(0).sum()),
        source_row_numbering='physical TSV line number; header is line 1; repeated rows do not imply replicates',
        source_sha256={str(p):_sha(p) for p in [obs_path,*paths.values()]},
        stage_policy='Complete integer 0..99 ranges with from<=to only; differing same-field/date ranges are unusable; no endpoint completion, interpolation or extrapolation',
        prediction_policy='Assessment-date stage is oracle diagnostic only; prospective history requires strictly earlier usable calendar dates',
        unresolved=['cultivar','resistance_class','observed_sowing_date','plot_id','replicate_id','severity_denominator','Corteva anatomical leaf reference'],
        source_coverage={},notable_targets=json.loads(prior[prior.global_target_index.isin([479,1750,413,159,1231])].to_json(orient='records')))
    for source,group in ledger.groupby('source'):
        cal = calendar[calendar.source.eq(source)]
        report['source_coverage'][source] = dict(n_targets=len(group),
            own_assessment_stage_valid=int(group.own_stage_status.eq('valid').sum()),
            same_date_calendar_stage_usable=int(group.usable.sum()),
            strictly_prior_stage_available=int(prior[prior.source.eq(source)].strictly_prior_stage_available.sum()),
            calendar_dates=len(cal),usable_calendar_dates=int(cal.usable.sum()),
            conflicting_calendar_dates=int(cal.stage_status.eq('conflicting_ranges').sum()))
    (output/'coverage.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',default='.')
    parser.add_argument('--output',default='outputs/raw_metadata_recovery_20261022')
    parser.add_argument('--basf-path',default='/Users/gangzhao/Documents/workspace/data/zenodo-6521175/basf-wheat-diseases.txt')
    args=parser.parse_args()
    print(json.dumps(run(args.root,args.output,args.basf_path),indent=2))
