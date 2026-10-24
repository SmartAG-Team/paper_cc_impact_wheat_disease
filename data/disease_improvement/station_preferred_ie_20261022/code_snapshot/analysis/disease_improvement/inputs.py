"""Retained observations and explicit unresolved measurement metadata."""
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

INPUT = 'data/final_study/evidence/disease_joint_forecast/inputs'
ARCHIVE = 'data/final_study/evidence/disease_coupled_ml/results/assessment_predictions.csv'
WEATHER = 'data/final_study/inputs/paper_study/field_weather_completed/daily_weather.parquet'
RECOVERED_METADATA = 'data/disease_improvement/metadata_recovered_20261022/field_treatment_ledger.csv'


def apply_field_metadata(fields,ledger):
    """Fill independently verified source metadata without overwriting conflicts."""
    if ledger.field_id.duplicated().any() or fields.field_id.duplicated().any():
        raise ValueError('Unique field metadata required')
    if (not ledger.treatment.eq('Untreated').all() or not ledger.num_sprays.eq(0).all()
        or ledger[['source_file','source_sha256']].isna().any().any()):
        raise ValueError('Verified untreated zero-spray source metadata required')
    out=fields.copy();recovered=ledger.set_index('field_id')
    for name in ['treatment','num_sprays']:
        supplied=out.field_id.map(recovered[name])
        if name not in out:out[name]=pd.NA
        known=out[name].notna()&supplied.notna()
        conflict=pd.Series(False,index=out.index)
        conflict.loc[known]=out.loc[known,name].ne(supplied.loc[known])
        if conflict.any():raise ValueError('Recovered field metadata conflict: '+name)
        out[name]=out[name].where(out[name].notna(),supplied)
    available=out.field_id.isin(recovered.index)
    out['metadata_recovery_status']=np.where(available,'source_verified_untreated_zero_sprays','not_recovered')
    out['metadata_source_file']=out.field_id.map(recovered.source_file)
    out['metadata_source_sha256']=out.field_id.map(recovered.source_sha256)
    return out


def quality_screen(observations, fields):
    """Quarantine independently flagged records, never prediction residuals.

    An assessment after a grain endpoint is quarantined pending source
    adjudication, rather than assigned an invented replacement date.
    """
    frame = observations.copy()
    dates = pd.to_datetime(frame.date, errors='coerce', format='mixed')
    field_year = frame.field_id.map(fields.set_index('field_id').season_year)
    rules = {
        'invalid_percentage': ~pd.to_numeric(frame.observed_percent, errors='coerce').between(0, 100),
        'invalid_assessment_date': dates.isna(),
        'assessment_year_disagrees_with_field_season': dates.dt.year.ne(field_year) & dates.notna(),
        'unsupported_recorded_leaf_code': ~frame.leaf_index.isin(range(7)),
        'assessment_after_grain_endpoint_pending_adjudication': frame.after_grain_record.eq(True),
    }
    reasons = [[] for _ in range(len(frame))]
    for name, mask in rules.items():
        for position in np.flatnonzero(mask):
            reasons[position].append(name)
    frame['reason'] = [';'.join(row) for row in reasons]
    excluded = frame.loc[frame.reason.ne('')].copy()
    excluded['exclusion_basis'] = 'source_metadata_or_physical_range; independent_of_model_error'
    kept = frame.loc[frame.reason.eq('')].drop(columns='reason').reset_index(drop=True)
    return kept, excluded


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_observations(root):
    root = Path(root)
    observations = pd.read_csv(root / INPUT / 'observations.csv')
    fields = pd.read_csv(root / INPUT / 'fields.csv')
    if observations.global_target_index.duplicated().any() or fields.field_id.duplicated().any():
        raise ValueError('Unique observations and fields required')
    if observations.duplicated(['field_id', 'endpoint_series', 'date']).any():
        raise ValueError('Duplicate source-series dates')
    if not observations.observed_percent.between(0, 100).all():
        raise ValueError('Finite severity percentages required')
    observations['season_year'] = pd.to_datetime(observations.date).dt.year
    observations = observations.merge(fields[['field_id', 'latitude', 'longitude']],
                                      on='field_id', validate='many_to_one', how='left')
    if observations[['latitude', 'longitude']].isna().any().any():
        raise ValueError('Complete field coordinates required')
    definitions = pd.read_csv(root / 'data/final_study/evidence/disease_observation_revision/results/measurement_definitions.csv')
    observations = observations.merge(definitions[['source', 'measurement', 'leaf_definition']],
                                      on='source', validate='many_to_one', how='left')
    observations['measurement_qualified'] = False
    observations['anatomical_leaf_verified'] = False
    observations['recorded_leaf_category'] = observations.source + ':' + observations.leaf_index.astype(str)
    lineage_path = root / 'data/final_study/evidence/disease_endpoint_audit/results/assessment_lineage.csv'
    lineage = pd.read_csv(lineage_path)
    observations = observations.merge(lineage[['global_target_index', 'source_rows', 'source_records',
        'latest_grain_date', 'assessment_after_grain_record']], on='global_target_index', how='left', validate='one_to_one')
    observations['after_grain_record'] = observations.assessment_after_grain_record.eq(True)
    observations['adjudication'] = np.where(observations.after_grain_record,
        'source_date_and_season_confirmation_required', 'no_retained_chronology_flag')
    observations['replicate_identity_status'] = 'not_recovered'
    observations['source_trial_id'] = fields.set_index('field_id').source_unit.reindex(observations.field_id).to_numpy()
    for name in ['cultivar', 'resistance_class', 'observed_sowing_date', 'treatment', 'plot_id', 'replicate_id']:
        if name not in fields:
            fields[name] = pd.NA
    recovered_path=root/RECOVERED_METADATA
    if recovered_path.exists():
        fields=apply_field_metadata(fields,pd.read_csv(recovered_path))
    paths = [root / INPUT / name for name in ['observations.csv', 'fields.csv', 'features.npz']]
    paths += [root / ARCHIVE, root / WEATHER, lineage_path,
              root / 'data/final_study/evidence/disease_observation_revision/results/measurement_definitions.csv',
              root / 'data/final_models/verification/disease_queries.csv',
              root / 'data/final_models/verification/disease_queries.npz']
    if recovered_path.exists():
        paths.append(recovered_path)
    provenance = dict(independent_validation_available=False,
        target_policy='Preserve original means, dates and all benchmark identities; no automatic deduplication',
        source_sha256={str(p.relative_to(root)): sha(p) for p in paths},
        unresolved=['severity_denominator', 'Corteva_anatomical_leaf', 'plot_and_replicate_identity',
                    'observed_sowing', 'cultivar_resistance', 'external_unexamined_cohort'])
    return observations, fields, provenance
