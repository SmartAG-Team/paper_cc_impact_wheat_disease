"""Explicit susceptibility scenarios and provenance-qualified variety metadata.

A name alone never supplies a resistance rating. The multiplier is an assumed
effective infection-rate parameter, not a conversion of an official rating scale.
"""
from dataclasses import replace

import numpy as np
import pandas as pd

from model.seasonal_septoria.overwinter import OverwinterParameters


def resistance_parameters(parameters, susceptibility_multiplier=1.):
    try:
        value = float(susceptibility_multiplier)
    except (TypeError, ValueError):
        raise ValueError('Finite nonnegative susceptibility multiplier required') from None
    if not isinstance(parameters, OverwinterParameters) or not np.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative susceptibility multiplier required')
    return replace(parameters, primary_scale=parameters.primary_scale*value,
        secondary_scale=parameters.secondary_scale*value)


def link_varieties(fields, metadata):
    required = ['field_id', 'cultivar', 'source_file', 'source_key']
    if not set(required).issubset(metadata) or 'field_id' not in fields:
        raise ValueError('Exact field identity and cultivar provenance required')
    if fields.field_id.isna().any() or metadata.field_id.isna().any() or fields.field_id.duplicated().any() or metadata.field_id.duplicated().any():
        raise ValueError('Field identities and metadata must be unique')
    data = metadata.copy()
    for column in required:
        if data[column].isna().any() or data[column].astype(str).str.strip().eq('').any():
            raise ValueError('Nonempty cultivar identity and provenance required')
    data['cultivar'] = data.cultivar.astype(str).str.strip()
    if 'resistance_multiplier' in data:
        if not {'resistance_source', 'resistance_verified'}.issubset(data):
            raise ValueError('Explicit resistance provenance and verification required')
        values = pd.to_numeric(data.resistance_multiplier, errors='coerce')
        if not np.isfinite(values).all() or (values < 0).any() or not data.resistance_verified.map(lambda x: x is True).all() or data.resistance_source.isna().any() or data.resistance_source.astype(str).str.strip().eq('').any():
            raise ValueError('Explicit resistance provenance and verification required')
    else:
        data['resistance_multiplier'] = 1.
        data['resistance_verified'] = False
    # Existing unresolved export columns are replaced only in this returned copy.
    result = fields.drop(columns=[c for c in data if c != 'field_id' and c in fields]).merge(data, on='field_id', how='left', validate='one_to_one')
    result['resistance_multiplier'] = result.resistance_multiplier.fillna(1.)
    result['resistance_verified'] = result.resistance_verified.eq(True)
    return result
