from dataclasses import replace
import numpy as np
import pandas as pd
import pytest


def test_resistance_is_explicit_and_neutral_default_replays_physics():
    from models.monitoring.variety_resistance import resistance_parameters
    from models.seasonal_septoria.overwinter import OverwinterParameters
    p = OverwinterParameters()
    assert resistance_parameters(p) == p
    resistant = resistance_parameters(p, .4)
    assert resistant.primary_scale == .4*p.primary_scale
    assert resistant.secondary_scale == .4*p.secondary_scale
    assert resistant.latent_reference_days == p.latent_reference_days
    for invalid in [-1, np.nan, np.inf, 'unknown']:
        with pytest.raises(ValueError): resistance_parameters(p, invalid)


def test_exact_variety_link_requires_provenance_and_missing_stays_unknown():
    from models.monitoring.variety_resistance import link_varieties
    fields = pd.DataFrame(dict(field_id=['a', 'b']))
    metadata = pd.DataFrame(dict(field_id=['a'], cultivar=['  Example A '], source_file=['trial.xlsx'], source_key=['row 4']))
    result = link_varieties(fields, metadata)
    assert result.cultivar.tolist()[0] == 'Example A'
    assert pd.isna(result.cultivar.iloc[1])
    assert result.resistance_multiplier.tolist() == [1., 1.]
    assert not result.resistance_verified.any()
    with pytest.raises(ValueError, match='provenance'):
        link_varieties(fields, metadata.assign(source_key=''))
    with pytest.raises(ValueError, match='unique'):
        link_varieties(fields, pd.concat([metadata, metadata.assign(cultivar='Other')]))
