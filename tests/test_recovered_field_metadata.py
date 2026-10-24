import pandas as pd
import pytest


def test_verified_treatment_fills_missing_without_erasing_other_metadata():
    from models.monitoring.inputs import apply_field_metadata
    fields=pd.DataFrame(dict(field_id=['a','b'],cultivar=['known',None],treatment=[None,'Untreated']))
    ledger=pd.DataFrame(dict(field_id=['a','b'],treatment=['Untreated']*2,num_sprays=[0,0],
        source_file=['raw.tsv']*2,source_sha256=['abc']*2))
    changed=apply_field_metadata(fields,ledger)
    pd.testing.assert_series_equal(changed.cultivar,fields.cultivar)
    assert changed.treatment.tolist()==['Untreated']*2
    assert changed.num_sprays.tolist()==[0,0]
    assert changed.metadata_recovery_status.eq('source_verified_untreated_zero_sprays').all()
    assert fields.treatment.isna().sum()==1


def test_metadata_conflicts_fail_instead_of_replacing_existing_field_values():
    from models.monitoring.inputs import apply_field_metadata
    fields=pd.DataFrame(dict(field_id=['a'],treatment=['Treated']))
    ledger=pd.DataFrame(dict(field_id=['a'],treatment=['Untreated'],num_sprays=[0],source_file=['raw.tsv'],source_sha256=['abc']))
    with pytest.raises(ValueError,match='conflict'):
        apply_field_metadata(fields,ledger)
