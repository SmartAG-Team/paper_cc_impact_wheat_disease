import pandas as pd

from paper.analyses.food_security.evidence import assess_claims


def evidence(validated=False):
    return pd.DataFrame([
        dict(quantity=q, observed_quantity_supported=q in ['disease', 'grain_yield'],
             claim_ready=q in ['disease', 'grain_yield'],
             end_to_end_physiological_validation=validated)
        for q in ['disease', 'HAD', 'grain_yield', 'grain_loss', 'tonnes', 'adaptation']
    ])


def test_harvest_targets_do_not_establish_cause_specific_grain_loss():
    claims = assess_claims(evidence()).set_index('claim')
    assert claims.loc['baseline_production_exposure', 'supported']
    assert not claims.loc['functional_canopy_loss', 'supported']
    assert not claims.loc['grain_loss', 'supported']
    assert not claims.loc['tonnes_lost', 'supported']
    assert not claims.loc['adaptation_benefit', 'supported']


def test_one_validation_flag_cannot_fill_missing_measurements():
    claims = assess_claims(evidence(True)).set_index('claim')
    assert not claims.loc['grain_loss', 'supported']
    assert not claims.loc['tonnes_lost', 'supported']
