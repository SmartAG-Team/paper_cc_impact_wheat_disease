"""Conditional economics must preserve exposure scope and percentage units."""
import numpy as np
import pandas as pd
import pytest

from paper.analyses.food_security import exposure


def production_exposure():
    return pd.DataFrame([
        dict(domain_type='Europe', domain='Europe', scenario='ssp585', period='2071-2100',
             weight_basis='total', member_count=3, threshold_days=0., minimum_paired_seasons=1,
             baseline_production_tonnes=10000., unanimous_increase_tonnes=1000.,
             ensemble_increase_tonnes=8000., unanimous_increase_share_percent=10.),
        dict(domain_type='Europe', domain='Europe', scenario='ssp585', period='2071-2100',
             weight_basis='total', member_count=3, threshold_days=2., minimum_paired_seasons=1,
             baseline_production_tonnes=10000., unanimous_increase_tonnes=500.,
             ensemble_increase_tonnes=2000., unanimous_increase_share_percent=5.),
    ])


def test_one_percent_loss_uses_consensus_exposure_and_price_per_tonne():
    result = exposure.economic_sensitivity(production_exposure(), loss_fractions=[.01], prices_eur_t=[200.])
    row = result.iloc[0]
    assert row.exposed_baseline_production_tonnes == 1000.
    assert row.assumed_additional_yield_loss_percent == 1.
    assert row.conditional_grain_loss_tonnes == 10.
    assert row.conditional_gross_value_loss_eur == 2000.
    assert row.exposed_baseline_production_share_percent == 10.


def test_zero_assumed_loss_produces_zero_losses_at_every_price():
    result = exposure.economic_sensitivity(production_exposure(), loss_fractions=[0.], prices_eur_t=[150., 250.])
    assert len(result) == 2
    assert result.conditional_grain_loss_tonnes.eq(0).all()
    assert result.conditional_gross_value_loss_eur.eq(0).all()


@pytest.mark.parametrize('loss,price', [(-.01,200), (1.01,200), (np.nan,200), (.01,-1), (.01,np.inf)])
def test_invalid_assumptions_are_rejected(loss,price):
    with pytest.raises(ValueError):
        exposure.economic_sensitivity(production_exposure(), loss_fractions=[loss], prices_eur_t=[price])


def test_ambiguous_or_nonconserving_exposure_is_rejected():
    frame=production_exposure()
    with pytest.raises(ValueError,match='one'):
        exposure.economic_sensitivity(pd.concat([frame,frame.iloc[:1]]))
    frame.loc[0,'unanimous_increase_tonnes']=11000.
    with pytest.raises(ValueError,match='production'):
        exposure.economic_sensitivity(frame)


def test_saved_analysis_identifies_assumptions_and_source(tmp_path):
    source=tmp_path/'exposure.csv';production_exposure().to_csv(source,index=False)
    result=exposure.write_economic_sensitivity(source,tmp_path/'economics')
    assert result['status']=='passed'
    assert not result['grain_loss_predicted'] and not result['economic_loss_predicted']
    assert result['scenario_rows']==9
    assert (tmp_path/'economics/scenarios.csv').is_file()
    assert (tmp_path/'economics/assumptions.json').is_file()
    assert len(result['source_sha256'])==64
