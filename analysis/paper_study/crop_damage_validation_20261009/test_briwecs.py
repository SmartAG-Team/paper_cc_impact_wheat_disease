"""Protection-response comparisons preserve water regimes and held-out sites."""
import numpy as np
from .briwecs import prepare
from .run import grouped_predictions


def test_management_matching_keeps_irrigation_separate():
    contrasts, quality=prepare()
    assert not contrasts.contrast_id.duplicated().any()
    gge=contrasts[contrasts.Location.eq('GGE')&contrasts.Year.isin([2015,2018,2019])]
    assert gge.water_regime.eq('IR').all()
    assert not contrasts.Location.eq('DKI').any()
    assert contrasts.nitrogen.isin(['HN','LN']).all()


def test_missing_protected_severity_remains_missing():
    contrasts, quality=prepare()
    missing=contrasts.severity_protected.isna()
    assert missing.any()
    assert contrasts.loc[missing,'severity_reduction'].isna().all()
    assert contrasts.response_fraction.lt(0).any()


def test_numeric_disease_precision_is_preserved():
    contrasts, quality=prepare()
    assert ((contrasts.severity_unprotected*100)%1 != 0).any()
    assert quality['non_numeric_severity_excluded']==1
    assert quality['reference_yield_basis']=='dry mass'


def test_held_out_site_does_not_leak_training_yields():
    contrasts,_=prepare()
    a=grouped_predictions(contrasts,'Location','training_mean',[],'response_fraction')
    changed=contrasts.copy();changed.loc[changed.Location.eq('HAN'),'response_fraction']+=1
    b=grouped_predictions(changed,'Location','training_mean',[],'response_fraction')
    np.testing.assert_allclose(a.loc[a.Location.eq('HAN'),'prediction'],b.loc[b.Location.eq('HAN'),'prediction'])


def test_drought_suffix_is_not_pooled_into_plain_reference():
    import pandas as pd
    from .briwecs import SOURCE
    contrasts,_=prepare()
    raw=pd.read_csv(SOURCE/'data/locations/GGE_2016.csv',sep=';')
    for row in contrasts[contrasts.Location.eq('GGE')&contrasts.Year.eq(2016)].itertuples():
        values=raw[raw.BRISONr.eq(row.BRISONr)&raw.Treatment.eq(row.nitrogen+'_WF')]
        np.testing.assert_allclose(row.Seedyield_protected,values.Seedyield.mean())
