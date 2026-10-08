import numpy as np
import pytest

from model.seasonal_septoria.climate_alignment import apply_monthly_alignment
from calibration.seasonal_septoria.climate_alignment import fit_monthly_alignment


def test_alignment_uses_historical_forcing_and_preserves_declared_changes():
    # Months have two distinct daily values, so the correction cannot reorder days.
    month = np.repeat(np.arange(1,13),2)
    reference = np.tile([10.,14.],12)[None,:]
    baseline = reference-2
    era = dict(tmean_c=reference,tmax_c=reference+8,rh_mean_pct=np.full_like(reference,70.),
               precipitation_mm=np.tile([0.,4.],12)[None,:])
    model = dict(tmean_c=baseline,tmax_c=baseline+8,rh_mean_pct=np.full_like(reference,60.),
                 precipitation_mm=np.tile([0.,2.],12)[None,:])
    coefficients = fit_monthly_alignment(era,model,month)
    corrected = apply_monthly_alignment(model,month,coefficients)
    np.testing.assert_allclose(corrected['tmean_c'],era['tmean_c'])
    np.testing.assert_allclose(corrected['precipitation_mm'],era['precipitation_mm'])
    np.testing.assert_allclose(corrected['rh_mean_pct'],era['rh_mean_pct'])
    future = {key:value.copy() for key,value in model.items()}
    future['tmean_c'] += 3
    future['tmax_c'] += 3
    future['precipitation_mm'] *= 1.5
    adjusted_future = apply_monthly_alignment(future,month,coefficients)
    np.testing.assert_allclose(adjusted_future['tmean_c']-corrected['tmean_c'],3)
    np.testing.assert_allclose(adjusted_future['precipitation_mm'],corrected['precipitation_mm']*1.5)
    np.testing.assert_equal(adjusted_future['precipitation_mm']==0,future['precipitation_mm']==0)
    assert np.all(adjusted_future['tmax_c']>=adjusted_future['tmean_c'])


def test_zero_model_rainfall_cannot_be_silently_bias_corrected_to_positive_mean():
    months = np.arange(1,13)
    base = dict(tmean_c=np.full((1,12),10.),tmax_c=np.full((1,12),15.),
                rh_mean_pct=np.full((1,12),70.),precipitation_mm=np.zeros((1,12)))
    reference = {key:value.copy() for key,value in base.items()}
    reference['precipitation_mm'][:] = 1.
    with pytest.raises(ValueError,match='zero'):
        fit_monthly_alignment(reference,base,months)


def test_missing_month_is_rejected_instead_of_default_correction():
    base = dict(tmean_c=np.ones((1,11)),tmax_c=np.full((1,11),2.),
                rh_mean_pct=np.full((1,11),70.),precipitation_mm=np.ones((1,11)))
    with pytest.raises(ValueError,match='12'):
        fit_monthly_alignment(base,base,np.arange(1,12))
