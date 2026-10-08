import numpy as np


def test_accumulated_float32_lai_rounding_has_derived_bound():
    true=np.full((198,3),1/3,dtype=np.float64)
    stored=true.astype(np.float32)
    error=abs(stored.astype(float).sum()-true.sum())
    bound=(np.spacing(stored).astype(float)/2).sum()
    assert error>4e-6
    assert error<=bound


def test_product_bound_accounts_for_both_float32_inputs():
    true_area=np.array([1/3,.234567891234],dtype=np.float64)
    true_damage=np.array([.765432198765,.987654321],dtype=np.float64)
    area=true_area.astype(np.float32);damage=true_damage.astype(np.float32)
    ea=np.spacing(area).astype(float)/2;ed=np.spacing(damage).astype(float)/2
    error=abs(area.astype(float)*damage.astype(float)-true_area*true_damage)
    bound=area.astype(float)*ed+damage.astype(float)*ea+ea*ed
    assert np.all(error<=bound)
