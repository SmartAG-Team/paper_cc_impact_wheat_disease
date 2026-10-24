import numpy as np
import pandas as pd
import pytest

from paper.analyses.food_security.verify import values, ratio_summary


def test_independent_decomposition_closes_without_extension_helpers():
    frame = pd.DataFrame({'y00': [1.], 'y10': [6.], 'y01': [3.], 'y11': [10.]})
    terms = values(frame)
    np.testing.assert_array_equal(terms['weather'] + terms['host'], terms['net'])
    np.testing.assert_array_equal(terms['interaction'], [2.])


def test_independent_ensemble_uses_equal_model_ratios_with_unequal_support():
    frame = pd.DataFrame({'model': ['a', 'a', 'b'], 'area_mean_weight': [1., 1., 1.],
                          'keep': [True, False, True], 'y00': [0., 0., 0.],
                          'y10': [2., 0., 10.], 'y01': [0., 0., 0.], 'y11': [2., 0., 10.]})
    _, result = ratio_summary(frame, 'keep', ['a', 'b'])
    assert result['weather'] == pytest.approx(6.)
    assert result['net'] == pytest.approx(6.)
