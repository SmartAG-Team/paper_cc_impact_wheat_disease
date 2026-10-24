import numpy as np
from sklearn.ensemble import RandomForestRegressor


def test_conditional_sowing_prior_retains_observed_spread_when_tree_means_agree():
    from models.monitoring.sowing_distribution import conditional_sowing_prior
    x = np.zeros((40, 2)); y = np.repeat([260., 300.], 20); weight = np.repeat([9., 1.], 20)
    forest = RandomForestRegressor(n_estimators=4, bootstrap=False, random_state=1)
    forest.fit(x, y, sample_weight=weight)
    np.testing.assert_allclose([tree.predict(x[:1])[0] for tree in forest.estimators_], [264.]*4)
    result = conditional_sowing_prior(forest, x, y, weight, x[:1], quantiles=[.05, .5, .95])
    np.testing.assert_array_equal(result['days'], [[260., 260., 300.]])
    np.testing.assert_allclose(result['mean'], [264.])
    assert result['standard_deviation'][0] > 10


def test_sowing_prior_preserves_scientific_mass_under_row_splitting():
    from models.monitoring.sowing_distribution import conditional_sowing_prior
    x = np.zeros((8, 2)); y = np.array([260., 270., 280., 290.]*2); w = np.arange(1., 9.)
    forest = RandomForestRegressor(n_estimators=3, bootstrap=False, random_state=1).fit(x, y, sample_weight=w)
    first = conditional_sowing_prior(forest, x, y, w, x[:1])
    second = conditional_sowing_prior(forest, np.repeat(x, 2, axis=0), np.repeat(y, 2),
        np.repeat(w/2, 2), x[:1])
    np.testing.assert_array_equal(first['days'], second['days'])
    np.testing.assert_allclose(first['mean'], second['mean'])
    np.testing.assert_allclose(first['standard_deviation'], second['standard_deviation'])


def test_sowing_quantile_endpoints_use_only_positive_conditional_support():
    from models.monitoring.sowing_distribution import conditional_sowing_prior
    x = np.repeat([[0., 0.], [1., 1.]], 5, axis=0)
    y = np.array([260., 260., 265., 270., 270., 300., 300., 305., 310., 310.])
    forest = RandomForestRegressor(n_estimators=1, bootstrap=False, max_depth=1, random_state=1).fit(x, y)
    result = conditional_sowing_prior(forest, x, y, np.ones(10), x[[0, 9]], quantiles=[0., 1.])
    np.testing.assert_array_equal(result['days'], [[260., 270.], [300., 310.]])
