import numpy as np


def test_observation_anchor_is_retained_when_modeled_visible_state_changes():
    from models.monitoring.leaf_transition_anchor import fit_ensemble, predict_ensemble
    from test_leaf_transition_model import training_fixture
    x, h, z, frame = training_fixture()
    fitted = fit_ensemble(x, h, z, frame, members=2)
    q = frame.drop(columns='observed_percent')
    before = predict_ensemble(fitted, x, h, z, q)
    changed = z.copy(); changed[:, 0] -= .1; changed[:, 4] += .1
    after = predict_ensemble(fitted, x, h, changed, q)
    # Constant learned change is added to the measured anchor, not to state damage.
    np.testing.assert_allclose(before, 40., atol=1e-7)
    np.testing.assert_array_equal(before, after)
