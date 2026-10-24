"""Guard the calibration boundary and whole-cluster exclusion."""
import numpy as np
import pandas as pd
import pytest


def test_oob_excludes_entire_coordinate_year_and_rejects_partial_members():
    from models.monitoring.oob_history_blend import membership_masks
    frame = pd.DataFrame(dict(global_target_index=[1, 2, 3, 4], coordinate_year=['a', 'a', 'b', 'b']))
    ensemble = dict(training_ids=[1, 2, 3, 4], members=[dict(training_ids=[1, 2]), dict(training_ids=[3, 4])])
    np.testing.assert_array_equal(membership_masks(ensemble, frame), [[False, False, True, True], [True, True, False, False]])
    ensemble['members'][0]['training_ids'] = [1]
    with pytest.raises(ValueError, match='whole'):
        membership_masks(ensemble, frame)


def test_group_weight_is_training_only_shrunk_and_ignores_insufficient_oob():
    from models.monitoring.oob_history_blend import fit_weights, apply_weights
    # Two equally weighted coordinate-years: unregularized optimum .8;
    # ten equivalent neutral groups shrink it to (2*.8 + 10*.5)/12 = .55.
    frame = pd.DataFrame(dict(global_target_index=[1, 2, 3], coordinate_year=['a', 'b', 'c'],
        field_id=['a', 'b', 'c'], leaf_index=[0, 0, 0], observed_percent=[80., 80., 0.]))
    h = pd.DataFrame(dict(prior_assessment_available=[0., 0., 1.], other_leaf_available=[0., 0., 0.]))
    fitted = fit_weights([0., 0., np.nan], [100., 100., np.nan], h, frame, [2, 4, 1])
    assert fitted['groups']['first_field']['history_weight'] == pytest.approx(.55)
    assert fitted['groups']['same_leaf']['history_weight'] == .5
    query = frame.drop(columns='observed_percent')
    prediction = apply_weights(fitted, [0., 0., 0.], [100., 100., 100.], h, query)
    np.testing.assert_allclose(prediction, [55., 55., 50.])
    with pytest.raises(ValueError, match='labels'):
        apply_weights(fitted, [0.]*3, [100.]*3, h, frame)


def test_same_leaf_takes_precedence_and_all_blends_remain_convex():
    from models.monitoring.oob_history_blend import apply_weights
    fitted = dict(groups={key:dict(history_weight=value) for key,value in
        [('first_field', .2), ('same_leaf', .8), ('other_leaf_only', 1.)]})
    h = pd.DataFrame(dict(prior_assessment_available=[0., 1., 0.], other_leaf_available=[0., 1., 1.]))
    queries = pd.DataFrame(dict(global_target_index=[10, 11, 12]))
    np.testing.assert_allclose(apply_weights(fitted, [10., 10., 10.], [90., 90., 90.], h, queries), [26., 74., 90.])
    h.loc[0, 'other_leaf_available'] = .2
    with pytest.raises(ValueError, match='binary'):
        apply_weights(fitted, [10.]*3, [90.]*3, h, queries)
