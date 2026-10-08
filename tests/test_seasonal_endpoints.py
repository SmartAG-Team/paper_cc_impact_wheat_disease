import math

import pandas as pd
import pytest

from calibration.seasonal_septoria import endpoints


def test_onset_interval_uses_last_zero_as_exclusive_lower_bound():
    # A prediction on the observed zero day cannot count as correct onset.
    result = endpoints.onset_distance('2019-04-10', '2019-04-10', '2019-04-20')
    assert result == {'compatible': False, 'delta_days': -1, 'missing_prediction': False}
    assert endpoints.onset_distance('2019-04-11', '2019-04-10', '2019-04-20')['compatible']
    assert endpoints.onset_distance('2019-04-22', '2019-04-10', '2019-04-20')['delta_days'] == 2


def test_left_censored_onset_does_not_become_an_exact_date():
    assert endpoints.onset_distance('2019-03-01', None, '2019-03-15')['delta_days'] == 0
    assert endpoints.onset_distance('2019-03-17', None, '2019-03-15')['delta_days'] == 2


def test_no_predicted_event_is_compatible_only_with_right_censoring():
    assert endpoints.onset_distance(None, '2019-06-01', None) == {
        'compatible': True, 'delta_days': 0, 'missing_prediction': True}
    assert endpoints.onset_distance(None, None, '2019-06-01') == {
        'compatible': False, 'delta_days': None, 'missing_prediction': True}


def test_invalid_bounds_and_unobserved_onset_are_rejected():
    with pytest.raises(ValueError):
        endpoints.onset_distance('2019-04-10', '2019-04-10', '2019-04-10')
    with pytest.raises(ValueError):
        endpoints.onset_distance('2019-04-10', None, None)


def test_chronological_partition_keeps_all_assessments_in_a_field_season_together():
    source = pd.DataFrame({'field_id': ['a', 'a', 'a', 'b'],
                           'season_year': [2018, 2018, 2019, 2019]})
    result = endpoints.chronological_partition(source, [2019])
    assert result.tolist() == ['calibration', 'calibration', 'validation', 'validation']


def test_future_data_cannot_leak_into_a_chronological_calibration_partition():
    source = pd.DataFrame({'season_year': [2017, 2019, 2020]})
    result = endpoints.chronological_partition(source, [2019])
    assert result.tolist() == ['calibration', 'validation', 'excluded_future']


def test_one_class_occurrence_does_not_claim_specificity_or_auc():
    score = endpoints.occurrence_scores([True, True], [.8, .4])
    assert score['sensitivity'] == .5
    assert score['specificity'] is None
    assert score['balanced_accuracy'] is None
    assert score['brier_score'] == pytest.approx(.2)


def test_occurrence_scores_use_field_weights_and_actual_probabilities():
    score = endpoints.occurrence_scores([False, True, True], [.8, .8, .2], [2., 1., 1.])
    assert score['sensitivity'] == .5
    assert score['specificity'] == 0.
    assert score['brier_score'] == pytest.approx(.49)
    with pytest.raises(ValueError):
        endpoints.occurrence_scores([False, True], [20., 80.])
