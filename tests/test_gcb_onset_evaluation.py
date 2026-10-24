import numpy as np
from paper.scripts.replay_onset import interval_excess,state_day_date


def test_last_negative_day_is_outside_the_first_symptom_interval():
    # Disease is still absent on day247, so earliest admissible onset is248.
    np.testing.assert_array_equal(interval_excess(247,255,[242,247,248,255,256]),[6,1,0,0,1])


def test_state_day_includes_the_sowing_date_as_day_one():
    assert str(state_day_date('2018-10-03',242).date())=='2019-06-01'
    assert str(state_day_date('2018-10-31',215).date())=='2019-06-02'


def test_onset_eligibility_uses_first_positive_and_prior_zero_in_date_order():
    import pandas as pd
    from paper.scripts.figure_validation_onset import onset_eligibility
    overview=pd.DataFrame(dict(field_id=['a','b','c','d'],source=['BASF']*4,site_id=['s1','s2','s3','s4'],season_year=[2019]*4))
    # Deliberately unsorted dates; a later zero cannot bracket an earlier positive.
    flag=pd.DataFrame(dict(field_id=['a','a','b','b','c','c'],
        date=['2019-06-10','2019-06-01','2019-06-01','2019-06-10','2019-06-01','2019-06-10'],
        observed_percent=[2,0,3,0,0,0]))
    status=onset_eligibility(overview,flag).set_index('field_id').onset_eligibility.to_dict()
    assert status=={'a':'two_sided_F1_onset','b':'F1_positive_at_first_assessment',
                    'c':'no_positive_F1_assessment','d':'no_qualified_F1_observation'}


def test_cohort_onset_bracket_ends_at_first_positive_before_later_zero():
    import pandas as pd
    from paper.scripts.replay_onset_cohorts import observation_interval
    history=pd.DataFrame(dict(date=['2019-06-07','2019-06-04','2019-06-01'],observed_percent=[0,2,0]))
    assert observation_interval(history,'2019-06-01')==('zero_to_positive',1,4)


def test_one_sided_histories_keep_the_unobserved_boundary_missing():
    import pandas as pd
    from paper.scripts.replay_onset_cohorts import observation_interval
    first_positive=pd.DataFrame(dict(date=['2019-06-01','2019-06-07'],observed_percent=[2,3]))
    status,lower,upper=observation_interval(first_positive,'2019-06-01')
    assert status=='positive_at_first' and np.isnan(lower) and upper==1
    zeros=first_positive.assign(observed_percent=0)
    status,lower,upper=observation_interval(zeros,'2019-06-01')
    assert status=='no_positive' and lower==7 and np.isnan(upper)
