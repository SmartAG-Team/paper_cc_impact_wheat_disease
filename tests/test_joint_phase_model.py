import numpy as np
import pandas as pd


def test_joint_phase_model_learns_growth_and_conditions_on_earlier_days_only():
    from scipy.special import expit
    from models.monitoring.joint_phase_model import fit_phase, predict_phase
    clock = np.tile([1.6, 1.8, 2.], 4)
    train = pd.DataFrame(dict(field_id=np.repeat(np.arange(4).astype(str), 3), source='BASF',
        leaf_index=0, date=np.tile(['2014-05-01', '2014-05-11', '2014-05-21'], 4),
        observed_percent=100*expit(8*(clock-2)), global_target_index=np.arange(12)))
    fit = fit_phase(np.zeros((12, 2)), clock, train, np.ones(12), epochs=150)
    query = pd.DataFrame(dict(field_id='new', source='BASF', leaf_index=[0, 0, 1, 0],
        date=['2015-05-21', '2015-05-31', '2015-05-31', '2015-06-10'],
        observed_percent=[50., 90., 100., 0.]))
    x = np.zeros((4, 2)); clock = np.array([2., 2.2, 2.2, 2.4]); baseline = np.full(4, 40.)
    value = predict_phase(fit, x, clock, query, baseline)
    assert value['anchor'][0] == value['anchor'][2] == 40
    assert value['anchor'][1] > 65
    changed = query.copy(); changed.loc[1:, 'observed_percent'] = 0
    replay = predict_phase(fit, x, clock, changed, baseline)
    np.testing.assert_array_equal(value['posterior'][:3], replay['posterior'][:3])
    np.testing.assert_array_equal(value['free'], replay['free'])
    assert value['anchor'][3] != replay['anchor'][3]
