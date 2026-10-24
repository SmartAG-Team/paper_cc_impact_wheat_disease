import numpy as np
import pandas as pd


class _Constant:
    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


def _fit():
    members = [dict(clusters=[key], shared=_Constant(value), stage=[_Constant(value)]*4)
        for key, value in zip(['A', 'B', 'C'], [10., 30., 70.])]
    return dict(baseline=[_Constant(v) for v in [10., 30., 70.]],
        growth=dict(members=members, phase_weight_floor=.01))


def test_oob_uses_only_members_excluding_entire_coordinate_year():
    from models.monitoring.autonomous_monitoring import coordinate_oob
    frame = pd.DataFrame(dict(coordinate_year=['A', 'B', 'C', 'D'], observed_percent=[0., 30., 70., 99.]))
    x = np.zeros((4, 2)); phase = np.tile([0., 0., 1., 0.], (4, 1))
    result = coordinate_oob(_fit(), x, x, phase, frame)
    np.testing.assert_array_equal(result['count'], [2, 2, 2, 3])
    for values in result['predictions'].values():
        np.testing.assert_allclose(values, [50., 40., 20., 110/3])
    altered = frame.assign(observed_percent=[100., 0., 0., 0.], bbch=99.)
    again = coordinate_oob(_fit(), x, x, phase, altered)
    for name in result['predictions']:
        np.testing.assert_array_equal(result['predictions'][name], again['predictions'][name])
    fitted = _fit()
    for member in fitted['growth']['members']:
        member['clusters'] = ['A', 'B', 'C', 'D']
    missing = coordinate_oob(fitted, x, x, phase, frame)
    assert (missing['count'] == 0).all()
    assert all(np.isnan(v).all() for v in missing['predictions'].values())


def test_optional_monitoring_keeps_same_day_future_other_field_withheld():
    from models.monitoring.autonomous_monitoring import monitoring_forecast
    frame = pd.DataFrame(dict(field_id=['a', 'a', 'a', 'a', 'b'], leaf_index=[0, 0, 1, 0, 0],
        date=['2018-06-01', '2018-06-10', '2018-06-10', '2018-06-20', '2018-06-01'],
        observed_percent=[20., 60., 80., 90., 100.]))
    baseline = np.array([10., 30., 40., 50., 60.]); parameters = np.log([1., 1., 30., 30.])
    for link in ['linear', 'logit']:
        before = monitoring_forecast(parameters, frame, baseline, link)
        assert before[0] == baseline[0] and before[-1] == baseline[-1]
        changed = frame.copy(); changed.loc[[1, 2, 3, 4], 'observed_percent'] = [0., 0., 0., 0.]
        after = monitoring_forecast(parameters, changed, baseline, link)
        np.testing.assert_array_equal(before[:3], after[:3])
        changed = frame.copy(); changed.loc[0, 'observed_percent'] = 80.
        after = monitoring_forecast(parameters, changed, baseline, link)
        assert np.all(after[1:3] > before[1:3])
