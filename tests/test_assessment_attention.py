import numpy as np
import pandas as pd


def test_assessment_memory_excludes_same_day_and_other_fields():
    from models.monitoring.assessment_attention import make_memory
    frame = pd.DataFrame(dict(field_id=['a', 'a', 'a', 'a', 'b'], source='BASF',
        leaf_index=[0, 0, 1, 0, 0], date=['2015-05-01', '2015-05-02', '2015-05-02',
            '2015-05-03', '2015-05-03'], observed_percent=[20., 80., 70., 0., 100.]))
    baseline = np.full(5, 30.); clock = np.arange(5) / 10
    first = make_memory(frame, baseline, clock)
    altered = frame.copy(); altered.loc[1:3, 'observed_percent'] = [0., 0., 100.]
    second = make_memory(altered, baseline, clock)
    np.testing.assert_array_equal(first['tokens'][:3], second['tokens'][:3])
    assert first['mask'].sum(axis=1).tolist() == [0, 1, 1, 3, 0]
    assert not np.array_equal(first['tokens'][3], second['tokens'][3])


def test_attention_learns_context_response_and_preserves_first_visits():
    from models.monitoring.assessment_attention import fit_attention, predict_attention
    frame = pd.DataFrame(dict(field_id=np.repeat(np.arange(24).astype(str), 2), source='BASF',
        leaf_index=0, date=np.tile(['2014-05-01', '2014-05-11'], 24),
        observed_percent=np.repeat(np.tile([10., 90.], 12), 2), global_target_index=np.arange(48)))
    x = np.zeros((48, 3)); base = np.full(48, 50.); clock = np.tile([1., 1.1], 24)
    fit = fit_attention(x, frame, base, clock, np.ones(48), config=dict(epochs=120, members=1))
    value = predict_attention(fit, x, frame, base, clock)
    np.testing.assert_array_equal(value[::2], base[::2])
    assert np.sqrt(np.mean((value[1::2]-frame.observed_percent.to_numpy()[1::2])**2)) < 10
