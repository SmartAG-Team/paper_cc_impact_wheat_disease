import numpy as np
import pandas as pd
import pytest


def test_first_visit_training_and_prediction_ignore_disease_history_metadata():
    from models.monitoring.history_blind_model import fit_blind,predict_blind
    from models.monitoring.history import SAME_NAMES,OTHER_NAMES
    n=80
    x=np.zeros((n,53));x[:,0]=np.linspace(0,1,n)
    training=pd.DataFrame(dict(global_target_index=np.arange(n),source='Corteva',leaf_index=1,
        coordinate_year=np.repeat(np.arange(8),10),field_id=np.repeat(np.arange(8),10),
        observed_percent=10+80*x[:,0]))
    with_history=training.assign(**{name:1. for name in SAME_NAMES+OTHER_NAMES})
    without_history=training.assign(**{name:0. for name in SAME_NAMES+OTHER_NAMES})
    a=fit_blind(x,with_history,members=2)
    b=fit_blind(x,without_history,members=2)
    query=training.drop(columns='observed_percent').iloc[[0,-1]]
    first=predict_blind(a,x[[0,-1]],query)
    np.testing.assert_array_equal(first,predict_blind(b,x[[0,-1]],query))
    assert first[1]>first[0]+20
    changed=query.assign(**{name:1. for name in SAME_NAMES+OTHER_NAMES})
    np.testing.assert_array_equal(first,predict_blind(a,x[[0,-1]],changed))
    assert all(member['models']['history'] is None for member in a['members'])
    with pytest.raises(ValueError,match='labels'):
        predict_blind(a,x[[0,-1]],query.assign(observed_percent=99.))
