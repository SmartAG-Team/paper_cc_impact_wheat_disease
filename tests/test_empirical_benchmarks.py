from types import SimpleNamespace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from calibration.seasonal_septoria.empirical import assessment_features, WeightedRidge


def test_features_ignore_observed_disease_and_crop_stage_and_future_weather():
    params = json.loads((Path(__file__).parents[1]/'process_model/parameters/calibration.json').read_text())
    targets = pd.DataFrame(dict(field_index=[0],day_index=[3],leaf_index=[0],
        date=[pd.Timestamp('2018-10-03')],value=[0.],stage_from=[10],stage_to=[10]))
    data = SimpleNamespace(metadata=pd.DataFrame(dict(field_index=[0],forcing_days=[8],
        sowing_date=['2018-10-01'],latitude=[50.])),targets=targets,
        temperature=np.full((1,8),12.),maximum_temperature=np.full((1,8),16.),
        humidity=np.full((1,8),85.),rain=np.full((1,8),2.),
        host_active=np.ones((1,8,8),bool))
    original = assessment_features(data,params)
    data.targets['value'],data.targets['stage_from'],data.targets['stage_to'] = 99.,90,90
    for name in ['temperature','maximum_temperature','humidity','rain']:
        getattr(data,name)[:,3:] = 0.
    pd.testing.assert_frame_equal(original,assessment_features(data,params))


def test_ridge_standardization_uses_only_training_rows():
    x = np.array([[0.,1.],[2.,1.],[4.,1.]])
    model = WeightedRidge(1.).fit(x,[0.,50.,100.],[1.,2.,1.])
    np.testing.assert_allclose(model.mean,[2.,1.])
    before = model.predict(x)
    model.predict([[1e9,-1e9]])
    np.testing.assert_array_equal(model.predict(x),before)
    assert ((before>=0)&(before<=100)).all()
