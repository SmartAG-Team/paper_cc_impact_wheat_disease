import numpy as np
import pandas as pd
import pytest

from model.seasonal_septoria.field_data import FieldData
from model.seasonal_septoria.calibrate import fit


def field_data():
    days = 60
    t = np.full((3,days),18.)
    h = np.full_like(t,90.)
    r = np.tile(np.array([1.,4.,0.])[:,None],(1,days))
    active = np.ones((3,days,1),bool)
    targets = pd.DataFrame(dict(field_index=[0,1,2],day_index=[60]*3,leaf_index=[0]*3,
        observation_operator=['damage_proxy']*3,value=[30.,40.,95.],weight=[1.]*3))
    return FieldData(t,h,r,active,np.zeros(active.shape),pd.DataFrame(),targets,pd.DataFrame())


def test_calibration_parameters_do_not_depend_on_withheld_disease_scores():
    data = field_data()
    first = fit(data,[0,1],beta_free=False)
    data.targets.loc[2,'value'] = 0.
    second = fit(data,[0,1],beta_free=False)
    assert first['parameters'] == second['parameters']
    assert first['target_indices'] == [0,1]
    assert first['parameters']['beta'] == 0.
    assert first['training_rmse'] < 12.


def test_calibration_cannot_split_assessments_within_a_field_season():
    data = field_data()
    data.targets.loc[1,'field_index'] = 0
    with pytest.raises(ValueError,match='field-season'):
        fit(data,[0],beta_free=False)
