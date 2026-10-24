import numpy as np
import pandas as pd


def test_spatial_subgroup_context_comes_from_saved_history_without_changing_predictions():
    from models.monitoring.rf_sowing_finalize import assessment_populations
    frame=pd.DataFrame(dict(source=['BASF','Corteva'],observed_percent=[80.,20.],outer_fold=['spatial0','spatial0']))
    history=pd.DataFrame(dict(prior_assessment_available=[1.,0.],other_leaf_available=[0.,1.],latest_severity_fraction=[.2,0.]))
    masks=assessment_populations(frame,history)
    np.testing.assert_array_equal(masks['warm'],[True,False])
    np.testing.assert_array_equal(masks['cold'],[False,True])
    np.testing.assert_array_equal(masks['new_site'],[True,True])
    np.testing.assert_array_equal(masks['increase_above40'],[True,False])
    assert 'new_site' not in frame and 'prior_assessment_available' not in frame
