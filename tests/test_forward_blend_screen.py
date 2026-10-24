import pandas as pd
import pytest


def test_calibration_uses_only_earlier_outer_training_forecasts():
    from models.monitoring.forward_blend_screen import calibration_rows
    forecasts = pd.DataFrame(dict(global_target_index=[10, 11, 12, 13],
        season_year=[2015, 2016, 2015, 2017], date=['2015-06-01','2016-06-01','2015-06-01','2017-06-01']))
    selected = calibration_rows(forecasts, [10, 11, 13], 2016)
    assert selected.global_target_index.tolist() == [10]
    forecasts.loc[0,'date'] = '2017-06-01'
    with pytest.raises(ValueError,match='date'):
        calibration_rows(forecasts,[10,11,13],2016)
