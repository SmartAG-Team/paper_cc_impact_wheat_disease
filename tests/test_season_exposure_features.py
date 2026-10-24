"""Check complete causal prefixes and hand-calculated cumulative exposure."""
import numpy as np
import pandas as pd
import pytest


def test_complete_prefix_has_hand_calculated_totals_and_excludes_assessment_day():
    from models.monitoring.season_exposure_features import build_season_exposure
    dates=pd.date_range('2016-11-01','2016-12-25')
    weather=pd.DataFrame(dict(location_id='a',date=dates,tmean_c=18.,tmax_c=18.,rh_mean_pct=100.,
        rain_mm=2.,rain_hours_gt_0_1mm=3.,rh_hours_ge_90pct=24.))
    fields=pd.DataFrame(dict(field_id=['f'],site_id=['a'],sowing_date=['2016-11-01']))
    queries=pd.DataFrame(dict(global_target_index=[1],field_id=['f'],date=['2016-12-20']))
    actual,names,provenance=build_season_exposure(weather,fields,queries)
    # Nov1..Dec19:49days. Excluding the latest42leaves Nov1..Nov7:7days.
    np.testing.assert_allclose(actual,[[98.,49.,147.,1176.,49.,7.]])
    assert provenance.iloc[0].weather_end=='2016-12-19'
    changed=weather.copy()
    changed.loc[changed.date.ge('2016-12-20'),['rain_mm','rh_mean_pct']]=[999.,0.]
    np.testing.assert_array_equal(build_season_exposure(changed,fields,queries)[0],actual)
    with pytest.raises(ValueError,match='labels'):
        build_season_exposure(weather,fields,queries.assign(observed_percent=90.))
    with pytest.raises(ValueError,match='Incomplete'):
        build_season_exposure(weather.loc[weather.date.ne('2016-11-06')],fields,queries)
