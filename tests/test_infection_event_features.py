import numpy as np
import pandas as pd
import pytest


def fixture():
    dates = pd.date_range('2020-10-01', '2020-11-20')
    weather = pd.DataFrame(dict(location_id='site', date=dates, tmean_c=18., tmax_c=20.,
        rh_mean_pct=90., rain_mm=1., rain_hours_gt_0_1mm=3., rh_hours_ge_90pct=12.))
    fields = pd.DataFrame(dict(field_id=['field'], site_id=['site'], sowing_date=['2020-10-01']))
    queries = pd.DataFrame(dict(global_target_index=[1, 2], field_id=['field']*2, date=['2020-11-15']*2))
    provenance = pd.DataFrame(dict(global_target_index=[1, 2],
        latest_same_date=['2020-10-20', None], latest_any_date=['2020-11-01']*2))
    return weather, fields, queries, provenance


def test_event_interval_follows_the_selected_same_leaf_or_other_leaf_anchor():
    from models.monitoring.infection_event_features import build_event_features
    weather, fields, queries, provenance = fixture()
    values, names, ledger = build_event_features(weather, fields, queries, provenance)
    # Same-leaf history takes precedence; the first-leaf query uses earlier other-leaf context.
    np.testing.assert_array_equal(values[:, names.index('gap_thermal_units')], [25., 13.])
    np.testing.assert_array_equal(values[:, names.index('rain_hours_7d')], [21., 21.])
    assert ledger.weather_end.tolist() == ['2020-11-14']*2
    assert ledger.anchor_date.tolist() == ['2020-10-20', '2020-11-01']


def test_assessment_and_future_weather_and_query_labels_cannot_enter_event_inputs():
    from models.monitoring.infection_event_features import build_event_features
    weather, fields, queries, provenance = fixture()
    before, _, _ = build_event_features(weather, fields, queries, provenance)
    changed = weather.copy()
    changed.loc[changed.date.ge('2020-11-15'), ['rain_mm', 'rain_hours_gt_0_1mm', 'rh_hours_ge_90pct']] = [100., 24., 24.]
    after, _, _ = build_event_features(changed, fields, queries, provenance)
    np.testing.assert_array_equal(before, after)
    with pytest.raises(ValueError, match='labels'):
        build_event_features(weather, fields, queries.assign(observed_percent=100.), provenance)
    provenance.loc[1, 'latest_any_date'] = '2020-11-15'
    with pytest.raises(ValueError, match='earlier'):
        build_event_features(weather, fields, queries, provenance)
