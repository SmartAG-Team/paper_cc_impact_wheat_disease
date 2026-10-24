import numpy as np
import pandas as pd


def _training():
    n = 120
    frame = pd.DataFrame(dict(global_target_index=np.arange(n),
        coordinate_year=np.repeat(np.arange(n//2).astype(str), 2),
        field_id=np.repeat(np.arange(n//2).astype(str), 2), source='test', leaf_index=1,
        observed_percent=np.tile([95., 30.], n//2)))
    available = np.tile([False, True], n//2)
    x = np.zeros((n, 2))
    anchor = np.full(n, 20.)
    phase = np.tile([0., 0., 1., 0.], (n, 1))
    return frame, x, anchor, phase, available


def test_cold_labels_never_enter_warm_growth_fit():
    from models.monitoring.phenology_warm_oob import fit_warm_growth, predict_warm_growth
    frame, x, anchor, phase, available = _training()
    changed = frame.copy(); changed.loc[~available, 'observed_percent'] = 0.
    fits = [fit_warm_growth(x, f, anchor, np.ones(len(f)), phase, available,
        members=4, iterations=12) for f in [frame, changed]]
    for fitted in fits:
        assert set(fitted['growth']['training_ids']) == set(frame.loc[available, 'global_target_index'])
    forecasts = [predict_warm_growth(f, x, frame.drop(columns='observed_percent'), anchor,
        phase, np.full(len(frame), 17.), available) for f in fits]
    np.testing.assert_array_equal(forecasts[0]['stage'], forecasts[1]['stage'])
    assert np.mean(forecasts[0]['stage'][available]) > 25.


def test_excluded_cluster_labels_cannot_change_eligible_members():
    from models.monitoring.phenology_warm_oob import fit_warm_growth, predict_warm_growth
    frame, x, anchor, phase, available = _training()
    first = fit_warm_growth(x, frame, anchor, np.ones(len(frame)), phase, available,
        members=6, iterations=15)
    # Choose a genuinely omitted warm cluster, not a vacuous fallback row.
    counts = np.array([sum(c not in m['clusters'] for m in first['growth']['members'])
        for c in frame.coordinate_year])
    row = int(np.flatnonzero(available & (counts > 0) & (counts < 6))[0])
    changed = frame.copy()
    changed.loc[changed.coordinate_year.eq(frame.coordinate_year.iloc[row]), 'observed_percent'] = 100.
    second = fit_warm_growth(x, changed, anchor, np.ones(len(frame)), phase, available,
        members=6, iterations=15)
    query = frame.iloc[[row]].drop(columns='observed_percent')
    forecasts = [predict_warm_growth(f, x[[row]], query, anchor[[row]], phase[[row]],
        np.array([17.]), available[[row]], oob=True) for f in [first, second]]
    assert 0 < forecasts[0]['member_count'][0] < 6
    np.testing.assert_array_equal(forecasts[0]['stage'], forecasts[1]['stage'])
    # The perturbation must affect included-member fitting, proving sensitivity.
    full = [predict_warm_growth(f, x[[row]], query, anchor[[row]], phase[[row]],
        np.array([17.]), available[[row]])['stage'][0] for f in [first, second]]
    assert abs(full[0]-full[1]) > .05


def test_no_history_unseen_category_and_no_oob_support_preserve_reference():
    from models.monitoring.phenology_warm_oob import fit_warm_growth, predict_warm_growth
    frame, x, anchor, phase, available = _training()
    frame['coordinate_year'] = 'one'
    fitted = fit_warm_growth(x, frame, anchor, np.ones(len(frame)), phase, available,
        members=2, iterations=5)
    query = frame.iloc[:3].drop(columns='observed_percent').copy()
    query.loc[query.index[2], 'leaf_index'] = 6
    baseline = np.array([13., 27., 49.])
    oob = predict_warm_growth(fitted, x[:3], query, anchor[:3], phase[:3], baseline,
        np.array([False, True, True]), oob=True)
    np.testing.assert_array_equal(oob['stage'], baseline)
    full = predict_warm_growth(fitted, x[:3], query, anchor[:3], phase[:3], baseline,
        np.array([False, True, True]))
    np.testing.assert_array_equal(full['stage'][[0, 2]], baseline[[0, 2]])
    assert full['member_count'].tolist() == [0, 2, 0]


def test_saturated_soft_phase_forecast_remains_within_percentage_bounds():
    from models.monitoring.phenology_warm_oob import fit_warm_growth, predict_warm_growth
    frame, x, anchor, phase, available = _training()
    frame['observed_percent'] = 100.
    anchor[:] = 50.; available[:] = True
    # Actual first-fold gates that produced 100.00000000000001 after mixing.
    phase[:] = [0., .001620147489175383, .9683849580442672, .029994894466557422]
    fitted = fit_warm_growth(x, frame, anchor, np.ones(len(frame)), phase, available,
        members=2, iterations=1)
    values = predict_warm_growth(fitted, x, frame.drop(columns='observed_percent'),
        anchor, phase, np.full(len(frame), 17.), available)
    assert np.all((values['stage'] >= 0) & (values['stage'] <= 100))
