import numpy as np
import pandas as pd


def test_phase_experts_learn_distinct_development_regimes():
    from models.monitoring.phenology_phase_growth import fit_phase_growth, predict_phase_growth
    phase = np.zeros((80, 4)); phase[np.arange(80), np.tile([0, 1], 40)] = 1
    frame = pd.DataFrame(dict(field_id=np.arange(80).astype(str), coordinate_year=np.arange(80).astype(str),
        observed_percent=np.tile([10., 90.], 40), global_target_index=np.arange(80)))
    x = np.zeros((80, 2)); anchor = np.full(80, 50.)
    fit = fit_phase_growth(x, frame, anchor, np.ones(80), phase, members=2, iterations=30)
    value = predict_phase_growth(fit, x[:2], anchor[:2], phase[:2])
    assert value['stage'][0] < 25 and value['stage'][1] > 75
    assert np.all(np.abs(value['shared']-50) < 10)


def test_growth_preserves_reference_without_earlier_field_observations():
    from models.monitoring.phenology_phase_growth import fit_phase_growth, predict_phase_growth
    frame = pd.DataFrame(dict(field_id=np.arange(30).astype(str), coordinate_year=np.arange(30).astype(str),
        observed_percent=90., global_target_index=np.arange(30)))
    x = np.zeros((30, 2)); phase = np.tile([0., 0., 1., 0.], (30, 1))
    fit = fit_phase_growth(x, frame, np.full(30, 50.), np.ones(30), phase, members=1, iterations=20)
    value = predict_phase_growth(fit, x[:1], np.array([50.]), phase[:1],
        baseline=np.array([37.]), available=np.array([False]))
    assert value['shared'][0] == value['stage'][0] == 37
