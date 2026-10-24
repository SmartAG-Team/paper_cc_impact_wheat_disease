import numpy as np
import pandas as pd
import pytest


def test_zero_introduction_threshold_replays_shared_kernel_and_delays_are_causal():
    from models.monitoring.process_establishment import simulate_establishment
    from models.seasonal_septoria.overwinter import simulate_overwinter, OverwinterParameters
    t = np.full((1, 25), 18.); e = np.ones_like(t); r = np.ones_like(t)
    active = np.ones((1, 25, 8), bool); renewal = np.zeros(active.shape); area = np.ones(active.shape)
    mask = np.ones(t.shape, bool); params = OverwinterParameters(primary_scale=.1)
    expected = simulate_overwinter(t,e,r,active,renewal,area,mask,params,imported_pressure=np.full_like(t,.1))
    actual = simulate_establishment(t,e,r,active,renewal,area,mask,params,0.)
    np.testing.assert_array_equal(actual, expected.state)
    delayed = simulate_establishment(t,e,r,active,renewal,area,mask,params,6.)
    assert np.all(delayed[:, :7, :, 1:] == 0.)
    assert delayed[:, -1, :, 1:].sum() > 0
    assert np.max(np.abs(delayed.sum(-1)-1.)) < 1e-10
    changed = e.copy(); changed[:, 15:] = 0.
    altered = simulate_establishment(t,changed,r,active,renewal,area,mask,params,6.)
    np.testing.assert_array_equal(delayed[:, :16], altered[:, :16])
    dry = simulate_establishment(t,np.zeros_like(e),r,active,renewal,area,mask,params,0.)
    assert np.all(dry[..., 1:] == 0.)


def test_mixture_fits_known_establishment_variation_without_query_labels():
    from models.monitoring.process_establishment import fit_observation, predict_observation
    rng = np.random.default_rng(31)
    states = rng.uniform(0, 1, (150, 3, 8))
    truth = np.array([.2, .65, .1, .05])
    y = 100*(states[:, :, 0]@truth[:3])
    frame = pd.DataFrame(dict(source='BASF', leaf_index=0, observed_percent=y))
    fixed = fit_observation(states, frame, np.ones(len(frame)), mixture=False)
    fitted = fit_observation(states, frame, np.ones(len(frame)), mixture=True)
    q = frame.drop(columns='observed_percent')
    direct = predict_observation(fixed, states, q)
    learned = predict_observation(fitted, states, q)
    assert np.mean((learned-y)**2) < .05*np.mean((direct-y)**2)
    np.testing.assert_allclose(fitted['arrival_weight'].sum(), 1., atol=1e-12)
    assert (fitted['arrival_weight'] >= 0).all()
    assert ((learned >= 0)&(learned <= 100)).all()
    with pytest.raises(ValueError, match='outcomes'):
        predict_observation(fitted, states, frame)


def test_flag_mapping_is_fixed_and_unknown_codes_are_nominal():
    from models.monitoring.process_establishment import fit_observation, predict_observation
    rng = np.random.default_rng(5); states = rng.uniform(0, 1, (80, 3, 8))
    frame = pd.DataFrame(dict(source='Corteva', leaf_index=2, observed_percent=100*states[:, 0, 5]))
    fit = fit_observation(states, frame, np.ones(80), mixture=False)
    q = frame.drop(columns='observed_percent')
    learned = predict_observation(fit, states, q)
    assert np.sqrt(np.mean((learned-frame.observed_percent)**2)) < 3.
    flag = q.assign(source='BASF', leaf_index=0)
    np.testing.assert_allclose(predict_observation(fit, states, flag), 100*states[:, 0, 0], atol=1e-10)
    different = q.assign(leaf_index=6)
    unknown = predict_observation(fit, states, different)
    np.testing.assert_allclose(unknown, 100*(states[:, 0]@fit['pooled_operator']), atol=1e-10)
