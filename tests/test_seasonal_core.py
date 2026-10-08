from dataclasses import replace

import numpy as np
import pytest

from model.seasonal_septoria.core import Parameters, simulate_season


def forcing(days=80, leaves=2):
    t = np.full((1, days), 18.)
    h = np.full_like(t, 95.)
    r = np.full_like(t, 4.)
    active = np.ones((1, days, leaves), bool)
    renewal = np.zeros_like(active, float)
    return t, h, r, active, renewal


def test_season_start_has_no_unobserved_initial_disease():
    result = simulate_season(*forcing(), Parameters(alpha=0., beta=5.))
    assert np.count_nonzero(result.state[..., 1:]) == 0
    assert np.count_nonzero(result.primary_flow) == 0
    assert np.count_nonzero(result.secondary_flow) == 0


def test_latent_tissue_is_not_immediately_visible_and_mass_is_conserved():
    result = simulate_season(*forcing(), Parameters(alpha=.03, beta=2.))
    assert np.max(np.abs(result.state.sum(axis=-1)-1)) < 2e-13
    assert result.state.min() >= 0
    assert result.damage.max() <= 1
    assert result.damage[0, 0].sum() == 0
    assert result.state[0, 1, :, 1:].sum() > result.damage[0, 1].sum()
    assert result.secondary_flow.sum() > 0


def test_inactive_host_cannot_acquire_or_export_disease():
    args = list(forcing(days=40))
    args[3][:, :20, :] = False
    args[3][:, :, 1] = False
    result = simulate_season(*args, Parameters(alpha=.2, beta=5.))
    assert np.count_nonzero(result.state[0, :21, :, 1:]) == 0
    assert np.count_nonzero(result.state[0, :, 1, 1:]) == 0
    assert result.primary_flow[0, 20:, 0].sum() > 0


def test_external_and_secondary_routes_reconcile_with_total_disease():
    result = simulate_season(*forcing(), Parameters(alpha=.01, beta=3.))
    assert np.max(np.abs(result.origin_total.sum(axis=-1)+result.state[..., 0]-1)) < 1e-13
    only_external = simulate_season(*forcing(), Parameters(alpha=.01, beta=0.))
    assert np.count_nonzero(only_external.origin_total[..., 1]) == 0
    assert result.damage[0, -1].mean() > only_external.damage[0, -1].mean()


def test_complete_new_leaf_replacement_removes_old_disease():
    args = list(forcing(days=60))
    args[4][:, 40:, :] = 1.
    result = simulate_season(*args, Parameters(alpha=.1, beta=5.), time_step=1.)
    # With daily complete replacement and synchronous transitions, a newly
    # infected cohort cannot complete multiple latent steps in a single day.
    assert result.damage[0, 40].sum() > 0
    assert result.damage[0, 41:].sum() == 0


def test_bad_weather_and_inactive_renewal_are_not_silently_coerced():
    args = list(forcing())
    args[1][0, 0] = 120.
    with pytest.raises(ValueError):
        simulate_season(*args, Parameters())
    args = list(forcing())
    args[3][0, 0, 0] = False
    args[4][0, 0, 0] = .1
    with pytest.raises(ValueError):
        simulate_season(*args, Parameters())


def test_reduced_time_step_gives_stable_terminal_predictions():
    p = Parameters(alpha=.01, beta=1.)
    coarse = simulate_season(*forcing(), p, time_step=.25)
    fine = simulate_season(*forcing(), p, time_step=.125)
    assert np.max(np.abs(coarse.damage[0, -1]-fine.damage[0, -1])) < .01


def test_zero_weather_response_has_no_rain_splash_secondary_route():
    args = list(forcing())
    args[2][:] = 0.
    result = simulate_season(*args, Parameters(alpha=.02, beta=5.))
    assert result.primary_flow.sum() > 0
    assert np.count_nonzero(result.secondary_flow) == 0
