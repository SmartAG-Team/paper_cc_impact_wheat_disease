"""Mechanism tests: changing source accounting or observability breaks these cases."""
import importlib
import numpy as np
import pytest


def core():
    try:
        return importlib.import_module('model.primary_secondary.core')
    except ModuleNotFoundError:
        pytest.fail('Primary-secondary simulator is not implemented')


def forcing(days=80, rain=4.0, temperature=18.0, humidity=24.0, episodes=1):
    return tuple(np.full((episodes, days), x) for x in (temperature, humidity, rain))


def test_no_sources_and_no_initial_infection_remain_disease_free():
    c = core()
    out = c.simulate(np.zeros((1, 3)), *forcing(), c.Parameters(alpha=0, beta=0))
    assert np.array_equal(out.damage, np.zeros((1, 81, 3)))
    assert np.array_equal(out.primary_flux, np.zeros((1, 80, 3)))
    assert np.array_equal(out.secondary_flux, np.zeros((1, 80, 3)))


def test_primary_only_establishes_from_zero_and_labels_only_external_origin():
    c = core()
    out = c.simulate(np.zeros((1, 2)), *forcing(), c.Parameters(alpha=.08, beta=0))
    assert out.damage[0, -1, 0] > .7
    assert out.primary_flux.sum() > 0
    assert out.secondary_flux.sum() == 0
    assert np.allclose(out.origin_total[:, -1, :, 1], out.primary_flux.sum(axis=1))
    assert np.array_equal(out.origin_total[:, :, :, 2], np.zeros((1, 81, 2)))


def test_secondary_source_requires_infectious_tissue_and_splash():
    c = core()
    p = c.Parameters(alpha=0, beta=4)
    infected = np.array([[0.0, .5]])
    wet = c.simulate(infected, *forcing(), p)
    dry = c.simulate(infected, *forcing(rain=0), p)
    unseeded = c.simulate(np.zeros((1, 2)), *forcing(), p)
    assert wet.damage[0, -1, 0] > .5
    assert dry.secondary_flux.sum() == 0
    assert unseeded.secondary_flux.sum() == 0
    assert wet.primary_flux.sum() == 0


def test_latent_infection_is_not_visible_at_initialization():
    c = core()
    out = c.simulate(np.zeros((1, 1)), *forcing(), c.Parameters(alpha=0, beta=0), initial_latent=.6)
    assert out.damage[0, 0, 0] == 0
    assert out.pycnidia[0, 0, 0] == 0
    assert out.origin_total[0, 0, 0, 0] == pytest.approx(.6)
    assert out.damage[0, -1, 0] == pytest.approx(.6, abs=.01)
    assert out.primary_flux.sum() == 0
    assert out.secondary_flux.sum() == 0


def test_infection_origin_accounts_for_all_labeled_tissue_without_relabeling_initial():
    c = core()
    out = c.simulate(np.array([[.2, .1]]), *forcing(), c.Parameters(alpha=.02, beta=2), initial_latent=.3)
    assert np.allclose(out.state.sum(axis=-1), 1, atol=1e-12)
    assert out.state.min() >= -1e-14
    assert np.allclose(out.origin_total[..., 0], out.origin_total[:, :1, :, 0])
    assert np.allclose(out.origin_total[:, -1, :, 1], out.primary_flux.sum(axis=1), atol=1e-12)
    assert np.allclose(out.origin_total[:, -1, :, 2], out.secondary_flux.sum(axis=1), atol=1e-12)
    assert np.allclose(out.origin_total.sum(axis=-1) + out.state[..., 0], 1, atol=1e-12)


def test_removed_infected_tissue_remains_visible_but_stops_transmitting():
    c = core()
    out = c.simulate(np.array([[.4]]), *forcing(days=200), c.Parameters(alpha=0, beta=0, infectious_days=2))
    assert np.allclose(out.damage, .4)
    assert np.allclose(out.pycnidia, .4)
    assert out.infectious[0, -1, 0] < 1e-20


def test_future_forcing_does_not_change_past_states():
    c = core()
    t, h, r = forcing(days=30)
    changed_t, changed_h, changed_r = t.copy(), h.copy(), r.copy()
    changed_t[:, 10:] = 0
    changed_h[:, 10:] = 0
    changed_r[:, 10:] = 0
    p = c.Parameters(alpha=.02, beta=2)
    a = c.simulate(np.array([[.1]]), t, h, r, p)
    b = c.simulate(np.array([[.1]]), changed_t, changed_h, changed_r, p)
    assert np.array_equal(a.state[:, :11], b.state[:, :11])
    assert not np.allclose(a.damage[:, -1], b.damage[:, -1])


def test_quarter_day_solution_converges_toward_eighth_day_solution():
    c = core()
    p = c.Parameters(alpha=.02, beta=2)
    a = c.simulate(np.array([[.05, .4]]), *forcing(days=90), p, time_step=.25)
    b = c.simulate(np.array([[.05, .4]]), *forcing(days=90), p, time_step=.125)
    assert np.max(abs(a.damage - b.damage)) < .02


@pytest.mark.parametrize('which,value', [(0,np.nan),(1,25),(2,-1)])
def test_invalid_weather_is_rejected(which, value):
    c = core()
    inputs = list(forcing(days=2))
    inputs[which][0, 0] = value
    with pytest.raises(ValueError):
        c.simulate(np.array([[.1]]), *inputs, c.Parameters())


def test_invalid_visible_fraction_and_negative_parameter_are_rejected():
    c = core()
    with pytest.raises(ValueError):
        c.simulate(np.array([[1.01]]), *forcing(days=2), c.Parameters())
    with pytest.raises(ValueError):
        c.simulate(np.array([[.1]]), *forcing(days=2), c.Parameters(alpha=-.1))


def test_zero_temperature_prevents_latent_progress_not_source_accounting():
    c = core()
    out = c.simulate(np.zeros((1, 1)), *forcing(temperature=0), c.Parameters(alpha=.02, beta=0))
    assert out.primary_flux.sum() > 0
    assert np.array_equal(out.damage, np.zeros((1,81,1)))


def test_multiple_latent_stage_counts_preserve_mass_and_observation_definition():
    c = core()
    for stages in (1, 3, 12):
        p = c.Parameters(alpha=.03, beta=1, latent_stages=stages)
        out = c.simulate(np.array([[0.0, .3]]), *forcing(days=35), p, initial_latent=.2)
        assert out.state.shape[-1] == 1 + 3*(stages+3)
        assert np.allclose(out.state.sum(axis=-1), 1, atol=1e-12)
        assert np.all(out.pycnidia <= out.damage + 1e-12)


def test_unobserved_inactive_leaves_do_not_create_or_dilute_inoculum():
    c=core()
    p=c.Parameters(alpha=.02,beta=2)
    full=c.simulate(np.array([[.2,0.]]),*forcing(days=30),p,initial_latent=.3,
                    leaf_active=np.array([[True,False]]))
    one=c.simulate(np.array([[.2]]),*forcing(days=30),p,initial_latent=.3)
    assert np.allclose(full.damage[...,0],one.damage[...,0])
    assert np.all(full.state[:,:,1,0]==1)
    assert np.all(full.origin_total[:,:,1]==0)


def test_old_canopy_damage_has_reduced_current_infectious_pressure():
    c=core()
    p=c.Parameters(alpha=0,beta=2,infectious_days=10)
    visible=np.array([[0.,.5]])
    old=c.simulate(visible,*forcing(days=30),p,initial_visible_age=np.array([[0.,100.]]))
    fresh=c.simulate(visible,*forcing(days=30),p)
    assert old.damage[0,0,1]==pytest.approx(.5)
    assert old.infectious[0,0,1]==pytest.approx(.000022699964881242427)
    assert old.secondary_flux.sum()<fresh.secondary_flux.sum()/100


def test_full_weather_ablation_removes_thermal_development_information():
    c=core()
    p=c.Parameters(alpha=.02,beta=2,weather_response=False)
    cold=c.simulate(np.array([[.1]]),*forcing(days=30,temperature=5),p,initial_latent=.3)
    warm=c.simulate(np.array([[.1]]),*forcing(days=30,temperature=25),p,initial_latent=.3)
    assert np.array_equal(cold.state,warm.state)
