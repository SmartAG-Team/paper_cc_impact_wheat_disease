"""Behavioral checks for an explicit seasonal local-source model."""

from dataclasses import replace

import numpy as np
import pytest

from model.seasonal_septoria.overwinter import OverwinterModel, OverwinterParameters, simulate_overwinter


def forcing(days=100, leaves=3):
    t = np.full((1, days), 12.)
    exposure = np.full_like(t, .6)
    rain = np.full_like(t, 3.)
    active = np.ones((1, days, leaves), bool)
    area = active.astype(float)
    renewal = np.zeros_like(area)
    valid = np.ones_like(t, bool)
    return t, exposure, rain, active, renewal, area, valid


def test_absent_local_and_imported_sources_cannot_create_infection():
    result = simulate_overwinter(*forcing(), OverwinterParameters(), initial_local_source=0.)
    assert not result.state[..., 1:].any()
    assert not result.primary_flow.any()
    assert not result.secondary_flow.any()


def test_residue_development_is_slower_during_cold_weather():
    cold = list(forcing(days=10))
    cold[0][:] = 0.
    p = OverwinterParameters(initial_ready_fraction=0., residue_decay_reference_days=1e8)
    result_cold = simulate_overwinter(*cold, p)
    result_warm = simulate_overwinter(*forcing(days=10), p)
    assert result_cold.residue[0, -1, 1] == 0.
    assert result_warm.residue[0, -1, 1] > .05


def test_existing_infection_survives_a_cold_winter_without_a_spring_reset():
    args = list(forcing(days=160))
    args[0][:, 30:100] = 0.
    args[1][:, 30:100] = 0.
    args[2][:, 30:100] = 0.
    p = OverwinterParameters(primary_scale=.1, infectious_reference_days=40.)
    result = simulate_overwinter(*args, p)
    assert result.state[0, 30, :, 1:].sum() > 0.
    assert result.state[0, 100, :, 1:].sum() > 0.
    assert result.secondary_flow[:, 100:].sum() > 0.


def test_secondary_spread_requires_a_living_infectious_donor():
    early = simulate_overwinter(*forcing(days=1), OverwinterParameters(primary_scale=.2))
    assert early.primary_flow.sum() > 0.
    assert early.secondary_flow.sum() == 0.
    longer = simulate_overwinter(*forcing(days=60), OverwinterParameters(primary_scale=.2))
    assert longer.secondary_flow.sum() > 0.


def test_unsuitable_weather_stops_new_establishment():
    args = list(forcing(days=60))
    args[1][:, 30:] = 0.
    result = simulate_overwinter(*args, OverwinterParameters(primary_scale=.2))
    assert result.state[0, 30, :, 1:].sum() > 0.
    assert result.primary_flow[:, 30:].sum() == 0.
    assert result.secondary_flow[:, 30:].sum() == 0.


def test_contact_route_permits_dew_exposure_without_rain_on_unfolding_leaves():
    args = list(forcing(days=70))
    args[2][:, 35:] = 0.
    args[5][:, 35:, 0] = .5
    p = OverwinterParameters(primary_scale=.2, contact_fraction=.2)
    result = simulate_overwinter(*args, p)
    assert result.splash_flow[:, 35:].sum() == 0.
    assert result.contact_flow[:, 35:, 0].sum() > 0.


def test_unavailable_leaf_neither_receives_nor_exports_infection():
    args = list(forcing(days=70))
    args[3][:, :35, 0] = False
    args[5][:, :35, 0] = 0.
    result = simulate_overwinter(*args, OverwinterParameters(primary_scale=.1))
    assert result.state[0, :36, 0, 1:].sum() == 0.
    assert result.primary_flow[0, 35:, 0].sum() > 0.


def test_tissue_and_route_accounting_are_conserved():
    result = simulate_overwinter(*forcing(), OverwinterParameters(primary_scale=.04))
    np.testing.assert_allclose(result.state.sum(axis=-1), 1., atol=2e-13)
    assert result.state.min() >= -1e-14
    assert result.damage.max() <= 1.+1e-14
    np.testing.assert_allclose(result.primary_flow, result.local_flow+result.imported_flow)
    np.testing.assert_allclose(result.secondary_flow, result.splash_flow+result.contact_flow)
    assert np.all(result.residue >= 0.)


def test_imported_source_is_explicit_and_can_establish_without_local_residue():
    result = simulate_overwinter(*forcing(), OverwinterParameters(), initial_local_source=0.,
                                 imported_pressure=np.ones((1, 100)))
    assert result.local_flow.sum() == 0.
    assert result.imported_flow.sum() > 0.


def test_padding_cannot_age_residue_or_advance_disease():
    args = list(forcing(days=60))
    args[6][:, 30:] = False
    result = simulate_overwinter(*args, OverwinterParameters(primary_scale=.1))
    prefix = simulate_overwinter(*[a[:, :30] for a in args], OverwinterParameters(primary_scale=.1))
    np.testing.assert_array_equal(result.state[:, :31], prefix.state)
    np.testing.assert_array_equal(result.residue[:, :31], prefix.residue)
    np.testing.assert_array_equal(result.state[:, 31:], np.repeat(result.state[:, 30:31], 30, axis=1))
    assert result.primary_flow[:, 30:].sum() == 0.
    assert result.secondary_flow[:, 30:].sum() == 0.


def test_smaller_time_step_preserves_material_trajectory():
    p = OverwinterParameters(primary_scale=.03)
    coarse = simulate_overwinter(*forcing(days=60), p, time_step=.25)
    fine = simulate_overwinter(*forcing(days=60), p, time_step=.125)
    assert np.max(abs(coarse.damage-fine.damage)) < .012


@pytest.mark.parametrize('change', [dict(secondary_scale=-1.), dict(initial_ready_fraction=2.),
                                   dict(latent_reference_days=0.), dict(contact_fraction=-.1)])
def test_invalid_parameters_are_rejected(change):
    with pytest.raises(ValueError):
        simulate_overwinter(*forcing(), replace(OverwinterParameters(), **change))


def test_rate_calculation_does_not_change_state_and_can_only_be_committed_once():
    model = OverwinterModel(leaf_count=3)
    before = model.snapshot()
    rates = model.calc_rates(12., .6, 3., np.ones(3, bool), np.zeros(3), np.ones(3))
    assert model.snapshot() == before
    model.integrate(rates)
    assert model.state[:, 1:].sum() > 0.
    with pytest.raises(ValueError):
        model.integrate(rates)
    other = OverwinterModel(leaf_count=3)
    with pytest.raises(ValueError):
        other.integrate(rates)


def test_restored_daily_component_reproduces_the_uninterrupted_trajectory():
    original = OverwinterModel(leaf_count=3)
    arguments = (12., .6, 3., np.ones(3, bool), np.zeros(3), np.ones(3))
    for _ in range(20):
        original.integrate(original.calc_rates(*arguments))
    resumed = OverwinterModel(leaf_count=3)
    resumed.restore(original.snapshot())
    for _ in range(20):
        original.integrate(original.calc_rates(*arguments))
        resumed.integrate(resumed.calc_rates(*arguments))
    assert original.snapshot() == resumed.snapshot()
    batch = simulate_overwinter(*forcing(days=40), OverwinterParameters())
    np.testing.assert_array_equal(original.state, batch.state[0, -1])


def test_invalid_rate_proposal_does_not_corrupt_component_state():
    model = OverwinterModel(leaf_count=3)
    rates = model.calc_rates(12., .6, 3., np.ones(3, bool), np.zeros(3), np.ones(3))
    before = model.snapshot()
    corrupt = replace(rates, next_tissue=np.full((3, 7), -1.))
    with pytest.raises(ValueError):
        model.integrate(corrupt)
    assert model.snapshot() == before
