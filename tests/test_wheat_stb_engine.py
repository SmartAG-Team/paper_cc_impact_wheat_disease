"""Daily components preserve forcing, lifecycle and restart boundaries."""

from dataclasses import FrozenInstanceError, replace
from datetime import date, timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model.wheat_stb import (
    EngineConfig, PhenologyModel, TPVParameters, WeatherDay, WeatherProvider,
    WheatSTBSimulation,
)
from model.seasonal_septoria.regional import tpv_accumulation
from calibration.wheat_stb.configuration import make_configuration


ROOT = Path(__file__).resolve().parents[1]


def weather(days=90):
    first = date(2026, 9, 20)
    rows = []
    for index in range(days):
        mean = [-2., 4., 12., 18., 31., 36., 8.][index % 7]
        rows.append(WeatherDay(first+timedelta(days=index), mean, mean+4.,
            90. if index % 3 else 65., 2. if index % 4 == 0 else 0.))
    return WeatherProvider(rows)


def configuration(**changes):
    values = make_configuration(tpv_fit=ROOT/'configurations/wheat_stb/tpv_calibration.json',
        stage_fit=ROOT/'configurations/wheat_stb/calibrated_stage_thresholds.json',
        latitude=51.5, sowing_date=date(2026, 10, 1)).to_dict()
    values.update(changes)
    return EngineConfig.from_dict(values)


@pytest.mark.parametrize('vernalization_required', [True, False])
def test_incremental_tpv_matches_donor_with_prior_state_gates(vernalization_required):
    provider = weather(140)
    raw = json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    parameters = TPVParameters(**{key: raw[key] for key in TPVParameters.__dataclass_fields__})
    sowing = date(2026, 10, 1)
    component = PhenologyModel(parameters, latitude=51.5, sowing_date=sowing,
        vernalization_required=vernalization_required)
    actual = []
    for row in provider:
        actual.append(component.integrate(component.calc_rates(row)).cumulative_tpv)
    expected = tpv_accumulation(
        np.array([[row.tmean_c for row in provider]]),
        np.array([[row.tmax_c for row in provider]]), [51.5],
        np.array([row.date.timetuple().tm_yday for row in provider]),
        np.array([[row.date >= sowing for row in provider]]), raw,
        vernalization_required=vernalization_required)[0]
    np.testing.assert_allclose(actual, expected, rtol=0., atol=2e-12)


def test_phenology_rates_are_pure_and_cannot_be_reused_or_cross_instance():
    raw = json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    p = TPVParameters(**{key: raw[key] for key in TPVParameters.__dataclass_fields__})
    row = WeatherDay(date(2026, 10, 1), 10., 14., 80., 0.)
    first = PhenologyModel(p, latitude=51., sowing_date=row.date)
    second = PhenologyModel(p, latitude=51., sowing_date=row.date)
    before = first.snapshot()
    rates = first.calc_rates(row)
    assert first.snapshot() == before
    with pytest.raises(ValueError, match='owner|instance'):
        second.integrate(rates)
    first.integrate(rates)
    assert first.state.cumulative_gdd == 10.
    with pytest.raises(ValueError, match='current|stale'):
        first.integrate(rates)


def test_weather_rejects_invalid_ranges_and_nonconsecutive_dates():
    first = date(2026, 10, 1)
    with pytest.raises(ValueError, match='humidity'):
        WeatherDay(first, 12., 15., 101., 0.)
    with pytest.raises(ValueError, match='precipitation'):
        WeatherDay(first, 12., 15., 80., -1.)
    with pytest.raises(ValueError, match='maximum'):
        WeatherDay(first, 12., 10., 80., 0.)
    one = WeatherDay(first, 12., 15., 80., 0.)
    with pytest.raises(ValueError, match='consecutive'):
        WeatherProvider([one, WeatherDay(first+timedelta(days=2), 12., 15., 80., 0.)])
    with pytest.raises(ValueError, match='consecutive'):
        WeatherProvider([one, one])


def test_csv_weather_boundary_excludes_outcome_columns(tmp_path):
    path = tmp_path/'weather.csv'
    path.write_text('date,tmean_c,tmax_c,rh_mean_pct,precipitation_mm,value\n2026-10-01,12,15,80,0,55\n')
    with pytest.raises(ValueError, match='unsupported|outcome'):
        WeatherProvider.from_csv(path)


def test_external_configuration_preserves_authoritative_stage_fit_and_is_immutable():
    config = configuration()
    fit = json.loads((ROOT/'analysis/paper_study/overwinter_leaf_model_20261006/phenology/calibrated_stage_thresholds.json').read_text())
    assert dict(config.leaf.stage_thresholds) == {int(k): float(v) for k, v in fit['all_stage_thresholds'].items()}
    with pytest.raises(FrozenInstanceError):
        config.latitude = 40.
    with pytest.raises(FrozenInstanceError):
        config.phenology.photoperiod_onset_gdd = 99.


def test_forged_phenology_rates_cannot_corrupt_state_or_dates():
    config = configuration()
    component = PhenologyModel(config.phenology, latitude=config.latitude, sowing_date=config.sowing_date)
    rates = component.calc_rates(weather(1)[0])
    before = component.snapshot()
    for forged in [replace(rates, gdd=-1.), replace(rates, date=date(2030, 1, 1))]:
        with pytest.raises(ValueError, match='proposal|forcing|rate'):
            component.integrate(forged)
        assert component.snapshot() == before


def test_daily_engine_has_pure_rate_phase_and_end_day_outputs_without_default_yield():
    simulation = WheatSTBSimulation(configuration(), weather(120))
    simulation.initialize()
    before = simulation.snapshot()
    rates = simulation.calc_rates()
    assert simulation.snapshot() == before
    first = simulation.integrate(rates)
    assert first.date == weather(120)[0].date
    assert first.cumulative_tpv == simulation.phenology.state.cumulative_tpv
    assert len(first.affected_fraction) == 8
    with pytest.raises(ValueError, match='stale|current'):
        simulation.integrate(rates)
    simulation.run()
    results = simulation.finalize()
    assert len(results.daily) == 120
    assert results.summary['conditional_yield_loss_t_ha'] is None
    assert results.summary['actual_field_yield_forecast'] is None
    assert results.summary['state_boundary'] == 'end of each supplied forcing day'


def test_provenance_reports_the_effective_disease_substep():
    simulation = WheatSTBSimulation(configuration(time_step=.3), weather(1))
    simulation.run()
    provenance = simulation.finalize().provenance
    assert provenance['requested_time_step_days'] == .3
    assert provenance['numerical_time_step_days'] == .25


def test_engine_instances_and_weather_prefixes_are_independent():
    config, provider = configuration(), weather(100)
    first = WheatSTBSimulation(config, provider)
    second = WheatSTBSimulation(config, provider)
    first.run(30)
    assert len(second.outputs) == 0
    short = WheatSTBSimulation(config, WeatherProvider(provider[:30]))
    short.run()
    assert first.outputs == short.outputs
    second.run(30)
    assert first.outputs == second.outputs


def test_fresh_simulation_cannot_omit_weather_after_sowing():
    config = configuration(sowing_date='2026-10-01')
    late_weather = WeatherProvider([WeatherDay(date(2026, 10, 2), 12., 16., 80., 0.)])
    with pytest.raises(ValueError, match='sowing|coverage'):
        WheatSTBSimulation(config, late_weather)


def test_restart_matches_uninterrupted_and_rejects_changed_history_or_config():
    config, provider = configuration(), weather(100)
    uninterrupted = WheatSTBSimulation(config, provider)
    uninterrupted.run()
    interrupted = WheatSTBSimulation(config, WeatherProvider(provider[:35]))
    interrupted.run()
    checkpoint = json.loads(json.dumps(interrupted.snapshot()))
    resumed = WheatSTBSimulation(config, provider)
    resumed.restore(checkpoint)
    resumed.run()
    assert resumed.outputs == uninterrupted.outputs
    np.testing.assert_array_equal(resumed.disease.state, uninterrupted.disease.state)
    changed = list(provider)
    changed[10] = replace(changed[10], precipitation_mm=7.)
    with pytest.raises(ValueError, match='weather|prefix'):
        WheatSTBSimulation(config, WeatherProvider(changed)).restore(checkpoint)
    with pytest.raises(ValueError, match='configuration'):
        WheatSTBSimulation(configuration(latitude=50.), provider).restore(checkpoint)


def test_invalid_checkpoint_restoration_is_atomic():
    simulation = WheatSTBSimulation(configuration(), weather(100))
    simulation.run(20)
    before = simulation.snapshot()
    bad = json.loads(json.dumps(before))
    bad['components']['yield']['state']['lost_had3'] = -100.
    with pytest.raises(ValueError, match='yield|HAD'):
        simulation.restore(bad)
    assert simulation.snapshot() == before


@pytest.mark.parametrize('changed_component', ['clock_order', 'clock_output', 'leaf_accumulation',
    'leaf_area', 'disease_residue', 'disease_tissue', 'yield_had', 'yield_window'])
def test_internally_contradictory_checkpoints_are_rejected_atomically(changed_component):
    simulation = WheatSTBSimulation(configuration(), weather(100))
    simulation.run(30)
    before = simulation.snapshot()
    bad = json.loads(json.dumps(before))
    parts = bad['components']
    if changed_component == 'clock_order':
        state = parts['phenology']['state']
        state['cumulative_tpv'] = state['cumulative_tpp']+1.
    elif changed_component == 'clock_output':
        parts['phenology']['state']['cumulative_gdd'] += 1.
    elif changed_component == 'leaf_accumulation':
        parts['leaf']['previous_accumulation'] += 1.
    elif changed_component == 'leaf_area':
        parts['leaf']['previous_area'][0] = .5
    elif changed_component == 'disease_residue':
        parts['disease']['residue_state'][0] *= .9
    elif changed_component == 'disease_tissue':
        parts['disease']['tissue_state'][0][0] -= .01
        parts['disease']['tissue_state'][0][1] += .01
    elif changed_component == 'yield_had':
        parts['yield']['state']['reference_had3'] = 10.
        parts['yield']['state']['lost_had3'] = 1.
    else:
        parts['yield']['state']['window_start_date'] = '2026-10-01'
    with pytest.raises(ValueError, match='checkpoint|Checkpoint'):
        simulation.restore(bad)
    assert simulation.snapshot() == before


@pytest.mark.parametrize('seeded_component', ['phenology', 'disease'])
def test_empty_checkpoint_requires_canonical_initial_state(seeded_component):
    simulation = WheatSTBSimulation(configuration(), weather(20))
    before = simulation.snapshot()
    bad = json.loads(json.dumps(before))
    if seeded_component == 'phenology':
        bad['components']['phenology']['state'].update(cumulative_gdd=1.,
            cumulative_tpp=.5, cumulative_tpv=.3)
    else:
        bad['components']['disease']['tissue_state'][0][0] = .9
        bad['components']['disease']['tissue_state'][0][1] = .1
    with pytest.raises(ValueError, match='initial'):
        simulation.restore(bad)
    assert simulation.snapshot() == before


@pytest.mark.parametrize('mode', ['external_functional_loss', 'standardized_model_proxy'])
def test_restart_preserves_enabled_had_and_complete_window(mode):
    config = synthetic_configuration(sowing_date='2026-10-01', yield_model=dict(mode=mode))
    provider = WeatherProvider([WeatherDay(date(2026, 10, 1)+timedelta(days=i), 1., 2., 90., 2.,
        reference_leaf_area_index=(.3, .4, .3), functional_loss_fraction=(.2, .2, .2)) for i in range(12)])
    complete = WheatSTBSimulation(config, provider)
    complete.run()
    interrupted = WheatSTBSimulation(config, provider)
    interrupted.run(8)
    resumed = WheatSTBSimulation(config, provider).restore(
        json.loads(json.dumps(interrupted.snapshot())))
    resumed.run()
    assert resumed.outputs == complete.outputs
    assert resumed.finalize().summary == complete.finalize().summary


def test_checkpoint_runtime_source_identity_is_required():
    simulation = WheatSTBSimulation(configuration(), weather(20))
    simulation.run(10)
    before = simulation.snapshot()
    bad = json.loads(json.dumps(before))
    key = next(iter(bad['runtime_source_sha256']))
    bad['runtime_source_sha256'][key] = 'different implementation'
    with pytest.raises(ValueError, match='source version'):
        simulation.restore(bad)
    assert simulation.snapshot() == before


def test_conditional_yield_is_explicit_and_incomplete_windows_stay_missing():
    config = configuration(yield_model=dict(mode='standardized_model_proxy'))
    simulation = WheatSTBSimulation(config, weather(15))
    simulation.run()
    result = simulation.finalize()
    assert not result.summary['complete_grain_fill_window']
    assert result.summary['conditional_yield_loss_t_ha'] is None
    assert result.summary['yield_area_mode'] == 'standardized_model_proxy'


def test_cli_runs_and_preserves_existing_output_directory(tmp_path):
    from model.wheat_stb.cli import main
    config_path = tmp_path/'configuration.json'
    config_path.write_text(json.dumps(configuration().to_dict()))
    weather_path = tmp_path/'weather.csv'
    pd.DataFrame([dict(date=row.date.isoformat(), tmean_c=row.tmean_c,
        tmax_c=row.tmax_c, rh_mean_pct=row.rh_mean_pct,
        precipitation_mm=row.precipitation_mm) for row in weather(10)]).to_csv(weather_path, index=False)
    output = tmp_path/'result'
    assert main(['--config', str(config_path), '--weather', str(weather_path), '--output', str(output)]) == 0
    assert len(pd.read_csv(output/'daily.csv')) == 10
    summary = json.loads((output/'summary.json').read_text())
    assert summary['conditional_yield_loss_t_ha'] is None
    assert (output/'checkpoint.json').exists()
    with pytest.raises(FileExistsError):
        main(['--config', str(config_path), '--weather', str(weather_path), '--output', str(output)])


def synthetic_configuration(**changes):
    values = configuration().to_dict()
    values.update(phenology=dict(photoperiod_onset_gdd=1000., photoperiod_stop_gdd=2000.,
        vernalization_onset_tpp=1000., vernalization_stop_tpp=2000.), vernalization_required=False)
    values['leaf']['stage_thresholds'] = {str(key): float(value) for key, value in
        [(10, 1), (31, 2), (32, 3), (33, 3.5), (37, 4), (39, 5), (51, 6), (65, 7), (85, 10)]}
    values['leaf']['threshold_status'] = 'explicit synthetic unit-test scenario'
    values.update(changes)
    return EngineConfig.from_dict(values)


def test_excluded_crop_weather_freezes_disease_residue_and_tissue():
    first = date(2026, 9, 20)
    provider = WeatherProvider([WeatherDay(first+timedelta(days=i), 1., 2., 100., 2.,
        imported_pressure=.1, exposure_override=1.) for i in range(20)])
    config = synthetic_configuration(crop_end_date='2026-10-05')
    simulation = WheatSTBSimulation(config, provider)
    initial = simulation.disease.snapshot()
    simulation.run(11)
    np.testing.assert_array_equal(simulation.disease.residue, initial['residue_state'])
    np.testing.assert_array_equal(simulation.disease.state, initial['tissue_state'])
    simulation.run(5)
    assert simulation.disease.state[:, 1:].sum() > 0
    crop_end = simulation.disease.snapshot()
    simulation.run()
    np.testing.assert_array_equal(simulation.disease.residue, crop_end['residue_state'])
    np.testing.assert_array_equal(simulation.disease.state, crop_end['tissue_state'])
    for output in simulation.outputs[16:]:
        assert sum(output.local_flow+output.imported_flow+output.splash_flow+output.contact_flow) == 0.


def test_runtime_requires_explicit_stage_and_tpv_parameters():
    with pytest.raises(ValueError, match='Explicit phenology and leaf'):
        EngineConfig.from_dict(dict(latitude=51.5, sowing_date='2026-10-01'))


def test_external_fit_conversion_preserves_imported_background_and_daily_override(tmp_path):
    from dataclasses import asdict
    from model.seasonal_septoria.overwinter import OverwinterParameters
    path = tmp_path/'fit.json'
    path.write_text(json.dumps(dict(status='fit_complete', disease_magnitude_fit=False,
        validation_outcomes_used=False, calibration_fields=28,
        fitted=dict(parameters=asdict(OverwinterParameters(primary_scale=.001, secondary_scale=10.)),
            rank_spacing_units=80., constant_imported_pressure=.1,
            weather_preprocessing=dict(weather_operator='duration_proxy', rain_rate_mm_hour=.45)))))
    config = make_configuration(tpv_fit=ROOT/'configurations/wheat_stb/tpv_calibration.json',
        stage_fit=ROOT/'configurations/wheat_stb/calibrated_stage_thresholds.json', disease_fit=path,
        latitude=51.5, sowing_date='2026-10-01')
    assert config.background_imported_pressure == .1
    assert config.rain_rate_mm_hour == .45
    assert config.leaf.rank_spacing_units == 80.
    assert config.disease.primary_scale == .001
    provider = WeatherProvider([WeatherDay(date(2026, 10, 1), 10., 12., 90., 2.),
        WeatherDay(date(2026, 10, 2), 10., 12., 90., 2., imported_pressure=0.)])
    simulation = WheatSTBSimulation(config, provider)
    simulation.run()
    assert [day.imported_pressure for day in simulation.outputs] == [.1, 0.]


def test_external_functional_yield_integral_has_explicit_area_and_no_invented_reference_yield():
    config = synthetic_configuration(sowing_date='2026-10-01',
        yield_model=dict(mode='external_functional_loss'))
    provider = WeatherProvider([WeatherDay(date(2026, 10, 1)+timedelta(days=i), 1., 2., 50., 0.,
        reference_leaf_area_index=(.3, .4, .3), functional_loss_fraction=(.2, .2, .2)) for i in range(12)])
    simulation = WheatSTBSimulation(config, provider)
    simulation.run()
    summary = simulation.finalize().summary
    assert summary['complete_grain_fill_window']
    assert summary['reference_had3'] == 4.
    np.testing.assert_allclose(summary['lost_had3'], .8)
    np.testing.assert_allclose(summary['conditional_yield_loss_t_ha'], [.01128, .0144, .01656])
    assert summary['conditional_relative_loss_percent'] is None
    assert summary['actual_field_yield_forecast'] is None


def test_composed_daily_engine_matches_shared_batch_kernel_exactly():
    from model.seasonal_septoria.leaf_phenology import leaf_host
    from model.seasonal_septoria.overwinter import simulate_overwinter
    from model.seasonal_septoria.wetness import duration_exposure
    config, provider = configuration(crop_end_date='2026-12-01'), weather(100)
    raw = {key: value for key, value in config.to_dict()['phenology'].items()}
    t = np.array([[row.tmean_c for row in provider]])
    tx = np.array([[row.tmax_c for row in provider]])
    rh = np.array([[row.rh_mean_pct for row in provider]])
    rain = np.array([[row.precipitation_mm for row in provider]])
    valid = np.array([[config.sowing_date <= row.date <= config.crop_end_date for row in provider]])
    accumulation = tpv_accumulation(t, tx, [config.latitude],
        np.array([row.date.timetuple().tm_yday for row in provider]), valid, raw)
    host = leaf_host(accumulation, t, dict(config.leaf.stage_thresholds),
        rank_spacing_units=config.leaf.rank_spacing_units, forcing_mask=valid,
        juvenile_policy=config.leaf.juvenile_policy)
    exposure = duration_exposure(t, tx, rh, rain, config.rain_rate_mm_hour)['exposure']
    batch = simulate_overwinter(t, exposure, rain, host.active, host.renewal, host.area,
        valid, config.disease, initial_local_source=config.initial_local_source,
        imported_pressure=np.full_like(t, config.background_imported_pressure), time_step=config.time_step)
    simulation = WheatSTBSimulation(config, provider)
    simulation.run()
    np.testing.assert_array_equal(simulation.disease.state, batch.state[0, -1])
    np.testing.assert_array_equal(simulation.disease.residue, batch.residue[0, -1])
    np.testing.assert_array_equal(np.array([day.local_flow for day in simulation.outputs]), batch.local_flow[0])


def test_missing_external_area_day_cannot_partially_advance_components():
    config = synthetic_configuration(sowing_date='2026-10-01', yield_model=dict(mode='external_functional_loss'))
    rows = [WeatherDay(date(2026, 10, 1)+timedelta(days=i), 1., 2., 50., 0.,
        reference_leaf_area_index=(.3, .4, .3), functional_loss_fraction=(.2, .2, .2)) for i in range(12)]
    rows[6] = WeatherDay(date(2026, 10, 7), 1., 2., 50., 0.)
    simulation = WheatSTBSimulation(config, WeatherProvider(rows))
    simulation.run(6)
    before = simulation.snapshot()
    with pytest.raises(ValueError, match='external reference LAI'):
        simulation.step()
    assert simulation.snapshot() == before


def test_component_integration_failure_rolls_back_the_complete_day():
    simulation = WheatSTBSimulation(configuration(), weather(20))
    simulation.initialize()
    before = simulation.snapshot()
    rates = simulation.calc_rates()
    bad_yield = replace(rates.yield_model,
        next_state=replace(rates.yield_model.next_state, lost_had3=-1.))
    with pytest.raises(ValueError, match='yield|HAD'):
        simulation.integrate(replace(rates, yield_model=bad_yield))
    assert simulation.snapshot() == before
    assert simulation.step().date == weather(20)[0].date


def test_latent_infection_is_separate_from_symptoms_and_dates_follow_actual_days():
    from dataclasses import asdict
    from model.seasonal_septoria.overwinter import OverwinterParameters
    disease = asdict(OverwinterParameters(primary_scale=1., secondary_scale=0.,
        latent_reference_days=1., latent_stages=3))
    config = synthetic_configuration(sowing_date='2026-10-01', disease=disease,
        background_imported_pressure=1., detection_fraction=1e-6, time_step=1.)
    provider = WeatherProvider([WeatherDay(date(2026, 10, 1)+timedelta(days=i),
        1., 2., 100., 2., exposure_override=1.) for i in range(12)])
    simulation = WheatSTBSimulation(config, provider)
    simulation.run(5)
    infected = simulation.outputs[4]
    assert infected.crop_included
    assert infected.latent_fraction[0] > 0.
    assert infected.symptomatic_fraction[0] == 0.
    assert infected.affected_fraction[0] == infected.latent_fraction[0]
    np.testing.assert_allclose(np.array(infected.susceptible_fraction)
        +np.array(infected.latent_fraction)+np.array(infected.symptomatic_fraction), 1.)
    simulation.run()
    summary = simulation.finalize().summary
    # F1 visible at day4, becomes available at day5. A one-day step can
    # traverse only one of the three latent compartments on each later day.
    assert summary['first_infection_date_by_leaf']['F1'] == '2026-10-05'
    assert summary['first_symptom_date_by_leaf']['F1'] == '2026-10-08'
    assert summary['stage_dates']['31'] == '2026-10-02'
    assert summary['stage_dates']['32'] == '2026-10-03'
    assert summary['stage_dates']['33'] == '2026-10-04'
    assert summary['stage_dates']['37'] == '2026-10-04'
    assert summary['stage_dates']['39'] == '2026-10-05'
    assert summary['stage_dates']['65'] == '2026-10-07'
    assert summary['stage_dates']['85'] == '2026-10-10'
    assert not summary['model_event_dates_are_observed_infection_dates']
    frame = simulation.finalize().daily_frame()
    for variable in ['susceptible', 'latent', 'symptomatic', 'affected']:
        assert f'{variable}_fraction_F1' in frame


def test_unreached_events_are_null_and_outside_crop_days_do_not_create_events():
    simulation = WheatSTBSimulation(synthetic_configuration(sowing_date='2026-10-01'),
        WeatherProvider([WeatherDay(date(2026, 9, 20)+timedelta(days=i),
            1., 2., 100., 2., imported_pressure=10., exposure_override=1.) for i in range(10)]))
    simulation.run()
    summary = simulation.finalize().summary
    assert all(value is None for value in summary['first_infection_date_by_leaf'].values())
    assert all(value is None for value in summary['first_symptom_date_by_leaf'].values())
    assert all(value is None for value in summary['stage_dates'].values())
    assert all(not day.crop_included for day in simulation.outputs)


@pytest.mark.parametrize('cutoff', [0., -1., 1.01, float('nan')])
def test_invalid_detection_cutoff_is_rejected(cutoff):
    with pytest.raises(ValueError, match='detection'):
        configuration(detection_fraction=cutoff)
