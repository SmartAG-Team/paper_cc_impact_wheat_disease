"""Read-only mathematical and chronology audit of the parent epidemic model."""
import os
os.environ['NUMBA_DISABLE_JIT'] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
from datetime import datetime, timezone
import hashlib
import importlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit():
    snapshot = HERE / 'parent_model_audit_source'
    snapshot.mkdir(exist_ok=True)
    parents = ['__init__.py', 'core.py', 'field_data.py', 'calibrate.py', 'SPEC.txt']
    source_hashes = {}
    for filename in parents:
        source = PROJECT / 'model/primary_secondary' / filename
        shutil.copyfile(source, snapshot / filename)
        source_hashes[str(source.relative_to(PROJECT))] = digest(source)
        assert digest(source) == digest(snapshot / filename)
    contract = PROJECT / 'analysis/primary_secondary/evaluation_contract.json'
    shutil.copyfile(contract, snapshot / 'evaluation_contract.json')
    source_hashes[str(contract.relative_to(PROJECT))] = digest(contract)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(HERE))
    core = importlib.import_module('parent_model_audit_source.core')
    field = importlib.import_module('parent_model_audit_source.field_data')
    calibration = importlib.import_module('parent_model_audit_source.calibrate')
    passed = []

    def check(name, condition):
        if not condition:
            raise AssertionError(name)
        passed.append(name)

    rng = np.random.default_rng(20261004)
    temperature = rng.uniform(-5, 32, (3, 40))
    humidity = rng.uniform(0, 24, (3, 40))
    rain = rng.uniform(0, 15, (3, 40))
    visible = np.array([[.1, .2, 0], [.9, 0, .1], [0, .05, .3]])
    active = np.array([[True, True, False], [True, False, True], [True, True, True]])
    p = core.Parameters(alpha=.12, beta=4.2)
    trajectory = core.simulate(visible, temperature, humidity, rain, p, initial_latent=.6,
        leaf_ranks=[1, 3, 7], leaf_active=active)
    check('mass_conservation', np.max(abs(trajectory.state.sum(axis=-1) - 1)) < 1e-12)
    check('nonnegative_bounded_compartments', trajectory.state.min() >= -1e-14 and trajectory.state.max() <= 1 + 1e-14)
    check('pycnidia_never_exceed_damage', np.all(trajectory.pycnidia <= trajectory.damage + 1e-14))
    check('infectious_never_exceed_pycnidia', np.all(trajectory.infectious <= trajectory.pycnidia + 1e-14))
    check('damage_and_pycnidia_monotone_by_construction',
        np.all(np.diff(trajectory.damage, axis=1) >= -1e-14) and
        np.all(np.diff(trajectory.pycnidia, axis=1) >= -1e-14))
    check('initial_origin_total_conserved', np.max(abs(np.diff(trajectory.origin_total[..., 0], axis=1))) < 1e-12)
    for origin, flux in [(1, trajectory.primary_flux), (2, trajectory.secondary_flux)]:
        check(f'origin{origin}_daily_increment_equals_infection_flux',
            np.max(abs(np.diff(trajectory.origin_total[..., origin], axis=1) - flux)) < 1e-12)
    for episode, leaf in zip(*np.where(~active)):
        check(f'inactive_rank_constant_S1_episode{episode}_leaf{leaf}',
            np.all(trajectory.state[episode, :, leaf, 0] == 1) and
            np.all(trajectory.state[episode, :, leaf, 1:] == 0))
    t = np.full((1, 50), 18.)
    h = np.full_like(t, 24.)
    r = np.full_like(t, 2.)
    null = core.simulate([[0]], t, h, r, core.Parameters(), initial_latent=0)
    check('no_infection_without_any_source', np.all(null.origin_total == 0))
    primary = core.simulate([[0]], t, h, r, core.Parameters(alpha=.1), initial_latent=0)
    check('primary_only_preserves_origin_labels',
        primary.primary_flux.sum() > 0 and np.all(primary.secondary_flux == 0) and np.all(primary.origin_total[..., 0] == 0))
    no_splash = core.simulate([[.2]], t, h, np.zeros_like(r), core.Parameters(beta=3.))
    check('weather_enabled_secondary_requires_rain', np.all(no_splash.secondary_flux == 0))
    removal = core.simulate([[.2]], t, h, r, core.Parameters())
    check('infectious_removal_has_exact_calendar_day_exponential',
        np.max(abs(removal.infectious[0, :, 0] - .2 * np.exp(-np.arange(51) / 21))) < 1e-12)
    check('removal_retains_historical_damage_and_pycnidia',
        np.max(abs(removal.damage - .2)) < 1e-12 and np.max(abs(removal.pycnidia - .2)) < 1e-12)
    single = core.simulate([[.2]], t, h, r, core.Parameters(alpha=.1, beta=3.), initial_latent=.3, leaf_ranks=[1])
    masked = core.simulate([[.2, 0]], t, h, r, core.Parameters(alpha=.1, beta=3.),
        initial_latent=.3, leaf_ranks=[1, 7], leaf_active=[[True, False]])
    check('inactive_ranks_do_not_dilute_or_supply_transmission',
        np.max(abs(single.damage[..., 0] - masked.damage[..., 0])) < 1e-12)
    records = pd.DataFrame({'TrialId': ['X'] * 4, 'leaf_rank': [1, 1, 2, 2],
        'Date': pd.to_datetime(['2019-05-01', '2019-05-10', '2019-05-08', '2019-05-20']),
        'Value': [5., 90., 80., 100.]})
    first_v, first_a = field.canopy_snapshot(records, 'X', pd.Timestamp('2019-05-01'), [1, 2])
    perturbed = records.copy()
    perturbed.loc[perturbed.Date > pd.Timestamp('2019-05-01'), 'Value'] = 0.
    second_v, second_a = field.canopy_snapshot(perturbed, 'X', pd.Timestamp('2019-05-01'), [1, 2])
    check('future_disease_value_perturbation_does_not_change_initialization',
        np.array_equal(first_v, second_v) and np.array_equal(first_a, second_a))
    check('unobserved_future_rank_inactive_at_episode_start',
        np.array_equal(first_v, [.05, 0]) and np.array_equal(first_a, [True, False]))
    batch = field.prepare_basf(PROJECT)
    scored = batch.targets.loc[~batch.targets.conditioning]
    check('target_initial_points_are_excluded_from_scoring', bool(scored.day.gt(0).all()))
    check('active_initial_target_matches_source_first_assessment',
        np.allclose(batch.initial_visible[np.arange(len(batch.episodes)), batch.episodes.target_leaf_index],
            batch.episodes.initial_percent / 100))
    check('one_selected_source_year_per_episode',
        batch.targets.groupby('series_id').year.nunique().eq(1).all())
    folds = []
    repeated_location_cases = []
    for fold in calibration.make_folds(batch.episodes):
        train, test = batch.subset(fold['train']), batch.subset(fold['test'])
        overlap = set(train.episodes.location_id) & set(test.episodes.location_id)
        check(f"{fold['name']}_no_training_test_location_overlap", not overlap)
        if fold['type'] == 'forward_year':
            check(f"{fold['name']}_all_train_target_dates_precede_test_year",
                train.targets.Date.max().year < test.targets.Date.min().year)
        years_per_location = test.episodes.groupby('location_id').year.nunique()
        repeated = int(years_per_location.gt(1).sum())
        if repeated:
            repeated_location_cases.append(dict(fold=fold['name'], repeated_locations=repeated,
                coordinate_year_clusters=test.episodes.coordinate_year.nunique(),
                independent_location_clusters=test.episodes.location_id.nunique()))
        folds.append(dict(name=fold['name'], train_series=len(train.episodes), test_series=len(test.episodes),
            train_locations=train.episodes.location_id.nunique(), test_locations=test.episodes.location_id.nunique(),
            repeated_test_locations_across_years=repeated))
    for i, (train_ids, test_ids) in enumerate(calibration.inner_folds(batch.episodes)):
        train, test = batch.subset(train_ids), batch.subset(test_ids)
        check(f'inner{i}_locations_and_all_years_grouped',
            not set(train.episodes.location_id) & set(test.episodes.location_id))
    test_frame = scored[['coordinate_year', 'series_id', 'Value']].copy()
    test_frame['observed_percent'] = test_frame.Value
    test_frame['predicted_percent'] = test_frame.Value + np.arange(len(test_frame)) % 11 - 5
    weights = field.hierarchical_weights(scored)
    expected_rmse = np.sqrt(np.sum(weights * (test_frame.predicted_percent - test_frame.observed_percent) ** 2))
    check('fitting_weights_match_declared_hierarchical_score',
        np.isclose(expected_rmse, calibration.score(test_frame)['rmse_pp']))
    per_series = scored.groupby('series_id').size()
    drops = []
    for series, frame in batch.targets.groupby('series_id'):
        decrease = np.diff(frame.sort_values('Date').Value.to_numpy())
        if (decrease < 0).any():
            drops.append(dict(series_id=series, negative_increments=int((decrease < 0).sum()),
                largest_decrease_pp=float(-decrease.min())))
    pd.DataFrame(drops).to_csv(HERE / 'parent_model_decreasing_observed_series.csv', index=False)
    cold = core.simulate([[0]], np.full((1, 40), 5.), h[:, :40], r[:, :40],
        core.Parameters(weather_response=False), initial_latent=.5)
    warm = core.simulate([[0]], np.full((1, 40), 25.), h[:, :40], r[:, :40],
        core.Parameters(weather_response=False), initial_latent=.5)
    no_weather_difference = float(np.max(abs(cold.damage - warm.damage)) * 100)
    finer = core.simulate(visible, temperature, humidity, rain, p, initial_latent=.6,
        leaf_ranks=[1, 3, 7], leaf_active=active, time_step=.125)
    finer_difference = float(np.max(abs(trajectory.damage - finer.damage)) * 100)
    dt = .25
    latent_effective = p.latent_stages * dt / (-np.expm1(-dt * p.latent_stages / p.latent_days))
    n_effective = dt / (-np.expm1(-dt / p.nonsporulating_days))
    evidence = dict(status='verified_with_scientific_limitations', completed_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=source_hashes,
        source_files_unchanged_during_audit=all(digest(PROJECT / name) == value for name, value in source_hashes.items()),
        source_model_mutated=False, tests_passed=len(passed), named_checks=passed,
        maximum_mass_error=float(abs(trajectory.state.sum(axis=-1) - 1).max()),
        minimum_state=float(trajectory.state.min()),
        synthetic_quarter_vs_eighth_day_damage_difference_pp=finer_difference,
        latent_mean_at18c_fresh_stage0_finite_quarter_day=latent_effective,
        nonsporulating_mean_at18c_finite_quarter_day=n_effective,
        combined_fresh_latent_to_infectious_mean_at18c=latent_effective + n_effective,
        no_weather_variant_cold_warm_damage_difference_pp=no_weather_difference,
        episodes=len(batch.episodes), future_targets=len(scored), folds=folds,
        bootstrap_repeated_location_cases=repeated_location_cases,
        decreasing_observed_series=len(drops), decreasing_observed_transitions=sum(d['negative_increments'] for d in drops),
        longest_episode_days=int(batch.targets.day.max()),
        actual_episode_lengths=(batch.episodes.end - batch.episodes.start).dt.days.describe().to_dict(),
        fixed_parameter_contract=json.loads(contract.read_text()),
        limitations=[
            'No-weather variant retains thermal latent and nonsporulating development.',
            'Initial visible tissue is all infectious; N/R stage composition is not identified by the initial observation.',
            'Coordinate-year bootstrap separates repeated years at the same location in country folds.',
            'Damage and historical pycnidia are monotone; observed within-rank declines cannot be represented.',
            'Quarter-day serial-stage progression has geometric waiting times and finite-step delay; gamma/Erlang interpretation is a continuous-time limit.',
            'Unassessed ranks are inactive and have zero latent infection, differing from one sentence in the current SPEC.',
            'Padding beyond an episode end uses18degC, zero humidity and rain; only actual target dates are scored.',
            'Origin attribution is conditional simulated tissue, not measured source contribution or ascospore arrival.'
        ])
    (HERE / 'parent_model_audit.json').write_text(json.dumps(evidence, indent=2, allow_nan=False) + '\n')
    return evidence


if __name__ == '__main__':
    report = audit()
    print(json.dumps({key: report[key] for key in ['status', 'tests_passed', 'episodes', 'future_targets',
        'source_files_unchanged_during_audit', 'no_weather_variant_cold_warm_damage_difference_pp',
        'bootstrap_repeated_location_cases', 'decreasing_observed_series', 'decreasing_observed_transitions']}, indent=2))
