"""Independent fixed-baseline production exposure; no grain-loss conversion."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from .provenance import ROOT, auth_source, output_hashes, prepare_destination, write_json

MODELS = ['ACCESS-CM2', 'MPI-ESM1-2-HR', 'MRI-ESM2-0']
METRIC = 'GS65_85_lost_had3'
WEIGHTS = {'total': 'production_total_tonnes', 'rainfed': 'production_rainfed_tonnes'}


def _production(frame):
    if frame.cell_id.duplicated().any() or frame.cell_id.isna().any():
        raise ValueError('Duplicate or missing production cell identity')
    values = frame[list(WEIGHTS.values())].to_numpy(float)
    if not np.isfinite(values).all() or np.any(values < 0) or np.any(values[:, 1] > values[:, 0] + 1e-6):
        raise ValueError('Finite nonnegative and compatible production weights required')


def country_domains(production, fractions):
    _production(production)
    if fractions.duplicated(['cell_id', 'country']).any() or fractions.country.isna().any():
        raise ValueError('Duplicate or missing fractional country identity')
    if set(fractions.cell_id) - set(production.cell_id):
        raise ValueError('Unknown fractional country cell')
    sums = fractions.groupby('cell_id')[list(WEIGHTS.values())].sum().reindex(production.cell_id, fill_value=0)
    if not np.allclose(sums.to_numpy(float), production[list(WEIGHTS.values())].to_numpy(float), rtol=1e-10, atol=1e-5):
        raise ValueError('Fractional country weights do not conserve cell production')
    domains = {}
    for country, part in fractions.groupby('country', sort=True):
        _production(part)
        domains[('country', country)] = part[['cell_id', *WEIGHTS.values()]].copy()
    return domains


def summarize_exposure(changes, production, *, models=MODELS, thresholds=(0., 1., 2., 3., 5.),
                       min_pairs=1, extra_domains=None):
    """All member subsets use the original complete-model cell population."""
    _production(production)
    if not models or len(set(models)) != len(models) or not set(models) <= set(changes.model):
        raise ValueError('Unique available original models required')
    if changes.duplicated(['cell_id', 'model', 'scenario', 'period']).any():
        raise ValueError('Duplicate climate response identity')
    if set(changes.cell_id) - set(production.cell_id):
        raise ValueError('Unknown climate response cell')
    if not 1 <= min_pairs <= 30 or any(not np.isfinite(t) or t < 0 for t in thresholds):
        raise ValueError('Valid coverage count and nonnegative finite thresholds required')
    counts = changes.valid_year_pairs.to_numpy(float)
    if not np.isfinite(counts).all() or np.any(counts < 0) or np.any(counts > 30) or np.any(counts != np.floor(counts)):
        raise ValueError('Integer paired-season support in 0–30 required')
    if np.any(np.isfinite(changes.change) & (counts == 0)):
        raise ValueError('Finite climate response has no paired-season support')
    domains = {('Europe', 'Europe'): production}
    if 'environment_region' in production:
        if production.environment_region.isna().any():
            raise ValueError('Missing region assignment must have an explicit category')
        domains.update({('environment_region', str(name)): part
                        for name, part in production.groupby('environment_region', sort=True)})
    if extra_domains:
        domains.update(extra_domains)
    index = pd.Index(production.cell_id)
    domain_arrays = {}
    for key, cells in domains.items():
        _production(cells)
        positions = index.get_indexer(cells.cell_id)
        if np.any(positions < 0):
            raise ValueError('Unknown domain production cell')
        domain_arrays[key] = (positions, cells[list(WEIGHTS.values())].to_numpy(float))
    rows = []
    member_sets = [list(member) for n in range(len(models), 0, -1) for member in combinations(models, n)]
    for (scenario, period), part in changes.groupby(['scenario', 'period'], sort=True):
        x = part.pivot(index='cell_id', columns='model', values='change').reindex(index=production.cell_id, columns=models)
        n = part.pivot(index='cell_id', columns='model', values='valid_year_pairs').reindex(index=production.cell_id, columns=models)
        values_all, counts_all = x.to_numpy(float), n.to_numpy(float)
        complete = np.isfinite(values_all).all(axis=1) & np.isfinite(counts_all).all(axis=1) & (counts_all >= min_pairs).all(axis=1)
        for members in member_sets:
            columns = [models.index(member) for member in members]
            values = values_all[:, columns]
            continuous_by_domain = {}
            for key, (positions, weights) in domain_arrays.items():
                member_values, member_counts = values[positions], counts_all[positions][:, columns]
                eligible = complete[positions]
                member_means = []
                for basis_index in range(len(WEIGHTS)):
                    effective = np.where(eligible[:, None], weights[:, basis_index, None] * member_counts, 0.)
                    den = effective.sum(axis=0)
                    numerator = (np.where(eligible[:, None], member_values, 0.) * effective).sum(axis=0)
                    member_means.append(np.divide(numerator, den, out=np.full(len(columns), np.nan), where=den > 0))
                continuous_by_domain[key] = member_means
            for threshold in thresholds:
                positive = complete & (values > threshold).all(axis=1)
                negative = complete & (values < -threshold).all(axis=1)
                mixed = complete & ~positive & ~negative
                ensemble = complete & (np.where(complete[:, None], values, 0.).mean(axis=1) > threshold)
                flags = {'positive': positive, 'negative': negative, 'mixed': mixed,
                         'unavailable': ~complete, 'ensemble': ensemble}
                for (domain_type, domain), (positions, weights) in domain_arrays.items():
                    for basis_index, basis in enumerate(WEIGHTS):
                        weight = weights[:, basis_index]
                        total = float(weight.sum())
                        masses = {key: float(weight[mask[positions]].sum()) for key, mask in flags.items()}
                        error = masses['positive'] + masses['negative'] + masses['mixed'] + masses['unavailable'] - total
                        if abs(error) > max(1e-6, total * 1e-12):
                            raise AssertionError('Production exposure categories fail mass conservation')
                        continuous = continuous_by_domain[(domain_type, domain)][basis_index]
                        rows.append(dict(domain_type=domain_type, domain=domain, scenario=scenario, period=period,
                            weight_basis=basis, members='|'.join(members), member_count=len(members),
                            threshold_days=float(threshold), minimum_paired_seasons=min_pairs,
                            baseline_production_tonnes=total, unanimous_increase_tonnes=masses['positive'],
                            unanimous_decrease_tonnes=masses['negative'], mixed_or_zero_tonnes=masses['mixed'],
                            unavailable_tonnes=masses['unavailable'], ensemble_increase_tonnes=masses['ensemble'],
                            unanimous_increase_share_percent=100 * masses['positive'] / total if total else np.nan,
                            ensemble_increase_share_percent=100 * masses['ensemble'] / total if total else np.nan,
                            production_weighted_change_days=float(np.mean(continuous)),
                            model_min_days=float(np.min(continuous)), model_max_days=float(np.max(continuous)),
                            mass_balance_error_tonnes=error,
                            interpretation='Baseline production represented; no tonnes lost'))
    return pd.DataFrame(rows)


def run(output, *, root=ROOT):
    root, output = Path(root), Path(output)
    prepare_destination(output, root)
    relatives = ['data/final_study/evidence/full_grid_climate/results/full_grid_paired_changes_by_gcm.parquet',
                 'data/final_study/evidence/food_security_exposure/production_grid_input.csv',
                 'data/final_study/evidence/food_security_exposure/country_fraction_weights.csv']
    sources = [auth_source(root, path) for path in relatives]
    changes = pd.read_parquet(root / relatives[0], filters=[('metric', '=', METRIC)])
    production = pd.read_csv(root / relatives[1], float_precision='round_trip')
    fractions = pd.read_csv(root / relatives[2], float_precision='round_trip')
    countries = country_domains(production, fractions)
    frames = [summarize_exposure(changes, production, min_pairs=n, extra_domains=countries) for n in (1, 27)]
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(output / 'production_exposure.csv', index=False)
    primary = summary.loc[summary.domain_type.eq('Europe') & summary.members.eq('|'.join(MODELS))
                          & summary.minimum_paired_seasons.eq(1)]
    primary.to_csv(output / 'europe_thresholds.csv', index=False)
    headline = primary.loc[primary.scenario.eq('ssp585') & primary.period.eq('2071-2100')
                          & primary.weight_basis.eq('total') & primary.threshold_days.eq(0)].iloc[0]
    result = dict(status='calculated', metric=METRIC, sources=sources, rows=len(summary),
                  scenario_periods=int(changes[['scenario', 'period']].drop_duplicates().shape[0]),
                  original_models=MODELS, fixed_original_member_population=True,
                  baseline_production_million_tonnes=headline.baseline_production_tonnes / 1e6,
                  unanimous_increase_million_tonnes=headline.unanimous_increase_tonnes / 1e6,
                  unanimous_increase_share_percent=headline.unanimous_increase_share_percent,
                  maximum_mass_balance_error_tonnes=float(summary.mass_balance_error_tonnes.abs().max()),
                  physiological_loss_estimated=False, future_production_estimated=False,
                  outputs_sha256=output_hashes(output))
    write_json(output / 'receipt.json', result)
    return result
