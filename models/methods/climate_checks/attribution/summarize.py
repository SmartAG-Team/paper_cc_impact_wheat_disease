"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/review_response_20261010/attribution')
import numpy as np
import pandas as pd
from analysis.methods.climate_checks.attribution.utilities import components, domain_ratio, offset_percent
CORNER_ORDER = ['Wreference_Hreference', 'Wfuture_Hreference', 'Wreference_Hfuture', 'Wfuture_Hfuture']
COMPONENTS = ['weather', 'host', 'interaction', 'net', 'weather_alone', 'host_alone']

def variance(values, draws):
    return sum((float(g.area_mean_weight.sum()) ** 2 * np.var(values[g.index.to_numpy()], ddof=1) / len(g) for _, g in draws.groupby('stratum', sort=False)))

def summarize(corners, draws, models, regions, metrics):
    """Returns wide per-GCM/ensemble summaries and marginal-diagonal checks."""
    draws = draws.sort_values('spatial_draw_id').reset_index(drop=True)
    settings = sorted(corners.setting.unique())
    years = sorted(corners.reference_year.unique())
    n = len(draws)
    ny = len(years)
    weights = draws.area_mean_weight.to_numpy(float)
    records = []
    ensemble = []
    diagnostics = []
    influences = {}
    values = {}
    supports = {}
    diagonal = {}
    expected = pd.MultiIndex.from_product([draws.spatial_draw_id, years], names=['spatial_draw_id', 'reference_year'])
    for setting in settings:
        for model in models:
            part = corners[corners.setting.eq(setting) & corners.model.eq(model)]
            arrays = []
            excluded = []
            for name in CORNER_ORDER:
                frame = part[part.corner.eq(name)].set_index(['spatial_draw_id', 'reference_year'])
                if frame.index.has_duplicates or set(frame.index) != set(expected):
                    raise ValueError('Incomplete or duplicate four-corner support.')
                frame = frame.reindex(expected)
                arrays.append(frame[metrics].to_numpy(float).reshape(n, ny, len(metrics)))
                excluded.append(frame.excluded_unsupported_hybrid_forcing.to_numpy(bool).reshape(n, ny))
            if not all((np.array_equal(excluded[0], x) for x in excluded)):
                raise ValueError('Corner exclusion flags disagree.')
            for j, metric in enumerate(metrics):
                result = components(*[x[:, :, j] for x in arrays], supported=~excluded[0])
                values[setting, model, metric] = result
                supports[setting, model, metric] = result['common']
                diagonal[setting, model, metric] = [arrays[k][:, :, j] for k in [0, 3]]
                error = np.abs(result['weather'] + result['host'] - result['net'])
                assert not np.isfinite(error).any() or np.nanmax(error) < 1e-10
                for label, array in zip(['reference', 'future'], diagonal[setting, model, metric]):
                    mean, _, coverage = domain_ratio(array, np.isfinite(array), weights, np.ones(n))
                    diagnostics.append(dict(setting=setting, model=model, metric=metric, corner=label, mean_on_own_finite_support=mean, coverage=coverage))
    for scope in ['within_setting', 'common_all_settings']:
        for setting in settings:
            for model in models:
                for metric in metrics:
                    result = values[setting, model, metric]
                    common = supports[setting, model, metric].copy()
                    if scope == 'common_all_settings':
                        common = np.logical_and.reduce([supports[s, model, metric] for s in settings])
                    for region in regions:
                        indicator = np.ones(n) if region == 'Europe' else draws.environment_region.eq(region).to_numpy(float)
                        selected = indicator.astype(bool)
                        count = int(selected.sum())
                        unique = int(draws.loc[selected, 'cell_id'].nunique())
                        record = dict(setting=setting, model=model, metric=metric, environment_region=region, support_scope=scope, sample_draws=count, unique_sampled_cells=unique, valid_year_pairs=int((common & selected[:, None]).sum()), valid_draws=int((common.any(axis=1) & selected).sum()), valid_unique_cells=int(draws.loc[common.any(axis=1) & selected, 'cell_id'].nunique()))
                        for component in COMPONENTS:
                            mean, inf, coverage = domain_ratio(result[component], common, weights, indicator)
                            record[component] = mean
                            record[component + '_spatial_mcse'] = float(np.sqrt(variance(inf, draws))) if count >= 4 and np.isfinite(mean) else np.nan
                            influences[scope, setting, model, metric, region, component] = inf
                        region_weight = float(weights @ indicator)
                        record['common_area_time_weight'] = coverage
                        record['within_region_coverage'] = coverage / region_weight if region_weight > 0 else np.nan
                        record['host_offset_percent'] = offset_percent(record['weather'], record['host'])
                        records.append(record)
    by_gcm = pd.DataFrame(records)
    keys = ['setting', 'metric', 'environment_region', 'support_scope']
    for key, part in by_gcm.groupby(keys, sort=False):
        setting, metric, region, scope = key
        complete = len(part) == len(models) and np.isfinite(part[COMPONENTS].to_numpy()).all()
        row = dict(zip(keys, key))
        row.update(sample_draws=int(part.sample_draws.iloc[0]), unique_sampled_cells=int(part.unique_sampled_cells.iloc[0]), minimum_gcm_valid_pairs=int(part.valid_year_pairs.min()), maximum_gcm_valid_pairs=int(part.valid_year_pairs.max()), minimum_gcm_valid_draws=int(part.valid_draws.min()), maximum_gcm_valid_draws=int(part.valid_draws.max()), minimum_gcm_valid_unique_cells=int(part.valid_unique_cells.min()), maximum_gcm_valid_unique_cells=int(part.valid_unique_cells.max()), minimum_gcm_coverage=float(part.within_region_coverage.min()), maximum_gcm_coverage=float(part.within_region_coverage.max()))
        for component in COMPONENTS:
            row[component] = float(part[component].mean()) if complete else np.nan
            row[component + '_gcm_min'] = float(part[component].min()) if complete else np.nan
            row[component + '_gcm_max'] = float(part[component].max()) if complete else np.nan
            inf = np.mean([influences[scope, setting, model, metric, region, component] for model in models], axis=0)
            row[component + '_spatial_mcse'] = float(np.sqrt(variance(inf, draws))) if complete and row['sample_draws'] >= 4 else np.nan
        row['host_offset_percent'] = offset_percent(row['weather'], row['host'])
        row['offset_gcm_min'] = float(part.host_offset_percent.min()) if complete else np.nan
        row['offset_gcm_max'] = float(part.host_offset_percent.max()) if complete else np.nan
        ensemble.append(row)
    return (by_gcm, pd.DataFrame(ensemble), pd.DataFrame(diagnostics))
