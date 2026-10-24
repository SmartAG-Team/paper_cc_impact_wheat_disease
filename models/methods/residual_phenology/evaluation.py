"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_residual_ml_20261012')
import numpy as np
import pandas as pd
from analysis.methods.wheat_phenology.wheat_transfer import quantiles, STAGES
from analysis.methods.residual_phenology.residual import crps_values
METRICS = ('stage_equal_crps', 'stage_equal_mae', 'terminal_mae', 'terminal_bias', 'duration_mae', 'duration_bias')

def evaluate(cdfs, targets, cycles, references=('baseline', 'archived_tpv', 'wheat_refit_tpv'), replicates=10000, seed=20261012):
    targets = np.asarray(targets)
    if targets.shape != (len(cycles), 4) or set(references) - set(cdfs):
        raise ValueError('Complete observation/reference dimensions are required')
    if cycles.cycle_id.duplicated().any() or cycles.PEP_ID.isna().any() or (not np.isfinite(targets).all()):
        raise ValueError('Unique cycles and complete stations/targets are required')
    for cdf in cdfs.values():
        if cdf.shape[:2] != targets.shape or not np.isfinite(cdf).all() or np.any(cdf < 0) or np.any(cdf > 1) or np.any(np.diff(cdf, axis=2) < -1e-06) or np.any(np.diff(cdf, axis=1) > 1e-06):
            raise ValueError('Invalid stage CDF')
    medians = {name: quantiles(cdf, levels=(0.5,))[:, :, 0] for name, cdf in cdfs.items()}
    observed = targets >= 0
    support = observed & np.logical_and.reduce([np.isfinite(medians[name]) for name in references])
    pairs = support[:, 2] & support[:, 3]
    scores = {name: crps_values(cdf, targets) for name, cdf in cdfs.items()}
    errors = {name: pred - targets for name, pred in medians.items()}
    duration = {name: pred[:, 3] - pred[:, 2] - (targets[:, 3] - targets[:, 2]) for name, pred in medians.items()}
    cohorts = {'all': np.ones(len(cycles), dtype=bool), 'reserved_station': cycles.historical_station_group.eq('test').to_numpy()}
    metrics, stages, events, intervals = ([], [], [], [])
    for name, cdf in cdfs.items():
        for j, stage in enumerate(STAGES):
            frame = cycles.copy().reset_index(drop=True)
            frame['model'] = name
            frame['stage'] = stage
            frame['observed_day'] = targets[:, j]
            frame['observed'] = observed[:, j]
            frame['median_day'] = medians[name][:, j]
            frame['fixed_reference_support'] = support[:, j]
            frame['error_days'] = np.where(observed[:, j], errors[name][:, j], np.nan)
            frame['restricted_crps_days'] = scores[name][:, j]
            frame['tail_probability'] = 1 - cdf[:, j, -1]
            frame['lost_reference_crossing'] = support[:, j] & ~np.isfinite(medians[name][:, j])
            events.append(frame)
        for cohort, mask in cohorts.items():
            mae, bias, crps = ([], [], [])
            for j, stage in enumerate(STAGES):
                common, known = (support[:, j] & mask, observed[:, j] & mask)
                mae.append(float(np.mean(np.abs(errors[name][common, j]))) if common.any() else np.nan)
                bias.append(float(np.mean(errors[name][common, j])) if common.any() else np.nan)
                crps.append(float(np.mean(scores[name][known, j])) if known.any() else np.nan)
                stages.append(dict(cohort=cohort, model=name, stage=stage, observed_events=int(known.sum()), fixed_reference_events=int(common.sum()), mae=mae[-1], bias=bias[-1], crps=crps[-1], unresolved=int((known & ~np.isfinite(medians[name][:, j])).sum()), beyond_horizon=int((known & (targets[:, j] >= cdf.shape[2])).sum())))
            use = pairs & mask
            metrics.append(dict(cohort=cohort, model=name, cycles=int(mask.sum()), stations=int(cycles.loc[mask, 'PEP_ID'].nunique()), observed_events=int((observed & mask[:, None]).sum()), common_resolved_events=int((support & mask[:, None]).sum()), lost_reference_events=int((support & mask[:, None] & ~np.isfinite(medians[name])).sum()), unresolved_observed_events=int((observed & mask[:, None] & ~np.isfinite(medians[name])).sum()), stage_equal_crps=float(np.mean(crps)), stage_equal_mae=float(np.mean(mae)), terminal_mae=mae[3], terminal_bias=bias[3], duration_pairs=int(use.sum()), duration_mae=float(np.mean(np.abs(duration[name][use]))) if use.any() else np.nan, duration_bias=float(np.mean(duration[name][use])) if use.any() else np.nan))
    for cohort, mask in cohorts.items():
        if not mask.any():
            continue
        local_cycles = cycles.loc[mask].reset_index(drop=True)
        station_index, stations = pd.factorize(local_cycles.PEP_ID, sort=True)
        columns = [observed[mask], support[mask], pairs[mask, None]]
        names = list(cdfs)
        for name in names:
            columns.extend([np.where(observed[mask], scores[name][mask], 0), np.where(support[mask], np.abs(errors[name][mask]), 0), np.where(support[mask], errors[name][mask], 0), np.where(pairs[mask], np.abs(duration[name][mask]), 0)[:, None], np.where(pairs[mask], duration[name][mask], 0)[:, None]])
        per_cycle = np.concatenate(columns, axis=1).astype(float)
        per_station = np.zeros((len(stations), per_cycle.shape[1]))
        np.add.at(per_station, station_index, per_cycle)

        def summarize(total):
            result = np.full((len(total), len(names), 6), np.nan)
            with np.errstate(divide='ignore', invalid='ignore'):
                for m, name in enumerate(names):
                    offset = 9 + 14 * m
                    result[:, m, 0] = np.mean(total[:, offset:offset + 4] / total[:, :4], axis=1)
                    absolute = total[:, offset + 4:offset + 8] / total[:, 4:8]
                    signed = total[:, offset + 8:offset + 12] / total[:, 4:8]
                    result[:, m, 1] = np.mean(absolute, axis=1)
                    result[:, m, 2] = absolute[:, 3]
                    result[:, m, 3] = signed[:, 3]
                    result[:, m, 4:6] = total[:, offset + 12:offset + 14] / total[:, 8:9]
            return result
        point = summarize(per_station.sum(0, keepdims=True))[0]
        draws = np.empty((replicates if len(stations) > 1 else 0, len(names), 6))
        rng = np.random.default_rng(seed)
        for start in range(0, len(draws), 256):
            size = min(256, len(draws) - start)
            weights = rng.multinomial(len(stations), np.full(len(stations), 1 / len(stations)), size=size)
            draws[start:start + size] = summarize(weights @ per_station)
        a = names.index('selected')
        for reference in references:
            b = names.index(reference)
            for j, metric in enumerate(METRICS):
                difference = draws[:, a, j] - draws[:, b, j]
                valid = difference[np.isfinite(difference)]
                low, high = np.quantile(valid, [0.025, 0.975]) if len(valid) else (np.nan, np.nan)
                intervals.append(dict(cohort=cohort, candidate='selected', reference=reference, metric=metric, difference=point[a, j] - point[b, j], lower95=low, upper95=high, stations=len(stations), cycles=len(local_cycles), valid_replicates=len(valid), requested_replicates=replicates, seed=seed, all_years_per_station_retained=True, uncertainty_scope='Conditional on fitted models and observed years; no development or neighboring-station spatial-block uncertainty'))
    return dict(metrics=pd.DataFrame(metrics), stage_metrics=pd.DataFrame(stages), events=pd.concat(events, ignore_index=True), paired_intervals=pd.DataFrame(intervals))
