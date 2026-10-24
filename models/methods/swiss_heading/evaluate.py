"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/fip_heading_adaptation_20261011')
import numpy as np
import pandas as pd
from analysis.methods.swiss_heading.models import per_record_crps

def score_labels(labels, registry, predictions, *, phase):
    rows = []
    year_index = {int(y): i for i, y in enumerate(registry.year)}
    for model, curves in predictions.items():
        errors = per_record_crps(labels, registry, curves)
        horizon = curves.shape[-1]
        for position, row in enumerate(labels.itertuples()):
            curve = curves[year_index[row.year]]
            quantiles = []
            for probability in [0.1, 0.5, 0.9]:
                hits = np.flatnonzero(curve >= probability)
                quantiles.append(float(hits[0]) if len(hits) else np.nan)
            lower, median, upper = quantiles
            censored = bool(np.isnan(median))
            error = median - row.elapsed_day
            lower_bound = max(0, horizon - row.elapsed_day) if censored else abs(error)
            identified = bool(row.elapsed_day < horizon or np.isfinite(upper))
            if np.isfinite(upper) and row.elapsed_day > upper:
                covered = False
            elif row.elapsed_day < horizon:
                covered = bool(np.isfinite(lower) and lower <= row.elapsed_day and (not np.isfinite(upper) or row.elapsed_day <= upper))
            else:
                covered = np.nan
            rows.append(dict(phase=phase, model=model, plot_id=row.plot_id, year=int(row.year), genotype_id=str(row.genotype_id), heading_date=row.heading_date, observed_elapsed_day=int(row.elapsed_day), crps=float(errors[position]), median_elapsed_day=median, median_censored=censored, median_error=error, absolute_error=abs(error), squared_error=error ** 2, absolute_error_lower_bound=float(lower_bound), lower80_elapsed_day=lower, upper80_elapsed_day=upper, lower80_censored=bool(np.isnan(lower)), upper80_censored=bool(np.isnan(upper)), coverage_identified=identified, covered80=covered))
    return pd.DataFrame(rows)

def summarize(frame):
    rows = []
    for (phase, model), part in frame.groupby(['phase', 'model'], sort=True):
        for year, group in [('pooled', part), *list(part.groupby('year', sort=True))]:

            def mean(column):
                values = group[column].astype(float)
                return float(values.groupby(group.year).mean().mean()) if values.notna().any() else np.nan
            rows.append(dict(phase=phase, model=model, year=str(year), observations=len(group), genotypes=group.genotype_id.nunique(), environment_years=group.year.nunique(), crps=mean('crps'), median_finite=int((~group.median_censored).sum()), median_unreached=int(group.median_censored.sum()), mae_finite_days=mean('absolute_error'), rmse_finite_days=np.sqrt(mean('squared_error')), bias_finite_days=mean('median_error'), mae_lower_bound_days=mean('absolute_error_lower_bound'), central80_coverage=mean('covered80'), coverage_identified=int(group.coverage_identified.sum()), lower80_unreached=int(group.lower80_censored.sum()), upper80_unreached=int(group.upper80_censored.sum())))
    return pd.DataFrame(rows)

def paired_bootstrap(scores, selected, *, replicates=2000, seed=20261011):
    frame = scores[scores.phase.eq('evaluation')]
    candidate = frame[frame.model.eq(selected)].set_index('plot_id').sort_index()
    identity = ['year', 'genotype_id', 'observed_elapsed_day']
    if candidate.empty or candidate.index.duplicated().any():
        raise ValueError('Unique matched evaluation records required')
    references = ['frozen', 'calendar_das', 'calendar_doy']
    paired = {}
    for reference in references:
        other = frame[frame.model.eq(reference)].set_index('plot_id').sort_index()
        if other.index.duplicated().any() or not candidate.index.equals(other.index) or (not candidate[identity].equals(other[identity])):
            raise ValueError('Exactly matched observation support required')
        paired[reference] = other
    genotypes = sorted(candidate.genotype_id.unique())
    years = sorted(candidate.year.unique())
    membership = np.array([candidate[candidate.year.eq(y)].groupby('genotype_id').size().reindex(genotypes, fill_value=0) for y in years], float)
    rng = np.random.default_rng(seed)
    draws, attempted = ([], 0)
    while len(draws) < replicates and attempted < 100000:
        sample = rng.multinomial(len(genotypes), np.full(len(genotypes), 1 / len(genotypes)))
        attempted += 1
        if np.all(membership @ sample > 0):
            draws.append(sample)
    if len(draws) != replicates:
        raise ValueError('Insufficient whole-genotype draws retaining every evaluation year')
    draws = np.asarray(draws, float)
    denominators = draws @ membership.T
    summaries, replicate_rows = ([], [])
    for reference, other in paired.items():
        summary = dict(candidate=selected, reference=reference, observations=len(candidate), genotypes=len(genotypes), environment_years=len(years), replicates=replicates)
        output = pd.DataFrame({'replicate': np.arange(replicates), 'candidate': selected, 'reference': reference})
        for column, name in [('crps', 'crps_difference'), ('absolute_error_lower_bound', 'mae_lower_bound_difference')]:
            difference = candidate[column] - other[column]
            if not np.isfinite(difference).all():
                raise ValueError('Finite paired errors required')
            sums = np.array([difference[candidate.year.eq(y)].groupby(candidate.genotype_id).sum().reindex(genotypes, fill_value=0) for y in years], float)
            values = np.mean(draws @ sums.T / denominators, axis=1)
            summary[name] = float(difference.groupby(candidate.year).mean().mean())
            summary[name + '_lower'], summary[name + '_upper'] = map(float, np.quantile(values, [0.025, 0.975]))
            output[name] = values
        summaries.append(summary)
        replicate_rows.append(output)
    accounting = dict(seed=seed, attempted_draws=attempted, valid_draws=len(draws), rejected_missing_year_draws=attempted - len(draws), reused_draws_across_comparisons=True, scope='Whole-genotype conditional uncertainty with fixed fits and three observed evaluation seasons at one site')
    return (pd.DataFrame(summaries), pd.concat(replicate_rows, ignore_index=True), accounting)
