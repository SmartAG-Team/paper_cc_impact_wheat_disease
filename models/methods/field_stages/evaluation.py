"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/field_stage_extension_20261010')
import numpy as np
import pandas as pd
from analysis.methods.field_stages.bridge import EVENTS
MODELS = ('lstm_extension', 'archived_tpv', 'matched_tpv', 'calendar')

def assert_training(rows, registry):
    if {'value', 'Value'} & set(rows):
        raise ValueError('Numeric disease values entered the training boundary')
    allowed = registry.loc[registry.partition.eq('calibration') & registry.source.eq('BASF') & registry.season_year.isin([2017, 2018])]
    if rows.empty or not set(rows.field_id).issubset(set(allowed.field_id)):
        raise ValueError('Training includes fields outside original BASF 2017–2018')
    if not rows.partition.eq('calibration').all() or not rows.source.eq('BASF').all() or (not rows.season_year.isin([2017, 2018]).all()) or (not rows.event.isin(EVENTS).all()) or rows.duplicated(['field_id', 'event']).any():
        raise ValueError('Invalid training membership or duplicate event constraints')
    joined = rows.merge(allowed[['field_id', 'season_year']], on='field_id', validate='many_to_one', suffixes=('', '_registry'))
    if not joined.season_year.eq(joined.season_year_registry).all():
        raise ValueError('Training source identity differs from the registry')

def site_bootstrap(scores, repeats=2000, seed=20261010, counts=None):
    scores = scores.loc[scores.genuinely_bracketed].copy()
    keys = ['field_id', 'site_id', 'event']
    if scores.duplicated(keys + ['model']).any() or set(scores.model) != set(MODELS):
        raise ValueError('Paired unique field/event predictions from four models required')
    paired = scores.pivot(index=keys, columns='model', values='loss_days').reset_index()
    if not np.isfinite(paired[list(MODELS)].to_numpy()).all():
        raise ValueError('Every model must retain every constrained field/event')
    sites = sorted(paired.site_id.unique())
    if not sites or set(paired.event) != set(EVENTS):
        raise ValueError('All three stages require evaluable sites')
    site_index = {site: i for i, site in enumerate(sites)}
    sums = np.zeros((len(sites), 3, len(MODELS)))
    numbers = np.zeros((len(sites), 3))
    for row in paired.itertuples(index=False):
        i, j = (site_index[row.site_id], EVENTS.index(row.event))
        numbers[i, j] += 1
        sums[i, j] += [getattr(row, model) for model in MODELS]
    if counts is None:
        counts = np.random.default_rng(seed).multinomial(len(sites), np.full(len(sites), 1 / len(sites)), size=repeats)
    counts = np.asarray(counts)
    if counts.ndim != 2 or counts.shape[1] != len(sites) or np.any(counts < 0) or np.any(counts != np.floor(counts)) or (not np.all(counts.sum(axis=1) == len(sites))):
        raise ValueError('Each bootstrap count vector must resample all geographic sites')
    denominators = counts @ numbers
    numerator = np.einsum('rs,sem->rem', counts, sums)
    valid = (denominators > 0).all(axis=1)
    values = np.full((len(counts), len(MODELS)), np.nan)
    values[valid] = (numerator[valid] / denominators[valid, :, None]).mean(axis=1)
    point = (sums.sum(axis=0) / numbers.sum(axis=0)[:, None]).mean(axis=0)
    draws = pd.DataFrame(values, columns=MODELS)
    draws.insert(0, 'replicate', np.arange(len(counts)))
    draws['valid'] = valid
    for j, event in enumerate(EVENTS):
        draws[f'n_{event}'] = denominators[:, j].astype(int)
    draws['unique_sites'] = (counts > 0).sum(axis=1)
    summary = []
    for j, model in enumerate(MODELS[1:], 1):
        difference = values[:, 0] - values[:, j]
        draws['difference_' + model] = difference
        limits = np.quantile(difference[valid], [0.025, 0.975]) if valid.any() else [np.nan, np.nan]
        summary.append(dict(reference=model, candidate_loss_days=float(point[0]), reference_loss_days=float(point[j]), difference_days=float(point[0] - point[j]), lower_95=float(limits[0]), upper_95=float(limits[1]), sites=len(sites), valid_draws=int(valid.sum()), total_draws=len(counts), valid_fraction=float(valid.mean())))
    return (pd.DataFrame(summary), draws)

def decision(scores, bootstrap):
    brackets = scores.loc[scores.genuinely_bracketed]
    by_stage = brackets.groupby(['source', 'event', 'model']).loss_days.mean().unstack('model')
    by_source = by_stage.groupby('source').mean()
    coverage = scores.groupby(['source', 'model']).prediction_reached.mean().unstack('model')
    source_ok = all(((by_source.lstm_extension - by_source[ref] <= 1e-12).all() for ref in MODELS[1:]))
    stage_ok = all(((by_stage.lstm_extension - by_stage[ref] <= 1 + 1e-12).all() for ref in MODELS[1:]))
    coverage_ok = all(((coverage.lstm_extension >= coverage[ref] - 1e-12).all() for ref in MODELS[1:]))
    checks = dict(upper_95_below_zero_all_references=bool(bootstrap.upper_95.lt(0).all()), bootstrap_valid_fraction_at_least_95pct=bool(bootstrap.valid_fraction.ge(0.95).all()), no_source_average_worsening=bool(source_ok), no_source_stage_worsening_over_one_day=bool(stage_ok), no_event_coverage_loss=bool(coverage_ok))
    return dict(retrospective_comparison_support=all(checks.values()), criteria=checks, climate_adoption=False, independent_validation=False, limitations=['Previously accessed field sources; BASF sites recur across calibration and evaluation years.', 'Scenario sowing dates and incompletely documented crop-stage measurement semantics.', 'Conditional on realized weather and the frozen four-stage model; bootstrap excludes parameter and model-selection uncertainty.', 'Stage-interval agreement does not validate exact dates, functional canopy dynamics or cumulative disease response.'])

def summaries(scores):
    """Stage-equal scores plus event-specific support, never an exact-date MAE."""
    result = []
    for view, view_rows in scores.groupby('view', sort=True):
        cohorts = {name: group for name, group in view_rows.groupby('partition', sort=True)}
        cohorts['reused_pooled'] = view_rows.loc[~view_rows.partition.eq('calibration')]
        for cohort, group in cohorts.items():
            for subset in ('all_constraints', 'bracketed'):
                selected = group if subset == 'all_constraints' else group.loc[group.genuinely_bracketed]
                for model, model_rows in selected.groupby('model', sort=True):
                    for event in (0, *EVENTS):
                        sample = model_rows if event == 0 else model_rows.loc[model_rows.event.eq(event)]
                        if sample.empty:
                            continue
                        result.append(dict(view=view, cohort=cohort, subset=subset, model=model, event=event, mean_interval_excess_days=float(sample.groupby('event').loss_days.mean().mean()), constraints=len(sample), fields=sample.field_id.nunique(), sites=sample.site_id.nunique(), events=sample.event.nunique(), prediction_coverage=float(sample.prediction_reached.mean()), median_bracket_width_days=float(sample.bracket_width_days.median()) if sample.bracket_width_days.notna().any() else np.nan))
    return pd.DataFrame(result)
