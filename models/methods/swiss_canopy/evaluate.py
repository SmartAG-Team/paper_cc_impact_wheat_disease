"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/fip_canopy_transfer_20261011')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.methods.swiss_canopy.inputs import HERE, green_fraction_from_score, source_registry
from analysis.methods.canopy_senescence import features, model
from analysis.methods.crop_canopy import inputs as joint_inputs
from analysis.methods.crop_canopy import models as joint_models
from analysis.methods.crop_canopy.evaluation import scores, integrals
ROOT = PROJECT_ROOT
RESULTS = HERE / 'results'
PROTOCOL_SHA = '461766138b97f37a012ecc16acfd8de69077ca0f29d59f8bcf461be6d038cd0c'
AMENDMENT_SHA = 'bc1a0fddaa97b5fdbd6efe178dfc0e93657e5b7c568ef1199d17a6e1da7083c6'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def check_lock():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def score_frame(frame, prediction):
    result = scores(frame, prediction)
    window = integrals(frame, prediction)
    result['integral_bias_green_fraction_days'] = float(window.groupby('year').signed_error.mean().mean()) if len(window) else None
    result['genotypes'] = int(frame.cultivar_id.nunique())
    if len(window):
        result.update(window_days_min=int(window.duration_days.min()), window_days_median=float(window.duration_days.median()), window_days_max=int(window.duration_days.max()), visits_min=int(window.observations.min()), visits_max=int(window.observations.max()))
    return result

def endpoint_eligibility(predictions):
    ledger = predictions.groupby(['scope', 'model', 'year', 'plot_id', 'cultivar_id'], sort=True).agg(observations=('observation_id', 'size'), distinct_dates=('date', 'nunique')).reset_index()
    if not ledger.observations.eq(ledger.distinct_dates).all():
        raise ValueError('Duplicate plot visits in endpoint accounting')
    ledger['integral_eligible'] = ledger.distinct_dates.ge(2)
    ledger['timing_eligible'] = ledger.integral_eligible
    ledger['reason'] = np.where(ledger.integral_eligible, 'eligible', 'single_visit')
    return ledger

def coupled_weather_eligibility(registry, weather, frame):
    rows = []
    for row in registry.itertuples():
        part = frame[frame.year.eq(row.year)]
        eligible, reason, detail = (True, 'eligible', '')
        try:
            joint_inputs.weather_prefix(weather, row.sowing_date, int(row.horizon_days))
        except ValueError as error:
            eligible, reason, detail = (False, 'weather_contract_failure', str(error))
        rows.append(dict(year=int(row.year), sowing_date=row.sowing_date, horizon_days=int(row.horizon_days), plots=int(part.plot_id.nunique()), observations=len(part), eligible=eligible, reason=reason, detail=detail))
    return pd.DataFrame(rows)

def threshold_comparison(times, observed, predicted, threshold=0.5):
    t, y, p = map(lambda x: np.asarray(x, float), (times, observed, predicted))
    if len(t) < 2 or t.shape != y.shape or t.shape != p.shape or (not np.isfinite([t, y, p]).all()):
        raise ValueError('At least two finite paired visits required')
    if np.any(np.diff(t) <= 0) or np.any(y < 0) or np.any(y > 1) or np.any(p < 0) or np.any(p > 1):
        raise ValueError('Ordered times and unit-interval fractions required')

    def crossing(values):
        hits = np.flatnonzero(values <= threshold)
        if not len(hits):
            return ('right_censored', None, None, None)
        k = int(hits[0])
        if k == 0:
            return ('left_censored', None, None, None)
        point = t[k - 1] + (threshold - values[k - 1]) / (values[k] - values[k - 1]) * (t[k] - t[k - 1])
        return ('bracketed', float(t[k - 1]), float(t[k]), float(point))
    status, lower, upper, _ = crossing(y)
    pred_status, _, _, point = crossing(p)
    result = dict(observation_status=status, observed_lower=lower, observed_upper=upper, observed_bracket_width=None if lower is None else upper - lower, observed_recrossings=int(np.sum((y[:-1] <= threshold) & (y[1:] > threshold))), prediction_status=pred_status, predicted_time=point, interval_error_days=None, interval_error_lower_bound_days=None, bracket_covered=None)
    if status == 'bracketed':
        if pred_status == 'bracketed':
            error = max(lower - point, point - upper, 0.0)
            result.update(interval_error_days=error, interval_error_lower_bound_days=error, bracket_covered=bool(lower <= point <= upper))
        else:
            bound = max(lower - t[0], 0.0) if pred_status == 'left_censored' else max(t[-1] - upper, 0.0)
            result.update(interval_error_lower_bound_days=float(bound), bracket_covered=False)
    return result

def paired_bootstrap(predictions, comparisons, replicates=2000, seed=20261011):
    """Resample a genotype once across all years, retaining every associated plot."""
    summaries, output = ([], [])
    identity = ['plot_id', 'cultivar_id', 'year', 'date', 'green_fraction']
    for candidate, reference in comparisons:
        a = predictions.loc[predictions.model.eq(candidate)].set_index('observation_id').sort_index()
        b = predictions.loc[predictions.model.eq(reference)].set_index('observation_id').sort_index()
        if a.empty or a.index.duplicated().any() or b.index.duplicated().any() or (not a.index.equals(b.index)) or (not a[identity].equals(b[identity])):
            raise ValueError('Models must use the same observation population')
        ea, eb = (a.prediction - a.green_fraction, b.prediction - b.green_fraction)
        if not np.isfinite(np.r_[ea, eb]).all():
            raise ValueError('Finite paired predictions required')
        plot = a[identity].assign(a2=ea ** 2, b2=eb ** 2, mae=abs(ea) - abs(eb), bias=ea - eb).groupby(['year', 'cultivar_id', 'plot_id'])[['a2', 'b2', 'mae', 'bias']].mean().reset_index()
        ia, ib = (integrals(a, a.prediction), integrals(b, b.prediction))
        if ia.empty:
            raise ValueError('Observed-window integrals required')
        windows = ia[['year', 'cultivar_id', 'plot_id']].assign(integral=ia.absolute_error.to_numpy() - ib.absolute_error.to_numpy(), integral_bias=ia.signed_error.to_numpy() - ib.signed_error.to_numpy(), integral_plots=1.0)
        grouped = plot.assign(plots=1.0).groupby(['year', 'cultivar_id']).sum(numeric_only=True)
        grouped = grouped.join(windows.groupby(['year', 'cultivar_id'])[['integral', 'integral_bias', 'integral_plots']].sum()).fillna(0.0)
        genotypes = sorted(a.cultivar_id.unique())
        n = len(genotypes)
        years = sorted(a.year.unique())
        if n < 2:
            raise ValueError('At least two genotypes required')
        rng = np.random.default_rng(seed)
        counts = rng.multinomial(n, np.full(n, 1 / n), size=replicates)
        totals = {key: [] for key in ['a2', 'b2', 'mae', 'bias', 'integral', 'integral_bias']}
        for year in years:
            table = grouped.loc[year].reindex(genotypes, fill_value=0.0)
            den = counts @ table.plots.to_numpy()
            integral_den = counts @ table.integral_plots.to_numpy()
            if np.any(den <= 0) or np.any(integral_den <= 0):
                raise ValueError('Bootstrap draw lacks a required year; no silent year omission')
            for key in totals:
                totals[key].append(counts @ table[key].to_numpy() / (integral_den if key.startswith('integral') else den))
        average = {key: np.mean(value, axis=0) for key, value in totals.items()}
        draws = pd.DataFrame(dict(replicate=np.arange(replicates), rmse_difference=np.sqrt(average['a2']) - np.sqrt(average['b2']), mae_difference=average['mae'], bias_difference=average['bias'], integral_mae_difference=average['integral'], integral_bias_difference=average['integral_bias']))
        sa, sb = (score_frame(a, a.prediction), score_frame(b, b.prediction))
        values = {name: sa[name] - sb[name] for name in ['rmse', 'mae', 'bias']}
        values.update(integral_mae=sa['integral_mae_green_fraction_days'] - sb['integral_mae_green_fraction_days'], integral_bias=sa['integral_bias_green_fraction_days'] - sb['integral_bias_green_fraction_days'])
        row = dict(model=candidate, comparator=reference, genotypes=n, plots=len(plot), records=len(a), years=len(years), replicates=replicates, seed=seed, scope='Whole-genotype resampling across years; conditional on frozen fits and one observed site')
        for name, value in values.items():
            row[name + '_difference'] = float(value)
            row[name + '_lower95'], row[name + '_upper95'] = np.quantile(draws[name + '_difference'], [0.025, 0.975])
        summaries.append(row)
        output.append(draws.assign(model=candidate, comparator=reference))
    return (pd.DataFrame(summaries), pd.concat(output, ignore_index=True))

def fixed_parameters():
    base = ROOT / 'data/final_study/evidence'
    conditional = [x for x in json.loads((base / 'canopy_senescence/results/fits.json').read_text()) if x['phase'] == 'later_year' and x['fold'] == 2018]
    joint = [x for x in json.loads((base / 'phenology_canopy_coupling/results/canopy_fits.json').read_text()) if x['phase'] == 'evaluation' and x['policy'] == 'trial_doy' and (x['fold'] == 2018)]
    bridge = [x for x in json.loads((base / 'phenology_canopy_coupling/results/bridge_fits.json').read_text()) if x['phase'] == 'evaluation' and x['policy'] == 'trial_doy' and (x['fold'] == 2018)]
    if {x['model'] for x in conditional} != {'training_mean', 'calendar_logistic', 'thermal_logistic'} or len(conditional) != 3 or {x['model'] for x in joint} != set(joint_models.KINDS) or (len(joint) != 4) or (len(bridge) != 1) or (bridge[0]['fraction'] != 0.175) or any((x['training_years'] != [2015, 2016, 2017] for x in conditional + joint + bridge)):
        raise ValueError('Unexpected frozen parameter contract')
    return dict(conditional=conditional, coupled=joint, bridge=bridge[0])

def run():
    locked = check_lock()
    plots = pd.read_csv(RESULTS / 'plots.csv', dtype={'genotype_id': str})
    visits = pd.read_csv(RESULTS / 'raw_visits.csv')
    weather = pd.read_csv(RESULTS / 'weather_daily.csv')
    amendment = json.loads((HERE / 'source_semantics_amendment.json').read_text())
    visits['green_fraction'] = green_fraction_from_score(visits.raw_score, direction=amendment['score_direction'])
    visits['measurement_eligible'] = True
    p = plots.rename(columns={'plot_uid': 'plot_id', 'harvest_year': 'year', 'genotype_id': 'cultivar_id'})
    p['heading_doy'] = pd.to_datetime(p.heading_date).dt.dayofyear
    p['treatment'] = 'source_managed_trial'
    green = visits.rename(columns={'plot_uid': 'plot_id'})
    frame = green.merge(p[['plot_id', 'cultivar_id', 'sowing_date', 'harvest_date']], on='plot_id', validate='many_to_one')
    if (pd.to_datetime(frame.date) > pd.to_datetime(frame.harvest_date)).any():
        raise ValueError('Post-harvest observations require explicit qualification')
    feature = features.build_features(green, p, weather)
    feature.to_csv(RESULTS / 'conditional_eligibility.csv', index=False)
    conditional = feature[feature.eligible].copy()
    parameters = fixed_parameters()
    (RESULTS / 'fixed_parameters.json').write_text(json.dumps(parameters, indent=2) + '\n')
    registry = source_registry(plots[plots.harvest_year.isin(frame.year.unique())], json.loads((HERE / 'sources/site_location.json').read_text()))
    registry.to_csv(RESULTS / 'crop_registry.csv', index=False)
    weather_eligibility = coupled_weather_eligibility(registry, weather, frame)
    weather_eligibility.to_csv(RESULTS / 'coupled_weather_eligibility.csv', index=False)
    if not weather_eligibility.eligible.all():
        raise ValueError('Coupled weather contract failed; affected cycles retained in coupled_weather_eligibility.csv')
    parent = joint_inputs.parent_predictions(registry, weather)
    np.save(RESULTS / 'parent_cdf.npy', parent, allow_pickle=False)
    predictions = []
    keep = ['observation_id', 'year', 'plot_id', 'cultivar_id', 'date', 'green_fraction']
    for fit in parameters['conditional']:
        predictions.append(conditional[keep].assign(scope='observed_heading', model=fit['model'], prediction=model.predict(conditional, fit)))
    for fit in parameters['coupled']:
        basis = joint_models.canopy_basis(frame, registry, parent, weather, parameters['bridge']['fraction'], fit['model'])
        pred = frame[keep].assign(scope='coupled_all', model=fit['model'], prediction=joint_models.predict_canopy(basis, fit))
        predictions.append(pred)
        predictions.append(pred[pred.observation_id.isin(conditional.observation_id)].assign(scope='coupled_common_heading'))
    predictions = pd.concat(predictions, ignore_index=True)
    if not np.isfinite(predictions.prediction).all() or not predictions.prediction.between(0, 1).all():
        raise ValueError('Invalid frozen predictions')
    predictions.to_csv(RESULTS / 'predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    endpoint_eligibility(predictions).to_csv(RESULTS / 'endpoint_eligibility.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    metric_rows, window_rows, threshold_rows = ([], [], [])
    for (scope, kind), part in predictions.groupby(['scope', 'model'], sort=True):
        metric_rows.append(dict(scope=scope, model=kind, year='pooled', **score_frame(part, part.prediction)))
        for year, group in part.groupby('year'):
            metric_rows.append(dict(scope=scope, model=kind, year=str(year), **score_frame(group, group.prediction)))
        window_rows.append(integrals(part, part.prediction).assign(scope=scope, model=kind))
        for (year, plot), group in part.groupby(['year', 'plot_id']):
            group = group.sort_values('date')
            if len(group) < 2:
                continue
            day = (pd.to_datetime(group.date) - pd.Timestamp(group.date.iloc[0])).dt.days.to_numpy()
            threshold_rows.append(dict(scope=scope, model=kind, year=int(year), plot_id=plot, cultivar_id=group.cultivar_id.iloc[0], first_date=group.date.iloc[0], last_date=group.date.iloc[-1], **threshold_comparison(day, group.green_fraction, group.prediction)))
    pd.DataFrame(metric_rows).to_csv(RESULTS / 'metrics.csv', index=False)
    pd.concat(window_rows, ignore_index=True).to_csv(RESULTS / 'window_integrals.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    timing = pd.DataFrame(threshold_rows)
    timing.to_csv(RESULTS / 'thresholds.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    timing_rows = []
    for (scope, kind), part in timing.groupby(['scope', 'model']):
        for label, group in [('pooled', part), *list(part.groupby('year'))]:
            observed = group[group.observation_status.eq('bracketed')]

            def weighted_mean(column):
                return float(observed.groupby('year')[column].mean().mean()) if len(observed) else None
            timing_rows.append(dict(scope=scope, model=kind, year=str(label), plots=len(group), observed_bracketed=len(observed), observed_left_censored=int(group.observation_status.eq('left_censored').sum()), observed_right_censored=int(group.observation_status.eq('right_censored').sum()), prediction_bracketed=int(observed.prediction_status.eq('bracketed').sum()), prediction_left_censored=int(observed.prediction_status.eq('left_censored').sum()), prediction_right_censored=int(observed.prediction_status.eq('right_censored').sum()), bracket_coverage=weighted_mean('bracket_covered'), interval_error_lower_bound_days=weighted_mean('interval_error_lower_bound_days'), finite_prediction_interval_error_days=weighted_mean('interval_error_days'), bracket_width_days=weighted_mean('observed_bracket_width'), observed_recrossings=int(group.observed_recrossings.sum())))
    pd.DataFrame(timing_rows).to_csv(RESULTS / 'threshold_summary.csv', index=False)
    summaries, draws = ([], [])
    for scope, part in predictions.groupby('scope'):
        pairs = [('thermal_logistic', 'calendar_logistic'), ('thermal_logistic', 'training_mean')] if scope == 'observed_heading' else [('marginal', 'calendar'), ('marginal', 'median'), ('marginal', 'training_mean')]
        summary, sample = paired_bootstrap(part, pairs)
        summaries.append(summary.assign(scope=scope))
        draws.append(sample.assign(scope=scope))
    common = pd.concat([predictions[predictions.scope.eq('observed_heading') & predictions.model.eq('thermal_logistic')], predictions[predictions.scope.eq('coupled_common_heading') & predictions.model.eq('marginal')]])
    summary, sample = paired_bootstrap(common, [('marginal', 'thermal_logistic')])
    summaries.append(summary.assign(scope='all_model_common_heading'))
    draws.append(sample.assign(scope='all_model_common_heading'))
    pd.concat(summaries, ignore_index=True).to_csv(RESULTS / 'bootstrap_summary.csv', index=False)
    pd.concat(draws, ignore_index=True).to_csv(RESULTS / 'bootstrap_draws.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    locked_after = check_lock()
    result = dict(model_refit=False, model_selection=False, frozen_files_verified_before=locked, frozen_files_verified_after=locked_after, protocol_sha256=PROTOCOL_SHA, amendment_sha256=AMENDMENT_SHA, external_site_count=1, canopy_years=sorted((int(x) for x in frame.year.unique())), observations=len(frame), plots=int(frame.plot_id.nunique()), genotypes=int(frame.cultivar_id.nunique()), conditional_observations=len(conditional), conditional_plots=int(conditional.plot_id.nunique()), eligibility=feature.reason.value_counts().to_dict(), prediction_rows=len(predictions), source_direction=amendment['direction_status'], climate_adoption=False, bootstrap_replicates_per_comparison=2000, bootstrap_seed=20261011)
    (RESULTS / 'evaluation_receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    return result
