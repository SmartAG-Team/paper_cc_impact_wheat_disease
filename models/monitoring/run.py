"""Register, fit and evaluate disease-improvement development experiments."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from threadpoolctl import threadpool_limits

from analysis.final_models.disease_math import predict as legacy_predict
from analysis.final_study.metrics import hierarchical_weights
from .evaluation import (paired_interval, promotion_decision, score_predictions, score_subset,
                         temporal_spatial_splits)
from .features import build_features
from .history import SAME_NAMES, build_history
from .inputs import ARCHIVE, INPUT, WEATHER, load_observations, quality_screen, sha
from .models import fit_candidate, predict_candidate

SEED = 20261021
CANDIDATES = {
    'reference': dict(structure='residual', features='legacy', crossleaf=False),
    'events_residual': dict(structure='residual', features='events', crossleaf=True),
    'events_split': dict(structure='split', features='events', crossleaf=True),
    'events_no_crossleaf': dict(structure='split', features='events', crossleaf=False),
    'weather_history': dict(structure='split', features='weather', crossleaf=True),
    'weather_events_split': dict(structure='split', features='weather_events', crossleaf=True),
    'process_residual': dict(structure='residual', features='process', crossleaf=True),
    'process_corrected': dict(structure='process', features='process', crossleaf=True,
                              process_baseline_column=-3),
    'weather_process_corrected': dict(structure='process', features='weather_process', crossleaf=True,
                                      process_baseline_column=-3),
    'weather_process_source': dict(structure='process', features='weather_process', crossleaf=True,
                                   process_baseline_column=-3, source_specific=True),
    'weather_events_rf': dict(structure='split', features='weather_events', crossleaf=True,
                              estimator='rf'),
}
PRIMARY_CANDIDATES = ['weather_history', 'weather_events_split', 'weather_events_rf',
                      'weather_process_corrected', 'weather_process_source']
PRIMARY_REFERENCE = 'weather_process_corrected'


def write_json(path, value):
    def convert(v):
        if isinstance(v, np.generic):
            return v.item()
        raise TypeError(type(v).__name__)
    Path(path).write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n')


def select_candidate(predictions, names, reference_name='reference'):
    """Choose only from complete inner held-site predictions, with integral guard."""
    reference = score_predictions(predictions, reference_name)
    rows = []
    for name in names:
        value = score_predictions(predictions, name)
        admissible = (reference['integral_rmse'] is not None
            and value['integral_rmse'] is not None
            and value['integral_rmse'] <= reference['integral_rmse']+1e-10)
        ci = [0., 0.] if name == reference_name else paired_interval(predictions, name, reference_name)
        supported = name == reference_name or ci[1] < 0
        rows.append(dict(candidate=name, **value, integral_guard=bool(admissible),
                         paired_lower=ci[0], paired_upper=ci[1], improvement_supported=bool(supported)))
    eligible = [row for row in rows if row['integral_guard'] and row['improvement_supported']]
    # With no multi-assessment windows, retain the fixed reference recipe.
    choice = min(eligible, key=lambda r: (r['rmse'], r['candidate']))['candidate'] if eligible else reference_name
    return choice, rows


def assemble_designs(legacy, categories, events, event_names, weather_base=None):
    """Keep all primary input matrices independent of inherited crop columns."""
    process = [i for i, name in enumerate(event_names) if name.startswith('process_')]
    extras = [i for i, name in enumerate(event_names) if not name.startswith('weather_')]
    event = [i for i in extras if i not in process]
    weather_base = legacy[:, :12] if weather_base is None else weather_base
    return {
        'legacy': legacy.copy(),
        'events': np.column_stack([legacy, categories, events[:, event]]),
        'weather': np.column_stack([weather_base, categories]),
        'weather_events': np.column_stack([weather_base, categories, events[:, extras]]),
        'process': np.column_stack([legacy, categories, events[:, process]]),
        'weather_process': np.column_stack([weather_base, categories, events[:, process]]),
    }


def _designs(root, observations, fields, output, weather_cutoff_days=0):
    verification = pd.read_csv(root/'data/final_models/verification/disease_queries.csv')
    with np.load(root/'data/final_models/verification/disease_queries.npz') as arrays:
        stored = arrays['features'].copy()
    position = pd.Series(np.arange(len(verification)), index=verification.global_target_index)
    aligned = verification.set_index('global_target_index').loc[observations.global_target_index]
    for col in ['field_id', 'leaf_index', 'date', 'observed_percent']:
        if not np.array_equal(aligned[col].to_numpy(), observations[col].to_numpy()):
            raise ValueError('Frozen predictor identities or targets differ')
    legacy = stored[position.loc[observations.global_target_index].to_numpy()]
    history, hp = build_history(observations, observations, {})
    weather = pd.read_parquet(root/WEATHER)
    events, event_names, wp = build_features(weather, fields, observations,
        dict(prior_dates=dict(zip(hp.global_target_index, hp.latest_same_date)),
             weather_cutoff_days=weather_cutoff_days, rain_variable='rain_mm'))
    hp.to_json(output/'history_provenance.json', orient='records', indent=2)
    wp.to_csv(output/'feature_provenance.csv', index=False)
    categories = np.column_stack([(observations.source.eq(source)
        & observations.leaf_index.eq(code)).to_numpy(float)
        for source in ['BASF', 'Corteva'] for code in range(7)])
    base_columns = [i for i, name in enumerate(event_names) if name.startswith('weather_')]
    if len(base_columns) != 12:
        raise ValueError('Twelve independently rebuilt weather/calendar predictors required')
    designs = assemble_designs(legacy, categories, events, event_names, events[:, base_columns])
    np.savez_compressed(output/'predictors.npz', target_ids=observations.global_target_index,
        events=events, legacy=legacy, history=history.to_numpy())
    write_json(output/'feature_schema.json', dict(event_names=event_names,
        history_names=history.columns.tolist(), design_widths={k: v.shape[1] for k, v in designs.items()},
        later_trained_crop_in=['legacy', 'events', 'process'],
        crop_free=['weather', 'weather_events', 'weather_process'],
        missing_anatomical_leaf_direction=True,
        sowing_uncertainty='scenario not observed; crop-free and interval-weather ablations retained',
        process_parameters='model.seasonal_septoria.core.Parameters defaults; generic active cohort'))
    return designs, history


def _prepare(root, output, years, buffer_km, quality_policy, weather_cutoff_days):
    if output.exists():
        raise FileExistsError('A new output directory is required')
    observations, fields, provenance = load_observations(root)
    output.mkdir(parents=True)
    observations.to_csv(output/'original_observation_ledger.csv', index=False)
    screened, exclusions = quality_screen(observations, fields)
    exclusions.to_csv(output/'excluded_observations.csv', index=False)
    if quality_policy == 'quarantine':
        observations = screened
    elif quality_policy != 'retain':
        raise ValueError('Unknown observation quality policy')
    grain_path = root/'data/final_study/evidence/disease_endpoint_audit/results/crop_endpoint_source_rows.csv'
    source_flags = root/'data/final_study/evidence/disease_endpoint_audit/results/source_chronology_flags.csv'
    provenance['source_sha256'].update({str(p.relative_to(root)): sha(p) for p in [grain_path, source_flags]})
    upstream = pd.read_csv(source_flags)
    upstream.loc[~upstream.assessment_year_matches_field].assign(
        status='already_absent_from_original_active_targets').to_csv(output/'upstream_year_mismatches.csv', index=False)
    weather = pd.read_parquet(root/WEATHER)
    write_json(output/'weather_audit.json', dict(rows=len(weather), sites=weather.location_id.nunique(),
        missing_cells=int(weather.isna().sum().sum()), duplicate_location_dates=int(weather.duplicated(['location_id','date']).sum()),
        incomplete_days=int(weather.hour_count.ne(24).sum()),
        precipitation_rain_difference_days=int((abs(weather.precipitation_mm-weather.rain_mm) > 1e-8).sum()),
        maximum_total_minus_liquid_mm=float((weather.precipitation_mm-weather.rain_mm).max()),
        temperature_order_violations=int(((weather.tmin_c>weather.tmean_c+.01)|(weather.tmean_c>weather.tmax_c+.01)).sum()),
        definition_source='https://open-meteo.com/en/docs/historical-weather-api',
        liquid_rain_for_splash='rain_mm', total_precipitation_includes_snow=True,
        weather_cutoff_days=weather_cutoff_days, independent_station_accuracy_verified=False))
    code_paths = list(Path(__file__).parent.glob('*.py'))
    code_paths += [root/name for name in ['models/learned/disease_math.py',
        'paper/analyses/source_support/metrics.py', 'models/seasonal_septoria/core.py',
        'models/seasonal_septoria/wetness.py']]
    for path in code_paths:
        snapshot = output/'code_snapshot'/path.relative_to(root)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes(path.read_bytes())
    for year in years:
        choices = root/f'data/final_study/evidence/disease_coupled_ml/results/year_{year}/selected_choices.json'
        recipe = json.loads(choices.read_text())['selected_pipeline']
        model = choices.parent/f'outer/{recipe["family"]}_{recipe["setting"]}.joblib'
        provenance['source_sha256'].update({str(path.relative_to(root)): sha(path) for path in [choices, model]})
    registration = dict(status='registered_before_fitting', registered_utc=datetime.now(timezone.utc).isoformat(),
        years=list(years), candidate_registry=CANDIDATES, seed=SEED, independent_validation=False,
        quality_policy=quality_policy, excluded_target_ids=exclusions.global_target_index.astype(int).tolist()
            if quality_policy == 'quarantine' else [],
        weather_cutoff_days=weather_cutoff_days, rain_variable='rain_mm',
        primary_candidates=PRIMARY_CANDIDATES,
        primary_selection_reference=PRIMARY_REFERENCE,
        exploratory_only='all crop-conditioned candidates; inherited later-trained phenology',
        evidence_type='adaptive retrospective development; original outcomes previously examined',
        source_provenance=provenance, spatial_buffer_km=buffer_km,
        inner_selection='chronological and held-site; integral guard; fixed reference when no earlier-year validation',
        inner_spatial_buffer_km=50, inner_validation_years='all available years after earliest training year',
        selection_guard='Retain fixed process reference unless inner site-cluster RMSE upper interval <0 and integral guard passes',
        primary_weights='coordinate-year mass / fields / leaf categories / assessments',
        forecast_lead_days=0, use_observed_disease_history=True,
        severe_case_threshold_percent=75, rapid_rise_threshold_pp=40, low_severity_threshold_percent=5,
        bootstrap_replicates=2000,
        candidate_defaults=dict(max_iter=200, max_depth=3, max_leaf_nodes=15,
            min_samples_leaf=10, l2_regularization=1., learning_rate=.05),
        environment=dict(python=platform.python_version(),
            packages={name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'scipy', 'numba', 'joblib']}),
        adoption='requires untouched external validation and all prespecified error guards',
        code_sha256={str(p.relative_to(root)): sha(p) for p in sorted(code_paths)})
    write_json(output/'registration.json', registration)
    observations.to_csv(output/'observation_ledger.csv', index=False)
    fields.to_csv(output/'field_metadata_ledger.csv', index=False)
    template_columns = ['trial_id', 'site_id', 'latitude', 'longitude', 'season_year',
        'plot_id', 'replicate_id', 'assessment_date', 'anatomical_leaf', 'leaf_tracking_protocol',
        'severity_percent', 'severity_denominator', 'disease_confirmation', 'cultivar',
        'resistance_class', 'observed_sowing_date', 'winter_or_spring', 'treatment',
        'spray_date', 'weather_station_id', 'leaf_wetness_sensor_id',
        'external_partition_locked_at', 'source_record_identity']
    pd.DataFrame(columns=template_columns).to_csv(output/'acquisition_metadata_template.csv', index=False)
    readiness = [dict(requirement=name, status='missing_or_unqualified', action=action) for name, action in [
        ('independent_external_cohort', 'Reserve untouched later seasons/trials before outcome access'),
        ('cultivar_and_resistance', 'Recover original trial protocol or collect in new fields'),
        ('observed_sowing_and_type', 'Recover field records; keep scenario uncertainty explicit'),
        ('anatomical_leaf_and_denominator', 'Confirm source protocols before physical leaf pooling'),
        ('replicate_identity', 'Recover plot and replicate IDs before duplicate adjudication'),
        ('target_1036', 'Quarantined pending source date/season adjudication'),
        ('wetness_validation', 'Collect sensor observations to validate reanalysis exposure proxies')]]
    pd.DataFrame(readiness).to_csv(output/'data_readiness.csv', index=False)
    (output/'acquisition_protocol.txt').write_text(
        'Reserve independent trial/spatial-cluster groups before model development.\n'
        'Record exact anatomical leaf and percentage denominator; keep plot and replicate identifiers.\n'
        'Observe weekly during epidemic development, including low-disease/non-epidemic controls.\n'
        'Record sowing, cultivar/resistance, treatment dates, local weather and a wetness-sensor subset.\n'
        'Determine required independent field count from field-cluster error variance; leaf records are not independent fields.\n'
        'Do not relabel previously examined Nordic/Tunisia/Karisto data as an untouched external cohort.\n')
    return observations, fields, registration


def _fit_predict(designs, history, observations, train, test, names):
    predicted = observations.iloc[test].copy()
    fitted = {}
    for name in names:
        spec = CANDIDATES[name]
        x = designs[spec['features']]
        h = history if spec['crossleaf'] else history[SAME_NAMES]
        model = fit_candidate(x[train], h.iloc[train], observations.iloc[train], spec)
        predicted[name] = predict_candidate(model, x[test], h.iloc[test], predicted)
        fitted[name] = model
    return predicted, fitted


def _nested_choice(designs, history, observations, outer_train, output, fold_id):
    training = observations.iloc[outer_train].reset_index(drop=True)
    validation_years = sorted(training.season_year.unique())[1:]
    names = PRIMARY_CANDIDATES
    predictions, membership = [], []
    # For 2015 there is only one training year: no chronological inner test.
    if training.season_year.nunique() < 2:
        return PRIMARY_REFERENCE, [], [], 'fixed_recipe_no_earlier_year_validation'
    for inner_id, fit, validation in temporal_spatial_splits(training, validation_years, folds=3, buffer_km=50):
        if not len(fit):
            continue
        fit_global, validation_global = outer_train[fit], outer_train[validation]
        predicted, _ = _fit_predict(designs, history, observations, fit_global, validation_global, names)
        predicted['inner_fold'] = inner_id
        predictions.append(predicted)
        membership += [dict(outer_fold=fold_id, inner_fold=inner_id, role=role,
            target_ids=observations.iloc[indices].global_target_index.astype(int).tolist())
            for role, indices in [('train', fit_global), ('validation', validation_global)]]
    if not predictions:
        return PRIMARY_REFERENCE, [], membership, 'fixed_recipe_no_usable_inner_split'
    combined = pd.concat(predictions, ignore_index=True)
    # Grouping must preserve all later-year assessment windows in each source series.
    selected, rows = select_candidate(combined, names, PRIMARY_REFERENCE)
    combined.to_csv(output/f'inner_predictions_{fold_id.replace(":", "_")}.csv', index=False)
    return selected, rows, membership, 'inner_chronological_held_site_selection'


def _scores(predictions, names):
    results = []
    masks = {'all': np.ones(len(predictions), bool),
        'cold': ~predictions.prior_assessment_available, 'warm': predictions.prior_assessment_available,
        'BASF': predictions.source.eq('BASF'), 'Corteva': predictions.source.eq('Corteva'),
        'observed_above_75': predictions.observed_percent.gt(75),
        'observed_at_most_5': predictions.observed_percent.le(5)}
    if 'observed_change_pp' in predictions:
        masks['observed_increase_above_40'] = predictions.observed_change_pp.gt(40)
    if 'new_site' in predictions:
        masks['new_sites'] = predictions.new_site
    for population, mask in masks.items():
        if not mask.any():
            continue
        for name in names:
            scorer = (score_subset(predictions, mask, name) if population.startswith('observed_')
                      else score_predictions(predictions.loc[mask], name))
            results.append(dict(population=population, model=name, **scorer))
    return pd.DataFrame(results)


def run_experiment(root, output, *, years=range(2015, 2020), buffer_km=50,
                   prepare_only=False, spatial=True, quality_policy='quarantine', weather_cutoff_days=1):
    root, output = Path(root).resolve(), Path(output).resolve()
    years = tuple(years)
    observations, fields, registration = _prepare(root, output, years, buffer_km, quality_policy, weather_cutoff_days)
    designs, history = _designs(root, observations, fields, output, weather_cutoff_days)
    observations['prior_assessment_available'] = history.prior_assessment_available.gt(0).to_numpy()
    observations['observed_change_pp'] = np.where(observations.prior_assessment_available,
        observations.observed_percent.to_numpy()-100*history.latest_severity_fraction.to_numpy(), np.nan)
    if prepare_only:
        write_json(output/'status.json', dict(trained=False, complete_support=True,
            independent_validation_available=False, pending_data=registration['source_provenance']['unresolved']))
        return dict(trained=False, output=str(output))
    archive = pd.read_csv(root/ARCHIVE).set_index('global_target_index')
    names = list(CANDIDATES)
    all_predictions, members, selection, model_files = [], [], [], []
    max_replay_error = 0.
    with threadpool_limits(limits=1):
        for year in years:
            train = np.flatnonzero(observations.season_year.lt(year))
            test = np.flatnonzero(observations.season_year.eq(year))
            if not len(train) or not len(test):
                raise ValueError('Each outer year needs earlier training and test records')
            fold = f'year{year}'
            choice, scores, inner_members, reason = _nested_choice(designs, history, observations, train, output, fold)
            members += inner_members
            predicted, models = _fit_predict(designs, history, observations, train, test, names)
            predicted['selected_candidate'] = predicted[choice]
            predicted['selection'] = choice; predicted['outer_fold'] = fold
            predicted['new_site'] = ~predicted.site_id.isin(observations.iloc[train].site_id)
            predicted['current'] = archive.loc[predicted.global_target_index, 'selected_pipeline'].to_numpy()
            predicted['persistence'] = archive.loc[predicted.global_target_index, 'state_baseline'].to_numpy()
            choice_path = root/f'data/final_study/evidence/disease_coupled_ml/results/year_{year}/selected_choices.json'
            recipe = json.loads(choice_path.read_text())['selected_pipeline']
            fitted = joblib.load(root/f'data/final_study/evidence/disease_coupled_ml/results/year_{year}/outer/{recipe["family"]}_{recipe["setting"]}.joblib')
            replay = legacy_predict(fitted, designs['legacy'][test], history.iloc[test][SAME_NAMES].to_numpy(),
                                    predicted.leaf_index, recipe['strength'])
            max_replay_error = max(max_replay_error, float(np.max(abs(replay-predicted.current))))
            if max_replay_error > 1e-10:
                raise ValueError('Original model replay does not reproduce Figure 1')
            # Refit the archived recipe on the cleaned training population.
            training = observations.iloc[train]
            w = hierarchical_weights(training); w /= w.mean()
            target = training.observed_percent.to_numpy()/100
            means = np.full(7, np.average(target, weights=w))
            for leaf in training.leaf_index.unique():
                mask = training.leaf_index.eq(leaf).to_numpy()
                means[int(leaf)] = np.average(target[mask], weights=w[mask])
            hh = history.iloc[train][SAME_NAMES].to_numpy()
            base = np.where(hh[:, 0] > 0, hh[:, 1], means[training.leaf_index.to_numpy(int)])
            refitted = dict(fitted, baseline=means,
                estimator=clone(fitted['estimator']).fit(np.column_stack([designs['legacy'][train], hh]),
                    target-base, sample_weight=w),
                training_targets=training.global_target_index.astype(int).tolist())
            predicted['refit_current'] = legacy_predict(refitted, designs['legacy'][test],
                history.iloc[test][SAME_NAMES].to_numpy(), predicted.leaf_index, recipe['strength'])
            models['refit_current'] = refitted
            selection.append(dict(outer_fold=fold, selected=choice, reason=reason, inner_scores=scores))
            for role, indices in [('train', train), ('test', test)]:
                members.append(dict(outer_fold=fold, role=role,
                    target_ids=observations.iloc[indices].global_target_index.astype(int).tolist()))
            model_path = output/f'{fold}_models.joblib'
            joblib.dump(models, model_path); model_files.append(model_path.name)
            all_predictions.append(predicted)
            print(f'{fold}: selected={choice}; train={len(train)} test={len(test)}', flush=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    if predictions.global_target_index.duplicated().any():
        raise ValueError('Outer predictions overlap')
    expected = observations.loc[observations.season_year.isin(years), 'global_target_index']
    if set(predictions.global_target_index) != set(expected):
        raise ValueError('Incomplete outer prediction support')
    predictions.to_csv(output/'assessment_predictions.csv', index=False)
    scored_names = ['current', 'refit_current', 'persistence', *names, 'selected_candidate']
    summary = _scores(predictions, scored_names)
    summary.to_csv(output/'metrics.csv', index=False)
    full_archive = pd.read_csv(root/ARCHIVE)
    full_archive = full_archive.loc[full_archive.season_year.isin(years)]
    write_json(output/'cleaning_effect.json', dict(original_support=len(full_archive), cleaned_support=len(predictions),
        original_frozen_model=score_predictions(full_archive, 'selected_pipeline'),
        same_frozen_model_on_cleaned_support=score_predictions(predictions, 'current'),
        refitted_original_recipe_on_cleaned_support=score_predictions(predictions, 'refit_current'),
        interpretation='Population/filtering effect and model-refitting effect are reported separately'))
    write_json(output/'selection.json', selection)
    write_json(output/'split_membership.json', members)
    ref = score_predictions(predictions, 'current')
    candidate = score_predictions(predictions, 'selected_candidate')
    interval = paired_interval(predictions, 'selected_candidate', 'current')
    subgroup = summary.loc[summary.population.isin(['cold', 'warm', 'BASF', 'Corteva'])]
    delta = subgroup.pivot(index='population', columns='model', values='rmse')
    severe = predictions.observed_percent > 75
    severe_delta = (abs(score_subset(predictions, severe, 'selected_candidate')['bias'])
                    - abs(score_subset(predictions, severe, 'current')['bias'])) if severe.any() else None
    new = predictions.loc[predictions.new_site]
    ratio = (score_predictions(new, 'selected_candidate')['rmse']/score_predictions(new, 'current')['rmse']) if len(new) else None
    decision = promotion_decision(ref, candidate, interval, independent=False,
        subgroup_deltas=(delta.selected_candidate-delta.current).to_dict(),
        severe_bias_delta=severe_delta if severe_delta is not None else float('inf'), new_site_ratio=ratio)
    decision.update(reference=ref, candidate=candidate, paired_rmse_interval=interval,
                    severe_absolute_bias_change=severe_delta,
                    baseline_replay_maximum_error_pp=max_replay_error,
                    climate_adoption=False, model_files=model_files)
    write_json(output/'promotion.json', decision)
    diagnostic_intervals = []
    weights = hierarchical_weights(predictions)
    for population, mask in [('severe_above_75', predictions.observed_percent > 75),
        ('low_at_most_5', predictions.observed_percent <= 5),
        ('increase_above_40', predictions.observed_change_pp > 40)]:
        if mask.any():
            for name in [*names, 'selected_candidate']:
                ci = paired_interval(predictions.loc[mask], name, 'current', weights=weights[mask])
                diagnostic_intervals.append(dict(population=population, candidate=name,
                    n=int(mask.sum()), sites=predictions.loc[mask].site_id.nunique(),
                    lower_rmse_difference=ci[0], upper_rmse_difference=ci[1],
                    weighting='full_cohort_weights_conditioned_on_subset'))
    pd.DataFrame(diagnostic_intervals).to_csv(output/'diagnostic_intervals.csv', index=False)
    if spatial:
        _spatial_run(designs, history, observations, years, buffer_km, output)
    status = dict(trained=True, benchmark_support_complete=True, promote=decision['promote'],
        independent_validation_available=False,
        stages=dict(observation_audit='implemented; unresolved sources retained',
            event_and_history_features='implemented', models_and_nested_evaluation='implemented',
            field_acquisition='protocol_and_template_ready; observations_not_yet_available',
            publication_update='conditional_on_promotion; not_eligible'),
        pending_data=registration['source_provenance']['unresolved'])
    write_json(output/'status.json', status)
    _write_readout(output, summary, decision)
    return status


def _spatial_run(designs, history, observations, years, buffer_km, output):
    # Crop-free candidates only: no inherited later-trained phenology.
    names = PRIMARY_CANDIDATES
    rows, membership, unavailable = [], [], []
    with threadpool_limits(limits=1):
        for fold, train, test in temporal_spatial_splits(observations, years, folds=3, buffer_km=buffer_km):
            if not len(train):
                unavailable.append(dict(fold=fold, n=len(test), reason='no_training_after_spatial_buffer'))
                continue
            pred, _ = _fit_predict(designs, history, observations, train, test, names)
            pred['outer_fold'] = fold
            rows.append(pred)
            membership.append(dict(fold=fold, train=observations.iloc[train].global_target_index.astype(int).tolist(),
                test=observations.iloc[test].global_target_index.astype(int).tolist()))
            print(f'{fold}: crop-free training={len(train)} test={len(test)}', flush=True)
    write_json(output/'spatial_membership.json', membership)
    write_json(output/'spatial_unavailable.json', unavailable)
    expected = int(observations.season_year.isin(years).sum())
    scored = sum(len(row) for row in rows)
    write_json(output/'spatial_support.json', dict(expected_assessments=expected,
        scored_assessments=scored, unavailable_assessments=expected-scored,
        complete_support=scored == expected, buffer_km=buffer_km,
        evidence='retrospective crop-free fixed-candidate sensitivity; no outcome selection'))
    if rows:
        result = pd.concat(rows, ignore_index=True)
        result.to_csv(output/'spatial_predictions.csv', index=False)
        _scores(result, names).to_csv(output/'spatial_metrics.csv', index=False)


def _write_readout(output, summary, decision):
    all_rows = summary.loc[summary.population.eq('all')].set_index('model')
    lines = ['# Disease improvement development results', '',
        'Retrospective monitoring with realized weather and strictly earlier field observations. '
        'The outcomes were previously examined. Independent validation is unavailable.', '',
        '| Model | Assessment RMSE (percentage points) | Integral RMSE (percentage-point days) |',
        '|---|---:|---:|']
    for name, row in all_rows.iterrows():
        integral = f'{row.integral_rmse:.3f}' if pd.notna(row.integral_rmse) else 'unavailable'
        lines.append(f'| {name} | {row.rmse:.4f} | {integral} |')
    lines += ['', f'Nested-selected candidate minus current RMSE 95% site-bootstrap interval: '
        f'{decision["paired_rmse_interval"][0]:.3f} to {decision["paired_rmse_interval"][1]:.3f} percentage points.',
        '', 'Promotion gates:', '']
    lines += [f'- {name}: {"passed" if passed else "failed"}' for name, passed in decision['gates'].items()]
    lines += ['', 'New model adoption: '+str(decision['promote'])+'.', '',
        'Cultivar, verified anatomical leaf identity, observed sowing and plot/replicate metadata remain unavailable. '
        'The acquisition protocol and metadata template define the missing field evidence. '
        'Weather-only event candidates omit the inherited crop model; crop-conditioned candidates retain its temporal limitation.', '']
    (output/'Results.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--years', nargs='+', type=int, default=list(range(2015, 2020)))
    parser.add_argument('--buffer-km', type=float, default=50)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-spatial', action='store_true')
    parser.add_argument('--quality-policy', choices=['quarantine', 'retain'], default='quarantine')
    parser.add_argument('--weather-cutoff-days', choices=[0, 1], type=int, default=1)
    args = parser.parse_args()
    run_experiment(args.root, args.output, years=args.years, buffer_km=args.buffer_km,
                   prepare_only=args.prepare_only, spatial=not args.no_spatial,
                   quality_policy=args.quality_policy, weather_cutoff_days=args.weather_cutoff_days)


if __name__ == '__main__':
    main()
