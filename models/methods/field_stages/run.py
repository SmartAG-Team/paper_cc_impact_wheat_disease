"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/field_stage_extension_20261010')
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from analysis.methods.wheat_phenology.wheat_transfer import DEFAULT_GATES, WheatPhenology, build_features, predict_batches
from analysis.methods.field_stages.bridge import EVENTS, GRID, calendar_offsets, extend_cdf, interpolate, interval_loss, median_indices, ordered_fit
from analysis.methods.field_stages.data import load_sources, qualify, constraints
from analysis.methods.field_stages.evaluation import MODELS, assert_training, decision, site_bootstrap, summaries
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/field_stage_extension_20261010')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')

def predictor_arrays(registry):
    weather = pd.read_parquet(ROOT / 'data/final_study/inputs/paper_study/field_weather_completed/daily_weather.parquet')
    weather['date'] = pd.to_datetime(weather.date).dt.normalize()
    if weather.duplicated(['location_id', 'date']).any():
        raise ValueError('Duplicate location/date weather')
    groups = {str(k): g.set_index('date') for k, g in weather.groupby('location_id')}
    package = torch.load(ROOT / 'data/final_study/evidence/phenology_transfer/selected_wheat_ensemble.pt', map_location='cpu', weights_only=False)
    if package['stages'] != [10, 31, 51, 87] or package['model_family'] != 'ordered_lstm' or package['maize_weights_used'] is not False:
        raise ValueError('Expected the frozen wheat-specific ordered LSTM ensemble')
    metadata = package['members'][0]['metadata']
    if any((m['metadata'] != metadata for m in package['members'])):
        raise ValueError('Ensemble input schemas differ')
    features, progress, increments, original_progress = ([], [], [], [])
    for row in registry.itertuples(index=False):
        dates = pd.date_range(row.sowing_date, periods=330)
        frame = groups[str(row.site_id)].reindex(dates)
        supplied = frame[['tmean_c', 'tmax_c', 'tmin_c', 'precipitation_mm']].to_numpy()
        x, p, inc = build_features(supplied, dates, row.latitude, metadata['gates'])
        _, original, _ = build_features(supplied, dates, row.latitude, DEFAULT_GATES)
        features.append(x)
        progress.append(p)
        increments.append(inc)
        original_progress.append(original)
    features, progress, increments = map(np.asarray, [features, progress, increments])
    torch.set_num_threads(2)
    predictions = []
    for member in package['members']:
        model = WheatPhenology(member['metadata'], member['mode'])
        model.load_state_dict(member['state_dict'])
        predictions.append(predict_batches(model, features, progress, increments, np.arange(len(registry))))
    parents = np.mean(predictions, axis=0)
    extend_cdf(parents, [0.25, 0.75, 0.5])
    return (parents, np.asarray(original_progress))

def candidate_grids(parents, progress, registry):
    learned = np.empty((len(registry), 3, len(GRID)))
    process = np.empty_like(learned)
    for k, fraction in enumerate(GRID):
        learned[:, 0, k] = median_indices(interpolate(parents[:, 1], parents[:, 2], fraction))
        learned[:, 1, k] = learned[:, 0, k]
        learned[:, 2, k] = median_indices(interpolate(parents[:, 2], parents[:, 3], fraction))
        threshold_leaf = 542.3 + fraction * (986.65 - 542.3)
        threshold_flowering = 986.65 + fraction * (1760.39 - 986.65)
        for j, threshold in enumerate([threshold_leaf, threshold_leaf, threshold_flowering]):
            hit = progress >= threshold
            process[:, j, k] = np.where(hit.any(axis=1), hit.argmax(axis=1), np.nan)
    calendar = calendar_offsets(registry)
    calendar = np.where((calendar >= 0) & (calendar < 330), calendar, np.nan)
    return dict(lstm_extension=learned, matched_tpv=process, calendar=np.repeat(calendar[:, None, :], 3, axis=1))

def fit_candidates(training, registry, grids):
    assert_training(training, registry)
    field_index = dict(zip(registry.field_id, registry.prediction_index))
    fits, profiles, predictions = ({}, [], {})
    for name, predicted in grids.items():
        costs = []
        for j, event in enumerate(EVENTS):
            sample = training.loc[training.event.eq(event) & training.eligible_constraint]
            if sample.empty:
                raise ValueError(f'No original-training stage support for {event}')
            indices = sample.field_id.map(field_index).to_numpy()
            losses = interval_loss(predicted[indices, j], sample.lower_index.to_numpy()[:, None], sample.upper_index.to_numpy()[:, None], sample.horizon_index.to_numpy()[:, None])
            costs.append(losses.mean(axis=0))
        costs = np.asarray(costs) / 3
        if name == 'calendar':
            selected, profile = ordered_fit(costs)
            values = np.arange(365)
        else:
            leaf, leaf_profile = ordered_fit(costs[:2])
            flowering, flowering_profile = ordered_fit(costs[2:])
            selected = np.r_[leaf, flowering]
            leaf_minimum = costs[0, leaf[0]] + costs[1, leaf[1]]
            flowering_minimum = costs[2, flowering[0]]
            profile = np.vstack([leaf_profile + flowering_minimum, flowering_profile + leaf_minimum])
            values = GRID
        minimum = sum((costs[j, k] for j, k in enumerate(selected)))
        predictions[name] = predicted[:, np.arange(3), selected]
        fits[name] = dict(selected_indices=selected.tolist(), selected_values=values[selected].tolist(), training_loss_days=float(minimum), constrained_training_fields={str(event): int(training.loc[training.event.eq(event) & training.eligible_constraint, 'field_id'].nunique()) for event in EVENTS})
        for j, event in enumerate(EVENTS):
            for k, value in enumerate(values):
                profiles.append(dict(model=name, event=event, grid_index=k, grid_value=value, training_profile_loss_days=profile[j, k], excess_loss_days=profile[j, k] - minimum, selected=k == selected[j], within_0_days=bool(np.isclose(profile[j, k], minimum, rtol=0, atol=1e-12)), within_1_day=bool(profile[j, k] <= minimum + 1 + 1e-12), within_2_days=bool(profile[j, k] <= minimum + 2 + 1e-12)))
    return (fits, pd.DataFrame(profiles), predictions)

def archived_predictions(registry):
    old = pd.read_csv(ROOT / 'data/final_study/evidence/overwinter_leaf_model/manuscript/replay/full_season_top3_timing_and_conditional_yield.csv')
    joined = registry[['field_id', 'sowing_date']].merge(old[['field_id'] + [f'BBCH{e}_date' for e in EVENTS]], on='field_id', how='left', validate='one_to_one', indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('A registered field lacks an archived process prediction')
    result = []
    for event in EVENTS:
        offsets = (pd.to_datetime(joined[f'BBCH{event}_date']) - joined.sowing_date).dt.days.to_numpy(float)
        if np.any(offsets < 0):
            raise ValueError('An archived event predates its scenario sowing')
        result.append(np.where(offsets < 330, offsets, np.nan))
    return np.stack(result, axis=1)

def prediction_frame(registry, model_predictions):
    rows = []
    for name, predicted in model_predictions.items():
        for meta in registry.itertuples(index=False):
            for j, event in enumerate(EVENTS):
                index = predicted[meta.prediction_index, j]
                rows.append(dict(field_id=meta.field_id, event=event, model=name, predicted_index=index, predicted_date=(meta.sowing_date + pd.Timedelta(days=int(index))).date().isoformat() if np.isfinite(index) else None, prediction_reached=bool(np.isfinite(index))))
    return pd.DataFrame(rows)

def score_predictions(all_constraints, predictions):
    selected = all_constraints.loc[all_constraints.eligible_constraint]
    joined = selected.merge(predictions, on=['field_id', 'event'], how='left', validate='many_to_many')
    if len(joined) != len(selected) * len(MODELS) or joined.model.isna().any():
        raise ValueError('Prediction comparison lacks common field/event membership')
    joined['loss_days'] = interval_loss(joined.predicted_index, joined.lower_index, joined.upper_index, joined.horizon_index)
    joined['loss_is_lower_bound'] = ~joined.prediction_reached
    return joined

def run(output):
    output = Path(output)
    manuscript = ROOT / 'data/final_study/evidence/publication_source/Manuscript.docx'
    manuscript_start = sha(manuscript)
    protocol = json.loads((HERE / 'protocol.json').read_text())
    for relative, digest in protocol['input_sha256'].items():
        if sha(ROOT / relative) != digest:
            raise ValueError(f'Frozen input changed: {relative}')
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', protocol)
    registry, expanded, legacy = load_sources(ROOT)
    registry.to_csv(output / 'field_registry.csv', index=False)
    all_constraints = []
    quality = []
    for view, source_rows in [('expanded', expanded), ('legacy', legacy)]:
        normalized, rejected = qualify(source_rows, registry)
        normalized.to_csv(output / f'{view}_qualified_stage_dates.csv', index=False)
        rejected.to_csv(output / f'{view}_excluded_source_rows.csv', index=False)
        source_rows.to_csv(output / f'{view}_source_stage_rows.csv', index=False)
        stage_constraints = constraints(normalized, registry).assign(view=view)
        all_constraints.append(stage_constraints)
        quality.append(dict(view=view, source_rows=len(source_rows), qualified_dates=len(normalized), qualified_fields=int(normalized.field_id.nunique()), excluded_rows=len(rejected), exclusions_by_reason={str(k): int(v) for k, v in rejected.reason.value_counts().items()}))
    all_constraints = pd.concat(all_constraints, ignore_index=True)
    all_constraints.to_csv(output / 'stage_constraints.csv', index=False)
    write_json(output / 'source_quality.json', quality)
    print('Source qualification complete; generating frozen weather-only predictions.', flush=True)
    parents, progress = predictor_arrays(registry)
    grids = candidate_grids(parents, progress, registry)
    training = all_constraints.loc[all_constraints.view.eq('expanded') & all_constraints.partition.eq('calibration')].copy()
    fits, profiles, predictions = fit_candidates(training, registry, grids)
    write_json(output / 'fitted_parameters.json', fits)
    training.to_csv(output / 'training_constraints.csv', index=False)
    profiles.to_csv(output / 'training_profiles.csv', index=False)
    freeze = dict(status='frozen_before_evaluation_loss', created_utc=datetime.now(timezone.utc).isoformat(), fitted_parameters_sha256=sha(output / 'fitted_parameters.json'), training_constraints_sha256=sha(output / 'training_constraints.csv'), protocol_sha256=sha(output / 'protocol.json'), input_sha256=protocol['input_sha256'], computation_sha256={p.name: sha(p) for p in sorted(HERE.glob('*.py'))})
    write_json(output / 'fit_freeze_receipt.json', freeze)
    print('Three candidates fitted on original calibration fields; parameters frozen before evaluation scoring.', flush=True)
    predictions['archived_tpv'] = archived_predictions(registry)
    frame = prediction_frame(registry, predictions)
    frame.to_csv(output / 'predictions.csv', index=False)
    extended = extend_cdf(parents, fits['lstm_extension']['selected_values'])
    if not np.array_equal(median_indices(extended[:, [2, 3, 5]]), predictions['lstm_extension'], equal_nan=True):
        raise AssertionError('Selected grid and daily probability interface disagree')
    np.savez_compressed(output / 'daily_arrays.npz', field_ids=registry.field_id.to_numpy(str), parent_cdf=parents, extended_cdf=extended, original_tpv_progress=progress)
    scores = score_predictions(all_constraints, frame)
    scores.to_csv(output / 'scores.csv', index=False)
    table = summaries(scores)
    table.to_csv(output / 'summary.csv', index=False)
    primary = scores.loc[scores.view.eq('expanded') & ~scores.partition.eq('calibration')]
    bootstrap, draws = site_bootstrap(primary, repeats=protocol['bootstrap_repeats'], seed=protocol['bootstrap_seed'])
    bootstrap.to_csv(output / 'bootstrap_summary.csv', index=False)
    draws.to_csv(output / 'bootstrap_draws.csv', index=False)
    result = decision(primary, bootstrap)
    write_json(output / 'decision.json', result)
    support = all_constraints.groupby(['view', 'partition', 'event']).agg(registered_fields=('field_id', 'size'), constrained_fields=('eligible_constraint', 'sum'), bracketed_fields=('genuinely_bracketed', 'sum'), median_bracket_width_days=('bracket_width_days', 'median')).reset_index()
    support.to_csv(output / 'stage_support.csv', index=False)
    manuscript_end = sha(manuscript)
    receipt = dict(schema_version=2, status='complete', created_utc=datetime.now(timezone.utc).isoformat(), inputs_unchanged=all((sha(ROOT / p) == h for p, h in protocol['input_sha256'].items())), manuscript_start_sha256=manuscript_start, manuscript_end_sha256=manuscript_end, manuscript_unchanged=manuscript_start == manuscript_end, fields=len(registry), fitted_on_evaluation=False, disease_values_used=False, climate_host_changed=False, source_outcomes_previously_accessed=True, outputs={p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file()}, software_versions=dict(numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__))
    write_json(output / 'receipt.json', receipt)
    print(bootstrap.to_string(index=False), flush=True)
    print(json.dumps(result, indent=2), flush=True)
    return receipt
