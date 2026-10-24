"""Integrate qualified station data and records into the locked candidate recipe."""
import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import re

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import distances_km, paired_interval, score_predictions, score_subset
from .field_history_models import fit_ensemble as fit_history_ensemble
from .field_history_models import predict_ensemble as predict_history_ensemble
from .inputs import sha
from .robust_models import fit_cluster_ensemble, predict_cluster_ensemble
from .stage_experiment import augment

ROOT = Path(__file__).resolve().parents[2]
PROCESS_SPEC = dict(structure='process', process_baseline_column=-3, crossleaf=True)
HISTORY_SPEC = dict(structure='field_split')
CANDIDATE = 'station_candidate'


def load_station_design(path, target_ids):
    with np.load(path) as arrays:
        if not np.array_equal(arrays['target_ids'], target_ids):
            raise ValueError('Station predictor identity/order differs from the training pool')
        station = arrays['station']
        if station.shape != (len(target_ids), 53) or not np.isfinite(station).all():
            raise ValueError('Complete 53-column station predictor schema required')
        return station.copy()


def extend_spatial_split(frame, added_ids, split, assignment):
    """Append eligible additions without changing any original fold membership."""
    indexed = frame.set_index('global_target_index')
    train, test = list(split['train']), list(split['test'])
    held = indexed.loc[test]
    if held.season_year.nunique() != 1:
        raise ValueError('A single held assessment year is required')
    year = int(held.season_year.iloc[0])
    fold = int(re.search(r'sitefold(\d+)', split['fold']).group(1))
    extra = indexed.loc[list(added_ids)]
    eligible = extra.loc[extra.season_year.lt(year) & extra.site_id.map(assignment).ne(fold)]
    if len(eligible):
        distance = distances_km(eligible[['latitude', 'longitude']], held[['latitude', 'longitude']])
        eligible = eligible.loc[distance.min(axis=1) >= 50]
    additions = eligible.index.astype(int).tolist()
    if set(train) & set(added_ids) or set(test) & set(added_ids):
        raise ValueError('Additions must be disjoint from the original benchmark')
    return train + additions, test, additions


def predict_bundle(bundle, process, station, history, queries):
    """Predict a future qualified native-domain query without its disease label."""
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    dates = pd.to_datetime(queries.date, errors='coerce', format='mixed')
    if (dates.isna().any() or not dates.dt.year.eq(queries.season_year).all()
            or not queries.source.isin(['BASF', 'Corteva']).all()
            or not queries.leaf_index.isin(range(7)).all()
            or not queries.season_year.ge(bundle['minimum_assessment_year']).all()):
        raise ValueError('Query outside the frozen source, native-code or future-year domain')
    p, s = np.asarray(process, float), np.asarray(station, float)
    if (p.shape != (len(queries), 37) or s.shape != (len(queries), 53)
            or history.shape != (len(queries), len(bundle['history_names']))
            or history.columns.tolist() != bundle['history_names']
            or not np.isfinite(p).all() or not np.isfinite(s).all()
            or not np.isfinite(history.to_numpy(float)).all()):
        raise ValueError('Complete aligned frozen predictor schema required')
    weights = bundle['blend_weights']
    return (weights[0] * predict_cluster_ensemble(bundle['process'], p, history, queries)
            + weights[1] * predict_history_ensemble(bundle['history'], s, history, queries))


def run(root=ROOT, output=None, station_override=None):
    root = Path(root).resolve()
    out = Path(output or root/'outputs/disease_improvement/station_candidate_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    old = root/'outputs/disease_improvement/field_history_20261022'
    process_base = root/'outputs/disease_improvement/station_repairs_20261022_v4'
    raw_extra = root/'data/disease_improvement/coordinate_augmentation_20261022'
    station_extra = root/'data/disease_improvement/coordinate_augmentation_station_20261022'
    schema = root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json'
    lock = json.loads((old/'candidate_recipe_lock.json').read_text())
    assert lock['candidate'] == 'field_split_mean' and lock['blend_weights'] == [.5, .5]
    original = pd.read_csv(old/'observation_ledger.csv')
    added = pd.read_csv(station_extra/'observations.csv')
    pd.testing.assert_frame_equal(added, pd.read_csv(raw_extra/'observations.csv'))
    obs = pd.concat([original, added], ignore_index=True)
    assert obs.global_target_index.is_unique
    prior = np.load(old/'predictors.npz')
    process_old = np.load(process_base/'predictors.npz')
    process_extra = np.load(raw_extra/'predictors.npz')
    station_new = np.load(station_extra/'predictors.npz')
    for arrays, expected in [(prior, original), (process_old, original),
                             (process_extra, added), (station_new, added)]:
        np.testing.assert_array_equal(arrays['target_ids'], expected.global_target_index)
    names = json.loads(schema.read_text())['history_names']
    history = pd.DataFrame(np.vstack([prior['history'], station_new['history']]), columns=names)
    process = np.vstack([process_old['reanalysis'], process_extra['matrix']])
    station = np.vstack([prior['matrix'], augment(station_new['matrix'], station_new['stage'])])
    override = Path(station_override).resolve() if station_override is not None else None
    if override is not None:
        station = load_station_design(override, obs.global_target_index.to_numpy())
    assert process.shape == (len(obs), 37) and station.shape == (len(obs), 53)
    assert np.isfinite(process).all() and np.isfinite(station).all()
    sites = sorted(original.site_id.unique())
    ordered = sorted(sites, key=lambda s: hashlib.sha256(str(s).encode()).hexdigest())
    assignment = {s: i % 3 for i, s in enumerate(ordered)}
    for site in sorted(set(added.site_id) - set(sites)):
        assignment[site] = int(hashlib.sha256(str(site).encode()).hexdigest(), 16) % 3
    sources = [old/n for n in ['candidate_recipe_lock.json', 'observation_ledger.csv', 'predictors.npz',
                               'assessment_predictions.csv', 'spatial_predictions.csv', 'spatial_membership.json']]
    sources += [schema, process_base/'predictors.npz', raw_extra/'observations.csv', raw_extra/'predictors.npz']
    sources += [station_extra/n for n in ['manifest.json', 'observations.csv', 'predictors.npz',
                                        'daily_weather.parquet', 'stage_calendar.csv', 'fields.csv']]
    if override is not None:
        sources += [override, override.with_name('manifest.json')]
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in
        ['field_history_models', 'models', 'robust_models', 'evaluation', 'history', 'stage_experiment']]
    codes += [root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        recipe='Locked field_split_mean; no weight, feature, architecture or hyperparameter search',
        process_spec=PROCESS_SPEC, history_spec=HISTORY_SPEC, blend_weights=[.5, .5],
        ensemble=dict(members=20, fraction=.8, seed=20261022),
        original_records=len(original), added_records=len(added), evaluation_records=2017,
        additions_training_only=True, outcome_based_record_selection=False,
        fixed_old_spatial_membership=True, site_assignment=assignment,
        station_design_override=str(override.relative_to(root)) if override is not None else None,
        history='Strictly earlier native-code observations within the same field; allowed at held sites',
        station_preference='Qualified station precipitation, temperature and humidity in history component; missing/QC-failed days use reanalysis',
        process_component='Unchanged reanalysis feature recipe to preserve the locked blend',
        independent_external_validation=False, adaptive_retrospective_development=True,
        publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)): sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)): sha(p) for p in codes},
        environment={n: version(n) for n in ['numpy', 'pandas', 'scikit-learn', 'joblib']})
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in codes:
        dest = out/'code_snapshot'/path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    obs.to_csv(out/'training_observation_ledger.csv', index=False)
    np.savez_compressed(out/'predictors.npz', target_ids=obs.global_target_index,
                        process=process, station=station, history=history.to_numpy())
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    temporal, spatial, memberships = [], [], []

    def fit_fold(train_ids, test_ids, block, prefix):
        train, test = position.loc[train_ids].to_numpy(), position.loc[test_ids].to_numpy()
        queries = obs.iloc[test].drop(columns='observed_percent')
        first = fit_cluster_ensemble(process[train], history.iloc[train], obs.iloc[train], PROCESS_SPEC)
        second = fit_history_ensemble(station[train], history.iloc[train], obs.iloc[train], HISTORY_SPEC)
        block = block.copy()
        block['candidate_process'] = predict_cluster_ensemble(first, process[test], history.iloc[test], queries)
        block['candidate_history'] = predict_history_ensemble(second, station[test], history.iloc[test], queries)
        block[CANDIDATE] = .5*(block.candidate_process+block.candidate_history)
        joblib.dump(first, out/f'{prefix}_process.joblib')
        joblib.dump(second, out/f'{prefix}_history.joblib')
        return block

    for year, block in pd.read_csv(old/'assessment_predictions.csv').groupby('season_year'):
        train = obs.loc[obs.season_year.lt(year), 'global_target_index'].astype(int).tolist()
        test = block.global_target_index.astype(int).tolist()
        prefix = f'temporal{year}'
        temporal.append(fit_fold(train, test, block, prefix))
        memberships.append(dict(fold=prefix, train=train, test=test,
            added_train_ids=sorted(set(train)&set(added.global_target_index))))
        print(prefix, flush=True)
    old_spatial = pd.read_csv(old/'spatial_predictions.csv').set_index('global_target_index')
    for number, split in enumerate(json.loads((old/'spatial_membership.json').read_text())):
        if not split['train']:
            continue
        train, test, additions = extend_spatial_split(obs, added.global_target_index.tolist(), split, assignment)
        prefix = f'spatial{number}'
        spatial.append(fit_fold(train, test, old_spatial.loc[test].reset_index(), prefix))
        memberships.append(dict(fold=prefix, original_fold=split['fold'], train=train,
                                test=test, added_train_ids=additions))
        print(prefix, flush=True)
    (out/'split_membership.json').write_text(json.dumps(memberships, indent=2)+'\n')
    metrics, intervals, groups = [], [], []
    for label, frame, filename in [('temporal', pd.concat(temporal, ignore_index=True), 'assessment_predictions.csv'),
                                   ('spatial_50km', pd.concat(spatial, ignore_index=True), 'spatial_predictions.csv')]:
        frame.to_csv(out/filename, index=False)
        for name in [n for n in [CANDIDATE, 'candidate_process', 'candidate_history', 'field_split_mean', 'current', 'reference_replay'] if n in frame]:
            metrics.append(dict(evaluation=label, model=name, **score_predictions(frame, name)))
        for reference in [n for n in ['field_split_mean', 'current', 'reference_replay'] if n in frame]:
            interval = paired_interval(frame, CANDIDATE, reference)
            intervals.append(dict(evaluation=label, reference=reference, lower=interval[0], upper=interval[1]))
        if label == 'temporal':
            h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
            masks = dict(cold=~frame.prior_assessment_available, warm=frame.prior_assessment_available,
                first_field=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(0),
                first_leaf_other_history=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(1),
                BASF=frame.source.eq('BASF'), Corteva=frame.source.eq('Corteva'),
                severe_above75=frame.observed_percent.gt(75), new_site=frame.new_site)
            for population, mask in masks.items():
                for name in [CANDIDATE, 'field_split_mean', 'reference_replay']:
                    groups.append(dict(population=population, model=name, **score_subset(frame, mask, name)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    pd.DataFrame(groups).to_csv(out/'subgroup_metrics.csv', index=False)
    # This fit is frozen for future external use; its training predictions are never validation evidence.
    bundle = dict(process=fit_cluster_ensemble(process, history, obs, PROCESS_SPEC),
        history=fit_history_ensemble(station, history, obs, HISTORY_SPEC), blend_weights=[.5, .5],
        history_names=names, minimum_assessment_year=int(obs.season_year.max())+1,
        training_ids=obs.global_target_index.astype(int).tolist())
    joblib.dump(bundle, out/'fitted_candidate.joblib')
    contract = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        role='Frozen development candidate; independent endpoint-qualified validation required',
        training_records=len(obs), training_fields=int(obs.field_id.nunique()),
        maximum_training_year=int(obs.season_year.max()), minimum_validation_year=bundle['minimum_assessment_year'],
        source_domain=['BASF', 'Corteva'], native_codes=list(range(7)),
        cross_source_anatomical_equivalence_verified=False,
        new_provider_policy='Unsupported until a metadata-qualified input mapping and new version are frozen before disease outcomes',
        endpoint='Untreated, zero-spray percentage disease severity for a source-native numbered leaf; leaf-ranking and scoring denominator need documented compatibility',
        excluded_endpoints=['whole-plant coverage', 'incidence', 'ordinal score', 'selected infected leaves'],
        history='Only observations strictly before the query date; no same-day other-leaf outcomes',
        weather='Daily UTC through the day before query; station QC and timing qualification; per-variable/day reanalysis fallback',
        sowing='Explicit pre-outcome calendar scenario; not an observed sowing date',
        prediction_features=dict(process=37, station_stage=53, disease_history=len(names)),
        full_fit_predictions_are_validation=False, independent_external_validation=False,
        publication_model_promoted=False, goal_achieved=False,
        sha256={n: sha(out/n) for n in ['fitted_candidate.joblib', 'training_observation_ledger.csv',
                'predictors.npz', 'registration.json', 'split_membership.json']})
    (out/'fitted_candidate_lock.json').write_text(json.dumps(contract, indent=2)+'\n')
    print('full candidate frozen', flush=True)
    return pd.DataFrame(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--station-override', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output, station_override=args.station_override).to_string(index=False))
