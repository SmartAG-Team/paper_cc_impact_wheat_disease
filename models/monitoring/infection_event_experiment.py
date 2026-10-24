"""Fixed factorial tests of compact event inputs in the current component models."""
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval, score_predictions, score_subset
from .field_history_models import fit_ensemble as fit_history
from .field_history_models import predict_ensemble as predict_history
from .inputs import sha
from .robust_models import fit_cluster_ensemble, predict_cluster_ensemble

ROOT = Path(__file__).resolve().parents[2]
PROCESS_SPEC = dict(structure='process', process_baseline_column=-3, crossleaf=True)
HISTORY_SPEC = dict(structure='field_split')
BLENDS = dict(events_process_mean=['events_process', 'candidate_history'],
              events_history_mean=['candidate_process', 'events_history'],
              events_both_mean=['events_process', 'events_history'])


def populations(frame, history):
    return dict(cold=~frame.prior_assessment_available, warm=frame.prior_assessment_available,
        first_field=history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(0),
        first_leaf_other_history=history.prior_assessment_available.eq(0)&history.other_leaf_available.eq(1),
        BASF=frame.source.eq('BASF'), Corteva=frame.source.eq('Corteva'),
        severe_above75=frame.observed_percent.gt(75), new_site=frame.new_site,
        increase_above40=history.prior_assessment_available.eq(1)&
            (frame.observed_percent.to_numpy()-100*history.latest_severity_fraction.to_numpy() > 40))


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    base = root/'outputs/disease_improvement/knmi_station_candidate_20261022'
    data = root/'data/disease_improvement/infection_events_20261022'
    out = Path(output or root/'outputs/disease_improvement/infection_events_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    obs = pd.read_csv(base/'training_observation_ledger.csv')
    arrays = np.load(data/'predictors.npz')
    old_arrays = np.load(base/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    np.testing.assert_array_equal(arrays['history'], old_arrays['history'])
    parent_bundle = joblib.load(base/'fitted_candidate.joblib')
    history = pd.DataFrame(arrays['history'], columns=parent_bundle['history_names'])
    designs = dict(process=arrays['process_events'], history=arrays['history_events'])
    assert designs['process'].shape == (2304, 52) and designs['history'].shape == (2304, 68)
    assert np.isfinite(designs['process']).all() and np.isfinite(designs['history']).all()
    sources = [base/n for n in ['registration.json', 'training_observation_ledger.csv', 'predictors.npz',
        'assessment_predictions.csv', 'spatial_predictions.csv', 'split_membership.json',
        'fitted_candidate.joblib', 'fitted_candidate_lock.json', 'verification.json']]
    sources += [data/n for n in ['manifest.json', 'predictors.npz']]
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in
        ['infection_event_features', 'models', 'field_history_models', 'robust_models', 'features', 'history', 'evaluation']]
    codes += [root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        components=dict(process=PROCESS_SPEC, history=HISTORY_SPEC),
        blends={name: dict(components=components, weights=[.5, .5]) for name, components in BLENDS.items()},
        ensemble=dict(members=20, fraction=.8, seed=20261022),
        changed='One fixed15-feature timing set added to existing designs; no estimator, weight, cohort or split changes',
        process_weather='Existing reanalysis forcing', history_weather='Latest DWD/Meteo-France/KNMI station-preferred forcing',
        weather_cutoff_days=1,
        interval='After latest same-native-leaf observation if present, otherwise latest earlier other-leaf field date; no history uses existing21day fallback',
        earlier_event_models_unfavorable=True,
        distinction='Earlier event models used different unbagged same-leaf history structures; current controlled test changes inputs within the current field-history ensembles',
        design_informed_by_retrospective_development=True, independent_external_validation=False,
        hyperparameter_search=False, publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)): sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)): sha(p) for p in codes},
        environment={name: version(name) for name in ['numpy', 'pandas', 'scikit-learn', 'joblib']})
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in codes:
        dest = out/'code_snapshot'/path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    (out/'training_observation_ledger.csv').write_bytes((base/'training_observation_ledger.csv').read_bytes())
    (out/'split_membership.json').write_bytes((base/'split_membership.json').read_bytes())
    (out/'predictors.npz').write_bytes((data/'predictors.npz').read_bytes())
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    frames = {label: pd.read_csv(base/filename).set_index('global_target_index')
              for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]}
    predicted = dict(temporal=[], spatial_50km=[])
    for split in json.loads((base/'split_membership.json').read_text()):
        train = position.loc[split['train']].to_numpy()
        test = position.loc[split['test']].to_numpy()
        label = 'temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        part = frames[label].loc[split['test']].reset_index().copy()
        queries = obs.iloc[test].drop(columns='observed_percent')
        process = fit_cluster_ensemble(designs['process'][train], history.iloc[train], obs.iloc[train], PROCESS_SPEC)
        field = fit_history(designs['history'][train], history.iloc[train], obs.iloc[train], HISTORY_SPEC)
        part['events_process'] = predict_cluster_ensemble(process, designs['process'][test], history.iloc[test], queries)
        part['events_history'] = predict_history(field, designs['history'][test], history.iloc[test], queries)
        for component, fitted in [('process', process), ('history', field)]:
            joblib.dump(fitted, out/f"{split['fold']}_{component}.joblib")
        for name, components in BLENDS.items():
            part[name] = part[components].mean(axis=1)
        predicted[label].append(part)
        print(split['fold'], flush=True)
    metrics, intervals, groups = [], [], []
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame = pd.concat(predicted[label], ignore_index=True)
        frame.to_csv(out/filename, index=False)
        for name in ['events_process', 'events_history']+list(BLENDS)+['station_candidate', 'reference_replay']+(['current'] if label == 'temporal' else []):
            metrics.append(dict(evaluation=label, model=name, **score_predictions(frame, name)))
        for name in BLENDS:
            for reference in ['station_candidate', 'reference_replay']+(['current'] if label == 'temporal' else []):
                interval = paired_interval(frame, name, reference)
                intervals.append(dict(evaluation=label, model=name, reference=reference, lower=interval[0], upper=interval[1]))
        if label == 'temporal':
            h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
            for population, mask in populations(frame, h).items():
                for name in list(BLENDS)+['station_candidate', 'reference_replay']:
                    groups.append(dict(population=population, model=name, **score_subset(frame, mask, name)))
            frame.loc[frame.global_target_index.isin([479, 1750, 413, 159, 1231]),
                ['global_target_index', 'observed_percent', 'station_candidate']+list(BLENDS)].to_csv(out/'hard_cases.csv', index=False)
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    pd.DataFrame(groups).to_csv(out/'subgroup_metrics.csv', index=False)
    fitted = dict(base=parent_bundle,
        process=fit_cluster_ensemble(designs['process'], history, obs, PROCESS_SPEC),
        history=fit_history(designs['history'], history, obs, HISTORY_SPEC),
        history_names=history.columns.tolist(), blends=BLENDS, training_ids=obs.global_target_index.astype(int).tolist(),
        minimum_assessment_year=2020)
    joblib.dump(fitted, out/'fitted_candidates.joblib')
    contract = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        role='Fixed experimental component combinations; qualified external validation required',
        variants=list(BLENDS), selected_recipe=None, training_records=len(obs),
        minimum_validation_year=2020, source_domain=['BASF', 'Corteva'], native_codes=list(range(7)),
        predictor_widths=dict(base_process=37, base_history=53, augmented_process=52, augmented_history=68, disease_history=35),
        source_and_endpoint_contract=str((base/'fitted_candidate_lock.json').relative_to(root)),
        new_provider_mapping_required_before_outcomes=True, full_fit_predictions_are_validation=False,
        independent_external_validation=False, publication_model_promoted=False,
        sha256={n: sha(out/n) for n in ['fitted_candidates.joblib', 'registration.json', 'predictors.npz',
                                      'training_observation_ledger.csv', 'split_membership.json']})
    (out/'fitted_candidate_lock.json').write_text(json.dumps(contract, indent=2)+'\n')
    print('full candidates frozen', flush=True)
    return pd.DataFrame(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output).to_string(index=False))
