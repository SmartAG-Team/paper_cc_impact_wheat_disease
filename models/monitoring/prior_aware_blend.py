"""A fixed, pre-query-history weighting rule for the existing fitted components."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .evaluation import paired_interval, score_predictions, score_subset
from .inputs import sha
from .robust_models import predict_cluster_ensemble
from .field_history_models import predict_ensemble

ROOT = Path(__file__).resolve().parents[2]
RULE = dict(minimum_same_leaf_severity=.75, full_history_weight_at=1.,
            maximum_age_days=30., base_history_weight=.5,
            other_leaf_only_history_weight=.5)


def combine(process, history_prediction, history, queries):
    """Increase history weight only for recent high same-native-leaf observations."""
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    p, h = np.asarray(process, float), np.asarray(history_prediction, float)
    if (p.shape != (len(queries),) or h.shape != p.shape or len(history) != len(queries)
            or not np.isfinite(np.c_[p, h]).all() or np.any((p < 0)|(p > 100)|(h < 0)|(h > 100))):
        raise ValueError('Aligned bounded component predictions required')
    prior = history.latest_severity_fraction.to_numpy(float)
    age = history.days_since_latest_scaled.to_numpy(float)
    available = history.prior_assessment_available.to_numpy(float)
    if (not np.isfinite(np.c_[prior, age, available]).all() or not np.isin(available, [0, 1]).all()
            or np.any((prior < 0)|(prior > 1)|(age < 0))):
        raise ValueError('Finite bounded earlier history required')
    increase = np.clip((prior-RULE['minimum_same_leaf_severity']) /
                       (RULE['full_history_weight_at']-RULE['minimum_same_leaf_severity']), 0, 1)
    eligible = (available == 1) & (age <= RULE['maximum_age_days']/30)
    weight = RULE['base_history_weight'] + .5*increase*eligible
    return (1-weight)*p+weight*h, weight


def predict_bundle(fitted, process, station, history, queries):
    """Future native-domain inference; endpoint qualification remains upstream."""
    base = fitted['base']
    if fitted['blend_rule'] != RULE:
        raise ValueError('Frozen blending rule differs from the inference implementation')
    if 'observed_percent' in queries:
        raise ValueError('Query labels must be withheld')
    dates = pd.to_datetime(queries.date, errors='coerce', format='mixed')
    if (dates.isna().any() or not dates.dt.year.eq(queries.season_year).all()
            or not queries.source.isin(['BASF', 'Corteva']).all()
            or not queries.leaf_index.isin(range(7)).all()
            or not queries.season_year.ge(base['minimum_assessment_year']).all()):
        raise ValueError('Queries outside the frozen future native-source domain')
    p, s = np.asarray(process, float), np.asarray(station, float)
    if (p.shape != (len(queries), 37) or s.shape != (len(queries), 53)
            or history.columns.tolist() != base['history_names']
            or history.shape != (len(queries), len(base['history_names']))
            or not np.isfinite(p).all() or not np.isfinite(s).all()
            or not np.isfinite(history.to_numpy(float)).all()):
        raise ValueError('Complete aligned predictor schema required')
    first = predict_cluster_ensemble(base['process'], p, history, queries)
    second = predict_ensemble(base['history'], s, history, queries)
    return combine(first, second, history, queries)[0]


def run(root=ROOT, output=None):
    root = Path(root).resolve()
    base = root/'outputs/disease_improvement/station_candidate_20261022_v2'
    out = Path(output or root/'outputs/disease_improvement/prior_aware_blend_20261022').resolve()
    if out.exists():
        raise FileExistsError('A new output directory is required')
    obs = pd.read_csv(base/'training_observation_ledger.csv')
    arrays = np.load(base/'predictors.npz')
    fitted_base = joblib.load(base/'fitted_candidate.joblib')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    history = pd.DataFrame(arrays['history'], columns=fitted_base['history_names'])
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    sources = [base/n for n in ['fitted_candidate.joblib', 'fitted_candidate_lock.json', 'registration.json',
        'training_observation_ledger.csv', 'predictors.npz', 'assessment_predictions.csv',
        'spatial_predictions.csv', 'verification.json', 'split_membership.json']]
    codes = [Path(__file__)] + [Path(__file__).with_name(n+'.py') for n in
        ['models', 'robust_models', 'field_history_models', 'evaluation']]
    codes += [root/'analysis/final_study/metrics.py']
    out.mkdir(parents=True)
    protocol = dict(registered_utc=datetime.now(timezone.utc).isoformat(), rule=RULE,
        changed='Only convex prediction weighting; all fitted components and their training memberships unchanged',
        hypothesis='Equal weighting suppresses accurate recent high-severity same-native-leaf history',
        design_informed_by_retrospective_outcomes=True,
        descriptive_diagnostic='High-prior anchor group includes68assessments; process35.810pp/history8.421pp/equalblend18.773pp. This is not independent evidence and includes other-leaf anchors excluded from the new rule.',
        candidate_rules=1, hyperparameter_search=False, observations_removed=0,
        independent_external_validation=False, publication_model_promoted=False,
        source_sha256={str(p.relative_to(root)): sha(p) for p in sources},
        code_sha256={str(p.relative_to(root)): sha(p) for p in codes})
    (out/'registration.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for path in codes:
        dest = out/'code_snapshot'/path.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(path.read_bytes())
    metrics, intervals, groups = [], [], []
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame = pd.read_csv(base/filename)
        h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        pred, weight = combine(frame.candidate_process, frame.candidate_history, h,
                               frame.drop(columns='observed_percent'))
        frame['prior_aware_blend'], frame['history_weight'] = pred, weight
        frame['latest_same_leaf_percent'] = 100*h.latest_severity_fraction
        frame['latest_same_leaf_age_days'] = 30*h.days_since_latest_scaled
        frame.to_csv(out/filename, index=False)
        for name in [n for n in ['prior_aware_blend', 'station_candidate', 'field_split_mean', 'current', 'reference_replay'] if n in frame]:
            metrics.append(dict(evaluation=label, model=name, **score_predictions(frame, name)))
            if name != 'prior_aware_blend':
                ci = paired_interval(frame, 'prior_aware_blend', name)
                intervals.append(dict(evaluation=label, reference=name, lower=ci[0], upper=ci[1]))
        if label == 'temporal':
            masks = dict(cold=~frame.prior_assessment_available, warm=frame.prior_assessment_available,
                first_field=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(0),
                first_leaf_other_history=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(1),
                BASF=frame.source.eq('BASF'), Corteva=frame.source.eq('Corteva'),
                severe_above75=frame.observed_percent.gt(75), new_site=frame.new_site,
                rule_active=weight > .5)
            for population, mask in masks.items():
                for name in ['prior_aware_blend', 'station_candidate', 'reference_replay']:
                    groups.append(dict(population=population, model=name, **score_subset(frame, mask, name)))
    pd.DataFrame(metrics).to_csv(out/'metrics.csv', index=False)
    pd.DataFrame(intervals).to_csv(out/'paired_intervals.csv', index=False)
    pd.DataFrame(groups).to_csv(out/'subgroup_metrics.csv', index=False)
    fitted = dict(base=fitted_base, blend_rule=RULE, training_ids=fitted_base['training_ids'])
    joblib.dump(fitted, out/'fitted_candidate.joblib')
    contract = json.loads((base/'fitted_candidate_lock.json').read_text())
    contract.update(created_utc=datetime.now(timezone.utc).isoformat(),
        role='Frozen development candidate with fixed prior-aware blending; external validation required',
        blend_rule=RULE, base_candidate=str(base.relative_to(root)),
        design_informed_by_retrospective_outcomes=True, full_fit_predictions_are_validation=False,
        sha256={n: sha(out/n) for n in ['fitted_candidate.joblib', 'registration.json']},
        parent_contract_sha256=sha(base/'fitted_candidate_lock.json'))
    (out/'fitted_candidate_lock.json').write_text(json.dumps(contract, indent=2)+'\n')
    return pd.DataFrame(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(output=args.output).to_string(index=False))
