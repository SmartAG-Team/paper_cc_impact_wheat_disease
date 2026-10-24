"""Independent arithmetic replay of the fixed prior-aware weighting probe."""
import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .evaluation import paired_interval, score_predictions, score_subset
from .inputs import sha
from .verify_french_weather import write_verification_receipt

ROOT = Path(__file__).resolve().parents[2]


def verify(root, output):
    root, out = Path(root).resolve(), Path(output).resolve()
    registration = json.loads((out/'registration.json').read_text())
    for path, digest in registration['source_sha256'].items():
        assert sha(root/path) == digest, path
    for path, digest in registration['code_sha256'].items():
        assert sha(out/'code_snapshot'/path) == digest, path
    base = root/'outputs/disease_improvement/station_candidate_20261022_v2'
    bundle = joblib.load(out/'fitted_candidate.joblib')
    original = joblib.load(base/'fitted_candidate.joblib')
    assert joblib.hash(bundle['base']) == joblib.hash(original)
    assert bundle['training_ids'] == original['training_ids']
    contract = json.loads((out/'fitted_candidate_lock.json').read_text())
    for path, digest in contract['sha256'].items():
        assert sha(out/path) == digest, path
    assert contract['parent_contract_sha256'] == sha(base/'fitted_candidate_lock.json')
    arrays = np.load(base/'predictors.npz')
    obs = pd.read_csv(base/'training_observation_ledger.csv')
    history = pd.DataFrame(arrays['history'], columns=original['history_names'])
    position = pd.Series(np.arange(len(obs)), index=obs.global_target_index)
    frames, masks = {}, {}
    for label, filename in [('temporal', 'assessment_predictions.csv'), ('spatial_50km', 'spatial_predictions.csv')]:
        frame, old = pd.read_csv(out/filename), pd.read_csv(base/filename)
        pd.testing.assert_frame_equal(frame[old.columns], old, check_dtype=False)
        assert len(frame) == 2017 and frame.global_target_index.is_unique
        h = history.iloc[position.loc[frame.global_target_index]].reset_index(drop=True)
        # Independent expression of the single registered rule; no outcome values enter it.
        eligible = h.prior_assessment_available.eq(1)&h.days_since_latest_scaled.le(1)
        weights = .5 + .5*np.minimum(1, np.maximum(0, (h.latest_severity_fraction-.75)/.25))*eligible
        expected = (1-weights)*frame.candidate_process+weights*frame.candidate_history
        np.testing.assert_allclose(weights, frame.history_weight, atol=1e-12)
        np.testing.assert_allclose(expected, frame.prior_aware_blend, atol=1e-12)
        np.testing.assert_allclose(frame.latest_same_leaf_percent, 100*h.latest_severity_fraction, atol=1e-12)
        np.testing.assert_allclose(frame.latest_same_leaf_age_days, 30*h.days_since_latest_scaled, atol=1e-12)
        frames[label] = frame
        if label == 'temporal':
            masks = dict(cold=~frame.prior_assessment_available, warm=frame.prior_assessment_available,
                first_field=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(0),
                first_leaf_other_history=h.prior_assessment_available.eq(0)&h.other_leaf_available.eq(1),
                BASF=frame.source.eq('BASF'), Corteva=frame.source.eq('Corteva'),
                severe_above75=frame.observed_percent.gt(75), new_site=frame.new_site,
                rule_active=weights > .5)
    metrics = pd.read_csv(out/'metrics.csv')
    for row in metrics.itertuples(index=False):
        for key, value in score_predictions(frames[row.evaluation], row.model).items():
            if value is not None:
                np.testing.assert_allclose(value, getattr(row, key), atol=1e-9)
    intervals = pd.read_csv(out/'paired_intervals.csv')
    for row in intervals.itertuples(index=False):
        np.testing.assert_allclose(paired_interval(frames[row.evaluation], 'prior_aware_blend', row.reference),
                                   [row.lower, row.upper], atol=1e-9)
    groups = pd.read_csv(out/'subgroup_metrics.csv')
    for row in groups.itertuples(index=False):
        for key, value in score_subset(frames['temporal'], masks[row.population], row.model).items():
            if value is not None:
                np.testing.assert_allclose(value, getattr(row, key), atol=1e-9)
    result = dict(verified=True, evaluation_records_per_condition=2017,
        original_predictions_and_cohort_preserved=True, parent_fitted_components_identical=True,
        training_records=2304, query_labels_used_in_weighting=False,
        metrics_recomputed=len(metrics), intervals_recomputed=len(intervals), subgroups_recomputed=len(groups),
        independent_external_validation=False, selected_for_publication=False, verifier_sha256=sha(Path(__file__)))
    write_verification_receipt(out, result, Path(__file__))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(ROOT, args.output), indent=2))
