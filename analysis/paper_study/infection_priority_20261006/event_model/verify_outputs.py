"""Independent post-run arithmetic and immutable artifact verification."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_weights(frame):
    """Manual hierarchy shares without the model's weight helper."""
    hierarchy = {}
    for index, (year, field, leaf) in enumerate(frame[["coordinate_year", "field_id", "leaf_index"]].itertuples(index=False, name=None)):
        hierarchy.setdefault(year, {}).setdefault(field, {}).setdefault(leaf, []).append(index)
    result = np.zeros(len(frame), float)
    for fields in hierarchy.values():
        for leaves in fields.values():
            for indices in leaves.values():
                result[indices] = 1/(len(hierarchy)*len(fields)*len(leaves)*len(indices))
    return result


def main():
    fixture = pd.DataFrame(dict(coordinate_year=["X", "X", "X", "Y"],
        field_id=["A", "A", "B", "C"], leaf_index=[0, 1, 0, 0]))
    assert independent_weights(fixture).tolist() == [.125, .125, .25, .5]
    receipt = json.loads((HERE / "receipt.json").read_text())
    configuration = json.loads((HERE / "configuration_before_fitting.json").read_text())
    for relative, expected in receipt["output_sha256"].items():
        assert digest(HERE / relative) == expected, relative
    for relative, expected in configuration["input_and_source_sha256"].items():
        assert digest(ROOT / relative) == expected, relative
    a = pd.concat([pd.read_parquet(HERE / "out_of_fold_assessment_sign_predictions.parquet"),
                   pd.read_parquet(HERE / "frozen_v6_descriptive_assessment_sign_predictions.parquet")], ignore_index=True)
    b = pd.concat([pd.read_parquet(HERE / "out_of_fold_first_symptom_brackets.parquet"),
                   pd.read_parquet(HERE / "frozen_v6_descriptive_first_symptom_brackets.parquet")], ignore_index=True)
    detection = pd.read_csv(HERE / "assessment_detection_metrics.csv")
    onset = pd.read_csv(HERE / "onset_bracket_metrics.csv")
    for row in detection.itertuples():
        part = a.loc[a.evaluation_kind.eq(row.evaluation_kind) & a.model.eq(row.model)]
        if row.source != "pooled":
            part = part.loc[part.source.eq(row.source)]
        if row.leaf_scope == "top3":
            part = part.loc[part.leaf_index.lt(3)]
        weights = independent_weights(part)
        observed, predicted = part.observed_positive.to_numpy(bool), part.predicted_positive.to_numpy(bool)
        accuracy = np.sum(weights * (observed == predicted))
        assert abs(accuracy-row.weighted_accuracy) < 1e-12
        if observed.any():
            sensitivity = np.sum(weights[observed] * predicted[observed])/sum(weights[observed])
            assert abs(sensitivity-row.sensitivity) < 1e-12
        if (~observed).any():
            specificity = np.sum(weights[~observed] * ~predicted[~observed])/sum(weights[~observed])
            assert abs(specificity-row.specificity) < 1e-12
    for row in onset.itertuples():
        part = b.loc[b.evaluation_kind.eq(row.evaluation_kind) & b.model.eq(row.model) & b.censoring.eq(row.censoring)]
        if row.source != "pooled":
            part = part.loc[part.source.eq(row.source)]
        if row.leaf_scope == "top3":
            part = part.loc[part.leaf_index.lt(3)]
        weights = independent_weights(part)
        assert abs(np.sum(weights * part.onset_distance_days.to_numpy())-row.distance_days) < 1e-12
        assert abs(np.sum(weights * part.onset_compatible.to_numpy())-row.compatible_fraction) < 1e-12
    selections = sorted((HERE / "outer_folds").glob("*/*/selection.json")) + [HERE / "full_data_refit/selection.json"]
    checked_candidates = 0
    for path in selections:
        record = json.loads(path.read_text())
        frame = pd.read_parquet(path.parent / "all_inner_candidate_bracket_predictions.parquet")
        scores = {}
        for candidate, group in frame.loc[frame.censoring.eq("two_sided")].groupby("candidate", sort=False):
            scores[candidate] = float(np.sum(independent_weights(group) * group.onset_distance_days.to_numpy()))
        for row in record["ranking"]:
            assert abs(scores[row["candidate"]]-row["primary_distance_days"]) < 1e-12
        expected_selected = min(record["ranking"], key=lambda item: (scores[item["candidate"]], item["candidate_order"]))
        assert expected_selected["candidate"] == record["selected"]["name"]
        checked_candidates += len(scores)
    selected = pd.read_parquet(HERE / "out_of_fold_daily_event_dates.parquet")
    known = selected.predicted_symptom_day.ge(0)
    assert (selected.loc[known].predicted_symptom_day > selected.loc[known].predicted_effective_infection_day).all()
    report = dict(status="passed", source_hashes_checked=len(configuration["input_and_source_sha256"]),
        output_hashes_checked=len(receipt["output_sha256"]), independently_weighted_detection_rows=len(detection),
        independently_weighted_onset_rows=len(onset), independent_selection_records=len(selections),
        independently_recomputed_candidate_scores=checked_candidates,
        chronology_event_rows_checked=int(known.sum()), selection_unchanged=True)
    target = HERE / "post_run_independent_verification.json"
    content = json.dumps(report, indent=2)+"\n"
    if target.exists():
        assert target.read_text() == content
    else:
        target.write_text(content)
    print(content)


if __name__ == "__main__":
    main()
