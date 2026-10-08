"""Independent archive verification of the finite stage-only phenology study."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

DEST = Path(__file__).resolve().parent
ROOT = DEST.parents[3]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    config = json.loads((DEST / "configuration_before_calibration.json").read_text())
    frozen = json.loads((DEST / "frozen_q_before_evaluation.json").read_text())
    receipt = json.loads((DEST / "receipt.json").read_text())
    assert receipt["status"] == "complete"
    assert not receipt["disease_values_read"] and not receipt["model_q_changed"]
    assert receipt["partial_stage_endpoints_retained"]
    assert receipt["initial_complete_endpoint_archive_preserved"]
    initial_config = json.loads((DEST.parent / "configuration_before_calibration.json").read_text())
    for relative, expected in initial_config["input_and_source_sha256"].items():
        assert digest(ROOT / relative) == expected, relative
    for relative, expected in config["input_and_source_sha256"].items():
        assert digest(ROOT / relative) == expected, relative
    assert digest(DEST / "frozen_q_before_evaluation.json") == receipt["frozen_q_sha256"]
    assert digest(DEST / "calibration_q_grid.csv") == frozen["q_calibration_grid_sha256"]

    curve = pd.read_csv(DEST / "calibration_q_grid.csv")
    loss_matrix = pd.read_csv(DEST / "calibration_field_loss_by_q.csv", index_col="q")
    assert len(curve) == 401 and len(loss_matrix) == 401
    np.testing.assert_allclose(curve.q, np.linspace(.6, 1.6, 401), atol=1e-14)
    np.testing.assert_allclose(curve.loss_days, loss_matrix.mean(axis=1), atol=1e-14)
    minimum = curve.loss_days.min()
    tied = curve.loc[np.isclose(curve.loss_days, minimum, rtol=0, atol=1e-12), "q"]
    ordered = sorted(tied.tolist(), key=lambda q: (round(abs(q - 1), 12), q))
    assert np.isclose(ordered[0], frozen["selected_q"], atol=1e-14)

    inputs = pd.read_csv(DEST / "original_field_input_membership.csv")
    assert inputs.groupby("partition").size().to_dict() == {
        "calibration": 28, "reused_development_2019": 45, "reused_external_strict": 143}
    predictions = pd.read_csv(DEST / "frozen_event_predictions.csv",
        parse_dates=["predicted_date", "forcing_end"])
    prediction_checks = 0
    for partition, membership in inputs.groupby("partition"):
        accumulated = np.load(DEST / f"{partition}_development_inputs.npz")["accumulation"]
        for row in predictions.loc[predictions.partition.eq(partition)].itertuples(index=False):
            meta = membership.loc[membership.field_id.eq(row.field_id)].iloc[0]
            values = accumulated[int(meta.field_index), :int(meta.forcing_days)] * row.q
            reached = np.where(values >= frozen["thresholds"][str(row.event)])[0]
            expected = (pd.Timestamp(meta.sowing_date) + pd.Timedelta(days=int(reached[0]))
                        if len(reached) else pd.NaT)
            assert pd.isna(row.predicted_date) if pd.isna(expected) else row.predicted_date == expected
            prediction_checks += 1

    scores = pd.read_csv(DEST / "censoring_compatibility_and_distances.csv",
        parse_dates=["predicted_date", "forcing_end", "lower_exclusive", "upper_inclusive"])
    assert "value" not in scores
    assert not scores.duplicated(["model", "field_id", "event"]).any()
    distance_checks = 0
    for row in scores.itertuples(index=False):
        missing = pd.isna(row.predicted_date)
        if missing:
            earliest_possible = row.forcing_end + pd.Timedelta(days=1)
            expected = max(0, (earliest_possible - row.upper_inclusive).days) if pd.notna(row.upper_inclusive) else 0
        elif pd.notna(row.lower_exclusive) and row.predicted_date <= row.lower_exclusive:
            expected = (row.predicted_date - row.lower_exclusive).days - 1
        elif pd.notna(row.upper_inclusive) and row.predicted_date > row.upper_inclusive:
            expected = (row.predicted_date - row.upper_inclusive).days
        else:
            expected = 0
        assert row.signed_distance_days == expected
        assert row.distance_days == abs(expected)
        assert row.compatible == (expected == 0)
        assert row.distance_is_lower_bound == missing
        distance_checks += 1

    metrics = pd.read_csv(DEST / "event_censoring_metrics.csv", dtype={"event": str})
    for row in metrics.itertuples(index=False):
        group = scores.loc[scores.model.eq(row.model) & scores.partition.eq(row.partition)]
        if row.scope == "genuinely_bracketed":
            group = group.loc[group.genuinely_bracketed]
        if row.event != "all":
            group = group.loc[group.event.eq(int(row.event))]
        assert len(group) == row.n_field_events
        np.testing.assert_allclose(group.groupby("field_id").distance_days.mean().mean(),
                                   row.equal_field_event_distance_days, atol=1e-14)
        np.testing.assert_allclose(group.groupby("field_id").signed_distance_days.mean().mean(),
                                   row.mean_signed_distance_days, atol=1e-14)
    normalized = pd.read_csv(DEST / "normalized_field_date_stages.csv")
    rejected = pd.read_csv(DEST / "rejected_stage_rows.csv")
    assert not (normalized.stage_from.isna() & normalized.stage_to.isna()).any()
    complete = normalized.stage_from.notna() & normalized.stage_to.notna()
    assert normalized.loc[complete].stage_from.le(normalized.loc[complete].stage_to).all()
    assert not rejected.reason.eq("missing_stage_endpoint").any()
    endpoint_counts = pd.read_csv(DEST / "accepted_endpoint_counts.csv", index_col="partition")
    for partition, group in normalized.groupby("partition"):
        for column in ["source_row_count", "n_complete_rows", "n_stage_from_only_rows", "n_stage_to_only_rows"]:
            assert group[column].sum() == endpoint_counts.loc[partition, column]
    assert normalized.source_row_count.eq(normalized.n_complete_rows +
        normalized.n_stage_from_only_rows + normalized.n_stage_to_only_rows).all()
    verified = dict(status="verified", verified_utc=datetime.now(timezone.utc).isoformat(),
        n_frozen_dependencies=len(config["input_and_source_sha256"]),
        q_grid_points_verified=401, first_event_predictions_verified=prediction_checks,
        signed_censoring_distances_verified=distance_checks, metric_rows_verified=len(metrics),
        frozen_q=receipt["selected_q"], frozen_sources_unchanged=True,
        original_field_counts_verified=True, disease_values_absent_from_study_tables=True,
        partial_endpoint_counts_verified=True, initial_complete_endpoint_archive_preserved=True)
    output = DEST / "verification.json"
    if output.exists():
        raise FileExistsError("Independent verification receipt already exists")
    output.write_text(json.dumps(verified, indent=2) + "\n")
    print(json.dumps(verified), flush=True)


if __name__ == "__main__":
    main()
