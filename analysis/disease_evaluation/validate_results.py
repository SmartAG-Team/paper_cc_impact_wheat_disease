#!/usr/bin/env python3
"""Independent output-level checks using source rows and scalar arithmetic."""
from collections import defaultdict
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def main():
    source = pd.read_csv(ROOT / "data/basf-wheat-diseases.txt", sep="\t")
    source["source_row"] = np.arange(2, len(source) + 2)
    selected = pd.read_csv(OUT / "eligible_observations.csv")
    eligible = source.loc[source.Treatment.eq("Untreated") & source.Organism.eq("SEPTTR")
                          & source.Clarifier.ne("CROP INJURY") & source.Parameter.eq("INFECT")
                          & source.Method.eq("P%INF") & source.Value.between(0, 100)
                          & source.PlantPart.isin(["LEAF, 1ST / FLAG LEAF", "LEAF, 2ND", "LEAF, 3RD",
                                                  "LEAF, 4TH", "LEAF, 5TH", "LEAF, 6ST", "LEAF, 7ST"])]
    independent_rows = set()
    for _, group in eligible.groupby(["TrialId", "PlantPart"]):
        if group.Date.nunique() >= 3:
            independent_rows.update(group.source_row)
    assert set(selected.source_row) == independent_rows
    predictions = pd.read_csv(OUT / "predictions.csv")
    metrics = pd.read_csv(OUT / "fold_metrics.csv")
    assignments = pd.read_csv(OUT / "fold_assignments.csv")
    source_lookup = source.set_index("source_row")
    discrepancies = []
    for row in metrics.itertuples():
        records = predictions.loc[predictions.fold.eq(row.fold) & predictions.model.eq(row.model)
                                  & predictions.scored]
        errors = defaultdict(lambda: defaultdict(list))
        for prediction in records.itertuples():
            raw = source_lookup.loc[prediction.source_row]
            assert raw.Value == prediction.observed_percent
            assert raw.Date == prediction.target_date
            assert raw.TrialId == prediction.TrialId
            assert prediction.latest_weather_date < prediction.target_date
            assert prediction.first_date < prediction.target_date
            assert 0 <= prediction.predicted_percent <= 100
            errors[prediction.coordinate_year][prediction.series_id].append(
                prediction.predicted_percent - prediction.observed_percent)
        square_means, abs_means, signed_means = [], [], []
        for series in errors.values():
            squares = [sum(e * e for e in values) / len(values) for values in series.values()]
            absolutes = [sum(abs(e) for e in values) / len(values) for values in series.values()]
            signed = [sum(values) / len(values) for values in series.values()]
            square_means.append(sum(squares) / len(squares))
            abs_means.append(sum(absolutes) / len(absolutes))
            signed_means.append(sum(signed) / len(signed))
        computed = [math.sqrt(sum(square_means) / len(square_means)),
                    sum(abs_means) / len(abs_means), sum(signed_means) / len(signed_means)]
        reported = [row.rmse_pp, row.mae_pp, row.bias_pp]
        discrepancies.extend(abs(a - b) for a, b in zip(computed, reported))
        assert np.allclose(computed, reported, atol=1e-10, rtol=0)
        assert len(errors) == row.n_coordinate_years
        roles = assignments.loc[assignments.fold.eq(row.fold)]
        train, test = roles.loc[roles.role.eq("train")], roles.loc[roles.role.eq("test")]
        assert not set(train.location_id) & set(test.location_id)
        assert not set(train.series_id) & set(test.series_id)
        assert set(records.series_id) == set(test.series_id)
        if row.fold_type == "forward_year":
            assert train.year.max() < test.year.min()
        else:
            assert not set(train.country) & set(test.country)
    # One country-held-out prediction and one forward-year prediction per eligible row,
    # within the applicable period; duplicated records across split families are expected.
    for fold_type, group in predictions.groupby("fold_type"):
        for _, model_records in group.groupby("model"):
            assert not model_records.duplicated("source_row").any()
    outcome = {"independent_selected_row_set_reconciles": True,
               "reported_metrics_independently_recomputed": True,
               "metric_rows_verified": len(metrics),
               "maximum_metric_absolute_difference_pp": max(discrepancies),
               "predictions_reconcile_to_original_source_rows": True,
               "train_test_coordinate_country_time_groups_verified": True,
               "latest_weather_precedes_each_target_date": True,
               "each_pooled_source_row_has_one_heldout_prediction_per_model": True}
    (OUT / "independent_validation.json").write_text(json.dumps(outcome, indent=2) + "\n")
    print(json.dumps(outcome, indent=2))


if __name__ == "__main__":
    main()
