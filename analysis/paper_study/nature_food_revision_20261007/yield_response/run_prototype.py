"""Reproduce the predeclared Nordic disease-control grain-response evaluation."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from response_estimator import (
    arithmetic_metrics, endpoint_eligibility, evaluate_split, fit_origin, leave_one_trial_out,
    paired_integral, signed_response,
)

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "yield_data_audit"
KEYS = ["record", "registry_id", "untreated_treatment_code", "treated_treatment_code", "leaf_scope"]
DISEASES = ["stb", "yellow_rust", "brown_rust", "mildew"]
NONTARGETS = ["yellow_rust", "brown_rust", "mildew", "leaf_spot"]
SOURCE_FILES = [
    "nfts_final_disease_same_date_scope_covariates_and_flags.csv",
    "nfts_longitudinal_pair_coverage.csv", "nfts_disease_same_date_scope_panel.csv",
    "nfts_disease_observations_long.csv", "nfts_yield_treatment_means.csv",
    "nfts_fungicide_yield_contrasts_eligible.csv", "nfts_trial_candidate_catalogue.csv",
    "nfts_trial_metadata.json", "nfts_yield_precision_by_trial.csv",
    "nfts_native_units_provenance.json", "nfts_native_english_yield_value_checks.csv",
    "randomization_unit_audit.json", "source_redistribution_rights_review.json",
]


def scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def save_json(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False, default=scalar, allow_nan=False) + "\n")


def save_csv(name, frame):
    frame.to_csv(ROOT / name, index=False, float_format="%.17g")


def audit_hashes(metadata):
    paths = {SOURCE / f for f in SOURCE_FILES}
    for meta in metadata:
        paths.update([SOURCE / meta["source_html"], SOURCE / meta["native_unit_source"]])
    rows = []
    for path in sorted(paths):
        content = path.read_bytes()
        rows.append({"path": str(path.relative_to(ROOT.parent)), "bytes": len(content),
                     "sha256": hashlib.sha256(content).hexdigest()})
    protocol = ROOT / "predeclared_protocol.json"
    save_json("source_hashes.json", {
        "generated_utc": datetime.now(timezone.utc).isoformat(), "sources": rows,
        "protocol_sha256_before_fitting": hashlib.sha256(protocol.read_bytes()).hexdigest(),
        "prefit_amendment_sha256": hashlib.sha256((ROOT / "predeclared_amendment_01.json").read_bytes()).hexdigest(),
        "source_mutations": "None; inputs opened read-only",
    })


def reconstruct_cohorts():
    final = pd.read_csv(SOURCE / SOURCE_FILES[0])
    coverage = pd.read_csv(SOURCE / SOURCE_FILES[1])
    panel = pd.read_csv(SOURCE / SOURCE_FILES[2])
    observed = pd.read_csv(SOURCE / "nfts_disease_observations_long.csv")
    yields = pd.read_csv(SOURCE / "nfts_yield_treatment_means.csv")
    metadata = {m["record"]: m for m in json.loads((SOURCE / "nfts_trial_metadata.json").read_text())}
    precision = pd.read_csv(SOURCE / "nfts_yield_precision_by_trial.csv")
    final = final.merge(coverage, on=KEYS, validate="one_to_one")
    final["row_id"] = (final.record + "|" + final.untreated_treatment_code + "|" + final.treated_treatment_code + "|leaf" + final.leaf_scope.astype(int).astype(str))
    final["control_id"] = final.record + "|" + final.untreated_treatment_code
    assert not final.row_id.duplicated().any()
    final["R"] = signed_response(final.untreated_yield_native, final.treated_yield_native)
    assert np.allclose(final.R, final.untreated_shortfall_relative_to_treated, atol=1e-14)
    for disease in DISEASES:
        final[f"delta_{disease}_pp"] = final[disease] - final[disease + "_treated"]
    final["stb_pair_measured"] = final[["stb", "stb_treated"]].notna().all(axis=1)
    final["four_disease_both_arms_measured"] = final[[c for d in DISEASES for c in [d, d + "_treated"]]].notna().all(axis=1)
    final["all_nontargets_final_both_arms_measured_zero"] = final[[c for d in NONTARGETS for c in [d, d + "_treated"]]].eq(0).all(axis=1)
    lookup = {(record, code, int(scope)): group for (record, code, scope), group in panel.groupby(["record", "treatment_code", "leaf_scope"])}
    stage_groups = observed.groupby(["record", "treatment_code", "date"])["growth_stage"]
    stages = stage_groups.apply(lambda v: ";".join(str(int(x)) for x in sorted(v.dropna().unique()))).to_dict()
    selection_rows, integral_rows, knot_rows, yield_checks = [], [], [], []
    no_positive, any_missing = [], []
    for _, row in final.iterrows():
        u = lookup.get((row.record, row.untreated_treatment_code, int(row.leaf_scope)), panel.iloc[0:0])
        c = lookup.get((row.record, row.treated_treatment_code, int(row.leaf_scope)), panel.iloc[0:0])
        nontarget = pd.concat([u[NONTARGETS], c[NONTARGETS]])
        positive_seen = nontarget.gt(0).any().any()
        missing_seen = nontarget.isna().any().any()
        no_positive.append(not positive_seen)
        any_missing.append(bool(missing_seen))
        endpoint_ok = bool(row.stb_pair_measured and row.all_nontargets_final_both_arms_measured_zero and not positive_seen)
        paired = u[["date", "stb"]].merge(c[["date", "stb"]], on="date", suffixes=("_untreated", "_treated"), validate="one_to_one").dropna(subset=["stb_untreated", "stb_treated"])
        assert len(paired) == row.stb_same_date_pairs
        integral_ok = endpoint_ok and len(paired) >= 2
        selection_rows.append({"row_id": row.row_id, "record": row.record,
            "untreated_treatment_code": row.untreated_treatment_code, "treated_treatment_code": row.treated_treatment_code,
            "leaf_scope": row.leaf_scope, "same_date_stb_pair": bool(row.stb_pair_measured),
            "complete_four_diseases_both_arms": bool(row.four_disease_both_arms_measured),
            "all_final_nontargets_measured_zero": bool(row.all_nontargets_final_both_arms_measured_zero),
            "positive_nontarget_recorded_at_any_date": bool(positive_seen),
            "some_nontarget_measurements_missing_at_recorded_dates": bool(missing_seen),
            "common_numeric_dates": len(paired), "endpoint_eligible": endpoint_ok,
            "integral_eligible": integral_ok,
            "positive_stb_endpoint_eligible": bool(endpoint_ok and row.stb > 0),
            "positive_stb_integral_eligible": bool(integral_ok and row.stb > 0),
            "four_disease_eligible": bool(row.four_disease_both_arms_measured)})
        if integral_ok:
            values, knots = paired_integral(u, c)
            assert values["window_days"] == row.observed_common_window_days
            integral_rows.append({"row_id": row.row_id, **values})
            knots["row_id"], knots["record"] = row.row_id, row.record
            knots["leaf_scope"] = row.leaf_scope
            knots["untreated_growth_stage"] = [stages.get((row.record, row.untreated_treatment_code, date), "") for date in knots.date]
            knots["treated_growth_stage"] = [stages.get((row.record, row.treated_treatment_code, date), "") for date in knots.date]
            knot_rows.append(knots)
    final["no_positive_nontarget_at_any_recorded_date"] = no_positive
    final["nontarget_missing_some_recorded_dates"] = any_missing
    # Verify the previous source-audit flags against a fresh calculation.
    assert final.no_positive_nontarget_at_any_recorded_date.equals(final.no_positive_nontarget_in_any_recorded_same_scope_date)
    selection = pd.DataFrame(selection_rows)
    endpoint = final[endpoint_eligibility(final)].copy()
    assert set(endpoint.row_id) == set(selection.loc[selection.endpoint_eligible, "row_id"])
    integral = endpoint.merge(pd.DataFrame(integral_rows), on="row_id", validate="one_to_one")
    additive = final[final.four_disease_both_arms_measured].copy()
    assert (len(endpoint), endpoint.record.nunique()) == (67, 5)
    assert (len(integral), integral.record.nunique()) == (50, 5)
    assert (len(endpoint[endpoint.stb > 0]), endpoint[endpoint.stb > 0].record.nunique()) == (52, 4)
    assert (len(integral[integral.stb > 0]), integral[integral.stb > 0].record.nunique()) == (38, 4)
    assert (len(additive), additive.record.nunique()) == (131, 8)
    assert set(endpoint.leaf_scope) == set(integral.leaf_scope) == set(additive.leaf_scope) == {0}
    selected_records = set(additive.record) | set(endpoint.record)
    for dataset in [endpoint, integral, additive]:
        dataset["untreated_final_growth_stage"] = [stages.get((r.record, r.untreated_treatment_code, r.date), "") for _, r in dataset.iterrows()]
        dataset["treated_final_growth_stage"] = [stages.get((r.record, r.treated_treatment_code, r.date), "") for _, r in dataset.iterrows()]
        dataset["registry_design"] = dataset.record.map(lambda record: metadata[record]["design"])
        dataset["native_grain_unit"] = "dt/ha (=100 kg/ha), at 15% moisture"
        dataset["leaf_scope_interpretation"] = "Unspecified leaf rank; registry percent coverage"
    selected_rows = additive.drop_duplicates("row_id")
    for _, row in selected_rows.iterrows():
        for arm, code in [("untreated", row.untreated_treatment_code), ("treated", row.treated_treatment_code)]:
            candidates = yields[(yields.record == row.record) & (yields.treatment_code == code)]
            assert len(candidates) == 1
            y = candidates.iloc[0]
            assert y.native_unit_verified
            assert abs(y.value - row[arm + "_yield_native"]) < 1e-10
            yield_checks.append({"row_id": row.row_id, "record": row.record, "arm": arm,
                "treatment_code": code, "grain_native": y.value,
                "grain_t_ha": y.value * .1, "yield_date": y.date,
                "source_html": y.source_html, "source_table": y.source_table, "source_table_id": y.source_table_id,
                "source_row": y.source_row, "source_column": y.source_column,
                "native_yield_label_from_unit_verification": metadata[row.record]["native_swedish_yield_label"]})
    save_csv("selection_and_exclusions.csv", selection)
    save_csv("common_date_integral_knots.csv", pd.concat(knot_rows, ignore_index=True))
    save_csv("paired_yield_source_cells.csv", pd.DataFrame(yield_checks))
    save_csv("endpoint67_input.csv", endpoint)
    save_csv("integral50_input.csv", integral)
    save_csv("endpoint52_input.csv", endpoint[endpoint.stb > 0])
    save_csv("integral38_input.csv", integral[integral.stb > 0])
    save_csv("four_disease131_input.csv", additive)
    save_csv("selected_trial_precision.csv", precision[precision.registry_id.isin(additive.registry_id)])
    save_json("selected_source_metadata.json", [metadata[r] for r in sorted(selected_records)])
    audit_hashes([metadata[r] for r in sorted(selected_records)])
    candidate = pd.read_csv(SOURCE / "nfts_trial_candidate_catalogue.csv")
    contrast_source = pd.read_csv(SOURCE / "nfts_fungicide_yield_contrasts_eligible.csv")
    save_csv("source_candidate_trials.csv", candidate)
    save_json("cohort_denominators.json", {
        "registered_candidates": len(candidate), "trials_with_numeric_yield": int((candidate.numeric_yield_means > 0).sum()),
        "quality_eligible_trials": int(candidate.registry_quality_eligible.sum()),
        "quality_eligible_yield_contrasts": len(contrast_source), "negative_yield_contrasts_before_disease_selection": int((contrast_source.untreated_shortfall_relative_to_treated < 0).sum()),
        "final_date_leaf_scope_domains": len(final), "unique_contrasts_with_final_stb_domain": len(final.drop_duplicates(KEYS[:-1])),
        "same_date_both_arm_stb_domains": int(selection.same_date_stb_pair.sum()),
        "endpoint67": {"rows": len(endpoint), "trials": endpoint.record.nunique(), "negative_R": int((endpoint.R < 0).sum()), "by_trial": endpoint.groupby("record").size().to_dict()},
        "integral50": {"rows": len(integral), "trials": integral.record.nunique(), "negative_R": int((integral.R < 0).sum()), "by_trial": integral.groupby("record").size().to_dict()},
        "endpoint52_positive_stb": {"rows": 52, "trials": 4, "eligibility": "Endpoint67 plus untreated final STB>0; secondary domain"},
        "integral38_positive_stb": {"rows": 38, "trials": 4, "eligibility": "Integral50 plus untreated final STB>0; secondary domain"},
        "four_disease131": {"rows": len(additive), "trials": additive.record.nunique(), "negative_R": int((additive.R < 0).sum()), "by_trial": additive.groupby("record").size().to_dict()},
        "missingness_note": "Zero tests require numerical zero in all final co-pathogen cells. Earlier missing cells are explicitly flagged and never imputed; no-positive-recorded does not establish absence at unmeasured times.",
        "precision_note": "Registry treatment means and LSD are retained as provenance. Replicate covariance and response SE are unavailable; no confidence intervals are inferred.",
        "source_label_caution": "The archived per-cell native_swedish_measurement_label can be misaligned across languages because source columns are ordered differently. Native grain units are established from matching native yield headers and the independent native-English grain-value check, not those per-cell label strings.",
    })
    return endpoint, integral, additive


def domain_summary(frame, features):
    result = []
    for record, rows in frame.groupby("record"):
        result.append({"record": record, "year": int(rows.year.iloc[0]), "rows": len(rows),
            "cultivars": sorted(rows.cultivar.fillna("Entry name unavailable").unique().tolist()),
            "registry_entries": rows.cultivar_entry.nunique(),
            "entry_interpretation": "Treatment entry in single-factor trials; cultivar entry in two-factor trials",
            "named_cultivar_count": rows.cultivar.nunique(),
            "mean_R": float(rows.R.mean()), "min_R": float(rows.R.min()), "max_R": float(rows.R.max()),
            "negative_R": int((rows.R < 0).sum()), "date": sorted(rows.date.unique().tolist()),
            "final_growth_stage": sorted(rows.untreated_final_growth_stage.unique().tolist()),
            "feature_min": rows[features].min().to_dict(), "feature_max": rows[features].max().to_dict(),
            "nonzero_feature_rows": {f: int((rows[f] != 0).sum()) for f in features},
            "registry_design": rows.registry_design.iloc[0],
            "common_windows_days": sorted(rows.window_days.unique().tolist()) if "window_days" in rows else None,
            "latitude": float(rows.latitude.iloc[0]), "longitude": float(rows.longitude.iloc[0]),
            "replicates": int(rows.replicates.iloc[0]), "source_url": rows.source_url.iloc[0]})
    return result


def evaluate(name, frame, features, summary):
    preds, fits, members = leave_one_trial_out(frame, features)
    save_csv(name + "_loto_predictions.csv", preds)
    save_csv(name + "_loto_memberships.csv", members)
    save_json(name + "_loto_fits.json", fits)
    metrics = arithmetic_metrics(preds)
    trial_metrics = [{"record": record, **arithmetic_metrics(rows)} for record, rows in preds.groupby("record")]
    save_csv(name + "_loto_metrics_by_trial.csv", pd.DataFrame(trial_metrics))
    save_json(name + "_domain_summary.json", domain_summary(frame, features))
    summary[name + "_loto"] = {"features": features, "metrics": metrics, "per_trial_metrics": trial_metrics}
    if name in ["endpoint52", "integral38"]:
        train = frame[(frame.year <= 2021) & (frame.cultivar == "Informer")]
        test = frame[(frame.year == 2022) & (frame.cultivar == "Pondus")]
        assert set(train.record) == {"nfts_62693", "nfts_62768", "nfts_65793"}
        assert set(test.record) == {"nfts_68027"}
        predictions, fit, membership = evaluate_split(train, test, features, "2020_2021_Informer_to_2022_Pondus")
        save_csv(name + "_temporal_predictions.csv", predictions)
        save_csv(name + "_temporal_memberships.csv", membership)
        save_json(name + "_temporal_fit.json", fit)
        summary[name + "_temporal"] = {"training_rows": len(train), "training_trials": train.record.nunique(),
                                     "test_rows": len(test), "test_trials": test.record.nunique(),
                                     "features": features, "metrics": arithmetic_metrics(predictions)}
    # Full-data coefficients are descriptive, not validation predictions.
    save_json(name + "_descriptive_full_fit.json", fit_origin(frame, features))


def main():
    endpoint, integral, additive = reconstruct_cohorts()
    summary = {"target": "R = 1 - Y_U/Y_C", "metric_scale": "Relative fraction; multiply error by100 for percentage points",
               "weighting": "Each trial has equal total weight in fitting and scoring", "uncertainty_intervals": "Not estimated"}
    evaluate("endpoint67", endpoint, ["delta_stb_pp"], summary)
    evaluate("integral50", integral, ["delta_stb_percentage_days"], summary)
    evaluate("endpoint52", endpoint[endpoint.stb > 0].copy(), ["delta_stb_pp"], summary)
    evaluate("integral38", integral[integral.stb > 0].copy(), ["delta_stb_percentage_days"], summary)
    evaluate("four_disease131", additive, ["delta_" + d + "_pp" for d in DISEASES], summary)
    save_json("performance_summary.json", summary)
    print(json.dumps(summary, indent=2, default=scalar))


if __name__ == "__main__":
    main()
