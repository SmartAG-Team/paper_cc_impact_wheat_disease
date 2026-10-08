"""One fixed exploratory exponential candidate on the exact existing trial folds."""
from pathlib import Path
import hashlib
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from response_estimator import arithmetic_metrics, leave_one_trial_out

ROOT = Path(__file__).resolve().parent
FEATURES = {
    "endpoint67": ["delta_stb_pp"], "integral50": ["delta_stb_percentage_days"],
    "endpoint52": ["delta_stb_pp"], "integral38": ["delta_stb_percentage_days"],
    "four_disease131": ["delta_stb_pp", "delta_yellow_rust_pp", "delta_brown_rust_pp", "delta_mildew_pp"],
}


def save_json(name, data):
    (ROOT/name).write_text(json.dumps(data, indent=2, default=lambda v:v.item() if isinstance(v,np.generic) else None, allow_nan=False)+"\n")


def main():
    declaration = ROOT/"predeclared_damage_candidate.json"
    input_hashes = [{"file": name+"_input.csv", "sha256": hashlib.sha256((ROOT/(name+"_input.csv")).read_bytes()).hexdigest()} for name in FEATURES]
    save_json("damage_prefit_input_hashes.json", {
        "generated_utc_before_fitting": datetime.now(timezone.utc).isoformat(),
        "declaration_sha256": hashlib.sha256(declaration.read_bytes()).hexdigest(), "inputs": input_hashes,
        "fixed_specification": "One candidate; no tuning or alternative fits"})
    summary = {"target": "Signed R=1-Y_U/Y_C", "fit_target": "log(Y_C/Y_U)",
               "candidate": "1-exp(-X*beta), beta>=0", "analysis_role": "Exploratory conditional damage assumption; not verified biology"}
    for name, features in FEATURES.items():
        frame = pd.read_csv(ROOT/(name+"_input.csv"))
        predictions, fits, members = leave_one_trial_out(frame, features, family="damage")
        old_members = pd.read_csv(ROOT/(name+"_loto_memberships.csv"))
        columns = ["fold", "role", "row_id", "record", "control_id", "weight"]
        pd.testing.assert_frame_equal(members[columns].reset_index(drop=True), old_members[columns].reset_index(drop=True), check_dtype=False, check_exact=False, atol=1e-14, rtol=1e-14)
        prefix = name+"_damage"
        predictions.to_csv(ROOT/(prefix+"_loto_predictions.csv"), index=False, float_format="%.17g")
        members.to_csv(ROOT/(prefix+"_loto_memberships.csv"), index=False, float_format="%.17g")
        save_json(prefix+"_loto_fits.json", fits)
        trial_metrics = [{"record": trial, **arithmetic_metrics(rows)} for trial, rows in predictions.groupby("record")]
        pd.DataFrame(trial_metrics).to_csv(ROOT/(prefix+"_loto_metrics_by_trial.csv"), index=False, float_format="%.17g")
        metrics = arithmetic_metrics(predictions)
        metrics["prediction_at_numerical_upper_bound"] = int((predictions.prediction == 1).sum())
        summary[prefix+"_loto"] = {"features": features, "metrics": metrics, "per_trial_metrics": trial_metrics}
        if name in {"endpoint52", "integral38"}:
            # Temporal transfer is identical to the already fitted whole-trial fold.
            trial = "nfts_68027"
            temporal = predictions[predictions.record == trial].copy()
            temporal["fold"] = "2020_2021_Informer_to_2022_Pondus"
            temporal_members = members[members.fold == trial].copy()
            temporal_members["fold"] = "2020_2021_Informer_to_2022_Pondus"
            temporal.to_csv(ROOT/(prefix+"_temporal_predictions.csv"), index=False, float_format="%.17g")
            temporal_members.to_csv(ROOT/(prefix+"_temporal_memberships.csv"), index=False, float_format="%.17g")
            temporal_fit = next(f for f in fits if f["held_out_trial"] == trial)
            save_json(prefix+"_temporal_fit.json", temporal_fit)
            summary[prefix+"_temporal"] = {"features": features, "metrics": arithmetic_metrics(temporal),
                "training_rows": temporal_fit["training_rows"], "training_trials": 3,
                "test_rows": len(temporal), "test_trials": 1,
                "independence_note": "Same leave-Pondus-out fit; not an additional independent test"}
    save_json("damage_performance_summary.json", summary)
    for key, entry in summary.items():
        if isinstance(entry,dict) and "metrics" in entry:
            print(key, json.dumps(entry["metrics"]))


if __name__ == "__main__":
    main()
