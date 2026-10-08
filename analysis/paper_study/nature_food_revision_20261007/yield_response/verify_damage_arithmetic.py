"""Reconstruct the exponential candidate without importing either model script."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import json
import math

import numpy as np

from verify_arithmetic import close, read_csv, reconstruct_metrics

ROOT = Path(__file__).resolve().parent
NAMES = ["endpoint67", "integral50", "endpoint52", "integral38", "four_disease131"]


def main():
    receipt = json.loads((ROOT/"damage_prefit_input_hashes.json").read_text())
    assert hashlib.sha256((ROOT/"predeclared_damage_candidate.json").read_bytes()).hexdigest() == receipt["declaration_sha256"]
    for item in receipt["inputs"]:
        assert hashlib.sha256((ROOT/item["file"]).read_bytes()).hexdigest() == item["sha256"]
    summary = json.loads((ROOT/"damage_performance_summary.json").read_text())
    independent_metrics, verified_coefficients, verified_predictions = {}, 0, 0
    for name in NAMES:
        rows = read_csv(ROOT/(name+"_input.csv"))
        inputs = {row["row_id"]: row for row in rows}
        prefix = name+"_damage"
        predictions = read_csv(ROOT/(prefix+"_loto_predictions.csv"))
        memberships = read_csv(ROOT/(prefix+"_loto_memberships.csv"))
        original_memberships = read_csv(ROOT/(name+"_loto_memberships.csv"))
        assert memberships == original_memberships, "Whole-trial memberships/weights changed"
        assert Counter(row["row_id"] for row in predictions) == Counter(inputs.keys())
        fits = json.loads((ROOT/(prefix+"_loto_fits.json")).read_text())
        independent_metrics[prefix+"_loto"] = reconstruct_metrics(predictions)
        independent_metrics[prefix+"_loto"]["prediction_at_numerical_upper_bound"] = sum(float(row["prediction"]) == 1 for row in predictions)
        for metric, value in independent_metrics[prefix+"_loto"].items():
            close(value, summary[prefix+"_loto"]["metrics"][metric], metric)
        for fit in fits:
            fold = fit["held_out_trial"]
            train = [row for row in memberships if row["fold"] == fold and row["role"] == "train"]
            assert fold not in {row["record"] for row in train}
            assert set(row["row_id"] for row in train) == {key for key,row in inputs.items() if row["record"] != fold}
            counts = Counter(row["record"] for row in train)
            features, beta = fit["features"], fit["coefficients"]
            target = [math.log(float(inputs[row["row_id"]]["treated_yield_native"])/float(inputs[row["row_id"]]["untreated_yield_native"])) for row in train]
            x = [[float(inputs[row["row_id"]][feature]) for feature in features] for row in train]
            weights = [1/counts[row["record"]] for row in train]
            for trial,count in counts.items():
                close(math.fsum(float(row["weight"]) for row in train if row["record"] == trial),1,"training trial total")
            # Baseline stays in original R space; it is not the mean log ratio.
            baseline = math.fsum(math.fsum(float(inputs[row["row_id"]]["R"]) for row in train if row["record"]==trial)/count for trial,count in counts.items())/len(counts)
            close(baseline, fit["baseline"], "original-scale training-only mean")
            null_columns = [feature for j,feature in enumerate(features) if all(values[j]==0 for values in x)]
            assert null_columns == fit["null_predictors"]
            if len(features) == 1:
                numerator = math.fsum(w*values[0]*y for w,values,y in zip(weights,x,target))
                denominator = math.fsum(w*values[0]**2 for w,values in zip(weights,x))
                close(max(0,numerator/denominator) if denominator else 0,beta[0],"log-scale analytic slope")
            else:
                matrix, coefficients = np.asarray(x), np.asarray(beta)
                gradient = matrix.T @ (np.asarray(weights)*(matrix@coefficients-np.asarray(target)))
                assert (coefficients>=0).all()
                assert (gradient[coefficients==0]>=-1e-9).all()
                assert np.max(abs(gradient[coefficients>0]))<1e-9
            verified_coefficients += 1
            for row in predictions:
                if row["fold"] != fold:
                    continue
                score = math.fsum(float(row[feature])*b for feature,b in zip(features,beta))
                response = -math.expm1(-score)
                close(response,row["prediction"],"unclipped exponential prediction")
                close(score,row["linear_score"],"log-scale linear score")
                close(baseline,row["baseline_prediction"],"untouched calibration baseline")
                close(1-float(row["untreated_yield_native"])/float(row["treated_yield_native"]),row["R"],"original signed yield ratio")
                assert response <= 1
                verified_predictions += 1
        if name in {"endpoint52", "integral38"}:
            temporal = read_csv(ROOT/(prefix+"_temporal_predictions.csv"))
            fold_predictions = {row["row_id"]:row for row in predictions if row["record"]=="nfts_68027"}
            for row in temporal:
                close(row["prediction"],fold_predictions[row["row_id"]]["prediction"],"temporal prediction reused")
                close(row["baseline_prediction"],fold_predictions[row["row_id"]]["baseline_prediction"],"temporal baseline reused")
            independent_metrics[prefix+"_temporal"] = reconstruct_metrics(temporal)
            for metric,value in independent_metrics[prefix+"_temporal"].items():
                close(value,summary[prefix+"_temporal"]["metrics"][metric],"temporal "+metric)
    results = {"status": "All independently reconstructed exponential predictions, log-ratio fits, NNLS KKT conditions, memberships, weights and arithmetic metrics passed",
               "fixed_candidate_specifications_evaluated":1, "training_fold_fits_verified":verified_coefficients,
               "held_out_predictions_verified":verified_predictions, "input_hashes_verified":len(receipt["inputs"]),
               "metrics":independent_metrics,
               "method": "Standard-library logarithms, exponentials and sums; independent NNLS KKT gradients; no imports from response_estimator or candidate runner"}
    (ROOT/"independent_damage_arithmetic_verification.json").write_text(json.dumps(results,indent=2,allow_nan=False)+"\n")
    print(json.dumps({key:value for key,value in results.items() if key!="metrics"},indent=2))


if __name__ == "__main__":
    main()
