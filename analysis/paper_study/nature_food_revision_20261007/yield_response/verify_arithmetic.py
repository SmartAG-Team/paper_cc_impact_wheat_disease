"""Independent reconstruction: standard-library sums; no estimator imports."""
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
import csv
import hashlib
import json
import math

from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "yield_data_audit"
NAMES = ["endpoint67", "integral50", "endpoint52", "integral38", "four_disease131"]


def read_csv(path):
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def close(a, b, label, tolerance=2e-11):
    if a is None or b is None:
        assert a == b, label
    else:
        assert math.isclose(float(a), float(b), abs_tol=tolerance, rel_tol=tolerance), (label, a, b)


def reconstruct_metrics(rows):
    trials = defaultdict(list)
    for row in rows:
        trials[row["record"]].append(row)
    each_trial = []
    for record, observations in trials.items():
        model = [float(r["prediction"]) - float(r["R"]) for r in observations]
        baseline = [float(r["baseline_prediction"]) - float(r["R"]) for r in observations]
        each_trial.append({"record": record,
            "model_mse": math.fsum(e*e for e in model)/len(model),
            "model_mae": math.fsum(abs(e) for e in model)/len(model),
            "model_bias": math.fsum(model)/len(model),
            "baseline_mse": math.fsum(e*e for e in baseline)/len(baseline),
            "baseline_mae": math.fsum(abs(e) for e in baseline)/len(baseline),
            "baseline_bias": math.fsum(baseline)/len(baseline)})
    means = {k: math.fsum(t[k] for t in each_trial)/len(each_trial) for k in each_trial[0] if k != "record"}
    return {"rows": len(rows), "trials": len(trials),
            "model_rmse": math.sqrt(means["model_mse"]), "model_mae": means["model_mae"], "model_bias": means["model_bias"],
            "baseline_rmse": math.sqrt(means["baseline_mse"]), "baseline_mae": means["baseline_mae"], "baseline_bias": means["baseline_bias"],
            "mse_skill_against_calibration_mean": 1-means["model_mse"]/means["baseline_mse"],
            "negative_observed": sum(float(r["R"]) < 0 for r in rows),
            "negative_predicted": sum(float(r["prediction"]) < 0 for r in rows),
            "prediction_above_one": sum(float(r["prediction"]) > 1 for r in rows)}


def expanded_source_table(table):
    # Independent HTML-cell reader, used only to verify the saved source coordinates.
    occupied = {}
    for row_index, row in enumerate(table.find_all("tr")):
        column_index = 0
        for node in row.find_all(["td", "th"], recursive=False):
            while (row_index, column_index) in occupied:
                column_index += 1
            for dy in range(int(node.get("rowspan", "1"))):
                for dx in range(int(node.get("colspan", "1"))):
                    occupied[row_index+dy, column_index+dx] = node.get_text(" ", strip=True)
            column_index += int(node.get("colspan", "1"))
    return occupied


def read_indexed_source_table(soup, table_index, expected_table_id):
    table = soup.find_all("table")[int(table_index)]
    assert table.get("id") == expected_table_id, "Table index and source ID disagree"
    return expanded_source_table(table)


def source_cells():
    tables = {}
    cell_checks = []
    grain = pd.read_csv(ROOT / "paired_yield_source_cells.csv")
    diseases = pd.read_csv(SOURCE / "nfts_disease_observations_long.csv")
    input_rows = pd.read_csv(ROOT / "four_disease131_input.csv")
    source_measurements = []
    for _, row in input_rows.iterrows():
        for arm, code in [("untreated", row.untreated_treatment_code), ("treated", row.treated_treatment_code)]:
            candidates = diseases[(diseases.record == row.record) & (diseases.treatment_code == code) & (diseases.date == row.date)]
            for disease, pattern in [("stb", "Septoria|Wheat leaf blotch"), ("yellow_rust", "Yellow rust"), ("brown_rust", "Brown rust"), ("mildew", "Powdery mildew"), ("leaf_spot", "Leaf spot")]:
                exact = candidates[candidates.measurement.str.contains(pattern, case=False, regex=True) & ~candidates.measurement.str.contains("plants with", case=False)]
                assert len(exact) == 1, (row.row_id, arm, disease, len(exact))
                item = exact.iloc[0].to_dict()
                expected = row[disease if arm == "untreated" else disease + "_treated"]
                close(item["value"], expected, f"{row.row_id}:{arm}:{disease}")
                item.update({"expected": expected, "disease": disease, "arm": arm, "row_id": row.row_id})
                source_measurements.append(item)
    for item in grain.to_dict("records") + source_measurements:
        source_path = SOURCE / item["source_html"]
        table_id = item["source_table_id"]
        table_index = item["source_table"]
        table_key = (str(source_path), int(table_index), table_id)
        if table_key not in tables:
            soup = BeautifulSoup(source_path.read_text(), "html.parser")
            tables[table_key] = read_indexed_source_table(soup, table_index, table_id)
        table = tables[table_key]
        raw = table[int(item["source_row"]), int(item["source_column"])]
        value = float(raw.replace(",", "."))
        expected = item.get("expected", item.get("grain_native"))
        close(value, expected, f"raw HTML cell {table_key}")
        cell_checks.append({"row_id": item["row_id"], "arm": item["arm"], "measurement": item.get("disease", "grain_yield"),
                           "source_html": item["source_html"], "source_table_id": table_id,
                           "source_table": table_index,
                           "source_row": item["source_row"], "source_column": item["source_column"],
                           "source_raw": raw, "independent_value": value, "model_input_value": expected})
    pd.DataFrame(cell_checks).to_csv(ROOT / "independent_source_cell_checks.csv", index=False)
    return len(cell_checks)


def main():
    hashes = json.loads((ROOT / "source_hashes.json").read_text())
    for source in hashes["sources"]:
        path = ROOT.parent / source["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], path
    assert hashlib.sha256((ROOT/"predeclared_protocol.json").read_bytes()).hexdigest() == hashes["protocol_sha256_before_fitting"]
    assert hashlib.sha256((ROOT/"predeclared_amendment_01.json").read_bytes()).hexdigest() == hashes["prefit_amendment_sha256"]
    performance = json.loads((ROOT/"performance_summary.json").read_text())
    independently = {}
    coefficients_verified, weight_checks = 0, 0
    for name in NAMES:
        inputs = {r["row_id"]: r for r in read_csv(ROOT/f"{name}_input.csv")}
        predictions = read_csv(ROOT/f"{name}_loto_predictions.csv")
        memberships = read_csv(ROOT/f"{name}_loto_memberships.csv")
        fits = json.loads((ROOT/f"{name}_loto_fits.json").read_text())
        assert Counter(r["row_id"] for r in predictions) == Counter(inputs.keys()), name
        for row in predictions:
            close(float(row["R"]), 1-float(row["untreated_yield_native"])/float(row["treated_yield_native"]), "signed yield ratio")
            assert row["fold"] == row["record"]
        independently[name+"_loto"] = reconstruct_metrics(predictions)
        for metric, value in independently[name+"_loto"].items():
            close(value, performance[name+"_loto"]["metrics"][metric], f"{name}:{metric}")
        for fit in fits:
            fold = fit["held_out_trial"]
            train = [r for r in memberships if r["fold"] == fold and r["role"] == "train"]
            test = [r for r in memberships if r["fold"] == fold and r["role"] == "test"]
            assert all(r["record"] != fold for r in train)
            assert all(r["record"] == fold for r in test)
            assert set(r["row_id"] for r in train) == {key for key, row in inputs.items() if row["record"] != fold}
            assert set(r["row_id"] for r in test) == {key for key, row in inputs.items() if row["record"] == fold}
            train_counts = Counter(r["record"] for r in train)
            control_roles = defaultdict(set)
            for member in train+test:
                control_roles[member["control_id"]].add(member["role"])
            assert all(len(roles) == 1 for roles in control_roles.values())
            for trial, count in train_counts.items():
                close(math.fsum(float(r["weight"]) for r in train if r["record"] == trial), 1, "training trial total")
                for member in [r for r in train if r["record"] == trial]:
                    close(member["weight"], 1/count, "per-arm weight")
                weight_checks += 1
            baseline = math.fsum(math.fsum(float(inputs[m["row_id"]]["R"]) for m in train if m["record"] == trial)/count for trial, count in train_counts.items())/len(train_counts)
            close(baseline, fit["baseline"], "training-only baseline")
            for row in [r for r in predictions if r["fold"] == fold]:
                predicted = math.fsum(float(row[f])*beta for f, beta in zip(fit["features"], fit["coefficients"]))
                close(predicted, row["prediction"], "unclipped signed prediction")
                close(baseline, row["baseline_prediction"], "withheld baseline")
            if len(fit["features"]) == 1:
                feature = fit["features"][0]
                numerator = math.fsum(float(inputs[m["row_id"]][feature])*float(inputs[m["row_id"]]["R"])/train_counts[m["record"]] for m in train)
                denominator = math.fsum(float(inputs[m["row_id"]][feature])**2/train_counts[m["record"]] for m in train)
                beta = max(0, numerator/denominator) if denominator else 0
                close(beta, fit["coefficients"][0], "independent analytic slope")
                coefficients_verified += 1
            else:
                # Verify NNLS KKT conditions directly rather than reusing its solver.
                features, beta = fit["features"], np.array(fit["coefficients"], float)
                x = np.array([[float(inputs[m["row_id"]][f]) for f in features] for m in train])
                y = np.array([float(inputs[m["row_id"]]["R"]) for m in train])
                w = np.array([1/train_counts[m["record"]] for m in train])
                gradient = x.T @ (w*(x@beta-y))
                assert (beta >= 0).all()
                assert (gradient[beta == 0] >= -1e-9).all(), gradient
                assert np.max(abs(gradient[beta > 0])) < 1e-9, gradient
                coefficients_verified += 1
        if name in ["endpoint52", "integral38"]:
            rows = read_csv(ROOT/f"{name}_temporal_predictions.csv")
            member = read_csv(ROOT/f"{name}_temporal_memberships.csv")
            train = [r for r in member if r["role"] == "train"]
            test = [r for r in member if r["role"] == "test"]
            assert {r["record"] for r in train} == {"nfts_62693", "nfts_62768", "nfts_65793"}
            assert {r["record"] for r in test} == {"nfts_68027"}
            independently[name+"_temporal"] = reconstruct_metrics(rows)
            for metric, value in independently[name+"_temporal"].items():
                close(value, performance[name+"_temporal"]["metrics"][metric], f"{name}:temporal:{metric}")
    knots = read_csv(ROOT/"common_date_integral_knots.csv")
    input_integrals = {r["row_id"]: r for r in read_csv(ROOT/"integral50_input.csv")}
    grouped = defaultdict(list)
    for row in knots:
        grouped[row["row_id"]].append(row)
    assert set(grouped) == set(input_integrals)
    for row_id, observations in grouped.items():
        observations.sort(key=lambda r:r["date"])
        days = [(date.fromisoformat(r["date"])-date.fromisoformat(observations[0]["date"])).days for r in observations]
        integrals = {}
        for arm in ["untreated", "treated"]:
            areas = [(days[i]-days[i-1])*(float(observations[i]["stb_"+arm])+float(observations[i-1]["stb_"+arm]))/2 for i in range(1,len(observations))]
            integrals[arm] = math.fsum(areas)
            close(integrals[arm], input_integrals[row_id]["auc_"+arm+"_percentage_days"], "paired integral arithmetic")
        close(integrals["untreated"]-integrals["treated"], input_integrals[row_id]["delta_stb_percentage_days"], "signed integral difference")
        close(days[-1], input_integrals[row_id]["window_days"], "actual observed time window")
    verified_cells = source_cells()
    result = {"status": "All independent arithmetic and provenance checks passed",
              "source_hashes_verified": len(hashes["sources"]), "raw_HTML_numeric_cells_verified": verified_cells,
              "fold_coefficients_or_NNLS_KKT_verified": coefficients_verified,
              "training_trial_total_weight_checks": weight_checks,
              "paired_integrals_reconstructed": len(grouped), "metrics": independently,
              "verification_method": "CSV decimal values, standard-library sums, per-trial mean losses, analytic single slopes, NNLS KKT gradients, separate source HTML cell reader; no imports from response_estimator or run_prototype"}
    (ROOT/"independent_arithmetic_verification.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({key: value for key,value in result.items() if key != "metrics"}, indent=2))


if __name__ == "__main__":
    main()
