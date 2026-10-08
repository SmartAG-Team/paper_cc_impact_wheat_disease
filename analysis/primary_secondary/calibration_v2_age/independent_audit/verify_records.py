#!/usr/bin/env python3
"""Read-only audit of archived wheat crop-protection prediction records.

No model imports, simulations, parameter fitting, optimization, interventions,
or acquisition are performed. New receipts are written only beside this script.

Run: python analysis/primary_secondary/calibration_v2_age/independent_audit/verify_records.py
Dependencies: numpy, pandas, pyarrow.
"""
from collections import defaultdict
from datetime import datetime, timezone
import ast
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
BASE = OUT.parent
ROOT = BASE.parents[2]
FRENCH = ROOT / "analysis/primary_secondary/external_french"
TRANSFER = FRENCH / "transfer_v2"
LEAVES = {"LEAF, 1ST / FLAG LEAF": 1, "LEAF, FLAG": 1, "LEAF, 2ND": 2,
          "LEAF, 3RD": 3, "LEAF, 4TH": 4, "LEAF, 5TH": 5,
          "LEAF, 6ST": 6, "LEAF, 6TH": 6, "LEAF, 7ST": 7, "LEAF, 7TH": 7}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Audit:
    def __init__(self):
        self.checks = []
        self.findings = []
        self.metric_rows = []
        self.input_hashes = {}

    def watch(self, path):
        path = Path(path)
        self.input_hashes.setdefault(str(path.relative_to(ROOT)), digest(path))
        return path

    def csv(self, path):
        return pd.read_csv(self.watch(path))

    def json(self, path):
        return json.loads(self.watch(path).read_text())

    def check(self, name, passed, evidence=None):
        self.checks.append({"check": name, "passed": bool(passed), "evidence": evidence})

    def finding(self, category, text):
        self.findings.append({"category": category, "finding": text})


def nested_metrics(records):
    """Scalar nested averages independent of the score and vector-weight helpers."""
    groups = defaultdict(lambda: defaultdict(list))
    for row in records.itertuples():
        groups[row.coordinate_year][row.series_id].append(row.predicted_percent - row.observed_percent)
    square, absolute, signed = [], [], []
    for series in groups.values():
        square.append(sum(sum(e*e for e in x)/len(x) for x in series.values())/len(series))
        absolute.append(sum(sum(abs(e) for e in x)/len(x) for x in series.values())/len(series))
        signed.append(sum(sum(x)/len(x) for x in series.values())/len(series))
    return {"rmse_pp": math.sqrt(sum(square)/len(square)),
            "mae_pp": sum(absolute)/len(absolute), "bias_pp": sum(signed)/len(signed),
            "n_coordinate_years": len(groups), "n_series": sum(len(x) for x in groups.values()),
            "n_targets": sum(sum(len(x) for x in series.values()) for series in groups.values()),
            "n_trials": records.TrialId.nunique()}


def audit_metrics(audit, predictions, table, group_columns, domain):
    audit.check(domain + ":unique metric rows", not table.duplicated(group_columns).any())
    expected = set(map(tuple, predictions[group_columns].drop_duplicates().to_numpy()))
    supplied = set(map(tuple, table[group_columns].to_numpy()))
    audit.check(domain + ":metric and prediction groups match", expected == supplied,
                {"prediction_groups": len(expected), "metric_groups": len(supplied)})
    maximum_difference = 0.
    for _, row in table.iterrows():
        mask = np.ones(len(predictions), dtype=bool)
        for column in group_columns:
            mask &= predictions[column].eq(row[column]).to_numpy()
        records = predictions.loc[mask]
        computed = nested_metrics(records)
        name = domain + ":" + ":".join(str(row[c]) for c in group_columns)
        differences = {metric: float(computed[metric] - row[metric])
                       for metric in ("rmse_pp", "mae_pp", "bias_pp")}
        maximum_difference = max(maximum_difference, *(abs(x) for x in differences.values()))
        counts_match = all(int(row[key]) == computed[key]
                           for key in ("n_coordinate_years", "n_series", "n_targets", "n_trials")
                           if key in row.index)
        audit.check(name + ":nested scores", all(abs(x) < 1e-9 for x in differences.values()) and counts_match,
                    {"difference_pp": differences, "recomputed_denominators": computed})
        audit.metric_rows.append({"domain": domain, **{c: row[c] for c in group_columns},
                                  **{k: computed[k] for k in ("rmse_pp", "mae_pp", "bias_pp")},
                                  **{k + "_difference_pp": v for k, v in differences.items()}})
    return maximum_difference


def valid_basf(source):
    source = source.copy()
    source["source_row"] = np.arange(2, len(source) + 2)
    source["Date"] = pd.to_datetime(source.Date)
    selected = source.loc[source.Treatment.eq("Untreated") & source.Organism.eq("SEPTTR")
                          & source.Parameter.eq("INFECT") & source.Method.eq("P%INF")
                          & source.Clarifier.ne("CROP INJURY") & source.PlantPart.isin(LEAVES)
                          & source.Value.between(0, 100)].copy()
    selected["leaf_rank"] = selected.PlantPart.map(LEAVES)
    selected["series_id"] = selected.TrialId.astype(str) + "|leaf" + selected.leaf_rank.astype(str)
    return selected


def chronological_state(records, trial, start):
    history = records.loc[records.TrialId.eq(trial) & records.Date.le(start)].sort_values("Date")
    latest = history.groupby("leaf_rank").tail(1)
    return {int(row.leaf_rank): (float(row.Value)/100., row.Date) for row in latest.itertuples()}


def function_source(path, name):
    text = Path(path).read_text()
    tree = ast.parse(text)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(text, node)
    raise ValueError(f"No function {name} in {path}")


def main():
    a = Audit()
    predictions = a.csv(BASE / "predictions.csv")
    metrics = a.csv(BASE / "fold_metrics.csv")
    fits = a.json(BASE / "fits.json")
    assignments = a.csv(BASE / "fold_assignments.csv")
    inner = a.csv(BASE / "training_inner_selection.csv")
    episodes = a.csv(BASE / "episodes.csv")
    targets = a.csv(BASE / "target_observations.csv")
    validation = a.json(BASE / "validation.json")
    contract = a.json(BASE / "source_snapshot/evaluation_contract_v2.json")
    full_fit = a.json(BASE / "all_development_fit.json")
    source = pd.read_csv(
        a.watch(ROOT / "data/basf-wheat-diseases.txt"), sep="\t")
    valid = valid_basf(source)
    weather = pd.read_parquet(a.watch(ROOT / "data/era5/daily_weather.parquet"))
    weather["date"] = pd.to_datetime(weather.date)
    weather = weather.set_index(["location_id", "date"])
    a.check("weather:unique location-date keys", not weather.index.duplicated().any())
    current_hashes, snapshot_hashes = [], []
    for filename, expected in validation["source_sha256"].items():
        current = a.watch(ROOT / filename)
        observed = digest(current)
        a.check("hash:current:" + filename, observed == expected,
                {"expected_sha256": expected, "actual_sha256": observed})
        current_hashes.append(filename)
        snapshot = BASE / "source_snapshot" / current.name
        if filename.startswith(("model/", "analysis/")):
            a.watch(snapshot)
            observed = digest(snapshot)
            a.check("hash:snapshot:" + filename, observed == expected,
                    {"expected_sha256": expected, "actual_sha256": observed})
            snapshot_hashes.append(filename)
    a.check("BASF:episode unique identity", not episodes.series_id.duplicated().any())
    a.check("BASF:assignment unique identity", not assignments.duplicated(["fold", "side", "series_id"]).any())
    a.check("BASF:prediction unique identity", not predictions.duplicated(["fold", "model", "source_row"]).any())
    a.check("BASF:finite percent bounds", np.isfinite(predictions.predicted_percent).all()
            and predictions.predicted_percent.between(0, 100).all())
    selected_source_rows = set()
    for _, group in valid.groupby("series_id"):
        if group.Date.nunique() >= 3:
            selected_source_rows.update(group.source_row)
    a.check("BASF:target source eligibility", set(targets.source_row) == selected_source_rows)
    ep = episodes.set_index("series_id")
    conditioning_rows = []
    chronology_ok = True
    weather_ok = True
    for e in episodes.itertuples():
        start, end = pd.Timestamp(e.start), pd.Timestamp(e.end)
        series = targets.loc[targets.series_id.eq(e.series_id)]
        past = chronological_state(valid, e.TrialId, start)
        chronology_ok &= (pd.to_datetime(series.Date).min() == start and pd.to_datetime(series.Date).max() == end
                          and int(series.conditioning.sum()) == 1 and float(series.iloc[0].Value) == e.initial_percent
                          and e.leaf_rank in past and abs(past[e.leaf_rank][0] - e.initial_percent/100.) < 1e-12)
        for rank, (value, date) in past.items():
            conditioning_rows.append({"domain": "BASF", "series_id": e.series_id, "leaf_rank": rank,
                                      "cutoff": start.date().isoformat(), "latest_observed_date": date.date().isoformat(),
                                      "age_days": int((start-date).days), "known_fraction": value})
        days = pd.date_range(start, end-pd.Timedelta(days=1))
        frame = weather.reindex(pd.MultiIndex.from_product([[e.location_id], days]))
        weather_ok &= bool(frame[["tmean_c", "rh_hours_ge_90pct", "rain_mm", "hour_count"]].notna().all().all()
                           and frame.hour_count.eq(24).all())
    p_dates = pd.to_datetime(predictions.Date)
    p_starts = pd.to_datetime(predictions.start)
    a.check("BASF:conditional episode chronology", chronology_ok and p_dates.gt(p_starts).all()
            and ((p_dates-p_starts).dt.days == predictions.day).all() and not bool(predictions.conditioning.any()))
    a.check("BASF:complete preceding-day weather", weather_ok)
    raw_lookup = valid.set_index("source_row")
    provenance_ok = True
    for row in predictions.itertuples():
        raw = raw_lookup.loc[row.source_row]
        provenance_ok &= (raw.TrialId == row.TrialId and raw.series_id == row.series_id
                          and raw.Date == pd.Timestamp(row.Date) and raw.Value == row.observed_percent
                          and int(row.target_leaf_index) == int(raw.leaf_rank)-1)
    a.check("BASF:all scored rows reconcile to source", provenance_ok)
    fit_lookup = {(fit["fold"], fit["model"]): fit for fit in fits}
    a.check("BASF:one fit per score", len(fit_lookup) == len(fits)
            and set(fit_lookup) == set(zip(metrics.fold, metrics.model)))
    fold_rows = []
    inner_rows = []
    for fold, records in assignments.groupby("fold"):
        train_ids = set(records.loc[records.side.eq("train"), "series_id"])
        test_ids = set(records.loc[records.side.eq("test"), "series_id"])
        train, test = ep.loc[sorted(train_ids)], ep.loc[sorted(test_ids)]
        disjoint = not train_ids & test_ids and not set(train.location_id) & set(test.location_id)
        a.check(f"split:{fold}:location disjointness", disjoint)
        if fold.startswith("forward_"):
            year = int(fold.removeprefix("forward_"))
            expected_test = set(episodes.loc[episodes.year.eq(year), "series_id"])
            locations = set(ep.loc[sorted(expected_test), "location_id"])
            expected_train = set(episodes.loc[episodes.year.lt(year) & ~episodes.location_id.isin(locations), "series_id"])
            time_ok = int(train.year.max()) < int(test.year.min()) and pd.to_datetime(train.end).max() < pd.to_datetime(test.start).min()
            a.check(f"split:{fold}:forward chronology", time_ok)
        else:
            country = test.Country.iloc[0]
            expected_test = set(episodes.loc[episodes.Country.eq(country), "series_id"])
            locations = set(ep.loc[sorted(expected_test), "location_id"])
            expected_train = set(episodes.loc[~episodes.Country.eq(country) & ~episodes.location_id.isin(locations), "series_id"])
            a.check(f"split:{fold}:country separation", test.Country.nunique() == 1
                    and not set(train.Country) & set(test.Country))
        a.check(f"split:{fold}:exact declared membership", train_ids == expected_train and test_ids == expected_test)
        expected_source = set(targets.loc[targets.series_id.isin(test_ids) & ~targets.conditioning, "source_row"])
        for model, p in predictions.loc[predictions.fold.eq(fold)].groupby("model"):
            a.check(f"split:{fold}:{model}:complete common test targets", set(p.source_row) == expected_source
                    and set(p.series_id) == test_ids)
        train_targets = targets.loc[targets.series_id.isin(train_ids) & ~targets.conditioning]
        candidates = inner.loc[inner.outer_fold.eq(fold)].sort_values(["rmse_pp", "latent_days", "infectious_days"])
        winner = candidates.iloc[0]
        chosen = fit_lookup[(fold, "primary_secondary_selected_inner")]
        expected_candidates = {(x,y) for x in (10.,20.,30.) for y in (14.,21.,28.)}
        a.check(f"inner:{fold}:candidate grid", set(zip(candidates.latent_days, candidates.infectious_days)) == expected_candidates
                and len(candidates) == len(expected_candidates))
        a.check(f"inner:{fold}:winner matches archived fit", chosen["parameters"]["latent_days"] == winner.latent_days
                and chosen["parameters"]["infectious_days"] == winner.infectious_days)
        a.check(f"inner:{fold}:training-only candidate denominators",
                candidates.n_coordinate_years.eq(train_targets.coordinate_year.nunique()).all()
                and candidates.n_series.eq(len(train_ids)).all() and candidates.n_targets.eq(len(train_targets)).all())
        locations = sorted(train.location_id.unique())
        for inner_fold in range(min(3,len(locations))):
            held = {loc for i,loc in enumerate(locations) if i % min(3,len(locations)) == inner_fold}
            inner_test_ids = set(train.loc[train.location_id.isin(held)].index)
            inner_train_ids = train_ids-inner_test_ids
            inner_rows.append({"outer_fold": fold, "inner_fold": inner_fold,
                               "derived_inner_train_series": len(inner_train_ids), "derived_inner_test_series": len(inner_test_ids),
                               "outer_test_series_overlap": len((inner_train_ids|inner_test_ids)&test_ids),
                               "inner_location_overlap": len(set(ep.loc[sorted(inner_train_ids), "location_id"])
                                                             &set(ep.loc[sorted(inner_test_ids), "location_id"]))})
        fold_rows.append({"fold": fold, "train_series": len(train_ids), "test_series": len(test_ids),
                          "train_locations": train.location_id.nunique(), "test_locations": test.location_id.nunique(),
                          "selected_latent_days": winner.latent_days, "selected_infectious_days": winner.infectious_days})
    a.check("inner:derived training-only location splits", all(
        row["outer_test_series_overlap"] == 0 and row["inner_location_overlap"] == 0 for row in inner_rows))
    snapshot_calibrate = a.watch(BASE / "source_snapshot/calibrate.py")
    snapshot_field = a.watch(BASE / "source_snapshot/field_data.py")
    main_source = function_source(snapshot_calibrate, "main")
    select_source = function_source(snapshot_calibrate, "select_training_hyperparameters")
    state_source = function_source(snapshot_field, "canopy_snapshot")
    weather_source = function_source(snapshot_field, "weather_window")
    a.check("inner:archived call chain restricts selection to outer training",
            "select_training_hyperparameters(train)" in main_source
            and "inner_folds(batch.episodes)" in select_source and "batch.subset(train_ids)" in select_source)
    a.check("conditioning:archived cutoff and preceding-weather predicates",
            "frame.Date.le(cutoff)" in state_source and "end-pd.Timedelta(days=1)" in weather_source)
    a.finding("verification_limit", "Inner validation predictions, inner fits and actual inner assignments are not archived. Candidate-score arithmetic cannot be independently recomputed; training-only evidence consists of matching group/count records, winner consistency and the exact archived call chain.")
    basf_difference = audit_metrics(a, predictions, metrics, ["fold", "model"], "BASF")
    # Reconstruct archived BASF statistical baselines directly; no model code executes.
    maximum_baseline_difference = 0.
    baseline_ok = True
    cache = {}
    for row in predictions.loc[predictions.model.isin(["persistence", "global_logit_trend", "thermal_logit_trend", "canopy_logit_trend"])].itertuples():
        e = ep.loc[row.series_id]
        key = (row.series_id, row.Date)
        if key not in cache:
            state = chronological_state(valid, e.TrialId, pd.Timestamp(e.start))
            initial = state[int(e.leaf_rank)][0]
            canopy = sum(x[0] for x in state.values())/len(state)
            days = pd.date_range(e.start, pd.Timestamp(row.Date)-pd.Timedelta(days=1))
            daily = weather.reindex(pd.MultiIndex.from_product([[e.location_id], days]))
            thermal = sum(max(float(x),0.)/18. for x in daily.tmean_c)
            cache[key] = initial, canopy, thermal
        initial, canopy, thermal = cache[key]
        fit = fit_lookup[(row.fold,row.model)]
        if row.model == "persistence":
            expected = initial*100.
        else:
            slope = fit["slope"] + (fit["canopy_slope"]*canopy if row.model == "canopy_logit_trend" else 0.)
            exposure = thermal if row.model == "thermal_logit_trend" else row.day
            clipped = min(.995,max(.005,initial))
            z = math.log(clipped/(1-clipped)) + slope*exposure
            expected = 100./(1+math.exp(-z))
        difference = abs(expected-row.predicted_percent)
        maximum_baseline_difference = max(maximum_baseline_difference,difference)
        baseline_ok &= difference < 1e-9
    a.check("BASF:all archived statistical baseline predictions independently reconstructed", baseline_ok,
            {"maximum_absolute_difference_pp": maximum_baseline_difference})
    mechanism = predictions.loc[predictions.infectious_percent.notna()]
    a.check("BASF:mechanistic endpoint bounds and containment",
            mechanism.infectious_percent.between(0,100).all() and mechanism.pycnidia_proxy_percent.between(0,100).all()
            and (mechanism.infectious_percent <= mechanism.pycnidia_proxy_percent+1e-9).all()
            and (mechanism.pycnidia_proxy_percent <= mechanism.predicted_percent+1e-9).all())
    origin = mechanism[["unresolved_initial_infected_percent","external_infected_percent","secondary_infected_percent"]]
    a.check("BASF:origin fraction bounds", np.isfinite(origin).all().all() and origin.ge(-1e-9).all().all()
            and origin.sum(axis=1).le(100+1e-9).all())
    a.check("BASF:validation episode and target counts", validation["episodes"] == len(episodes)
            and validation["targets"] == len(targets) and validation["scored_targets"] == int((~targets.conditioning).sum()))
    a.finding("verification_limit", "Full compartment trajectories are not archived. The recorded mass-conservation, minimum-state and time-step sensitivity checks are not independently rerun; this audit verifies exported endpoint and origin bounds only.")
    french_summary = audit_french(a, weather, conditioning_rows)
    pd.DataFrame(a.metric_rows).to_csv(OUT / "recomputed_scores.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(OUT / "split_and_selection_checks.csv", index=False)
    pd.DataFrame(inner_rows).to_csv(OUT / "derived_inner_group_checks.csv", index=False)
    pd.DataFrame(conditioning_rows).to_csv(OUT / "chronological_conditioning_checks.csv", index=False)
    unchanged = all(digest(ROOT/path) == value for path,value in a.input_hashes.items())
    a.check("all original audited inputs preserved", unchanged)
    failures = [check for check in a.checks if not check["passed"]]
    receipt = {"created_utc": datetime.now(timezone.utc).isoformat(),
               "scope": "Independent statistical-record audit; no simulation, fitting, intervention or source acquisition",
               "status": "discrepancies detected" if failures else "verified within archived-record scope; verification limits retained",
               "checks": a.checks, "failed_checks": failures, "findings": a.findings,
               "n_checks": len(a.checks), "n_failed_checks": len(failures),
               "BASF": {"prediction_rows": len(predictions), "folds": assignments.fold.nunique(),
                        "metric_groups": len(metrics), "episodes": len(episodes),
                        "maximum_recomputed_score_difference_pp": basf_difference,
                        "maximum_reconstructed_baseline_prediction_difference_pp": maximum_baseline_difference,
                        "exact_current_source_hashes_checked": len(current_hashes),
                        "exact_snapshot_hashes_checked": len(snapshot_hashes),
                        "inner_score_arithmetic_verified": False,
                        "natural_onset_or_process_identification_validated": False},
               "French": french_summary, "original_files_preserved": unchanged,
               "input_sha256": a.input_hashes, "verification_script_sha256": digest(__file__)}
    (OUT / "audit_receipt.json").write_text(json.dumps(receipt, indent=2, ensure_ascii=False, default=native) + "\n")
    print(json.dumps({"status": receipt["status"], "checks": len(a.checks), "failed": len(failures),
                      "BASF": receipt["BASF"], "French": french_summary}, indent=2, default=native))
    if failures:
        print(json.dumps(failures, indent=2, default=native))
        sys.exit(1)


def native(value):
    if isinstance(value,np.generic):
        return value.item()
    return str(value)


def audit_french(a, weather, conditioning_rows):
    p = a.csv(TRANSFER / "frozen_transfer_predictions.csv")
    metrics = a.csv(TRANSFER / "frozen_transfer_metrics.csv")
    receipt = a.json(TRANSFER / "transfer_validation.json")
    episodes = a.csv(FRENCH / "episodes.csv")
    targets = a.csv(FRENCH / "targets.csv")
    observations = a.csv(FRENCH / "all_plot_leaf_observations.csv")
    inventory = a.json(FRENCH / "inventory.json")
    fit_path = a.watch(ROOT / receipt["BASF_fit_path"])
    fit_hash_verified = digest(fit_path) == receipt["BASF_fit_sha256"]
    a.check("French:exact frozen mechanistic fit hash", fit_hash_verified)
    simulator = a.watch(ROOT / "model/primary_secondary/core.py")
    a.check("French:exact simulator hash", digest(simulator) == receipt["simulator_sha256"])
    source_path = a.watch(ROOT / "data/public_septoria/orellana-torrejon-2022-field/F1_Field_disease_severity_rawdata.csv")
    a.check("French:exact raw source hash", digest(source_path) == inventory["source_sha256"])
    prepare_source = a.watch(FRENCH / "prepare_external.py")
    transfer_source = a.watch(FRENCH / "evaluate_transfer.py")
    code = function_source(transfer_source, "main")
    a.check("French:current transfer code fits no French responses", "fit_baseline(development,name)" in code
            and "fit_mechanism" not in code and "target_values(batch,trajectory,'pycnidia')" in code)
    a.finding("verification_limit", "French transfer receipts hash the mechanistic BASF fit and simulator but do not hash their original preprocessing/transfer scripts or archive the all-development baseline coefficients. Current scripts are hashed by this audit; frozen statistical-baseline fitting provenance is supported by the current call chain, not a contemporaneous baseline-fit artifact.")
    difference = audit_metrics(a,p,metrics,["year","model"],"French")
    a.check("French:unique scored identity", not p.duplicated(["model","series_id","Date"]).any())
    a.check("French:finite percentage bounds", np.isfinite(p.predicted_percent).all() and p.predicted_percent.between(0,100).all())
    a.check("French:single station and two station-years", p.location_id.nunique() == 1 and p.coordinate_year.nunique() == 2
            and metrics.n_coordinate_years.eq(1).all() and receipt["independent_stations"] == 1 and receipt["station_years"] == 2)
    expected = set(zip(targets.loc[~targets.conditioning,"series_id"],targets.loc[~targets.conditioning,"Date"]))
    for model, records in p.groupby("model"):
        a.check("French:"+model+":complete common future targets", set(zip(records.series_id,records.Date)) == expected)
    ep = episodes.set_index("series_id")
    chronology_ok, weather_ok, persistence_ok = True, True, True
    observations["Date"] = pd.to_datetime(observations.Date)
    known = observations.loc[observations.n_numeric.gt(0)]
    for e in episodes.itertuples():
        start,end = pd.Timestamp(e.start),pd.Timestamp(e.end)
        group = targets.loc[targets.series_id.eq(e.series_id)]
        past = chronological_state(known,e.TrialId,start)
        chronology_ok &= pd.to_datetime(group.Date).min() == start and pd.to_datetime(group.Date).max() == end
        chronology_ok &= int(group.conditioning.sum()) == 1 and int(e.leaf_rank) in past
        chronology_ok &= abs(past[int(e.leaf_rank)][0]-e.initial_percent/100) < 1e-12
        for rank,(value,date) in past.items():
            conditioning_rows.append({"domain":"French","series_id":e.series_id,"leaf_rank":rank,
                                      "cutoff":start.date().isoformat(),"latest_observed_date":date.date().isoformat(),
                                      "age_days":int((start-date).days),"known_fraction":value})
        days = pd.date_range(start,end-pd.Timedelta(days=1))
        frame = weather.reindex(pd.MultiIndex.from_product([[e.location_id],days]))
        weather_ok &= frame[["tmean_c","rh_hours_ge_90pct","rain_mm","hour_count"]].notna().all().all()
        weather_ok &= frame.hour_count.eq(24).all()
    target_lookup = targets.set_index(["series_id","Date"])
    source_ok = True
    for row in p.itertuples():
        target = target_lookup.loc[(row.series_id,row.Date)]
        start = pd.Timestamp(ep.loc[row.series_id,"start"])
        chronology_ok &= pd.Timestamp(row.Date)>start and int(row.day)==int((pd.Timestamp(row.Date)-start).days) and not row.conditioning
        source_ok &= abs(row.observed_percent-target.Value)<1e-12 and row.TrialId==target.TrialId
        source_ok &= row.mixture==target.mixture and row.rep==target.rep and row.var_origin==target.var_origin
        if row.model == "persistence_BASF_frozen":
            persistence_ok &= abs(row.predicted_percent-ep.loc[row.series_id,"initial_percent"])<1e-12
    a.check("French:conditional chronology and target first observations",chronology_ok)
    a.check("French:complete preceding-day weather",weather_ok)
    a.check("French:predictions preserve target values and plot-cultivar identities",source_ok)
    a.check("French:persistence reconstructed from initial observations",persistence_ok)
    # Independently reconcile numeric means and denominators to original plant records.
    raw = pd.read_csv(source_path,sep=";",comment="#")
    raw["date_iso"] = pd.to_datetime(raw.date,dayfirst=True).dt.strftime("%Y-%m-%d")
    cells = defaultdict(list)
    for row in raw.to_dict("records"):
        for rank in (1,2,3):
            key = (row["date_iso"],row["mixture"],row["rep"],row["var_origin"],rank)
            cells[key].append((row["id"],row[f"%spor_F{rank}"]))
    denominators_ok,means_ok,identity_ok = True,True,True
    for row in observations.itertuples():
        key = (row.Date.strftime("%Y-%m-%d"),row.mixture,row.rep,row.var_origin,int(row.leaf_rank))
        source = cells[key]
        numbers,senescent,unassessed = [],0,0
        for _,value in source:
            if pd.isna(value) or str(value).strip() in ("NA","nan",""):
                unassessed+=1
            elif str(value).strip()=="S":
                senescent+=1
            else:
                numbers.append(float(value))
        denominators_ok &= row.n_numeric==len(numbers) and row.n_senescent==senescent and row.n_unassessed==unassessed
        means_ok &= (abs(row.Value-sum(numbers)/len(numbers))<1e-12) if numbers else pd.isna(row.Value)
        identity_ok &= set(str(x).removesuffix(".0") for x,_ in source)==set(str(row.source_rows).split(","))
    a.check("French:raw numeric denominators and senescence counts",denominators_ok)
    a.check("French:raw pycnidial means independently reconstructed",means_ok)
    a.check("French:raw plant-row identity preserved",identity_ok)
    series = p.loc[p.model.eq("primary_secondary_hidden_BASF_frozen")].drop_duplicates("series_id")
    plot_series = series.groupby(["year","mixture","rep"]).series_id.nunique()
    a.finding("weighting_scope", "French scores give equal weight to plot×cultivar×leaf-rank series within the station-year, then equal future assessments within each series. Mixture plots with two cultivar strata contain six series and pure-cultivar plots contain three; mixture plots therefore receive twice the aggregate weight of pure-cultivar plots. The reported scores are not equal-physical-plot scores; plots are not treated as independent stations.")
    return {"prediction_rows":len(p),"metric_groups":len(metrics),
            "maximum_recomputed_score_difference_pp":difference,"independent_stations":p.location_id.nunique(),
            "station_years":p.coordinate_year.nunique(),"plot_years":len(plot_series),
            "series_per_plot_year_distribution":{str(k):int(v) for k,v in plot_series.value_counts().items()},
            "frozen_mechanistic_fit_hash_verified":fit_hash_verified,
            "frozen_baseline_coefficients_archived":False,"natural_field_onset_validation":False}


if __name__ == "__main__":
    main()
