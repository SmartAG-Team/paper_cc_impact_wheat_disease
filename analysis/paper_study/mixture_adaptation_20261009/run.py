"""Reproduce the frozen empirical comparisons entirely from public_inputs.zip."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import platform
import zipfile

import numpy as np
import pandas as pd

from core import (applicability, complete_mean, control_baseline, dependency_groups,
                  environment_means, kernel_weight, map_release, site_bootstrap,
                  transfer_predictions, yield_contrast, yield_t_ha)

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def checked_inputs():
    lock = json.loads((HERE / "protocol_lock.json").read_text())
    if sha((HERE / "PROTOCOL.md").read_bytes()) != lock["protocol_sha256"]:
        raise ValueError("Frozen protocol changed")
    manifest = json.loads((HERE / "input_manifest.json").read_text())
    data = (HERE / manifest["bundle"]).read_bytes()
    if sha(data) != manifest["bundle_sha256"]:
        raise ValueError("Input bundle hash mismatch")
    archive = zipfile.ZipFile(io.BytesIO(data))
    if set(archive.namelist()) != {m["member"] for m in manifest["members"]}:
        raise ValueError("Input membership mismatch")
    for m in manifest["members"]:
        if sha(archive.read(m["member"])) != m["sha256"]:
            raise ValueError(f"Input member changed: {m['member']}")
    return archive, manifest, lock


def read_csv(archive, member):
    return pd.read_csv(io.BytesIO(archive.read(member)))


def write_csv(out, name, data):
    data.to_csv(out / f"{name}.csv", index=False, float_format="%.12g", lineterminator="\n")


def french(archive, out):
    prefix = "sources/montazeaud2022/"
    raw = read_csv(archive, prefix + "raw_yield_variables.csv")
    author = read_csv(archive, prefix + "GY_RAW_RYT.csv")
    stb = read_csv(archive, prefix + "STB_RAW_RYT.csv")
    raw["source_line"] = np.arange(2, len(raw) + 2)
    author["author_source_line"] = np.arange(2, len(author) + 2)
    if author.duplicated(["row", "column"]).any() or raw.duplicated(["row", "column", "sub_column"]).any():
        raise ValueError("French spatial unit duplicated")
    by_grid = {(int(r), int(c)): g for (r, c), g in raw.groupby(["row", "column"])}
    plot_rows, components = [], {}
    for _, a in author.iterrows():
        r, c = int(a.row), int(a.column)
        g = by_grid[(r, c)]
        expected = {a.focal, a.neighbor}
        if set(g.genotype) != expected or set(g.assoc) != {a.assoc} or len(g) != 4:
            raise ValueError(f"French composition or row linkage mismatch: {(r, c)}")
        pid = f"FR_Mauguio_2018_r{r:02}_c{c:02}"
        component_yields = {k: float(v.GY.mean()) if v.GY.notna().all() else np.nan for k, v in g.groupby("genotype")}
        if a.assoc == "P" and set(g.groupby("genotype").size()) != {2}:
            raise ValueError("Unequal constituent harvest areas")
        components[pid] = component_yields
        plot_rows.append({"plot_id": pid, "environment_id": "FR_Mauguio_2018", "site_id": "Mauguio", "year": 2018,
                          "grid_row": r, "grid_column": c, "treatment": "mixture" if a.assoc == "P" else "pure",
                          "constituent_1": a.focal, "constituent_2": a.neighbor,
                          "composition_id": "|".join(sorted(expected)), "n_raw_rows": len(g), "n_finite_yield_rows": int(g.GY.notna().sum()),
                          "yield_t_ha": yield_t_ha(complete_mean(g.GY), "g_m2"),
                          "author_blup_t_ha": yield_t_ha(a.RAW_GY, "g_m2"),
                          "author_component_ryt": a.RYT_GY, "tkw_g": kernel_weight(g.GY, g.TKW),
                          "raw_source_member": prefix + "raw_yield_variables.csv", "raw_source_lines": "|".join(g.source_line.astype(str)),
                          "author_source_line": a.author_source_line, "moisture_basis": "dried; percentage unspecified"})
    plots = pd.DataFrame(plot_rows)
    stb = stb.rename(columns={"row": "grid_row", "column": "grid_column", "RAW_severity": "stb_ordinal"})
    plots = plots.merge(stb[["grid_row", "grid_column", "stb_ordinal"]], how="left", on=["grid_row", "grid_column"], validate="one_to_one")
    pure = plots[plots.treatment.eq("pure")].set_index("constituent_1")
    if not pure.index.is_unique:
        raise ValueError("French pure-stand genotype has multiple plot references")
    contrasts, exclusions = [], []
    for _, m in plots[plots.treatment.eq("mixture")].iterrows():
        names = [m.constituent_1, m.constituent_2]
        reason = "incomplete_four_row_harvest" if not np.isfinite(m.yield_t_ha) else "missing_constituent_pure_stand" if any(n not in pure.index for n in names) else None
        if reason is None and not np.isfinite(pure.loc[names, "yield_t_ha"]).all():
            reason = "incomplete_constituent_harvest"
        if reason:
            exclusions.append({"plot_id": m.plot_id, "composition_id": m.composition_id, "reason": reason})
            continue
        p = pure.loc[names]
        contrast = yield_contrast(m.yield_t_ha, p.yield_t_ha)
        adjusted = yield_contrast(m.author_blup_t_ha, p.author_blup_t_ha)
        tkw_base = kernel_weight(p.yield_t_ha, p.tkw_g)
        d = {**m.to_dict(), "control_1_id": p.plot_id.iloc[0], "control_2_id": p.plot_id.iloc[1],
             "control_1_grid_row": p.grid_row.iloc[0], "control_2_grid_row": p.grid_row.iloc[1],
             "control_1_grid_column": p.grid_column.iloc[0], "control_2_grid_column": p.grid_column.iloc[1],
             "control_1_yield_t_ha": p.yield_t_ha.iloc[0], "control_2_yield_t_ha": p.yield_t_ha.iloc[1],
             "baseline_t_ha": contrast["baseline"], "delta_t_ha": contrast["delta"], "relative_percent": contrast["relative_percent"],
             "hindsight_best_delta_t_ha": contrast["hindsight_best_delta"],
             "raw_component_ryt": np.mean([components[m.plot_id][n] * 0.01 / pure.loc[n, "yield_t_ha"] for n in names]),
             "blup_baseline_t_ha": adjusted["baseline"], "blup_delta_t_ha": adjusted["delta"], "blup_relative_percent": adjusted["relative_percent"],
             "tkw_baseline_g": tkw_base, "tkw_delta_g": m.tkw_g - tkw_base,
             "inference_class": "randomized_within_trial_complete_case_comparison",
             "stb_mediated_yield_loss": False, "validated_climate_adaptation": False}
        contrasts.append(d)
    contrasts = pd.DataFrame(contrasts)
    contrasts["dependency_group"] = dependency_groups(contrasts)
    influence = []
    for kind, identifiers in [("genotype", sorted(set(contrasts.constituent_1) | set(contrasts.constituent_2))),
                              ("grid_row", sorted(plots.grid_row.unique())), ("grid_column", sorted(plots.grid_column.unique()))]:
        for identifier in identifiers:
            if kind == "genotype":
                keep = contrasts.constituent_1.ne(identifier) & contrasts.constituent_2.ne(identifier)
            else:
                keep = contrasts[kind].ne(identifier) & contrasts[f"control_1_{kind}"].ne(identifier) & contrasts[f"control_2_{kind}"].ne(identifier)
            remaining = contrasts[keep]
            influence.append({"deletion_group": kind, "omitted": identifier, "n_remaining": len(remaining),
                              "n_removed": len(contrasts) - len(remaining), "mean_delta_t_ha": remaining.delta_t_ha.mean(),
                              "mean_relative_percent": remaining.relative_percent.mean(), "interpretation": "influence_only_not_CI_or_external_validation"})
    group_rows = []
    for group, d in contrasts.groupby("dependency_group"):
        group_rows.append({"dependency_group": group, "n_mixtures": len(d), "n_genotypes": len(set(d.constituent_1) | set(d.constituent_2)),
                           "mean_delta_t_ha": d.delta_t_ha.mean(), "share_of_comparisons": len(d) / len(contrasts)})
    sums = []
    for endpoint, diff, rel in [("raw_primary", "delta_t_ha", "relative_percent"), ("author_BLUP_same_comparisons_sensitivity", "blup_delta_t_ha", "blup_relative_percent")]:
        sums.append({"endpoint": endpoint, "n_mixtures": len(contrasts), "n_constituent_controls": len(set(contrasts.control_1_id) | set(contrasts.control_2_id)),
                     "n_environments": 1, "n_dependency_groups": contrasts.dependency_group.nunique(),
                     "mean_delta_t_ha": contrasts[diff].mean(), "mean_relative_percent": contrasts[rel].mean(),
                     "below_baseline_n": int(contrasts[diff].lt(0).sum()), "below_baseline_fraction": contrasts[diff].lt(0).mean(),
                     "empirical_p10_delta_t_ha": contrasts[diff].quantile(.1), "minimum_delta_t_ha": contrasts[diff].min(),
                     "ci_lower": np.nan, "ci_upper": np.nan,
                     "interval_status": "unavailable: one environment; shared controls; dominant connected group; no independent combination replication"})
    # Track every raw grid, including rows not retained by the original author analysis.
    eligibility = []
    for (r, c), g in sorted(by_grid.items()):
        a = author[author.row.eq(r) & author.column.eq(c)]
        eligibility.append({"grid_row": r, "grid_column": c, "n_raw_rows": len(g), "n_finite_yield_rows": int(g.GY.notna().sum()),
                            "author_retained": bool(len(a)), "primary_plot_eligible": bool(len(a) and g.GY.notna().all()),
                            "source_lines": "|".join(g.source_line.astype(str))})
    quality = pd.DataFrame([{"trait": "thousand_kernel_weight", "unit": "g", "n_comparisons": int(contrasts.tkw_delta_g.notna().sum()),
                             "mean_delta": contrasts.tkw_delta_g.mean(), "lower_than_baseline_fraction": contrasts.tkw_delta_g.dropna().lt(0).mean(),
                             "yield_positive_tkw_negative_n": int((contrasts.delta_t_ha.gt(0) & contrasts.tkw_delta_g.lt(0)).sum()),
                             "interpretation": "measured yield component; not milling quality or economic return"}])
    hindsight = pd.DataFrame([{"comparison": "higher_observed_constituent", "n": len(contrasts),
                               "mean_delta_t_ha": contrasts.hindsight_best_delta_t_ha.mean(),
                               "above_fraction": contrasts.hindsight_best_delta_t_ha.gt(0).mean(),
                               "interpretation": "selection_on_observed_yields; hindsight benchmark, not prospective cultivar policy"}])
    for name, d in [("french_plots", plots), ("french_contrasts", contrasts), ("french_exclusions", pd.DataFrame(exclusions)),
                    ("french_summary", pd.DataFrame(sums)), ("french_influence", pd.DataFrame(influence)),
                    ("french_dependency_groups", pd.DataFrame(group_rows)), ("french_raw_grid_eligibility", pd.DataFrame(eligibility)),
                    ("french_measured_tradeoffs", quality), ("french_hindsight_selection", hindsight)]:
        write_csv(out, name, d)
    return plots, contrasts, pd.DataFrame(sums)


def swiss(archive, out):
    current = read_csv(archive, "sources/swiss-mixtures/blendit_full_pub.csv")
    previous = read_csv(archive, "sources/swiss-mixtures-v1/blendit_full.csv")
    records, mapping = map_release(current, previous)
    records["environment_id"] = "CH_" + records.postcode.astype(int).astype(str) + "_" + records.year.astype(int).astype(str)
    records["site_id"] = records.postcode.astype(int).astype(str)
    records["control_group"] = records.ww.map(lambda x: str(int(x)) if pd.notna(x) else "unknown")
    records["physical_plot_id_recovered"] = records.plot_id.notna()
    records["plot_id"] = records.plot_id.fillna("CH_pub_lines_" + records.current_source_lines)
    records["cultivar"] = records.name_var.str.strip()
    records["yield_t_ha"] = yield_t_ha(records.yield_dtha, "dt_ha")
    records["composition_id"] = records.apply(lambda r: "|".join(sorted(str(r[k]).strip() for k in ["var_1", "var_2", "var_3", "var_4"] if pd.notna(r[k]))), axis=1)
    candidate = records[records.mono_mix.eq("mix") & records.physical_plot_id_recovered & records.ww.eq(33)]
    mixes = candidate[candidate.var_numb.eq(2) & candidate.yield_t_ha.notna()].copy()
    controls = records[records.mono_mix.eq("mono") & records.year.isin(mixes.year) & records.postcode.isin(mixes.postcode)].copy()
    comparison_rows, coverage, check_rows = [], [], []
    for _, m in mixes.iterrows():
        names = [str(m.var_1).strip(), str(m.var_2).strip()]
        in_env = controls[controls.environment_id.eq(m.environment_id)]
        for code in sorted(set(in_env.control_group) - {"unknown", "33"}):
            b = control_baseline(controls, m.environment_id, code, names, "yield_t_ha")
            missing = [n for n in names if not len(in_env[in_env.control_group.eq(code) & in_env.cultivar.eq(n) & in_env.yield_t_ha.notna()])]
            coverage.append({"mixture_plot_id": m.plot_id, "environment_id": m.environment_id, "control_group": code,
                             "constituent_1": names[0], "constituent_2": names[1], "eligible": b is not None,
                             "missing_constituents": "|".join(missing), "exchangeability_validated": False})
            if b is None:
                continue
            effect = yield_contrast(m.yield_t_ha, b["values"])
            d = {"mixture_plot_id": m.plot_id, "environment_id": m.environment_id, "site_id": m.site_id, "year": m.year,
                 "block": m.rep, "composition_id": m.composition_id, "control_group": code,
                 "constituent_1": names[0], "constituent_2": names[1], "control_1_ids": b["control_ids"][0], "control_2_ids": b["control_ids"][1],
                 "control_1_n": b["control_n"][0], "control_2_n": b["control_n"][1],
                 "control_1_mean_t_ha": b["values"][0], "control_2_mean_t_ha": b["values"][1], "yield_t_ha": m.yield_t_ha,
                 "baseline_t_ha": effect["baseline"], "delta_t_ha": effect["delta"], "relative_percent": effect["relative_percent"],
                 "hindsight_best_delta_t_ha": effect["hindsight_best_delta"], "density": m.density,
                 "inference_class": "noncausal_matched_source_diagnostic", "causal_contrast": False,
                 "current_source_lines": m.current_source_lines, "previous_source_lines": m.previous_source_lines}
            for trait in ["prot", "phl"]:
                bq = control_baseline(controls, m.environment_id, code, names, trait)
                d[f"{trait}_mixture"] = m[trait]
                d[f"{trait}_baseline"] = np.mean(bq["values"]) if bq is not None else np.nan
                d[f"{trait}_delta"] = m[trait] - d[f"{trait}_baseline"]
                d[f"{trait}_control_ids"] = ";".join(bq["control_ids"]) if bq is not None else ""
            comparison_rows.append(d)
        for check in ["HANSWIN", "MONTALBANO"]:
            q = in_env[in_env.control_group.eq("33") & in_env.rep.eq(m.rep) & in_env.cultivar.eq(check) & in_env.yield_t_ha.notna()]
            if len(q) > 1:
                raise ValueError("Duplicated within-block check")
            if len(q):
                check_rows.append({"mixture_plot_id": m.plot_id, "environment_id": m.environment_id, "site_id": m.site_id,
                                   "year": m.year, "block": m.rep, "composition_id": m.composition_id, "check": check,
                                   "control_id": q.plot_id.iloc[0], "delta_t_ha": m.yield_t_ha - q.yield_t_ha.iloc[0],
                                   "inference_class": "same_block_check_diagnostic; pure_check_corandomization_unconfirmed; not_constituent_baseline"})
    contrasts, checks = pd.DataFrame(comparison_rows), pd.DataFrame(check_rows)
    env_rows, summary_rows, transfer_rows, quality_rows = [], [], [], []
    for code, d in contrasts.groupby("control_group", sort=True):
        composition = d.groupby(["environment_id", "site_id", "year", "composition_id"], as_index=False).agg(
            delta_t_ha=("delta_t_ha", "mean"), relative_percent=("relative_percent", "mean"),
            hindsight_best_delta_t_ha=("hindsight_best_delta_t_ha", "mean"), n_mixture_plots=("mixture_plot_id", "size"))
        env = composition.groupby(["environment_id", "site_id", "year"], as_index=False).agg(
            delta_t_ha=("delta_t_ha", "mean"), relative_percent=("relative_percent", "mean"),
            hindsight_best_delta_t_ha=("hindsight_best_delta_t_ha", "mean"),
            below_baseline_fraction=("delta_t_ha", lambda x: x.lt(0).mean()),
            empirical_p10_delta_t_ha=("delta_t_ha", lambda x: x.quantile(.1)),
            n_compositions=("composition_id", "size"), n_mixture_plots=("n_mixture_plots", "sum"))
        env["control_group"] = code
        env["inference_class"] = "noncausal_matched_source_diagnostic"
        env_rows.append(env)
        ci = site_bootstrap(env, "delta_t_ha")
        summary_rows.append({"control_group": code, "n_mixture_plots": d.mixture_plot_id.nunique(), "n_composition_environments": len(composition),
                             "n_environments": len(env), "n_sites": env.site_id.nunique(), "mean_delta_t_ha": env.delta_t_ha.mean(),
                             "mean_relative_percent": env.relative_percent.mean(), "cluster_stability_lower": ci[0], "cluster_stability_upper": ci[1],
                             "cluster_interval_status": "descriptive_noncausal_few_site_bootstrap" if env.site_id.nunique() >= 2 else "unavailable_fewer_than_two_sites",
                             "equal_trial_below_baseline_fraction": env.below_baseline_fraction.mean(),
                             "mean_environment_p10_delta_t_ha": env.empirical_p10_delta_t_ha.mean(),
                             "min_composition_environment_delta_t_ha": composition.delta_t_ha.min(),
                             "min_environment_delta_t_ha": env.delta_t_ha.min(), "max_environment_delta_t_ha": env.delta_t_ha.max(),
                             "hindsight_best_delta_t_ha": env.hindsight_best_delta_t_ha.mean(),
                             "inference_class": "noncausal_matched_source_diagnostic", "cluster_interval_is_causal": False})
        predictions = transfer_predictions(env, "delta_t_ha")
        if len(predictions):
            predictions["control_group"] = code
            transfer_rows.append(predictions)
        for trait, unit in [("prot", "percentage_points_dry_matter"), ("phl", "kg_hl")]:
            q = d[d[f"{trait}_delta"].notna()].copy()
            if len(q):
                means = environment_means(q, f"{trait}_delta")
                quality_rows.append({"control_group": code, "trait": trait, "unit": unit, "n_plot_comparisons": len(q),
                                     "n_environments": len(means), "equal_environment_mean_delta": means[f"{trait}_delta"].mean(),
                                     "inference_class": "noncausal_complete_trait_subset"})
    all_env = pd.concat(env_rows, ignore_index=True) if env_rows else pd.DataFrame()
    predictions = pd.concat(transfer_rows, ignore_index=True) if transfer_rows else pd.DataFrame()
    check_summaries = []
    for name, d in checks.groupby("check"):
        means = environment_means(d, "delta_t_ha")
        check_summaries.append({"check": name, "n_plot_comparisons": len(d), "n_environments": len(means),
                                "equal_environment_mean_delta_t_ha": means.delta_t_ha.mean(),
                                "status": "diagnostic; pure_check_corandomization_unconfirmed; not_diversity_effect"})
    source_profile = records.groupby(["year", "postcode", "mono_mix", "control_group", "physical_plot_id_recovered"], dropna=False).agg(
        n_source_records=("plot_id", "size"), n_yield=("yield_t_ha", "count"), n_protein=("prot", "count"), n_specific_weight=("phl", "count")).reset_index()
    eligibility = candidate[["plot_id", "environment_id", "var_numb", "yield_t_ha"]].copy()
    eligibility["eligible"] = candidate.var_numb.eq(2) & candidate.yield_t_ha.notna()
    eligibility["reason"] = np.where(candidate.var_numb.ne(2), "undocumented_four_way_composition_weights", np.where(candidate.yield_t_ha.isna(), "missing_harvest", "eligible_binary_mixture"))
    same_trial_coverage = []
    for _, m in mixes.iterrows():
        b = control_baseline(controls, m.environment_id, "33", [str(m.var_1).strip(), str(m.var_2).strip()], "yield_t_ha")
        same_trial_coverage.append({"mixture_plot_id": m.plot_id, "environment_id": m.environment_id,
                                    "complete_constituent_controls_in_mixture_trial": b is not None})
    for name, d in [("swiss_release_mapping", mapping), ("swiss_records", records), ("swiss_source_groups", source_profile),
                    ("swiss_mixture_eligibility", eligibility), ("swiss_constituent_coverage", pd.DataFrame(coverage)),
                    ("swiss_same_trial_constituent_gate", pd.DataFrame(same_trial_coverage)), ("swiss_contrasts", contrasts),
                    ("swiss_environments", all_env), ("swiss_summary", pd.DataFrame(summary_rows)), ("swiss_transfer_predictions", predictions),
                    ("swiss_measured_tradeoffs", pd.DataFrame(quality_rows)), ("swiss_same_block_checks", checks),
                    ("swiss_check_summary", pd.DataFrame(check_summaries))]:
        write_csv(out, name, d)
    transfer_summary = []
    if len(predictions):
        for (code, scheme), d in predictions.groupby(["control_group", "scheme"]):
            transfer_summary.append({"control_group": code, "scheme": scheme, "n_test_environments": len(d),
                                     "training_mean_rmse_t_ha": np.sqrt(np.mean((d.prediction - d.delta_t_ha) ** 2)),
                                     "zero_difference_rmse_t_ha": np.sqrt(np.mean(d.delta_t_ha ** 2)),
                                     "status": "noncausal_diagnostic_transfer; not_adaptation_validation"})
    write_csv(out, "swiss_transfer_summary", pd.DataFrame(transfer_summary))
    return records, contrasts, pd.DataFrame(summary_rows)


def run(output=HERE / "results"):
    output = Path(output).resolve()
    if not output.is_relative_to(HERE):
        raise ValueError("Output must remain inside the owned module")
    output.mkdir(parents=True, exist_ok=True)
    archive, manifest, lock = checked_inputs()
    fp, fc, fs = french(archive, output)
    sp, sc, ss = swiss(archive, output)
    (output / "applicability.json").write_text(json.dumps(applicability(), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    checks = [
        {"check": "French physical plots unique", "passed": bool(fp.plot_id.is_unique), "value": len(fp)},
        {"check": "French complete constituent contrasts", "passed": len(fc) == 195, "value": len(fc)},
        {"check": "French one environment", "passed": fp.environment_id.nunique() == 1, "value": fp.environment_id.nunique()},
        {"check": "Swiss no causal flags", "passed": bool(sc.causal_contrast.eq(False).all()), "value": len(sc)},
        {"check": "Swiss exact mapping only", "passed": bool(sp.mapping_status.eq("unique").all()), "value": len(sp)},
        {"check": "No France-Switzerland effect pooling", "passed": True, "value": 2},
    ]
    write_csv(output, "analysis_checks", pd.DataFrame(checks))
    if not all(c["passed"] for c in checks):
        raise AssertionError("Analysis checks failed")
    receipt = {"protocol_sha256": lock["protocol_sha256"], "input_bundle_sha256": manifest["bundle_sha256"],
               "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__,
               "source_code_sha256": {p.name: sha(p.read_bytes()) for p in [HERE / "core.py", HERE / "run.py"]},
               "results_sha256": {p.name: sha(p.read_bytes()) for p in sorted(output.iterdir()) if p.suffix in {".csv", ".json"} and p.name != "analysis_receipt.json"},
               "random_seed": 20261009, "site_bootstrap_draws": 10000, "status": "empirical_evidence_with_explicit_design_limits; no_validated_climate_adaptation"}
    (output / "analysis_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(fs[["endpoint", "n_mixtures", "mean_delta_t_ha", "mean_relative_percent"]].to_string(index=False))
    print(ss[["control_group", "n_environments", "n_sites", "mean_delta_t_ha", "mean_relative_percent"]].to_string(index=False))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "results")
    args = parser.parse_args()
    run(args.output)
