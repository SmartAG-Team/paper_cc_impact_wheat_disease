"""Independent source reconstruction, contracts, and byte-identical replay."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def near(a, b):
    assert math.isclose(float(a), float(b), rel_tol=2e-10, abs_tol=2e-10), (a, b)


def main():
    test = subprocess.run([sys.executable, "-B", str(HERE / "test_adaptation.py")], capture_output=True, text=True)
    print(test.stdout + test.stderr, end="")
    assert test.returncode == 0, "Contract suite failed"
    out = HERE / "results"
    receipt = json.loads((out / "analysis_receipt.json").read_text())
    checks = []

    def check(name, assertion, detail):
        assert assertion, name
        checks.append({"check": name, "passed": True, "detail": detail})

    for file, digest in receipt["source_code_sha256"].items():
        check("code_hash_" + file, sha(HERE / file) == digest, digest)
    for file, digest in receipt["results_sha256"].items():
        check("artifact_hash_" + file, sha(out / file) == digest, digest)
    with zipfile.ZipFile(HERE / "public_inputs.zip") as z:
        raw = list(csv.DictReader(io.StringIO(z.read("sources/montazeaud2022/raw_yield_variables.csv").decode())))
        by_grid = {}
        for r in raw:
            by_grid.setdefault((int(r["row"]), int(r["column"])), []).append(r)
        plots = pd.read_csv(out / "french_plots.csv").set_index("plot_id")
        contrasts = pd.read_csv(out / "french_contrasts.csv")
        manual_deltas, manual_percentages = [], []
        for _, c in contrasts.iterrows():
            values = []
            for pid in [c.plot_id, c.control_1_id, c.control_2_id]:
                p = plots.loc[pid]
                source = by_grid[(int(p.grid_row), int(p.grid_column))]
                assert len(source) == 4
                values.append(sum(float(r["GY"]) for r in source) / 4 * .01)
                near(values[-1], p.yield_t_ha)
            baseline = (values[1] + values[2]) / 2
            difference = values[0] - baseline
            percentage = 100 * difference / baseline
            near(baseline, c.baseline_t_ha)
            near(difference, c.delta_t_ha)
            near(percentage, c.relative_percent)
            near(values[0] - max(values[1:]), c.hindsight_best_delta_t_ha)
            manual_deltas.append(difference)
            manual_percentages.append(percentage)
        summary = pd.read_csv(out / "french_summary.csv").set_index("endpoint").loc["raw_primary"]
        near(sum(manual_deltas) / len(manual_deltas), summary.mean_delta_t_ha)
        near(sum(manual_percentages) / len(manual_percentages), summary.mean_relative_percent)
        check("French_direct_raw_reconstruction", len(manual_deltas) == 195, "All raw plot means, constituent baselines, differences and percentages reconstructed using CSV rows without pipeline estimators")
        check("French_no_pseudoreplicated_row_count", len(plots) == 381 and len(contrasts) == 195, "381 author-retained physical plots; 195 complete comparisons, not 4x counts")
        check("French_no_disease_conditioning", contrasts.stb_ordinal.isna().any(), f"{contrasts.stb_ordinal.isna().sum()} contrasts retained without a mixture STB score")
        dependency = pd.read_csv(out / "french_dependency_groups.csv")
        check("French_dependency_dominance_retained", dependency.n_mixtures.max() == 189 and len(dependency) == 4, "189/195 comparisons in largest shared-control component")
        check("French_CI_not_manufactured", pd.read_csv(out / "french_summary.csv")[["ci_lower", "ci_upper"]].isna().all().all(), "Intervals unavailable; influence ranges are separately labeled")
        influence = pd.read_csv(out / "french_influence.csv")
        for _, r in influence[influence.deletion_group.eq("genotype")].iterrows():
            kept = contrasts[contrasts.constituent_1.ne(r.omitted) & contrasts.constituent_2.ne(r.omitted)]
            assert len(kept) == r.n_remaining
            near(kept.delta_t_ha.mean(), r.mean_delta_t_ha)
        check("French_genotype_deletion_includes_all_shared_controls", True, "Every comparison sharing each omitted constituent removed")

        current = pd.read_csv(io.BytesIO(z.read("sources/swiss-mixtures/blendit_full_pub.csv")))
        previous = pd.read_csv(io.BytesIO(z.read("sources/swiss-mixtures-v1/blendit_full.csv")))
        records = pd.read_csv(out / "swiss_records.csv", dtype={"plot_id": str, "control_group": str, "current_source_lines": str, "previous_source_lines": str})
        for _, r in records.iterrows():
            source = current.iloc[int(r.current_source_lines.split("|")[0]) - 2]
            old = previous.iloc[int(r.previous_source_lines.split("|")[0]) - 2]
            for key in current.columns:
                left, right = source[key], old[key]
                if pd.isna(left) and pd.isna(right):
                    continue
                assert left == right, (r.plot_id, key, left, right)
            if pd.notna(source.yield_dtha):
                near(float(source.yield_dtha) / 10, r.yield_t_ha)
        check("Swiss_exact_source_line_reconstruction", len(records) == 3780, "All unique current payloads compared with exact prior-release rows; source yield units independently converted")
        controls = records.set_index("plot_id")
        assert controls.index.is_unique
        swiss = pd.read_csv(out / "swiss_contrasts.csv", dtype={"control_group": str})
        for _, r in swiss.iterrows():
            pure_means = []
            for j in (1, 2):
                ids = r[f"control_{j}_ids"].split("|")
                p = controls.loc[ids]
                assert p.environment_id.eq(r.environment_id).all()
                assert p.control_group.eq(r.control_group).all()
                assert p.cultivar.eq(r[f"constituent_{j}"]).all()
                assert p.mono_mix.eq("mono").all()
                pure_means.append(float(sum(p.yield_t_ha) / len(p)))
            near(sum(pure_means) / 2, r.baseline_t_ha)
            near(r.yield_t_ha - sum(pure_means) / 2, r.delta_t_ha)
        check("Swiss_controls_never_cross_environment_or_trial_code", len(swiss) == 147, "All 147 diagnostic plot-control-group comparisons directly recomputed from named references")
        check("Swiss_constituent_gate_fails_honestly", not pd.read_csv(out / "swiss_same_trial_constituent_gate.csv").complete_constituent_controls_in_mixture_trial.any(), "0/637 binary harvest plots have complete same-trial constituent baselines")
        check("Swiss_no_causal_flags", not swiss.causal_contrast.any(), "All external comparisons noncausal")
        env = pd.read_csv(out / "swiss_environments.csv", dtype={"control_group": str})
        for _, r in env.iterrows():
            d = swiss[swiss.environment_id.eq(r.environment_id) & swiss.control_group.eq(r.control_group)]
            per_composition = [sum(q.delta_t_ha) / len(q) for _, q in d.groupby("composition_id")]
            near(sum(per_composition) / len(per_composition), r.delta_t_ha)
        summaries = pd.read_csv(out / "swiss_summary.csv", dtype={"control_group": str})
        for _, r in summaries.iterrows():
            d = env[env.control_group.eq(r.control_group)]
            near(sum(d.delta_t_ha) / len(d), r.mean_delta_t_ha)
        check("Swiss_equal_trial_and_composition_weights", True, "Replicates -> composition -> environment -> equal environment mean, independently reconstructed")
        predictions = pd.read_csv(out / "swiss_transfer_predictions.csv", dtype={"control_group": str, "site_id": str})
        env.site_id = env.site_id.astype(str)
        for _, r in predictions.iterrows():
            train = env[env.control_group.eq(r.control_group) & env.environment_id.isin(r.training_environments.split("|"))]
            assert r.environment_id not in set(train.environment_id)
            if r.scheme == "leave_site_out":
                assert not train.site_id.eq(r.site_id).any()
            else:
                assert train.year.lt(r.year).all()
            near(train.delta_t_ha.mean(), r.prediction)
        check("Swiss_transfer_group_leakage_absent", True, "All training memberships and training-only predictions checked")

    app = json.loads((out / "applicability.json").read_text())
    for source in ["france", "swiss"]:
        for key in ["validated_climate_adaptation", "stb_mediated_grain_loss", "european_production_benefit", "independent_environment_validation"]:
            assert app[source][key] is False
    check("Unsupported_claims_machine_readable", True, "No validated climate adaptation, STB grain-loss mediation or European production claim")

    replay = HERE / ".verification_replay"
    assert not replay.exists(), "Replay directory already exists; inspect it before retrying"
    try:
        rerun = subprocess.run([sys.executable, "-B", str(HERE / "run.py"), "--output", str(replay)], capture_output=True, text=True)
        assert rerun.returncode == 0, rerun.stderr
        if rerun.stderr:
            raise AssertionError("Replay emitted warnings: " + rerun.stderr)
        for file in receipt["results_sha256"]:
            assert sha(replay / file) == sha(out / file), f"Nondeterministic output: {file}"
        check("Byte_identical_replay", True, f"{len(receipt['results_sha256'])} output tables/JSON files reproduce without external data or prior-module imports")
    finally:
        if replay.exists():
            shutil.rmtree(replay)
    verification = {"status": "passed", "contract_test_output": test.stderr.strip(), "checks": checks,
                    "analysis_receipt_sha256": sha(out / "analysis_receipt.json"),
                    "verifier_sha256": sha(HERE / "verify.py"), "test_sha256": sha(HERE / "test_adaptation.py"),
                    "limits": "Computational verification does not validate exchangeability, adaptation, disease-mediated loss or environmental transfer"}
    (HERE / "verification_receipt.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(f"Verified {len(checks)} checks; contract tests and byte-identical replay passed.")


if __name__ == "__main__":
    main()
