"""Recompute manuscript analyses, model replays, figures and documents.

The default uses the retained continental census and runs the saved monitoring
models, daily field process, and registered climate sample. ``--full-climate``
replaces the census with 810 newly simulated annual jobs. ``--deep-process``
also reruns the sampled factorial and sowing experiments from daily forcing.
All generated files are written to a fresh reproduction directory.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any, Callable, Iterator, Sequence
from uuid import uuid4


ROOT = Path(__file__).resolve().parents[1]
MODELS = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
STAGES = (
    "inputs", "monitoring", "process", "climate", "attribution", "exposure",
    "climate-response", "adaptation", "yield", "endpoints", "figures",
    "documents", "audit", "validate",
)
DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "inputs": (),
    "monitoring": ("inputs",),
    "process": ("inputs",),
    "climate": ("inputs",),
    "attribution": ("inputs",),
    "exposure": ("climate",),
    "climate-response": ("climate",),
    "adaptation": ("attribution",),
    "yield": ("inputs",),
    "endpoints": ("inputs",),
    "figures": ("monitoring", "process", "exposure", "climate-response", "adaptation", "yield"),
    "documents": ("figures",),
    "audit": ("figures",),
    "validate": ("documents", "audit", "endpoints"),
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if hasattr(value, "item"):
        return _json_value(value.item())
    if isinstance(value, float) and not __import__("math").isfinite(value):
        return None
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(_json_value(value), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def module(primary: str, legacy: str | None = None) -> Any:
    """Use the consolidated implementation; legacy names ease migration."""
    try:
        return importlib.import_module(primary)
    except ModuleNotFoundError as exc:
        if not legacy or not (exc.name == primary or primary.startswith(str(exc.name) + ".")):
            raise
        return importlib.import_module(legacy)


def scripts(name: str) -> Any:
    return module("paper.scripts." + name, "analysis.gcb_preparation." + name)


def monitoring(name: str) -> Any:
    return module("models.monitoring." + name, "analysis.disease_improvement." + name)


def climate(name: str) -> Any:
    return module("models.climate." + name, "analysis.final_climate." + name)


def food(name: str) -> Any:
    return module("paper.analyses.food_security." + name, "analysis.food_security_extension." + name)


@contextmanager
def configured(target: Any, **values: Any) -> Iterator[None]:
    """Restore module configuration after a stage, including on failure."""
    absent = object()
    original = {name: getattr(target, name, absent) for name in values}
    for name, value in values.items():
        setattr(target, name, value)
    try:
        yield
    finally:
        for name, value in original.items():
            if value is absent:
                delattr(target, name)
            else:
                setattr(target, name, value)


class Tee:
    def __init__(self, *streams: Any):
        self.streams = streams

    def write(self, value: str) -> int:
        for stream in self.streams:
            stream.write(value)
            stream.flush()
        return len(value)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def completion_flags(results):
    """Report executed scientific work rather than requested command options."""
    return dict(
        continental_census_recomputed=bool(results.get('climate',{}).get('full_census_recomputed',False)),
        sampled_process_experiments_rerun=bool(results.get('attribution',{}).get('scientific_simulations_rerun',False) and results.get('adaptation',{}).get('scientific_simulations_rerun',False)),
    )


def selected_stages(requested: Sequence[str] | None) -> list[str]:
    wanted: set[str] = set()

    def include(name: str) -> None:
        if name in wanted:
            return
        for dependency in DEPENDENCIES[name]:
            include(dependency)
        wanted.add(name)

    for name in requested or ["validate"]:
        include(name)
    return [name for name in STAGES if name in wanted]


@dataclass
class Workflow:
    root: Path
    source_package: Path
    output: Path
    full_climate: bool = False
    deep_process: bool = False
    verify_forcing_hashes: bool = False
    bootstrap_replicates: int = 500
    chunk_size: int = 256
    results: dict[str, Any] = field(default_factory=dict)
    source_hashes: dict[str, str] = field(default_factory=dict)

    @property
    def package(self) -> Path:
        return self.output / "package"

    @property
    def analyses(self) -> Path:
        return self.output / "analyses"

    @property
    def census(self) -> Path:
        if self.full_climate:
            return self.output / "climate" / "summary"
        return self.root / "data/final_study/evidence/full_grid_climate/results"

    def plan(self, requested: Sequence[str] | None = None) -> dict[str, Any]:
        return dict(
            schema_version=1, source_root=self.root, source_package=self.source_package,
            output=self.output, generated_package=self.package,
            stages=selected_stages(requested), dependencies=DEPENDENCIES,
            continental_census="new 810-job simulation" if self.full_climate else "retained paired continental census",
            sampled_experiments="new daily-forcing simulations" if self.deep_process else "recomputed from retained corner/season-option records",
            actual_model_replays=(["20 monitoring folds"] if "monitoring" in selected_stages(requested) else []) + (["115 F1 records", "55 onset histories"] if "process" in selected_stages(requested) else []) + (["62-cell climate sample"] if "climate" in selected_stages(requested) else []),
            climate_analysis_source=self.census, bootstrap_replicates=self.bootstrap_replicates,
            original_inputs_modified=False,
        )

    def remember(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        name = str(path.relative_to(self.root)) if path.is_relative_to(self.root) else str(path)
        value = digest(path)
        self.source_hashes[name] = value
        return dict(path=name, bytes=path.stat().st_size, sha256=value)

    def _check_original_inputs(self) -> None:
        for name, expected in self.source_hashes.items():
            path = Path(name)
            if not path.is_absolute():
                path = self.root / path
            if digest(path) != expected:
                raise ValueError("An original input changed during reproduction: " + name)

    def inputs(self) -> dict[str, Any]:
        required = [
            self.source_package / "Manuscript.md",
            self.root / "data/final_climate/scientific_manifest.json",
            self.root / "data/final_monitoring/manifest.json",
            self.root / "data/final_study/path_manifest.json",
            self.root / "data/repository_layout.json",
        ]
        for path in required:
            self.remember(path)
        source_data = self.source_package / "data"
        if not source_data.is_dir():
            raise FileNotFoundError("Missing manuscript data: " + str(source_data))
        # Materialize linked package data into the new run so no renderer or
        # figure builder can write through the canonical paper/data symlink.
        shutil.copytree(self.source_package, self.package, symlinks=False,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "exports"))
        self.analyses.mkdir()
        result = climate("run").check_inputs(
            self.root, verify_forcing_hashes=self.verify_forcing_hashes)
        write_json(self.output / "checks/climate_inputs.json", result)
        return dict(status="passed", package_inputs_copied=True,
                    forcing_content_hashes_reread=self.verify_forcing_hashes,
                    climate_input_check=result)

    def replay_monitoring(self) -> dict[str, Any]:
        import joblib
        import numpy as np
        import pandas as pd

        common = monitoring("innovation_common")
        baseline_module = monitoring("discrepancy_experiment")
        correction_module = monitoring("discrepancy_filter")
        obs, arrays, history, original, splits, position = common.load_cohort()
        source = self.root / "data/final_monitoring/folds"
        out = self.analyses / "monitoring"
        out.mkdir()
        frames = {
            label: pd.read_csv(source / name, float_precision="round_trip")
            for label, name in [("temporal", "assessment_predictions.csv"),
                                ("spatial_50km", "spatial_predictions.csv")]
        }
        outputs: dict[str, list[Any]] = {key: [] for key in frames}
        replayed, maximum = [], 0.0
        for split in splits:
            path = source / (split["fold"] + "_model.joblib")
            self.remember(path)
            fitted = joblib.load(path)
            if fitted["train_ids"] != split["train"] or fitted["test_ids"] != split["test"]:
                raise ValueError("Changed monitoring fold membership: " + split["fold"])
            qi = position.loc[split["test"]].to_numpy()
            ci = np.flatnonzero(obs.field_id.isin(obs.iloc[qi].field_id.unique()))
            within = pd.Series(np.arange(len(ci)), index=ci).loc[qi].to_numpy()
            baseline = baseline_module.forecast_baseline(fitted["base"], arrays, history, obs, ci)
            prediction = correction_module.predict_discrepancy(
                fitted["corrections"]["logit"]["parameters"], obs.iloc[ci], baseline, "logit")[within]
            label = "temporal" if split["fold"].startswith("temporal") else "spatial_50km"
            recorded = frames[label].set_index("global_target_index").loc[split["test"]]
            np.testing.assert_allclose(prediction, recorded.discrepancy_logit, rtol=1e-14, atol=1e-12)
            difference = float(np.max(np.abs(prediction - recorded.discrepancy_logit.to_numpy())))
            maximum = max(maximum, difference)
            fresh = recorded.reset_index().copy()
            fresh["prediction_percent"] = prediction
            outputs[label].append(fresh)
            replayed.append(dict(fold=split["fold"], records=len(fresh), maximum_absolute_difference=difference))
            write_json(out / "progress.json", dict(status="running", folds=replayed))
            print("Replayed monitoring fold", split["fold"], flush=True)
        for label, parts in outputs.items():
            frame = pd.concat(parts, ignore_index=True)
            if len(frame) != 2017 or frame.global_target_index.nunique() != 2017:
                raise ValueError("Monitoring replay changed the complete evaluation population")
            frame.to_csv(out / (label + "_predictions.csv"), index=False)
        # The manuscript reference table contains the display metadata and exact
        # original hierarchy weights. Every forecast above is independently
        # replayed before these verified records enter newly calculated figures.
        reference = self.root / "data/final_monitoring/reference"
        shutil.copytree(reference, out / "reference")
        published = pd.read_csv(reference / "data/fig1/predictions.csv", float_precision="round_trip")
        for label in outputs:
            fresh = pd.concat(outputs[label]).set_index("global_target_index")
            expected = published[published.evaluation.eq(label)].set_index("global_target_index")
            np.testing.assert_allclose(fresh.loc[expected.index, "prediction_percent"],
                                       expected.prediction_percent, rtol=1e-14, atol=1e-12)
        result = dict(status="passed", saved_models_replayed=len(replayed),
                      records_per_design=2017, maximum_absolute_prediction_difference=maximum,
                      models_fitted=False, information_setting="strictly earlier own-field monitoring",
                      model_implementation=str(Path(baseline_module.__file__).relative_to(self.root)))
        if len(replayed) != 20:
            raise ValueError("Expected all twenty saved monitoring models")
        write_json(out / "checks.json", result)
        return result

    def replay_process(self) -> dict[str, Any]:
        prefix = monitoring("climate_prefix")
        if (self.root / "models").is_dir() and not Path(prefix.__file__).resolve().is_relative_to(self.root / "models"):
            raise ValueError("Process replay did not load the consolidated models implementation")
        # Preloading the canonical module also prevents legacy portable scripts
        # from importing their bundled snapshots in preference to live models.
        sys.modules["analysis.disease_improvement.climate_prefix"] = prefix
        results = {}
        for name, folder, destination in [
            ("replay_f1", "f1", None),
            ("replay_onset", "onset", "onset_evaluation"),
            ("replay_onset_cohorts", "onset_cohorts", "onset_cohorts"),
        ]:
            implementation = scripts(name)
            out = self.analyses / folder
            implementation.replay(self.package, out)
            result = json.loads((out / "checks.json").read_text())
            if result.get("status") != "passed":
                raise ValueError("Process replay failed: " + name)
            results[name] = result
            target_code = self.package / "code" / (name + ".py")
            target_code.parent.mkdir(exist_ok=True)
            shutil.copy2(implementation.__file__, target_code)
            if destination:
                shutil.copytree(out, self.package / "data" / destination, dirs_exist_ok=True)
            elif name == "replay_f1":
                shutil.copy2(out / "checks.json", self.package / "verification/portable_f1_replay.json")
                shutil.copy2(out / "scores.csv", self.package / "verification/portable_f1_scores.csv")
        return dict(status="passed", implementation=Path(prefix.__file__),
                    records=results["replay_f1"]["records"],
                    field_seasons=results["replay_onset_cohorts"]["field_seasons"],
                    bracketed_evaluation_histories=results["replay_onset"]["records"],
                    fitting_performed=False)

    def replay_climate(self) -> dict[str, Any]:
        runtime = climate("run")
        result = runtime.replay_sample(self.root)
        write_json(self.output / "checks/climate_sample.json", result)
        if self.full_climate:
            manifest = runtime.read_manifest(self.root)
            annual = self.output / "climate"
            jobs = 0
            for model in manifest["models"]:
                for scenario in manifest["scenarios"]:
                    for year in manifest["harvest_years"]:
                        runtime.run_year(model, scenario, year, annual, root=self.root, chunk_size=self.chunk_size)
                        jobs += 1
                        write_json(self.output / "checks/climate_progress.json",
                                   dict(completed_annual_jobs=jobs, expected_annual_jobs=810,
                                        last_job=dict(model=model, scenario=scenario, year=year)))
                        print("Climate annual jobs", jobs, "/ 810", flush=True)
            summary = climate("summary").summarize(annual, self.census, root=self.root)
            if jobs != 810:
                raise ValueError("Full continental rerun did not execute all 810 annual jobs")
            result = dict(sample_replay=result, annual_jobs=jobs, census=self.census,
                          full_census_recomputed=True, aggregation_status=summary["status"])
        else:
            result = dict(sample_replay=result, census=self.census, full_census_recomputed=False)
        return result

    def attribution(self) -> dict[str, Any]:
        import pandas as pd

        implementation = food("attribution")
        out = self.analyses / "attribution"
        if self.deep_process:
            result = implementation.run(out, root=self.root, models=MODELS, reference_years=list(range(1991, 2021)))
            return dict(status=result["status"], mode="daily forcing corner simulations",
                        scientific_simulations_rerun=True,pair_records=result["pair_records"], output=out)
        source = self.root / "data/final_analyses/attribution_replay/attribution_pairs.parquet"
        self.remember(source)
        pairs = pd.read_parquet(source).drop(columns=["common_across_settings"], errors="ignore")
        summary, pairs = implementation.summarize_pairs(pairs, expected_models=MODELS)
        out.mkdir()
        pairs.to_parquet(out / "attribution_pairs.parquet", index=False)
        summary.to_csv(out / "attribution_summary.csv", index=False)
        result = dict(status="passed", mode="recomputed from retained four-corner records",
                      scientific_simulations_rerun=False, pair_records=len(pairs), settings=8,
                      source=self.remember(source), output_hashes={
                          p.name: digest(p) for p in out.iterdir() if p.is_file()})
        write_json(out / "attribution_receipt.json", result)
        return result

    def exposure(self) -> dict[str, Any]:
        import pandas as pd

        implementation = food("exposure")
        source = self.census / "full_grid_paired_changes_by_gcm.parquet"
        changes = pd.read_parquet(source, filters=[("metric", "=", implementation.METRIC)])
        original = self.root / "data/final_study/evidence/food_security_exposure"
        production = pd.read_csv(original / "production_grid_input.csv", float_precision="round_trip")
        fractions = pd.read_csv(original / "country_fraction_weights.csv", float_precision="round_trip")
        for name in ["production_grid_input.csv", "country_fraction_weights.csv"]:
            self.remember(original / name)
        countries = implementation.country_domains(production, fractions)
        summary = pd.concat([
            implementation.summarize_exposure(changes, production, min_pairs=n, extra_domains=countries)
            for n in (1, 27)
        ], ignore_index=True)
        out = self.analyses / "exposure"
        out.mkdir()
        summary.to_csv(out / "production_exposure.csv", index=False)
        primary = summary.loc[summary.domain_type.eq("Europe") &
                              summary.members.eq("|".join(MODELS)) &
                              summary.minimum_paired_seasons.eq(1)]
        primary.to_csv(out / "europe_thresholds.csv", index=False)
        economic = implementation.write_economic_sensitivity(
            out / "europe_thresholds.csv", out / "economic_sensitivity")
        result = dict(status="passed", recalculated_rows=len(summary), census=source,
                      fixed_production=True, grain_loss_estimated=False,
                      conditional_economic_sensitivity=economic,
                      maximum_mass_balance_error_tonnes=float(summary.mass_balance_error_tonnes.abs().max()))
        write_json(out / "checks.json", result)
        return result

    def climate_response(self) -> dict[str, Any]:
        implementation = scripts("climate_response_insights")
        out = self.analyses / "climate_response"
        with configured(implementation, SOURCE=self.census):
            result = implementation.main(out, self.bootstrap_replicates)
        write_json(out/"run_receipt.json",dict(status="passed",census=self.census,analysis="Recomputed from cell/member paired changes",source_audit=out/"analysis_audit.json",bootstrap_replicates=self.bootstrap_replicates))
        return dict(status="passed", output=out, census=self.census, findings=result)

    def adaptation(self) -> dict[str, Any]:
        out = self.analyses / "adaptation"
        if self.deep_process:
            implementation = scripts("adaptation_analysis")
            with configured(implementation, PRIOR=(self.analyses / "attribution").relative_to(self.root)):
                implementation.run(out, root=self.root)
        else:
            source = self.root / "data/final_analyses/adaptation"
            shutil.copytree(source, out)
            for name in ["adaptation_pairs.parquet", "registration.json", "receipt.json", "leave_gcm_out_region_policy.csv"]:
                self.remember(source / name)
            # Derive new support records without inheriting a completed-stage
            # receipt from the archived analysis.
            for path in out.glob("support_sensitivity_*"):
                if path.is_file():
                    path.unlink()
        result = scripts("adaptation_support_sensitivity").run(out)
        return dict(status="passed", scientific_simulations_rerun=self.deep_process,
                    mode="daily forcing sowing simulations" if self.deep_process else "recomputed from retained season-option records",
                    output=out, complete_season_draw_years=result["complete_season_common_draw_years"],
                    primary_policy=result["primary_complete_season_policy"])

    def yield_context(self) -> dict[str, Any]:
        implementation = scripts("yield_evidence_extension")
        out = self.analyses / "yield_context"
        source = self.root / "data/final_analyses/yield_context"
        shutil.copytree(source / "sources", out / "sources")
        pdf = source / "sources/arvalis_performance_2025_champagne_ardenne.pdf"
        original = self.remember(pdf)
        if original["sha256"] != implementation.EXPECTED_PDF_SHA256:
            raise ValueError("The source annual-yield chart changed")
        # Source-scope records are archived evidence. Only the annual chart
        # values and their descriptive analysis underlie the plotted figure;
        # unrelated historical trial-profile files are not runtime inputs.
        for name in ["source_metadata.json", "fig1_caption.txt"]:
            shutil.copy2(source / name, out / name)
        data = out / "data/fig1"
        data.mkdir(parents=True)
        shutil.copy2(source / "data/fig1/variables.json", data / "variables.json")
        shutil.copy2(source / "data/additional_published_endpoints.csv",
                     out / "data/additional_published_endpoints.csv")
        implementation.literature_inventory().to_csv(out / "data/evidence_screening_inventory.csv", index=False)
        annual = implementation.annual_yield_gap_data()
        annual.to_csv(data / "annual_yield_gap.csv", index=False)
        result = implementation.descriptive_summary(annual)
        write_json(out / "descriptive_summary.json", result)
        write_json(out / "verification.json", dict(
            status="passed", passed=True, annual_rows=len(annual),
            original_pdf_sha256_matches=True, source=original,
            annual_values_from_printed_labels=True, annual_summaries_recomputed=True,
            missing_trial_counts_retained=True, missing_sampling_variances_retained=True,
            new_meta_analysis_claim=False, normalized_HAD_to_yield_conversion=False,
            historical_unplotted_trial_profiles_recomputed=False,
        ))
        return dict(status="passed", output=out, summary=result, new_meta_analysis=False)

    def endpoints(self) -> dict[str, Any]:
        implementation = scripts("recompute_endpoint_scores")
        out = self.analyses / "endpoint_scores"
        implementation.recompute(self.package, out)
        result = json.loads((out / "checks.json").read_text())
        shutil.copy2(out / "checks.json", self.package / "verification/portable_endpoint_scores.json")
        shutil.copy2(implementation.__file__, self.package / "code/recompute_endpoint_scores.py")
        return result

    def figure_inputs(self) -> dict[str, Path]:
        return dict(
            monitoring=self.analyses / "monitoring/reference",
            chronology=self.root / "data/final_monitoring/cohort/training_observation_ledger.csv",
            attribution=self.analyses / "attribution", exposure=self.analyses / "exposure",
            climate_summary=self.census / "full_grid_ensemble_paired_changes.parquet",
            climate_members=self.census / "full_grid_paired_changes_by_gcm.parquet",
            climate_cells=self.root / "data/final_climate/cells.parquet",
            insights=self.analyses / "climate_response", adaptation=self.analyses / "adaptation",
            **{"yield": self.analyses / "yield_context"},
        )

    def figures(self) -> dict[str, Any]:
        implementation = scripts("figure_pipeline")
        parameters = inspect.signature(implementation.build).parameters
        if "output" not in parameters or "inputs" not in parameters:
            raise RuntimeError("The figure builder must accept build(output=None, inputs=None) before running this workflow")
        implementation.build(output=self.package, inputs=self.figure_inputs())
        manifest = json.loads((self.package / "figure_manifest.json").read_text())
        records = manifest["figures"]
        if len(records) != 14:
            raise ValueError("The full manuscript requires fourteen figure outputs")
        write_json(self.output / "figure_inputs.json", self.figure_inputs())
        return dict(status="passed", figures=[row["name"] for row in records], output=self.package / "figures")

    def documents(self) -> dict[str, Any]:
        implementation = scripts("documents")
        with configured(implementation, OUT=self.package):
            implementation.build()
        profile = json.loads((self.package / "document_profiles.json").read_text())
        return dict(status="passed", documents=profile["documents"])

    def audit(self) -> dict[str, Any]:
        implementation = scripts("figure_scientific_audit")
        findings = scripts("findings_audit")
        with configured(implementation, OUT=self.package), configured(findings, CENSUS=self.census):
            result = implementation.run()
        return dict(status=result["status"], figures=list(result["figures"]))

    def validate(self) -> dict[str, Any]:
        implementation = scripts("validate")
        with configured(implementation, PACKAGE=self.package):
            result = implementation.validate(write_manifest=True)
        self._check_original_inputs()
        return result

    def run(self, requested: Sequence[str] | None = None) -> dict[str, Any]:
        stages = selected_stages(requested)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "logs").mkdir()
        write_json(self.output / "plan.json", self.plan(requested))
        state: dict[str, Any] = dict(status="running", started_utc=utc_now(), stages=[], current_stage=None)
        functions: dict[str, Callable[[], Any]] = {
            "inputs": self.inputs, "monitoring": self.replay_monitoring,
            "process": self.replay_process, "climate": self.replay_climate,
            "attribution": self.attribution, "exposure": self.exposure,
            "climate-response": self.climate_response, "adaptation": self.adaptation,
            "yield": self.yield_context, "endpoints": self.endpoints,
            "figures": self.figures, "documents": self.documents,
            "audit": self.audit, "validate": self.validate,
        }
        try:
            for name in stages:
                state["current_stage"] = name
                write_json(self.output / "progress.json", state)
                print("Starting", name, flush=True)
                started = time.monotonic()
                with (self.output / "logs" / (name + ".log")).open("w") as log:
                    with redirect_stdout(Tee(sys.stdout, log)), redirect_stderr(Tee(sys.stderr, log)):
                        result = functions[name]()
                self.results[name] = result
                record = dict(stage=name, status="passed", elapsed_seconds=time.monotonic() - started, result=result)
                state["stages"].append(record)
                write_json(self.output / "stages" / (name + ".json"), record)
                write_json(self.output / "progress.json", state)
                print("Completed", name, flush=True)
            self._check_original_inputs()
            state.update(status="passed", completed_utc=utc_now(), current_stage=None,
                         original_input_hashes=self.source_hashes,
                         package_complete="validate" in self.results,
                         **completion_flags(self.results))
            write_json(self.output / "progress.json", state)
            write_json(self.output / "receipt.json", state)
            return state
        except BaseException as exc:
            state.update(status="failed", failed_utc=utc_now(), error_type=type(exc).__name__, error=str(exc))
            write_json(self.output / "progress.json", state)
            write_json(self.output / "failure.json", dict(state, traceback=traceback.format_exc()))
            raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Fresh directory under data/reproductions/; default is timestamped")
    parser.add_argument("--stage", nargs="+", choices=STAGES, help="Run selected stages and all their prerequisites; default completes validation")
    parser.add_argument("--full-climate", action="store_true", help="Rerun all 810 continental annual jobs and use their new summaries downstream")
    parser.add_argument("--deep-process", action="store_true", help="Rerun sampled factorial and sowing experiments from daily forcing")
    parser.add_argument("--verify-forcing-hashes", action="store_true", help="Reread all 687 forcing contents in the input stage")
    parser.add_argument("--bootstrap-replicates", type=int, default=500, help="Climate-response diagnostic spatial bootstrap replicates")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true", help="Print the complete stage/input/output plan without creating files")
    args = parser.parse_args(argv)
    if min(args.bootstrap_replicates, args.chunk_size, args.threads) < 1:
        parser.error("Bootstrap replicates, chunk size and threads must be positive")
    default = ROOT / "data/reproductions" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid4().hex[:6])
    args.output = (args.output or default).resolve()
    if not args.output.is_relative_to(ROOT / "data/reproductions"):
        parser.error("Figure source provenance requires a repository-local fresh output; choose data/reproductions/<run>")
    if args.output.exists():
        parser.error("The output already exists; use a fresh reproduction directory")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    source = ROOT / "paper"
    if not (source / "Manuscript.md").is_file():
        source = ROOT / "publication/gcb_20261023"
    workflow = Workflow(ROOT, source, args.output, full_climate=args.full_climate,
                        deep_process=args.deep_process, verify_forcing_hashes=args.verify_forcing_hashes,
                        bootstrap_replicates=args.bootstrap_replicates, chunk_size=args.chunk_size)
    if args.dry_run:
        result = workflow.plan(args.stage)
        print(json.dumps(_json_value(result), ensure_ascii=False, indent=2))
        return result
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[name] = str(args.threads)
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=args.threads):
        result = workflow.run(args.stage)
    print(json.dumps(dict(status=result["status"], output=str(workflow.output),
                          generated_package=str(workflow.package), package_complete=result["package_complete"])))
    return result
