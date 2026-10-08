# Training-only process calibration

The frozen partition in `configs/station_split.json` contains 187 training, 1,222 validation, and 1,223 reserved test stations. All seven neural architectures and GDD, T-P, and T-P-V use this partition. Scoring uses the same 1,140 eligible test sites; the other 83 remain reserved and are excluded only from scoring. Validation and test sites never contribute to calibration or scaler fitting.

Sowing dates are management records known when sowing occurs. `calibrate.py` uses previous-day simulated accumulation to gate photoperiod and vernalization, with every gate and stage threshold fitted only on training stations. Transforming validation/test stations requires no observed stage labels. The [forecast-input contract](forecast_input_contract.md) defines availability, neural inputs, and interpretation limits.

The explicit sowing record export is a one-time migration from retained sowing events under the approved known-at-sowing assumption:

```sh
python process_model/sowing.py \
  --source dataset/data_cleaned.csv \
  --output dataset/sowing_records.csv
```

The existing export contains 35,977 records across 2,632 stations. Its sidecar records the original source hash and availability assumption. Subsequent calibration reads these records directly; it never derives cycle boundaries from outcome labels.

```sh
python process_model/calibrate.py \
  --data dataset/data_cleaned.csv \
  --sowing dataset/sowing_records.csv \
  --split configs/station_split.json \
  --output dataset/calibrated_causal
```

The local `dataset/calibrated_causal/` already contains all 11,862,724 regenerated daily rows. Output paths must be new for reruns. Original datasets, checkpoints and result tables remain historical artifacts. Superseded pre-P1-3 calibration metadata is retained under `configs/archive/pre_p1_3_calibration/`; obsolete generated daily CSVs were removed during cleanup.

Outputs include known-sowing records, the frozen split, schema-2 calibration parameters, stage thresholds, rebuilt daily features/simulations, all-reserved-site daily diagnostics, and a completed manifest with hashes. `gdd_median.py` and the CLI of `simulate.py` delegate to this workflow. Historical notebooks retain observed-stage gates and global calibration; they are reference evidence, not the supported regeneration path.

# Shared model runs

`configs/evaluation_sites.json` binds the approved 1,140 eligible and 83 excluded sites to the calibrated dataset identity. Fresh regeneration requires checking coverage against these exact site lists before updating the data-manifest binding. The current binding and dataset-local copy refer to `dataset/calibrated_causal/`; the previous binding is preserved in `configs/archive/evaluation_sites_before_p1_3.json`.

The current package already contains verified common-site process outputs. Regeneration uses the command below with a fresh output path. Prepare a new efficient neural plan with:

```sh
python process_model/evaluate_common_sites.py \
  --data dataset/calibrated_causal \
  --evaluation configs/evaluation_sites.json \
  --output dataset/calibrated_causal/common_evaluation

python experiments/run_efficient_models.py \
  --data dataset/calibrated_causal \
  --output runs/efficient_prepared \
  --gpu-type cuda --prepare-only
```

These outputs already exist locally; use new output paths for reruns. Common process scoring uses frozen parameters and 5,049,790 daily rows across exactly the approved 1,140 sites. The 83 exclusions are listed in `experiments/evaluation_exclusions.md`; all 1,223 reserved sites remain excluded from fitting. Coverage after P1-3 remains 4,780,697 valid test windows.

The efficient runner verifies artifact hashes, schema-2 input semantics, site membership, and 120-day history plus 30-day target windows. All seven architectures compare widths 128 and 256 using sampled training and a fixed validation panel, followed by full validation and selection before testing. Earlier prepared exhaustive plans are archived under `configs/archive/superseded_run_plans/`. The GPU handoff and launch commands are in [HOW_TO_RUN.md](../HOW_TO_RUN.md). No local production run is active.

Both neural `run.py` entry points default to `dataset/calibrated_causal/`. They require frozen site manifests and verified process artifacts. Observed historical stage labels are replaced with simulated T-P-V state in both encoder and decoder context. Training uses training/validation data only, with no per-epoch test-loss inspection. Checkpoints bind data, scaler, input semantics, reserved/evaluation-site identities, and weights. Standalone testing rejects changed inputs, splits, weights, and unverified historical checkpoints. Informer/Reformer evaluation uses a fixed seed for repeatable reloads.

# Interpretation limits

The corrected day-based process evaluation is available separately:

```sh
python process_model/evaluate_event_dates.py
python paper/scripts/plot_corrected_process_fig03.py
```

The existing `dataset/calibrated_causal/common_event_evaluation/` requires a new output path for another evaluation. Events use first daily threshold crossings matched by station, known sowing cycle, and BBCH; observed dates never select a predicted hit. The primary score excludes sowing and pools BBCH 10, 31, 51 and 85 over the same matched event keys for all three models. Missing predictions and predictions without observed events remain in the event table, with coverage counts in `event_metrics.json`. Incomplete daily cycles do not receive exact predicted event dates.

Current event-date MAE is **12.10329 days for GDD, 8.56818 for T-P, and 8.04495 for T-P-V**, on **42,115 of 43,177 observed physiological events (97.54%)**. The 1,140-site cohort is unchanged; 1,138 sites contribute matched events. Sites 2280 and 3658 each have an observed BBCH51 date but no matching threshold crossing within the available record. Their events remain in the missing-prediction counts. `AFM/figures/Fig_03.png` and `.pdf` show these corrected process results in days, with source data and provenance beside them. Known sowing dates and realized weather condition these event-date hindcasts; fixed-horizon neural event evaluation remains pending.

Process daily scores use known sowing and realized daily weather. They are weather-driven hindcasts, not process forecasts made with future weather unknown. Fixed-horizon process forecasts require issue-time weather forecasts or an explicitly specified weather scenario. Historical canopy preprocessing and observation availability remain unverified. Daily process MAE uses ordinal stage-code units; it is not event-date error in days and is not directly comparable with neural forecast-window metrics.

Auxiliary process-guided regressions remain blocked because their archived event tables and summaries retain global calibration. The subsequent P1-4/P1-5 pipeline update implements verified neural exports and independent issue-specific event evaluation; production prediction regeneration awaits execution on the target GPU. Archived manuscript scores are unchanged.

# Verification

```sh
python -m pytest -q process_model/tests experiments/tests dl_model/tests
python -m pytest -q AGC-Transformer/dl_model/tests
python -m pytest -q --import-mode=importlib
```

Tests cover held-out fitting invariance; label-free process transformation; sowing availability and future-input prefix invariance; neural encoder/decoder label isolation; checked station/date joins; frozen site membership; rejection of old calibration semantics; artifact/checkpoint provenance; and all seven neural CLI training/reload paths. Separate model-tree runs exercise both source copies. Default combined pytest collection still encounters duplicate module names. CUDA checks require available hardware.

# Verified neural forecast and event pipeline

P1-4/P1-5 repairs add explicit station/history/issue/target/lead mappings and completed, hashed forecast exports in both neural source trees. Fresh checkpoints bind training arguments and source identities. The neural event evaluator uses the approved midpoint thresholds, retains each issue independently, and reports stage/lead errors with missing and censored counts. A shared-model comparison requires identical windows and scores common matched issue/stage pairs.

The current entry point is `gpu/run.sh`, which performs package and CUDA checks, benchmarks all candidates, and executes the efficient comparison. The proposed v2 architecture remains design-only. Earlier exhaustive run plans are archived; corrected process figures remain unchanged. See [forecast and event protocol](../experiments/forecast_evaluation.md) for commands and scientific limits. Historical neural arrays and event tables remain unverified.
