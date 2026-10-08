# Retrospective training-selection reproducibility

A retrospective replay reproduced all 117 archived hyperparameter-candidate score rows and all 13 selected winners for the v2 crop-disease forecast calibration. The execution comprised 351 fitted inner models: 13 outer folds, nine latent/infectious-duration combinations and three inner location folds. Each fit used `primary_secondary_hidden`, three optimizer starts and seed 20261004. The code was imported from the current `model.primary_secondary` modules after their four hashes matched the archived source snapshot exactly.

The largest RMSE or MAE difference from the original table was 3.55 × 10⁻¹⁵ percentage points; the largest bias difference was 4.44 × 10⁻¹⁶ percentage points. The prespecified numerical tolerance was 10⁻⁸ percentage points plus 10⁻¹⁰ times the absolute original metric. Coordinate-year, series and target counts matched exactly. Selected nominal latent durations at 18°C and calendar infectious durations matched `fits.json` exactly in every outer fold. All 53 watched original source and result files retained their SHA256 hashes.

The replay used `prepare_basf`, `make_folds`, `inner_folds`, `fit_mechanism`, `prediction_frame` and `score` from the archived v2 implementation. Inner calibration and selection used only outer-training episodes. All years and leaf series at each location remained together within each inner split. Inner training and test locations were disjoint, and no outer-test episode entered an inner fit. The nominal latent-duration grid was 10, 20 and 30 days at 18°C; the infectious-duration grid was 14, 21 and 28 calendar days. The same RMSE ranking and duration tie-break ordering reproduced `select_training_hyperparameters`. At three starts, the optimizer uses the three fixed initial vectors; the seed generates no additional random starts.

The trace contains 2,940 episode-membership records and 10,197 target-membership records. Shared membership tables identify the 39 inner splits; all nine candidates in an outer fold reference the same three splits. Conditioning observations remain marked and are excluded from held-out scoring. The 117 compressed prediction archives contain 21,771 held-out prediction rows. An independent audit of the persisted archives reconstructs all metrics by averaging future observations within series, series within coordinate-year, and coordinate-years equally. It checks every prediction's source-row membership and observed value, every saved fit record and all selected winners. Reconstructed scores agree with both replay and original tables within the fixed tolerance.

The fitting trace is evidence from a new retrospective execution. It does not provide contemporaneous inner-fit records for the original calibration. Reproducibility of this development-era selection does not establish external field validation, mechanistic identifiability or European transfer performance.

The principal artifacts are:

- `replay_receipt.json`: comparison outcomes, numerical tolerances and source-preservation result.
- `replay_configuration.json`: exact versions, seed, grids, replay-code hash and all original-file hashes.
- `inner_membership.csv` and `inner_target_membership.csv`: complete train/test series and source-row membership for each inner split.
- `inner_fits.json` and `candidate_checkpoints/`: 351 fitted parameter records and optimizer diagnostics.
- `predictions/`: 117 compressed candidate-level prediction archives with inner-fold identity.
- `replay_candidate_scores.csv`, `candidate_comparison.csv` and `selected_winner_comparison.csv`: recomputed scores, metric differences, count checks and exact winner checks.
- `persisted_artifact_verification.json`: independent verification of the saved trace.

```sh
shasum -a 256 -c analysis/primary_secondary/calibration_v2_age/selection_replay/SHA256SUMS
uv run --no-project --python 3.12 --with numpy==2.5.3 --with pandas==3.0.6 --with pyarrow==25.0.1 --with scipy==1.18.1 --with numba==0.68.0 python -B analysis/primary_secondary/calibration_v2_age/selection_replay/replay_selection.py
uv run --no-project --python 3.12 --with numpy==2.5.3 --with pandas==3.0.6 python -B analysis/primary_secondary/calibration_v2_age/selection_replay/verify_archived_replay.py
```

The checksum manifest identifies the delivered snapshot. A repeated execution refreshes replay timestamps and result checksums. Matching checkpoints can be reused with `--resume`; resumption requires unchanged original hashes, software versions, seed, grids and replay-code hash. Python bytecode generation is disabled and compiled caches remain inside the replay directory. Original model, data and calibration archives remain unmodified.
