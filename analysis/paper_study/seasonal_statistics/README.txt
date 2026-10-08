Frozen seasonal-model statistics

Working directory: repository root.
Environment: project .venv, Python 3.12; NumPy 2.5.3, pandas 3.0.6, SciPy 1.18.1, numba 0.68.0.

Commands:
.venv/bin/python -B analysis/paper_study/seasonal_statistics/analyze_frozen_records.py
.venv/bin/python -B analysis/paper_study/seasonal_statistics/bootstrap_calibration.py
.venv/bin/python -B analysis/paper_study/seasonal_statistics/verify_statistics.py

analyze_frozen_records.py reads archived BASF predictions and produces 10,000 paired coordinate-year bootstrap draws per final-severity subset, together with independently reconstructed onset-bound compatibility stratified by censoring. No model fitting occurs in that script.

bootstrap_calibration.py filters the archived source assessments to harvest years 2017 and 2018 before model input preparation. It calls the byte-identical v1 model.seasonal_septoria.calibrate.fit function for 100 fixed-seed coordinate-year resamples with complete field membership and a fixed 30-day latent period. frozen_source.py pins numerical imports to source_snapshot/ and verifies the snapshot hashes. Integer cluster multiplicities multiply source weights, preserving equal cluster/equal field/equal leaf/equal assessment weighting. The script checkpoints every fit in data/paper_study/seasonal_statistics/bootstrap_fits/. Existing checkpoints are reused only when draws and source SHA-256 records match. It also profiles fixed beta on an explicit grid with training-only reoptimization of alpha. Numba cache files are confined to the owned data folder; Python bytecode generation is disabled.

verify_statistics.py independently selects final numerical leaf endpoints, reconstructs all reported intervals and onset bounds, checks seed draws and complete membership, and simulates each fitted bootstrap parameter vector to reproduce its residual objective. It performs no parameter optimization. Independent verification checks source hashes and records six coordinate locations shared by calibration and forward-year development validation.

Primary statistical tables:
paired_final_severity_intervals.csv: point estimates and 95% percentile intervals in percentage points; paired differences are model minus frozen leaf-rank baseline.
onset_compatibility_by_censoring.csv: censoring-specific compatibility with nested equal weights, source counts, missing predictions and finite bound distances. Distances are not errors against known infection dates.
calibration_parameter_bootstrap.csv: one row per calibration-only parameter resample.
calibration_parameter_intervals.csv: conditional percentile parameter intervals and boundary frequencies.
secondary_beta_training_profile.csv: fixed-beta grid and reoptimized alpha, original weighted SSE and training RMSE; no likelihood threshold is defined.
physical_unit_inventory.csv: coordinate-year and coordinate-location counts and overlap.

Receipts:
frozen_prediction_statistics_receipt.json: hashes, prediction-bootstrap definitions and retained archive checks.
parameter_uncertainty_receipt.json: hashes, calibration-only parameter bootstrap definitions, parameter bounds and fixed assumptions.
independent_verification.json: 4,082 independently executed checks, no optimization, retained source checks and executable hashes.
retained_source_snapshot.json: retained numerical modules and dependencies, exact-match scope and live-code drift.
Statistical_methods_and_findings.txt: formal methods, numerical findings and interpretation limits.

The shared field_data.py gained stricter leaf-rank and metric/unit eligibility checks after the 100 bootstrap fits and initial independent verification. The live source hash therefore differs from the original execution hash. The six v1 model modules were subsequently preserved from the independent review snapshot, checked byte-for-byte against the hashes recorded during those original fits, and imports were pinned to that retained source. Other process-model dependencies were copied after execution; the manifest explicitly distinguishes their retrospective provenance. Initial execution receipts remain available in *_at_initial_execution.json. No recorded fit or score was repaired or replaced because of the source drift.

Data artifacts in data/paper_study/seasonal_statistics/ retain final predictions, all paired bootstrap draws, exact draw indices, cluster scores and memberships, all fit checkpoints, and all reconstructed onset rows. Their sourcekeys retain links to original BASF observations.

Neither statistics script reads Corteva disease outcomes. No source archive, harmonized partition, model code or manuscript is modified. Parameter uncertainty is conditional on the frozen latent-period selection and host/weather/observation assumptions. Prediction intervals condition on the frozen model and baseline and do not include parameter-estimation uncertainty.
