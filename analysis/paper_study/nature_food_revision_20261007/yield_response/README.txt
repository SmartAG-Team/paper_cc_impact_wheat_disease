Nordic crop disease-control grain-response prototype

Target: signed R=1-Y_U/Y_C, against the measured disease-control arm with residual disease.
Units: grain dt/ha or hkg/ha at15% moisture; native*0.1=t/ha. Final disease differences are percentage points; genuine common-date trapezoidal integrals are percentage-days. Leaf rank is unspecified. Healthy-area duration is unavailable.

Primary populations include observed-zero STB: endpoint67/five environments and integral50/five environments. The original52 and38 populations require positive untreated final STB and are secondary domains. Four-disease complete cases provide131 contrasts/eight environments. Eligibility is outcome-blind; missing covariates never become zero. One midpoint co-pathogen assessment remains explicitly missing.

Whole-trial withholding keeps shared controls and all cultivars together. Each training and evaluation trial has total weight1. The baseline uses the training-only trial-balanced mean of original-scale R. Both linear and fixed exploratory exponential candidates fail against this baseline in the two primary domains. No transferable yield-response or absolute yield-loss model is demonstrated.

Declarations: predeclared_protocol.json; predeclared_amendment_01.json documents the zero-STB cohort correction before estimation; predeclared_damage_candidate.json documents the one fixed log-ratio/exponential structure after retaining linear failures and before its fit.
Inputs and selection: endpoint67_input.csv, integral50_input.csv, endpoint52_input.csv, integral38_input.csv, four_disease131_input.csv, selection_and_exclusions.csv, cohort_denominators.json, common_date_integral_knots.csv.
Model comparison: model_comparison_metrics.csv. Error columns with suffix_pp are percentage points of relative grain response. MSE skill compares equal-trial model error against calibration-only mean error.
Exact results: *_loto_predictions.csv, *_loto_memberships.csv, *_loto_fits.json, *_loto_metrics_by_trial.csv, *_temporal_predictions.csv and *_temporal_memberships.csv. Temporal Informer-to-Pondus predictions are the same leave-Pondus-out fold in the positive-STB domains, not extra independent validation.
Diagnostics: model_failure_diagnostics.json, *_domain_summary.json, selected_trial_precision.csv.
Additional co-pathogen provenance: additional_nordic_disease_provenance.csv and native_nontarget_value_checks_67111.csv preserve Swedish67111 Wheat leaf spot/Brunfläcksjuka (septoria) observations (all82 final both-arm values positive,2-35%). They are outside the four fixed predictors and did not trigger refitting or outcome-driven exclusions. native_STB_value_checks_67111.csv verifies the82 actual Svartpricksjuka/Wheat leaf blotch final values against the saved STB input. Native ISO dates and English m/d/y dates are decoded separately.
Source provenance: source_hashes.json, selected_source_metadata.json, paired_yield_source_cells.csv and independent_source_cell_checks.csv. Repeated HTML table IDs require source_table ordinal as well as source_table_id,row,column. Per-cell archived Swedish label strings are not reliable label mappings; native grain units have independent verified provenance.
Verification: independent_arithmetic_verification.json, independent_damage_arithmetic_verification.json, tests/test_response_estimator.py, tdd_* logs and final_verification_receipt.json.
Interpretation: yield_response_results_zh.txt. Redistribution status: source_redistribution_status.json; no CC license claim or external notifications.

Reproduction from the project root, using the existing .venv:
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m pytest analysis/paper_study/nature_food_revision_20261007/yield_response/tests/test_response_estimator.py -q -p no:cacheprovider
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/paper_study/nature_food_revision_20261007/yield_response/run_prototype.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/paper_study/nature_food_revision_20261007/yield_response/verify_arithmetic.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/paper_study/nature_food_revision_20261007/yield_response/run_damage_candidate.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/paper_study/nature_food_revision_20261007/yield_response/verify_damage_arithmetic.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python analysis/paper_study/nature_food_revision_20261007/yield_response/extract_additional_disease_provenance.py

Treatment codes must be loaded as strings when joining a subset containing only numeric codes to the larger source tables. Registry entry denotes a treatment entry in single-factor trials and a cultivar entry in two-factor trials. Every source file remains in the existing read-only yield_data_audit directory; this directory contains derived calculations and separate prototype code only. Current model, calibration and manuscripts are outside the prototype scope.
