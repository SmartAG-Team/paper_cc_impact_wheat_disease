The independent seasonal audit applies to the precise source hashes in `review_source_hashes.json`. The copied source snapshot is retrospective; original model files and the prior calibration archive remain read-only. Numerical receipts and formal findings are in `independent_check_receipt.json`, `existing_tests_receipt.json` and `READ_ONLY_REVIEW.md`.

Reproduce the independent checks:

```sh
NUMBA_CACHE_DIR=analysis/paper_study/model_review/numba_cache PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with numpy --with pandas --with scipy --with numba --with pyarrow python -B analysis/paper_study/model_review/independent_checks.py
```

The script checks numerical mass and route accounting, an exact continuous-time reference, causal weather prefixes, inactive-host behavior, synthetic parameter recovery, withheld-input isolation, archived membership, frozen hashes and all candidate-selection scores. It reports later changes to reviewed source files separately. Source-contract error demonstrations are synthetic, with no external disease outcomes.

`time_step_convergence.csv` contains solver errors in percentage points. `recomputed_inner_selection.csv` contains exact score reconciliation. `archived_field_weights.csv` records implemented field weights without disease values. The Numba cache is confined to the review directory. No new source code, fit or prior-archive artifact is written outside that directory.
