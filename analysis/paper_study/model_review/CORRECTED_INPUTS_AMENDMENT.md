The corrected source assembler rejects count metrics, unsupported percentage definitions and measurement units other than `percent` or `%`. Explicit measurement units are required. Multi-digit leaf labels, including `F10` and `LEAF, 14TH`, are excluded rather than truncated to leaf 1. The corrected source hash and its original reviewed hash are recorded in `corrected_inputs_receipt.json`.

Thirty independent correction checks passed. The two error demonstrations fail eligibility under the corrected assembler and retain their original behavior under the preserved review snapshot. All five daily weather and host arrays and all metadata, target and exclusion tables are identical between the original and corrected assemblers for the 73 BASF field-seasons and 584 eligible targets. The 34 frozen calibration-archive files retain their original SHA256 values. Numerical solver, host, observation-endpoint and calibration code hashes are unchanged. No external disease outcomes enter this correction check.

The six field-input regression tests passed, including both newly added input-error tests. The original independent numerical review, source snapshot and receipts remain version-specific evidence. The correction receipt is a retrospective verification and does not replace the original calibration execution record.

The protocol amendment identifies the implemented hierarchy as equal coordinate-year weights, followed by equal fields within coordinate-year, equal leaf series within field and equal dates within leaf series. Frozen fitted parameters and their archived selection scores retain their previous values.

Reproduction uses the read-only correction checker:

```sh
PYTHONDONTWRITEBYTECODE=1 uv run --no-project --python 3.12 --with numpy --with pandas --with scipy --with numba --with pyarrow python -B analysis/paper_study/model_review/verify_corrected_inputs.py
```
