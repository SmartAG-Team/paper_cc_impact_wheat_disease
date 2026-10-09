# Nordic/Baltic archived management–grain evidence

**307 grain contrasts, 263 reported trial identifiers, 25 country–year groups. No observed severity–yield pairs.** Mean management-associated response: 0.980 t ha⁻¹ (95% cluster interval 0.804–1.142). Primary RMSE: training mean 0.628, management mean 0.636, crop–management mean 0.638 t ha⁻¹.

- [Methods, results and claim boundary](METHODS_AND_RESULTS.md).
- [Research-plan evidence text](RESEARCH_PLAN_EVIDENCE.md), for Priority 1 and the remaining Priority 3 limitations.
- [Fixed pre-fit protocol](PROTOCOL.md) and [timestamp/hash receipt](receipts/protocol_fixed.json).
- [Exact source subset and provenance](source_bundle/manifest.json); [CC BY 4.0 attribution](source_bundle/LICENSE_AND_PROVENANCE.md).
- [Analysis receipt](outputs/analysis_receipt.json), [paired observed yields](outputs/paired_management_grain.csv), [benchmark metrics](outputs/benchmark_metrics.csv), [missing-input gates](outputs/compatibility.json).
- [Execution and reproduction receipt](receipts/verification.json).

## Reproduce

Run from this directory using Python 3.12 with the versions in `requirements.txt`:

```sh
python run_analysis.py
python -m pytest -q -p no:cacheprovider --basetemp receipts/test_tmp test_analysis.py
python verify_reproduction.py
```

The verified interpreter is `/Users/gangzhao/Documents/workspace/paper_cc_impact_wheat_disease/.venv/bin/python`. `verify_reproduction.py` reruns the analysis and all module tests, copies only the script, fixed protocol and bundled sources into an isolated directory, blocks original-archive and network access, and checks numerical output identity. Temporary verification files remain under this module and are removed after successful verification. `--output-dir` selects another output directory for `run_analysis.py`.

The default run needs only the nine-member, 167,763-byte source ZIP, its manifest and the fixed protocol receipt. Missing or changed sources, incompatible schemas, invalid joins, invalid yield denominators and leaking folds fail closed. Raw source tables, recommendations, 2017 records and ambiguous records remain in auditable outputs; weather recommendations never become observed disease.

All deliverables are confined to `analysis/paper_study/nordic_archive_validation_20261009/`. The root research plan, publication builder, previous analyses, branch/index and commits are outside this module's changes. No journal literature search was required; citations are the two primary public datasets.
