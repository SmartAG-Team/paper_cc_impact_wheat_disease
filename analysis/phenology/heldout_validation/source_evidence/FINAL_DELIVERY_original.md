# Whole-season process-model results

Updated: 2026-09-28. Folder name retained: `AGC-PhenFormer-v2-final-results-20260926`.

The result collection contains GDD, T–P and T–P–V only. Neural training runs, checkpoints, prediction arrays and neural evaluation outputs are absent. No new seasonal neural performance result is available.

Each sample spans known sowing through the day before the next recorded sowing or the station record end. Process event dates are the first crossings of the frozen physiological thresholds. The recorded BBCH 10, 31, 51 and 85 events are matched by station, sowing cycle and stage. Sowing contributes no physiological event error. All inputs use observed seasonal weather; results are realized-weather hindcasts.

## Shared-event MAE, days

| Cohort | GDD | T–P | T–P–V |
|---|---:|---:|---:|
| Calibration | 10.78706 | 7.41862 | 6.95542 |
| Validation | 11.94393 | 8.47995 | 7.94157 |
| Testing | 12.06974 | 8.54044 | 8.02297 |

All three models use the same matched-event keys within each cohort. Overall MAE pools event errors and is not an unweighted mean of stage-level MAEs. Calibration results are in-sample.

## Coverage

| Cohort | Sites | Complete / total cycles | Shared matched / observed events | Shared coverage |
|---|---:|---:|---:|---:|
| Calibration | 187 | 5,797 / 5,797 | 18,057 / 18,179 | 99.33% |
| Validation | 1,222 | 13,726 / 14,713 | 38,133 / 40,537 | 94.07% |
| Testing | 1,140 | 14,445 / 15,360 | 40,878 / 43,177 | 94.68% |

Testing preserves the approved 1,140 sites; all 1,223 reserved test sites remain excluded from fitting and model selection, including the 83 historical scoring exclusions. The 915 incomplete testing cycles and their 1,685 observed events remain in coverage with missing predictions. Whole-cycle eligibility differs from the earlier process evaluation, so its historical 42,115-event denominator and scores are separate results. The current three-model intersection will also differ from the future five-model intersection whenever a neural model lacks a matched prediction.

Calibration parameters remain the existing frozen, training-only fit on 187 sites. This update changes the seasonal evaluation and result collection; it does not refit parameters on validation or test data. Original observations remain unchanged, including flagged reversals in physiological ordering. Missing stage observations supply no negative evidence.

## Files

- `process_models/calibration/`, `validation/`, `testing/`: event-date CSVs, all-cycle coverage, individual and common-matched metrics, and integrity manifests.
- `process_models/summary.csv`: 45 cohort/model/stage summaries, with individual coverage and shared-event errors.
- `process_models/validation.json`: independent arithmetic and exact-key checks.
- `process_models/seasonal_preflight.json`: verified cohort, input identities and seasonal contract.
- `configs/`: frozen process parameters, threshold table, station split, evaluation group and calibrated-data manifest.
- `evaluation_source/`: exact evaluator and a standalone verification command.
- `RESULTS_MANIFEST.json`: file inventory and SHA-256 hashes.

From this folder, an environment with Python and NumPy can verify every artifact and recalculate all evaluation tables and metrics from the retained cycle records:

```bash
python evaluation_source/verify_results.py
```

The result collection does not duplicate the multi-gigabyte daily input CSV. Source hashes identify the calibrated data retained in the repository and the sowing-onward runnable package. The evaluator uses the shared `whole-season-known-weather-midpoint-v1` event format; the input contract is `whole-season-sowing-onward-v2`. No neural midpoint predictions are present in this process-only collection.
