# Wheat-fitted DL phenology comparison

A wheat-specific temporal-convolution model (TCN) reduces pooled date MAE from **8.02 to 7.16 days** across **40,874 identical test events** shared by T-P-V, LSTM and TCN. The station comparison is retrospective; its years overlap calibration. Three-seed TCN and LSTM ensembles use architectures adapted from `maize_phen_dl`, with fresh wheat weights, wheat T-P-V daily increments and four native wheat thresholds. Original maize code, weights and results remain unchanged.

The TCN family was selected using validation stations before test prediction. On the T-P-V/TCN pairwise intersection of 40,883 events, stage-equal MAE decreases by **0.799 days** (station-bootstrap 95% interval **0.729–0.874 days**). This interval conditions on the fitted models and does not represent crop/disease or future-climate uncertainty.

| Native stage | T-P-V MAE (days) | LSTM MAE (days) | TCN MAE (days) | Common test events |
| --- | ---: | ---: | ---: | ---: |
| First leaf emerged, GS10 | 6.03 | 4.43 | 4.44 | 13,459 |
| First node, GS31 | 11.59 | 11.06 | 11.06 | 12,115 |
| Beginning of heading, GS51 | 6.84 | 6.42 | 6.39 | 12,546 |
| Soft dough, GS85 | 7.45 | 6.88 | 6.83 | 2,754 |

[Comparison figure](results/Wheat_DL_phenology_comparison.pdf), [event-level predictions](results/event_predictions.csv.gz), [native-stage decision](results/decision.json) and [field-transfer evidence](field_evidence/evaluation/interval_metrics.csv) retain unfavorable outcomes and distinct denominators.

The full replacement condition is **not met**. TCN supplies 2,758 soft-dough medians versus 2,759 for T-P-V on 2,781 weather-qualified observed soft-dough events: five baseline predictions are lost and four new predictions are gained. Overall native-event coverage improves, but the fixed requirement of no stage-specific coverage loss fails. In strict external disease fields, transfer of the original auxiliary flowering threshold increases mean distance outside 64 paired two-sided observation intervals from 1.81 to 2.13 days. Flag-leaf timing improves from 1.13 to 0.53 days in 30 such comparisons. Only one paired, two-sided external soft-dough prediction is available. These are interval diagnostics using calendar-scenario sowing, not exact-date errors or direct German flowering calibration. The current STB engine and completed climate archive retain their original phenology implementation.

## Evidence and model contract

The fixed split reserves 187 calibration, 1,222 validation and 1,223 test stations. The established archive restricts scored test membership further; excluded test stations do not enter fitting. The prepared arrays retain 5,797 calibration, 14,713 validation and 15,360 test sowing cycles. Calibration uses 18,179 observed targets. After the fixed weather/eligibility screen, validation and test have 38,749 and 41,492 targets. Source observations outside that support remain in the exclusion and prediction inventories.

Inputs contain only daily weather and simulated process history available through the prediction date. Observed stage labels, canopy indices, site identifiers and future-window summaries are absent from predictors. Weather starts on sowing day, unlike the maize project's day-after-sowing convention. The normalizer uses calibration weather only. A common marginal uncertainty scale preserves stage-CDF ordering. Missing quantiles are retained, without a harvest cap or fixed-date fallback. The native models output GS10, GS31, GS51 and GS85; GS39 and GS65 remain separate auxiliary-threshold transfer tests.

A native-stage forecast is conditional on supplied weather. Realized-weather hindcast skill does not establish an issue-time operational forecast or climate-extrapolation skill. Phenology improvement does not validate disease-severity predictions, functional canopy loss, grain loss, or adaptation benefits.

## Reproduction

The recorded environment is in `runtime_versions.json`. PyTorch, NumPy, pandas, matplotlib and pytest are needed for inference, fitting and verification; `pyarrow` is additionally needed to rebuild the training and field inputs from Parquet references. The existing `maize_phen_dl/.venv` supplies the Torch runtime. The wheat project's existing environment supplies the Parquet preparation runtime. No installed maize package is imported by the wheat models.

Git LFS objects must be downloaded (`git lfs pull`) for the NPZ and Parquet inputs. From the repository root, with a Python environment containing the relevant dependencies:

```sh
python -m analysis.paper_study.wheat_dl_phenology_20261009.verify
python -m pytest -q --import-mode=importlib analysis/paper_study/wheat_dl_phenology_20261009
```

Verification reconciles stored errors against original source event dates and replays all six checkpoints on 32 source-key-selected environments. The compact replay is an inference check; it is not a replacement for the full training population.

A source-backed raw-weather example is included:

```sh
python -m model.wheat_phenology_dl.predict \
  --weather analysis/paper_study/wheat_dl_phenology_20261009/example_weather.csv \
  --sowing 1985-09-19 --latitude 54.6167 \
  --checkpoints analysis/paper_study/wheat_dl_phenology_20261009/fits/tcn_s17.pt \
                analysis/paper_study/wheat_dl_phenology_20261009/fits/tcn_s29.pt \
                analysis/paper_study/wheat_dl_phenology_20261009/fits/tcn_s43.pt \
  --output /tmp/wheat_dl_prediction.json
```

Full fitting uses the external, hashed 4.15-GB causal daily archive identified in `input_metadata.json`, together with the already published wheat event tables and station split. It does not require the maize training data or checkpoints:

```sh
python -m analysis.paper_study.wheat_dl_phenology_20261009.prepare \
  --source /path/to/calibrated_causal/data_cleaned.csv --output /tmp/wheat_dl_inputs
python -m analysis.paper_study.wheat_dl_phenology_20261009.train \
  --inputs /tmp/wheat_dl_inputs --output /tmp/wheat_dl_fits --device mps
python -m analysis.paper_study.wheat_dl_phenology_20261009.evaluate \
  --inputs /tmp/wheat_dl_inputs --fits /tmp/wheat_dl_fits --output /tmp/wheat_dl_results
```

CPU fitting is available with `--device cpu`; it constitutes a distinct numerical fit. The saved checkpoints were fitted on MPS and evaluated on CPU. Preparation independently reproduced the 318-MB input bundle byte-for-byte; frozen T-P-V first crossings agree at all 127,556 finite source comparisons after array conversion. The complete training array is external to this repository. Source hashes, event units, exclusions, all attempted fits, training histories and the fixed protocol are retained.

Field transfer can be repeated from the compact bundled field inputs without the original weather directory:

```sh
python -m analysis.paper_study.wheat_dl_phenology_20261009.field_transfer evaluate \
  --prepared analysis/paper_study/wheat_dl_phenology_20261009/field_evidence/prepared \
  --checkpoints analysis/paper_study/wheat_dl_phenology_20261009/fits/tcn_s17.pt \
                analysis/paper_study/wheat_dl_phenology_20261009/fits/tcn_s29.pt \
                analysis/paper_study/wheat_dl_phenology_20261009/fits/tcn_s43.pt \
  --output /tmp/wheat_dl_field_transfer
```

The [release reproduction receipt](public_reproduction.json) records a separate-copy check of all six checkpoint replays, 83,714 source event dates, the raw-weather example, and complete rebuilding of the field inputs and predictions. All seven field data/result files are byte-identical. The original field-evaluation receipt retains its initial code hash; [the reproduced receipt](field_evidence/reproduced_evaluation.json) records the portable output-path and restricted checkpoint-loading implementation with identical scientific results. Verification passed 33 DL tests and 424 existing model/analysis tests.
