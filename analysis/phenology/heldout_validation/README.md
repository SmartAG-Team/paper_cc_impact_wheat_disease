# German winter-wheat phenology validation

The frozen T–P–V model was calibrated on 187 German winter-wheat stations. The archived split contains 1,222 validation stations and 1,223 reserved test stations, with disjoint station identities. The fitted station identities and calibration SHA256 agree exactly with the copied model. The approved test cohort comprises 1,140 stations; 83 reserved stations failed the historical requirement for a 150-day consecutive same-station window after common-feature complete-case filtering. All 1,223 reserved stations remained outside fitting.

Event-date evaluation matches station, sowing cycle and BBCH exactly. Predictions are the first daily crossings of fixed physiological thresholds. Known sowing anchors each cycle and contributes no physiological error. The primary comparisons use identical events matched by all three process models.

| Cohort | Shared matched events | T–P–V MAE (days) | RMSE (days) | Bias (days) |
| --- | --- | --- | --- | --- |
| calibration | 18,057 | 6.955 | 10.084 | 0.647 |
| validation | 38,133 | 7.942 | 11.750 | 0.681 |
| testing | 40,878 | 8.023 | 11.750 | 0.504 |

The test cohort contains 15,360 sowing cycles and 43,177 observed physiological events. The 40,878 common matched events represent 94.6754% of observations. T–P–V alone matches 40,898 events (94.7217%) and has MAE 8.020 days and RMSE 11.748 days on its own matches. The 915 incomplete test cycles retain 1,685 observed events as missing predictions; total T–P–V missing predictions number 2,279. Error scores are conditional on matched events. Predicted stages without observations cannot be classified as false positives because absent stage records do not establish event absence.

| Model | Shared test events | MAE (days) | RMSE (days) |
| --- | --- | --- | --- |
| GDD | 40,878 | 12.070 | 18.663 |
| T-P | 40,878 | 8.540 | 12.793 |
| T-P-V | 40,878 | 8.023 | 11.750 |

| BBCH | Shared test events | T–P–V MAE (days) | RMSE (days) | Bias (days) |
| --- | --- | --- | --- | --- |
| 10 | 13,471 | 6.030 | 10.285 | 1.904 |
| 31 | 12,109 | 11.591 | 15.266 | -0.838 |
| 51 | 12,541 | 6.846 | 9.542 | 0.467 |
| 85 | 2,757 | 7.444 | 9.613 | -0.269 |

BBCH85 denotes soft dough. The model contains no flag-leaf emergence, anthesis or physiological-maturity event output. BBCH10 errors coincide across GDD, T–P and T–P–V because photoperiod and vernalization modifiers begin after the emergence threshold.

Training, validation and testing contain observations in every calendar year from 1985 to 2015. The holdout is by station; no withheld-year evaluation is present. The archive supports retrospective hindcasts conditioned on realized full-season weather. Cycle ends use the next recorded sowing or station record end. The sowing-known-at-date assumption derives from retrospective CODE0 records; independent issue-time management timestamps are unavailable. Exact event matching and unconditioned process accumulation exclude observed-date nearest-hit selection. Archived split and parameter checks establish station separation, while independently blinded prospective model selection and regional extrapolation remain unverified. Repeated events within stations and seasons are correlated; the point metrics contain no uncertainty intervals or significance tests. Twenty-one test cycles retain observed-stage order-conflict flags.

The evaluation concerns German phenology. European Septoria disease outcomes, cultivar transfer, durum wheat and regional transfer remain unvalidated by these results.

The authoritative process-only archive is `AGC-Transformer/results/AGC-PhenFormer-v2-final-results-20260926`, refreshed 2026-09-28. Despite its directory name, this collection contains GDD, T–P and T–P–V rather than neural-model results. Its original verifier reconciles 27 inventory files. Independent recalculation from 430,440 retained event rows reproduces 270 MAE, RMSE and bias values to absolute tolerance 1e-12. The earlier 42,115-event evaluation and the later five-model intersection use different denominators and are separate results.

The 12 rows in `heldout_test_figure_data.csv` define the standalone PNG/PDF figure. `metrics_long.csv` retains 90 cohort/model/stage/denominator combinations. Original configurations, metrics, extraction code and manifests are byte-identical in `source_evidence/`; `PROVENANCE.json` records source hashes, and `SHA256SUMS` records the delivered artifacts.

```sh
/Users/gangzhao/Documents/workspace/agri_water_management/.venv/bin/python -B analysis/phenology/verify_archived_validation.py --directory analysis/phenology/heldout_validation
/Users/gangzhao/Documents/workspace/agri_water_management/.venv/bin/python -B analysis/phenology/plot_heldout_validation.py --directory analysis/phenology/heldout_validation
```
