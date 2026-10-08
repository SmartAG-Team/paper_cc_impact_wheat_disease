# Forecast-input contract: known sowing and simulated development

Schema version: 2. Effective date: 2026-09-24. Supported artifacts: `dataset/calibrated_causal/`.

Neural forecast issuance occurs at the end of the final encoder day. The standard window contains 120 historical days and predicts the next 30 days, with 30 historical decoder-context days. Future decoder value channels are zero; deterministic calendar marks remain available.

| Input | Availability and permitted use |
| --- | --- |
| Station coordinates and calendar | Fixed station metadata and known dates. |
| Sowing records | Each sowing date is known when sowing occurs. Records contain `PEP_ID`, `SOWING_DATE`, and `SOWING_KNOWN_AT`. No sowing event applies before its occurrence. |
| Daily weather | Process features for a date use only weather through that date. Realized weather after neural issuance is excluded from its encoder/context. Daily process scoring uses realized weather and therefore represents hindcasting. |
| Developmental state | Recomputed from explicit sowing records, historical weather, fixed response constants, and training-only calibrated gates/thresholds. |
| NDVI and EVI | Historical values through the final encoder day. Original preprocessing and timestamp availability remain unverified. |
| Observed phenological labels | Training event labels fit process thresholds on the 187 training sites. Training daily targets fit the target scaler and supervised neural loss. Validation/test targets serve loss or scoring only; they never supply encoder/decoder-context values. |
| Future labels | Supervision and scoring only. They do not determine process gates, simulation joins, historical neural values, or cycle resets. |

## Sowing records

The retained management table contains 35,977 records across 2,632 sites. `process_model/sowing.py` exported the original `CODE==0` dates with `SOWING_KNOWN_AT = SOWING_DATE`, under the explicitly approved availability assumption. The export and its sidecar preserve source provenance. Independent management-record timestamps have not been recovered.

As-of assignment selects the latest sowing that has occurred at each station/date. Later planned sowing cannot affect earlier rows. Records whose availability date is later than sowing, duplicate station/sowing records, and incomplete records are rejected. Before the first known sowing, process state is missing rather than initialized from a future observation.

## Process state

Gate fitting proceeds sequentially using training stations only:

1. Unadjusted cumulative GDD determines the emergence and BBCH51 thresholds. Photoperiod adjustment applies when previous-day GDD is at least the emergence threshold and below the BBCH51 threshold.
2. Recomputed cumulative T-P determines its emergence and BBCH31 thresholds. Vernalization accumulation starts after the emergence threshold; growth sensitivity ends when previous-day T-P reaches the BBCH31 threshold.
3. Recomputed GDD, T-P, and T-P-V supply the five final stage thresholds. Validation/test transformations use frozen parameters.

This replaces observed-stage gates with empirical simulated-state approximations. Fixed photoperiod and temperature-response constants retain the historical formulas. The current stop thresholds are GDD 1426.78 and T-P 582.80; both emergence thresholds are 142.01. The transformation and daily simulator accept inputs without `CODE`, `CODE_new`, or `BBCH`.

## Neural inputs and provenance

The simulation join uses station/date only. The eleven-channel encoder retains coordinates, process accumulations/simulations, NDVI, and EVI; its final channel is simulated T-P-V stage mapped from BBCH `0, 10, 31, 51, 85` to ordinal `0..4`. That proxy uses the training target scaler's units. The decoder's historical target-context channel uses the same proxy. Observed `CODE_new` remains the future prediction target.

Schema-2 data/checkpoint contracts record the issue-time, sowing, weather, simulated-state, and target-use policies. The older observation-conditioned calibration is rejected even if its artifact hashes match. Historical checkpoints retain their original input semantics and do not become corrected models through reuse of compatible parameter shapes.

All models share the frozen 187/1,222/1,223 train/validation/reserved-test partition. Exactly 1,140 eligible test sites are scored; 83 sites without valid 150-day windows remain excluded from both scoring and fitting. The corrected data preserve those exact lists and 4,780,697 valid test windows.

## Scope of validation

Regression checks establish that removing/changing observed process labels leaves process outputs unchanged; future weather or future sowing cannot alter earlier process features; held-out changes cannot affect fitted parameters; and changing held-out daily labels leaves encoder and decoder-context values unchanged while changing supervision targets. Old observation-conditioned artifacts are rejected at run entry.

Known sowing is a supplied management input, so stage-zero initialization is not evidence of independently predicted sowing. Daily realized-weather process scores are not fixed-horizon forecasting or event-date accuracy. Independent issue-date/lead-preserving event evaluation remains P1-5. Historical canopy provenance remains unresolved. Full neural retraining is deferred, and archived scientific scores have not been replaced.
