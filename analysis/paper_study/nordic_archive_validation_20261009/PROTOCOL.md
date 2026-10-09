# Nordic/Baltic archived management–grain evidence: fixed protocol

Protocol version: 1.0; fixed on 2026-10-09 after inspection of source schemas and arithmetic, before fitting or evaluating any benchmark. This is a retrospective analysis protocol, not a prospectively registered study. Scope: the 2012–2016 records in Figshare 19203377.v4 and 19203398.v3. The original workbooks also contain 2017 records, which remain in the source and audit outputs but are outside the analysis cohort.

## Evidence and unit of analysis

`Yield.xlsx` identifies trial (`SpotIT_ID`), year, country, crop, treatment, and grain yield in kg ha⁻¹ at 15% moisture. Its notes define A as untreated, B as one treatment, and C as two treatments. The release lacks individual plot, block, replicate, cultivar, fungicide product/dose and application-date identifiers. A record is a reported trial–treatment yield summary; no plot-level sample size, variance or randomization is inferred. The same untreated trial summary may support both B−A and C−A contrasts. Both contrasts remain, with a shared control identifier and shared validation group.

`Yield.xlsx` is the authoritative yield table. The SAS workbooks contain rounded copies and weather-derived recommendation counts, not independent yields or measured disease. Exact year–country–trial–treatment joins are checked for uniqueness before attachment; ambiguous keys are never joined many-to-many. Numeric agreement within 0.51 kg ha⁻¹ accommodates independently rounded SAS values. Missing SAS matches remain explicit and do not invalidate an otherwise internally complete pair in the primary workbook. Missing fields are never filled by neighboring rows, inferred gains, or guessed trial identifiers.

The source documentation and workbook schemas contain no observed disease species, severity, leaf identity, disease assessment date, actual LAI, senescence or healthy-yield measurement. The severity–yield, endpoint versus repeated/leaf-aware severity, functional-canopy and cause-specific STB analyses are unavailable. Weather-derived recommendation columns cannot substitute for disease observations. Regional average development-stage dates in `Development stage.xlsx` remain reference metadata, not observed dates for any trial-year. Both its published day numbers and 2022 date cells are preserved; calendar discrepancies are flagged, not silently corrected.

## Eligibility and arithmetic

1. Preserve every source record and its workbook, sheet and Excel row. Normalize only surrounding whitespace and crop-code case; retain raw fields. Archive checksums and exact source bytes establish provenance.
2. Restrict the analytic cohort to 2012–2016. Require a unique A and unique B and/or C within year–country–trial, internally consistent country, crop, station, region and treatment indicator, and finite positive yields. Duplicate treatment keys exclude the entire trial. A missing control excludes all contrasts for that trial. Missing/nonpositive treatment yields exclude affected contrasts. A missing control yield excludes the trial. Exclusions remain row-addressable.
3. Recompute every contrast directly from the two observed yields. Source-reported gain is an audit field, never an input predictor or substitute control. Differences exceeding 1 kg ha⁻¹ are substantive gain inconsistencies; preserve the calculated pair, flag it, and run a prespecified sensitivity excluding that trial. Smaller discrepancies are recorded as rounding. No winsorization, outlier deletion or removal of negative responses is permitted.
4. Primary response: ΔY = (Y_treated − Y_untreated)/1000, in t ha⁻¹ at 15% moisture. Descriptive percentages: 100ΔY/Y_untreated and 100ΔY/Y_treated, using matching t ha⁻¹ denominators. These are different observed management contrasts; neither is a measured fraction of STB-caused yield loss. Zero and negative denominators fail closed.
5. In descriptive and evaluation estimates, each trial has total weight one, split equally among its retained B−A and C−A contrasts. Distinct treatment and crop strata are also reported. Contrasts are not independent replicates.

## Fixed benchmarks and environmental evaluation

Only management benchmarks are estimable; there is no disease-response candidate to fit or select. The fixed response for all benchmark evaluation is ΔY in t ha⁻¹. No held-out yield or gain appears in a predictor, training weight, transformation or fallback estimate.

- **Training mean:** trial-weighted mean ΔY of the training fold.
- **Management mean:** training-only weighted mean by B/C treatment category; an unseen category falls back to the training mean.
- **Crop–management mean:** training-only weighted mean by WW/SW × B/C; an unseen combination falls back to the management mean and then the overall training mean. This is a descriptive management benchmark, not a causal adjustment model.

Primary evaluation leaves an entire country–year out, a deliberately conservative environmental group that contains all trials, shared controls, weather stations and repeated treatment summaries from that country in that year. It does not establish independence between years at a recurring site. Secondary evaluations leave an entire country out, leave an entire year out, and use expanding historical training windows: 2012–2013→2014, 2012–2014→2015, and 2012–2015→2016. Forward-year results are retrospective temporal holdouts, not prospective climate validation. Country/crop coverage is uneven and partially confounded.

Every fold requires nonempty, disjoint train/test trials, controls and country–year groups. Forward folds additionally require max(training year) < min(test year). There is no hyperparameter search, feature selection or benchmark selection. Predictions, memberships, training sizes, fallback counts and metrics remain inspectable. Overall trial-weighted RMSE, MAE, mean error and country–year-balanced RMSE are reported. Relative skill is 1−MSE_benchmark/MSE_training_mean on identical held-out records; negative values remain visible.

For descriptive mean response and primary-evaluation paired RMSE differences, 2,000 country–year cluster bootstrap draws (NumPy seed 20261009) provide percentile 95% intervals. A draw resamples whole groups and all shared controls within them. Evaluation intervals are conditional on the fitted out-of-fold predictions; they do not refit models, quantify model-selection uncertainty or correct residual cross-year spatial dependence. No significance-based model choice is allowed. The prespecified sensitivity removes trials with substantive source gain inconsistencies, recalculates all weights, and repeats primary evaluation.

## Reproducibility and claim boundary

A compact ZIP contains exact, unmodified public source workbooks, documentation and both versioned metadata records used here. `Weather data.txt` is omitted because raw weather exposure is not analyzed. The manifest records archive members, Figshare file identifiers and download URLs, publisher MD5, SHA-256, sizes, attribution and CC BY 4.0. Both supplied archive copies are checked for byte identity. A standalone run reads only this subset bundle; missing or changed inputs and incompatible schemas fail closed.

Tests cover duplicate and unmatched join handling, missing controls, shared controls, grouping, training-only fallback estimates, temporal leakage, positivity and denominator arithmetic, preservation of unmeasured disease fields, and source-gain inconsistencies. Numerical outputs, execution/test receipts, package versions and protocol/code/source checksums accompany the analysis.

This evidence advances trial–treatment harmonization and establishes a retrospective observed management-yield benchmark. It does not validate disease severity, healthy yield, functional LAI, physiological damage, cause-specific STB loss, European production effects, adaptation efficacy or future climate responses. Disease/leaf/stage comparisons remain explicitly blocked by missing compatible observations. The root research plan and publication builder are outside this worker's write scope.
