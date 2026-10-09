# Nordic/Baltic 2012–2016 management-associated grain response

The archived Nordic/Baltic data support **307 observed management–grain contrasts from 263 reported trial identifiers**, distributed across Denmark, Finland, Lithuania, Norway and Sweden and **25 country–year groups**. The trial-weighted mean treated-minus-untreated grain response was **0.980 t ha⁻¹** at 15% moisture, with a country–year cluster-bootstrap 95% interval of **0.804–1.142 t ha⁻¹**. The release contains no observed disease severity, disease species, leaf identity, repeated disease assessments, measured canopy area or trial-specific stage observations. Its contribution is retrospective management-yield evidence; severity–yield and physiological-damage validation remain unavailable. Sources: [Andersson, Figshare 19203377.v4](https://doi.org/10.6084/m9.figshare.19203377.v4) and [Andersson, Figshare 19203398.v3](https://doi.org/10.6084/m9.figshare.19203398.v3).

## Source reconciliation and experimental units

`Yield.xlsx` contains 662 reported trial–treatment records. Its notes define A as untreated, B as one treatment and C as two treatments; `YieldKgHa15%` identifies grain yield in kg ha⁻¹ at 15% moisture. Individual plots, replicate/block identifiers, cultivars, product/dose, treatment dates and randomization information are absent. The aggregation underlying each reported yield is not documented. Trial identifiers therefore do not establish 263 independent physical experiments or a plot-level sample size.

The year–country–trial–treatment key permits unambiguous source linkage except for a duplicated Finnish treatment key. Of 662 primary records, 657 have a unique SAS match, three have none and two share an ambiguous key. Comparable, uniquely matched numerical yields differ by at most 0.5 kg ha⁻¹, consistent with rounding in the SAS workbooks. The SAS files contain 677 records, including 664 trial records and 13 weather-only records. Five trial records have no exact primary-table counterpart: four records for Danish trials absent from the yield workbook and one Norwegian control with a different year. These secondary records are not appended as new observations or used to repair identifiers. The internally complete primary Norwegian pair remains eligible. The source `Group` field is unique within source-file–country–year scope, but its numerical values recur across years or countries and cannot serve as globally unique trial keys.

| Source handling | Primary records | Consequence |
| --- | ---: | --- |
| Year 2017, outside fixed 2012–2016 scope | 86 | Preserved in source and exclusion tables |
| `2012FI01`, duplicated B key | 3 | Entire reported trial excluded; no averaging or guessed relabeling |
| `2012FI02`, no A control | 1 | Excluded; control not reconstructed from reported gain |
| `2012DK19`, both yields missing | 2 | Excluded |
| Eligible observed yield records | 570 | 263 controls and 307 treated records |

Forty-four Swedish trial identifiers each support B−A and C−A contrasts from one shared untreated value. Each reported trial receives total weight one, divided equally between its eligible contrasts. Nine contrasts are negative and one is zero; all remain in the analysis. Identical untreated values occur under different trial identifiers in another 43 source records, representing 20 country–year–station–crop–yield signatures. Equal values cannot establish physical shared controls. These records remain distinct, with an explicit audit flag, and all belong to the same country–year holdout whenever a signature recurs.

The source gain field for `2016DK31` reports 1,690 kg ha⁻¹, whereas its untreated and treated yields are 7,310 and 8,150 kg ha⁻¹, giving **840 kg ha⁻¹**. The primary endpoint uses the observed yields; the source discrepancy is retained. Two smaller discrepancies, 0.130 and −0.650 kg ha⁻¹, fall within the fixed 1 kg ha⁻¹ gain-rounding tolerance. Excluding `2016DK31` leaves 306 contrasts and 262 reported trial identifiers, with mean response 0.980 t ha⁻¹ and no reversal of the benchmark comparison.

## Dates, stages, weather recommendations and disease compatibility

The development-stage workbook supplies 72 regional reference entries: winter wheat DC32/DC71 and spring wheat DC30/DC65 for 18 regions. Its metadata describes average dates, not measurements in each field or trial-year. All 72 date cells use 2022 and have a calendar day of year one greater than the published day-number field. Both values and the discrepancy remain explicit; neither field becomes an observed trial-stage date. Regional metadata can be attached to all 307 contrasts after whitespace normalization of region names, without creating observed phenology.

Exact, case-normalized full station identifiers join 604 of 664 SAS trial records to the station workbook; 60 remain unresolved. Short station aliases and truncated regional names are preserved rather than used to manufacture weather joins. Country–year grouping contains any same-year station sharing without depending on those unresolved aliases. Raw weather exposures are outside this analysis.

The SAS workbook annotations and example script identify columns such as `r85h14` and `day6` as **numbers of weather-derived treatment recommendations**. These are neither observed disease severity nor repeated disease measurements. The archive has zero compatible severity–yield pairs and no recorded disease-species field; STB, other foliar diseases and direct management effects cannot be separated. The endpoint-versus-leaf/stage/repeated-severity comparison, functional-LAI reconstruction, healthy-yield validation and cause-specific STB-loss estimation are unavailable. These conditions are recorded in `outputs/compatibility.json`; no corresponding model is fitted.

## Endpoints and fixed evaluation

The primary endpoint is ΔY = (Y_treated − Y_untreated)/1,000 in t ha⁻¹. The descriptive relative gain is 100(Y_treated − Y_untreated)/Y_untreated. The alternative observed contrast percentage uses Y_treated in the denominator. Trial-weighted means are **14.068%** and **11.657%**, respectively; these are means of paired percentages, not ratios of population mean yields. The latter percentage does not quantify disease-attributable loss or assume that treatment produces healthy yield. Finite positive denominators are required.

The fixed protocol precedes fitting and specifies three benchmarks: the training-only trial-weighted mean response; the training mean by treatment category; and the training mean by crop type and treatment category. Unseen combinations fall back to the treatment mean and then the overall training mean. Test outcomes never enter these estimates. There is no feature selection, tuning or model selection. Reported source gains, observed control yields, weather recommendations and regional reference dates are not predictors.

The primary 25-fold evaluation leaves out whole country–years, retaining all treatments, shared controls and same-year station observations together. Secondary checks leave out a whole country or year; expanding historical training windows test 2014, 2015 and 2016 using earlier years only. Every fold verifies disjoint trial, control and country–year memberships. The independent-environment interpretation is conservative within years; repeated locations across years and broader regional correlations remain possible.

| Holdout | Evaluated contrasts / reported trials | Training mean RMSE | Management mean RMSE | Crop–management mean RMSE |
| --- | ---: | ---: | ---: | ---: |
| Country–year, primary | 307 / 263 | **0.628** | 0.636 | 0.638 |
| Country | 307 / 263 | **0.693** | 0.691 | 0.733 |
| Year | 307 / 263 | **0.625** | 0.633 | 0.637 |
| Forward year, 2014–2016 | 194 / 166 | **0.623** | 0.638 | 0.641 |

RMSE units are t ha⁻¹; each reported trial has total evaluation weight one. Bold values identify the fixed reference benchmark, not a selected best model. Primary MAE values are 0.487, 0.492 and 0.495 t ha⁻¹. Management and crop–management benchmarks have primary MSE skill of **−2.515%** and **−3.241%** relative to the training mean. Their paired RMSE differences are +0.00785 t ha⁻¹ (95% interval −0.00874 to +0.02360) and +0.01010 t ha⁻¹ (−0.00738 to +0.02482). Excluding the inconsistent Danish trial gives primary RMSE values of 0.629, 0.637 and 0.639 t ha⁻¹, respectively.

Uncertainty uses 2,000 country–year bootstrap draws with seed 20261009. Entire groups and their shared controls are resampled together. The evaluation intervals condition on fixed out-of-fold predictions; they do not incorporate refitting uncertainty or remove cross-year spatial dependence. Mean-response uncertainty describes the sampled field evidence, not a causal treatment effect or a representative European wheat population.

## Evidence boundary

The observed treatment contrasts provide quantitative grain-response evidence and reproducible management baselines. The small, unfavorable primary benchmark differences do not support improved response prediction from treatment count and crop type alone. The two crop classes are unevenly distributed: winter wheat contributes 258 contrasts from 214 reported trials in Denmark, Lithuania and Sweden; spring wheat contributes 49 contrasts from 49 reported trials in Finland and Norway. Crop and country effects are consequently confounded, and a larger mean response for a treatment category is not evidence of an incremental application effect.

Retrospective forward-year performance is not prospective validation of responses to climate change. Protected yields are not independently established healthy yields. The grain contrasts cannot calibrate a disease-specific HAD-to-yield conversion, support production-tonnage aggregation, validate functional canopy damage or identify cause-specific STB loss. Physiological and severity-response milestones in the research plan remain open.

## Inspectable evidence

`PROTOCOL.md` and `receipts/protocol_fixed.json` establish the pre-fit specification and its SHA-256. `source_bundle/nordic_public_sources.zip` contains exact source bytes under CC BY 4.0; `source_bundle/manifest.json` records attribution, publisher checksums and both original archive copies. `outputs/paired_management_grain.csv`, source-row exclusion and join audits, fold memberships, predictions, metrics and uncertainty files provide the numerical evidence. `receipts/verification.json` records execution, tests and isolated reproduction. Standalone reproduction requires no ignored original archive or network access.
