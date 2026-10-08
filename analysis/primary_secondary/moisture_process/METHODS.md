# Pycnidial coverage under early moisture treatments

The source dataset contains controlled seedling inoculation measurements from 48 genetically distinct *Zymoseptoria tritici* isolates in Irish (IR) and Israeli (IS) populations. Four moisture treatments were imposed during the first three days after inoculation. The workbook defines C0, C1, C2 and C3 as zero, one, two and three days of bagging, with mean relative humidity of 88.3%, 92.2%, 96.1% and 100%, respectively. PYC denotes the percentage of inoculated area covered by pycnidia; DPI denotes calendar days after inoculation. These categorical treatments combine bagging duration and reported humidity. They do not establish a continuous humidity response function. Source evidence consists of the original workbook and its repository dictionary and metadata ([Recherche Data Gouv, DOI 10.15454/FK7WHW](https://doi.org/10.15454/FK7WHW)).

Each of the three experimental series contains 16 isolates, comprising eight isolates from each population. No isolate occurs in more than one series. Within each series, every isolate occurs in two experimental blocks, four moisture treatments and three repetition labels, F1–F3, assessed at 14, 17 and 20 dpi. The workbook defines `bloc` as an experimental block within the growth chamber and `rep` as a repetition. Physical plant or leaf identities are not separately documented. A repeated trajectory is defined operationally by series, block, treatment, population, isolate and repetition label. Matching these labels across dates preserves their recorded dependence without establishing physical leaf identity.

The archive contains 3,456 score slots and 3,326 valid percentages; 130 cells contain the explicit missing marker `na`. Observed percentages span 0–100. All 72 score slots for isolate ir7 in series 1 are missing. Consequently, 47 source isolates have at least one valid response, including 31 assessed training isolates. Missing observations remain missing and receive no imputed zero. The probability of missingness conditional on disease progression cannot be determined from the workbook. Temperature, wheat cultivar, inoculum concentration, spore origin, deposited dose and penetrated infection counts are absent. No temperature, thermal-time or infection-efficiency parameter is estimated. The mean humidity values do not supply hours above 90% RH or leaf-wetness duration; those weather-response quantities remain uncalibrated. Source weather and coverage definitions are retained in `weather_coverage_constraints.json`.

## Predictive model and target separation

All fitted parameters use only series 1–2 at 14 and 17 dpi, comprising 1,456 valid scores. The three disjoint target partitions are series 3 at 14/17 dpi (761 scores, 16 isolates), series 1–2 at 20 dpi (728 scores, 31 isolates), and series 3 at 20 dpi (381 scores, 16 isolates). Holding out series 3 combines an experimental-series shift with transfer to previously unseen isolates. These effects cannot be separated because the isolate sets do not overlap. Day-20 prediction is a three-day extrapolation beyond the last fitting visit. No response from series 3 or day 20 contributes to fitting or hyperparameter selection. All source dates were inspected for schema, ranges and descriptive auditing; the targets are computationally excluded rather than claimed to be previously uninspected data.

The phenomenological coverage model is

\[
\widehat y_r(t)=A_r F_\Gamma(t;k,m/k),\qquad 0\leq A_r\leq100,
\]

where \(y_r\) is pycnidial area in percent of inoculated area, \(t\) is calendar DPI, \(F_\Gamma\) is a gamma cumulative distribution function, \(k\) is its shape, \(m\) is a shared progression-curve mean in calendar days, and \(A_r\) is a treatment-specific coverage capacity. The assumed time origin is inoculation; penetration time is unobserved. No onset lag is fitted. This cumulative area curve is a mathematical description of coverage accumulation, not a fitted distribution of individual infection latency or pycnidium appearance. A fixed-shape version uses \(k=3\). A second version selects among the prespecified shapes 3, 6, 12, 24, 48 and 96 by leave-one-training-series-out validation restricted to 14/17 dpi. Mean progression is bounded between 1 and 100 days. For each mean, weighted least squares gives the four capacities analytically, truncated to their percentage bounds; a grid-assisted one-dimensional search determines the mean.

Competitive baselines comprise a global constant, treatment-specific constants, a common linear time curve, treatment-specific linear time curves, and treatment-specific logistic curves with a fixed 100% asymptote. Linear predictions are bounded to 0–100%. Logistic coefficients reproduce the two weighted treatment-time means after truncating the means to 0.5–99.5% for the logit calculation. Their 100% asymptote is an extrapolation assumption. All seven models use the same training records. Their withheld-target scores are reported together; withheld responses select no parameter, shape or baseline.

Fitting and scoring give equal weight successively to experimental series, populations within series, assessed isolates within population, available treatment-date cells within isolate, available blocks within cell, and valid repetition scores within block. Thus neither isolates with more observations nor a population retaining more assessed isolates dominates the loss. RMSE and MAE are expressed in percentage points and include the variation among repetition scores. A second implementation independently reconstructs scores through nested arithmetic averages rather than reusing the fitting-weight helper.

## Dependence and uncertainty

Prediction-score uncertainty uses 2,000 percentile bootstrap replicates that resample entire isolate clusters within each observed series and population. All treatments, blocks, repetition labels and dates belonging to a sampled isolate remain together. Paired model comparisons use the same resampled clusters. This preserves lower-level correlation without treating repetition measurements as independent experimental units. The score intervals condition on the fitted training models and observed test series. They describe variation among sampled isolates within those series, not uncertainty across new experimental series or field environments.

Coverage-curve and parameter intervals use 200 training-isolate cluster bootstrap fits, stratified within series and population, with the selected shape held fixed. They condition on two observed training series and omit model-selection uncertainty. Two training dates cannot separately identify an onset lag, progression distribution and moisture-specific final capacity. Boundary capacities and parameter intervals therefore remain conditional model quantities rather than mechanistic biological estimates.

## Predictive results

Training-series cross-validation selected shape 24. Its fitted shared progression mean is 16.47 calendar days, with a conditional training-isolate bootstrap interval of 15.86–17.14 days. The four capacities are 24.87%, 51.78%, 77.71% and 100% for C0–C3. The C3 estimate reaches the imposed 100% upper bound in all 200 bootstrap fits. Its degenerate 100–100% interval reflects the constraint and does not establish a known final biological capacity. The late targets show that the shared progression with treatment-specific capacities extrapolates poorly, particularly under treatments with low early coverage.

| Model | Series 3, days 14/17: MAE / RMSE (pp) | Series 1–2, day 20: MAE / RMSE (pp) | Series 3, day 20: MAE / RMSE (pp) |
|---|---:|---:|---:|
| global constant | 28.10 / 35.54 | 45.70 / 52.56 | 59.49 / 63.75 |
| regime constant | 25.79 / 33.25 | 43.17 / 49.80 | 58.97 / 63.11 |
| time linear | 23.53 / 29.87 | 31.28 / 35.70 | 28.77 / 30.78 |
| regime linear | 19.95 / 26.80 | 23.67 / 32.10 | 25.21 / 33.81 |
| regime logistic 100 | 19.95 / 26.80 | 25.53 / 33.19 | 20.98 / 25.36 |
| gamma shape3 | 23.50 / 30.53 | 35.55 / 41.14 | 48.69 / 53.10 |
| gamma selected shape | 20.52 / 27.27 | 26.70 / 33.01 | 34.55 / 40.77 |

At series 3, day 20, the selected gamma coverage model has RMSE 40.77 pp, compared with 25.36 pp for the treatment-specific logistic baseline. Their paired RMSE difference is +15.41 pp, with a 95% isolate-cluster bootstrap interval of +8.54 to +21.08 pp, conditional on the observed series and fixed training fit. The shape-3 gamma curve has RMSE 53.10 pp at this combined series/time target. These comparisons provide no support for treating a universal three-stage gamma latent distribution, or a shared timing curve with moisture affecting only final capacity, as validated by these data. They remain comparisons of the specified predictive area curves.

The figure `pycnidial_coverage_transfer.png` and its PDF counterpart show treatment curves, observed series-specific means and the selected gamma curve's conditional training-isolate interval. Exact plotted observations and curve coordinates are retained in `figure_observed_means.csv` and `figure_curve_points.csv`. Filled markers denote the fitting dates in series 1–2; open markers denote excluded targets. Population- and isolate-balanced means preserve the scoring population, while the underlying score metrics retain repetition-level variability.

## Source-label onset constraints and scope

For each of 1,152 source-label trajectories, onset is defined descriptively as the first valid PYC score greater than zero. When an earlier valid zero exists, onset lies in the interval from that last zero to the first positive visit. A first positive without an earlier valid zero is left-censored between assumed inoculation time zero and the first positive visit. A zero-only assessed trajectory is right-censored beyond its last visit and can represent delayed development, failed establishment or absence of visible disease; resistance is not inferred. An unassessed trajectory is unobserved and receives no censoring time. The resulting counts are 791 left-censored, 279 interval-censored, 41 right-censored and 41 unobserved trajectories. The workbook contains 25 trajectories with decreasing area and two with a later zero after a positive score. These flags are retained. Persistent detectability and strictly cumulative coverage are working assumptions contradicted by those records, rather than properties imposed on the observations.

The source-label onset intervals are separate from the fitted area curves. A percentage of tissue bearing pycnidia is not the probability that an individual plant has reached onset, the fraction of viable deposited spores establishing infections, or the amount of conidia emitted. The data contain no natural airborne inoculum exposure, field secondary-transmission observations or temperature series. They constrain predictive pycnidial coverage under the recorded moisture treatments and laboratory-series shifts, with no direct validation of European field epidemics.

## Provenance and reproduction

The original XLSX contains 128,392 bytes, repository MD5 `54a222c41868d31d1ac8be21d60a5b12`, and SHA256 `f42f9cdbd4eeb4ed71706b0ac52f4b23bb697c64cb2c3471a93c17b4596647f8`. The original acquisition manifest is preserved byte-identically as `source_download_manifest.json`; `verification_receipt.json` also embeds its original fields. Source-row numbers, missing markers, identities and percentages remain recoverable in `source_records.csv` and `source_dictionary.csv`. The source workbook and shared data are unmodified.

```sh
shasum -a 256 -c analysis/primary_secondary/moisture_process/SHA256SUMS
uv run --no-project --python 3.12 --with numpy --with pandas --with scipy --with openpyxl --with matplotlib python -B analysis/primary_secondary/moisture_process/analyze_moisture.py
uv run --no-project --python 3.12 --with numpy --with pandas --with scipy --with openpyxl python -B analysis/primary_secondary/moisture_process/verify_moisture.py
```

The checksum manifest describes the delivered snapshot. Regeneration produces new receipt timestamps and can alter figure-file metadata without altering the numerical results. The independent verifier checks all 3,456 source records, all 21 model-partition prediction/metric combinations and all 1,152 source-label onset intervals. An excluded-target perturbation check replaces every series-3 and day-20 response while leaving all fitted parameters and training-only shape-selection scores unchanged.
