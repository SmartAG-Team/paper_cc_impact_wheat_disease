# Fixed-baseline wheat production exposure

SPAM2020 V2r2 production supplies a fixed exposure measure in metric tonnes. The provider defines production as harvested area multiplied by yield ([SPAM methodology](https://www.mapspam.info/methodology/)); this analysis reads production directly, without applying a yield-unit conversion. The archived source is [Harvard Dataverse, DOI 10.7910/DVN/SWPENT](https://doi.org/10.7910/DVN/SWPENT), dataset version 6. Its archive size and MD5, downloaded SHA-256, extracted raster SHA-256 and all 111,171 included native pixel values are checked. Native pixel production is multiplied once by the recorded European fraction. Native, grid and country contributions conserve mass. Total production is authoritative: small discrepancies between total and irrigated-plus-rainfed rasters are recorded without rescaling.

The fixed operational European domain contains 14,941 cells and 241,844,854.56171423 tonnes. Source-country contributions are summed within each current cell and retain source administrative membership. A cell intersecting multiple countries contributes only its respective production to each country. Countries crossing the operational European boundary therefore represent their included portions. Environmental regions inherit the current coarse-cell classification, including `Unassigned` and `Outside EEA regions`. The data describe all wheat; winter/spring production is unresolved.

The climate input is the current full-grid `GS65_85_lost_had3` result. HAD is normalized top-three-leaf healthy-area duration lost during GS65–85, expressed in normalized canopy-area days. Positive changes indicate greater simulated canopy damage. The historical reference is 1991–2020; comparisons are 2031–2060 and 2071–2100 under SSP126, SSP245 and SSP585. Year pairs match positions within each thirty-year period. A pair is valid only when both metric values are finite. The supplied cell change is the mean future-minus-reference difference over those pairs. The paired counts, differences, current grid identities and supplied equal-three-GCM ensemble results are checked before aggregation.

For baseline cell production \(P_i\), model-specific valid-pair count \(n_{im}\), and cell paired-mean change \(d_{im}\), the domain estimator for GCM \(m\) is

\[
D_m=\frac{\sum_i P_i(n_{im}/30)d_{im}}{\sum_i P_i(n_{im}/30)}.
\]

The ensemble domain estimate is \((D_1+D_2+D_3)/3\). The reported range is the minimum and maximum of those three domain estimates, not a confidence interval. Primary continuous means use each model's own metric-valid support. Neither pooling all model-year observations nor production-weighting an equal-GCM cell mean gives this estimator when missing-year counts differ. Historical and future weighted values accompany the changes. A zero denominator produces an unavailable mean, not zero damage.

Cell classification is separate from domain averaging. A complete cell requires a finite change in all three GCMs; its ensemble direction uses their equally weighted mean. `all3positive` requires each change to be strictly positive; `all3negative` requires each to be strictly negative. `mixed_sign` requires at least one positive and one negative change. Exact zeros are explicit: all-three-zero, nonnegative-with-zero, and nonpositive-with-zero are separate in `exposure_categories.csv`. The four plot categories are `all3increase`, `mixed_or_zero`, `all3decrease`, and `unavailable`. Here `mixed_or_zero` is the sum of all nonunanimous or zero-containing cases. No sign tolerance is applied.

Exposed production is the sum of fixed \(P_i\) for cells in a class, without multiplying production by disease severity or valid-year fraction. ALL3-positive production and ensemble-positive production overlap and are not additive. Every share uses the entire fixed baseline production of that domain, including unavailable cells. Production tonnes quantify where simulated changes occur; they do not estimate tonnes lost, saved or produced in the future.

The `robust_all3_ge27` sensitivity retains only cells with at least 27 of 30 metric-valid pairs in every GCM. The same cells enter all three continuous estimates, with the original model-specific \(n_{im}/30\) weights retained. Excluded cells count as unavailable in the fixed-baseline classification denominator. This is a count-based coverage filter: the summarized data cannot establish whether the same calendar-pair identities are valid across models.

Regional priorities are ordered by ALL3-positive exposed production within each scenario, period and analysis mode. Absolute exposure, baseline share, model disagreement and availability remain separate dimensions. These quantities support surveillance and field-validation prioritization and scenario-contingent planning; they do not identify adaptation efficacy, economic vulnerability or a composite priority score. No recommendations for particular interventions are inferred. The GCM range excludes structural-model, management and production-map uncertainty.

`table1.csv` contains late-century results for Europe and eight named environmental regions. Europe includes the two residual geographical categories. Consequently, the eight regional baseline values sum to less than Europe by the production in `Unassigned` and `Outside EEA regions`. Full summaries preserve both categories.

## Callable API

```python
from analysis.paper_study.food_security_exposure_20261009 import read, source_paths

tables = read()  # dict[str, pandas.DataFrame]; receipt/output hashes checked
primary = tables['domain_summary'].query("analysis_mode == 'primary_any_paired_year'")
regional = primary.query("domain_type == 'environment_region'")
csv_paths = source_paths()  # verified absolute Paths, including compact CSV inputs
one_table = read('region_summary')
```

`domain_summary` includes all geographical levels (`Europe`, `environment_region`, `country`), six scenario-period combinations and two analysis modes. It has 624 rows. The compact region and country tables contain 120 and 492 rows. Their values are identical to the corresponding columns in `domain_summary`. The GCM table contains 1,872 rows. `read(verify=False)` explicitly disables hash checks; normal integration uses the default. Hashes establish local integrity against the receipt, not digital authenticity. `source_paths()` includes compressed CSV input paths; pandas reads them directly.

## Reproduction

Commands run from the repository root with its existing environment:

```sh
.venv/bin/python -B -m analysis.paper_study.food_security_exposure_20261009.run
.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/paper_study/food_security_exposure_20261009/test_exposure.py
.venv/bin/python -B -m analysis.paper_study.food_security_exposure_20261009.run --bundled --destination /tmp/exposure-reproduction
```

The original-source command requires the local full-grid and SPAM inputs. The bundled command uses only files in this namespace: `production_grid_input.csv`, `country_fraction_weights.csv`, metric-only `paired_changes_input.csv.gz`, `native_source_fixture.csv` and their receipts. It requires pandas, NumPy and PyArrow; only original-TIFF verification requires Pillow. The destination must be empty. Generated CSVs and classifications reproduce the scientific tables without the giant production archive or TIFFs.

The native fixture contains complete selected coarse cells, including country borders, partial Europe fractions and zero-production cells. It supports small numerical examples and source reconstruction tests, not a continental production total. Country fractional weights retain both tonnes and the production fraction of the cell. The fraction is undefined for a zero-production cell; `positive_cell_production` distinguishes that case and tonnes remain zero. Baseline harvested and physical areas are independently reconstructed from native source fractions.

`receipt.json` records source and result hashes, units, availability rules, inference limits and software versions. `bundle_receipt.json` records compact-input hashes. The publication builder reads these verified outputs through the local API.
