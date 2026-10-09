The additional public sources contain 948 unique disease–harvest plot pairs: 724 Swiss winter-wheat plots from five sites and ten site-years, and 224 French durum-wheat plots from one site-year. Neither cohort supplies a compatible observed functional-canopy time series or an identified STB-specific yield counterfactual. Actual grain-loss estimates and adaptation benefits require matched physiological, disease, phenological and harvest observations.

The French trait archive contains 179 monoculture profiles. Of these, 166 link by field-grid position and genotype to paired disease–harvest plots. Each profile has one LAI value and Heading/Maturity trait values. Measurement dates, the functional-green-area interpretation, leaf-rank coverage and calendar interpretation of the phenology values are unresolved in the retained compact archive. These observations support trait associations, not measured HAD. The Swiss cohort has 724 heading day-of-year values, but no dated disease or functional-canopy sequence. French disease was assessed near GS30 before subsequent fungicide protection; final harvest cannot identify untreated season-long disease damage.

The Swiss common-field match recovers 724 unique plot identifiers from the earlier release. All 19 measurement fields agree across versions. The full latest release contains 68 excess exact rows, with none in the paired subset. The 948-row common table is an index of the two cohorts and contributes no additional observations. The French raw archive has 1,592 subrows representing 398 plots, of which 379 have four nonmissing yield representations. All 224 paired French plots are complete. Adjacent pure-stand row values are mirrored in all 179 complete pure stands; these are pooled-sample representations. The author column `RAW_GY` contains spatial BLUPs and is retained separately from reconstructed raw harvest. Two of 226 scored French plots lack harvest yield.

Swiss source yield in dt/ha is multiplied by 0.1; French raw yield in g/m² is multiplied by 0.01. The existing package reports Swiss yields at 15% moisture and dried French samples with an unspecified moisture percentage. No cross-source moisture normalization is applied. No ordinal score is converted to percent leaf area. The Swiss README's scale and collection-date statements conflict with other retained source information. It also describes 350 seeds/m² and two-cultivar mixtures, whereas the paired records include 350/400 seeds/m² and 29 four-cultivar mixture plots. Original values remain intact.

The exploratory Swiss comparison uses categorical STB scores and management/cultivar covariates, fixed ridge regularization, fold-local preprocessing, and equal training weight per site-year. The primary test excludes each complete site; the secondary test excludes each site-year. The protocol was specified after source and coverage inspection and before fitting. It is a local exploratory protocol, not an externally preregistered analysis. Retrospective disease scores have no verified assessment date, so the predictions are not validated prospective forecasts.

Whole-site validation gives environment-weighted RMSEs of 1.142 t/ha for the training-mean baseline, 0.926 t/ha for STB alone, 0.914 t/ha for management/cultivar covariates, and 0.877 t/ha when STB is added to those covariates. The incremental squared-error reduction is 7.88%, with improvement at four of five sites. The mean site-level MSE difference is −0.0658 (t/ha)²; the descriptive five-site bootstrap stability interval is −0.2371 to 0.1039 (t/ha)². The interval includes zero and is conditional on the fitted folds, rather than a confirmatory confidence interval. With recorded co-diseases included, RMSE changes from 0.951 to 0.932 t/ha when STB is added; three sites improve. Only two seasons are represented. Cultivars repeat across sites, density is fixed by site, three site-years have only zero STB scores, and 26 disease-scored plots lack harvest. These results do not establish causal disease damage, independent future-climate transfer or adaptation benefit.

The existing Tunisia, German BRIWECS and Nordic benchmark receipts are frozen in the compact bundle. Their validations are not repeated. The prior assessment table has 423 ratings but 272 harvest keys; 151 additional rating rows repeat a harvest. Previous eligibility rules, including Nordic aliases and post-harvest ratings, remain governed by those retained receipts. The existing crop-yield implementation is a conditional HAD proxy; none of these supplementary regressions validates physiological grain production.

The Source Data and optional Supplementary Table S24 inputs are:

- `source_profile.csv`, `column_profile.csv`, `source_readiness.csv`: source dimensions, completeness, measurement grain and physiological compatibility.
- `evidence_gate.csv`: supportable disease, HAD, grain-yield, grain-loss, tonnes and adaptation quantities, with missing observations and claim boundaries. `claim_ready=true` applies only to the explicitly restricted supportable scope; it never certifies end-to-end physiology.
- `swiss_model_comparison.csv`, `site_paired_errors.csv`, `site_bootstrap_stability.csv`: grouped metrics and uncertainty limitations.
- `heldout_predictions.csv`, `fold_diagnostics.csv`, `unseen_categories.csv`, `environment_errors.csv`: all predictions and leakage/coverage diagnostics. The 8,688 prediction rows represent repeated model/split evaluations of 724 plots, not additional observations.
- `plot_provenance.csv`, `source_files.csv`, `identical_file_copies.csv`, `bundle_member_hashes.csv`, `duplicate_checks.csv`, `audit_checks.csv`: record links, source hashes and reconstruction checks.
- `audit_receipt.json`, `prediction_receipt.json`, `protocol.json`: machine-readable scope, checks, protocol timing, runtime and output hashes.
- `input_contract.json`, `validate_contract.py`: future-input structural contract and validator. A passing screen does not establish mechanistic validation or causal grain loss; source-method, sampling, independent-prediction and intervention-design review remain necessary.

Primary dataset citations are Swiss [Zenodo 10.5281/zenodo.17432866](https://doi.org/10.5281/zenodo.17432866), with [10.5281/zenodo.10209176](https://doi.org/10.5281/zenodo.10209176) for identifier recovery, and French [Zenodo 10.5281/zenodo.5393959](https://doi.org/10.5281/zenodo.5393959). All three retained records specify CC BY 4.0. Full creators, titles, publication dates and licence fields are in `dataset_citations.csv` and the retained Zenodo metadata. The Swiss release's metadata publication date and record-creation date are distinct fields and are not conflated.

The Swiss associated paper is in Crop Science, DOI `10.1002/csc2.21151`. Its journal identity is retained as provenance only; eligibility under the strict top-disciplinary-journal rule is not established, and the article is not added as scholarly evidence. The French associated paper is Montazeaud et al. (2022), New Phytologist 233, 2573–2584, [DOI 10.1111/nph.17915](https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.17915); its publisher data-availability statement links the retained Zenodo record. `journal_provenance.json` records DOI-registry verification. No MDPI or Frontiers sources are added.

The 268,702-byte `public_source_subset.zip` contains exact compact public-source files, repository metadata, existing extraction code, selected original French R scripts, attribution, member checksums and frozen prior-benchmark receipts. The French phenotype files are byte-identical to members of the original 30,506,448-byte source archive, whose retained repository MD5, acquisition SHA-256 and ZIP integrity were checked. The complete original archive is unnecessary for audit or prediction reproduction. The standalone scripts require only Python, NumPy and pandas; tests additionally require pytest. The local `.gitignore` exposes source data and CSV/JSON outputs for subsequent version control. No staging or commit is performed.

From the repository root:

```sh
.venv/bin/python -B analysis/paper_study/yield_transfer_audit_20261009/audit.py
.venv/bin/python -B analysis/paper_study/yield_transfer_audit_20261009/prediction.py
.venv/bin/python -B -m pytest -q -p no:cacheprovider analysis/paper_study/yield_transfer_audit_20261009/test_audit.py
```

`verify.py` repeats the audit, comparison and tests, then reproduces all numerical artifacts from an isolated copy containing only the compact bundle and local scripts/configuration. Verification records 79 passing source checks, 16 passing tests, and 23 byte-identical standalone output files in `verification_receipt.json`.

```sh
.venv/bin/python -B analysis/paper_study/yield_transfer_audit_20261009/verify.py
```

For a future independently observed input package:

```sh
.venv/bin/python -B analysis/paper_study/yield_transfer_audit_20261009/validate_contract.py /path/to/observed_inputs
```

The validator emits JSON to stdout and returns status 2 for missing, incompatible or incomplete inputs. Required tables are `plots.csv`, `canopy.csv`, `weather.csv` and `management.csv`. Three dates per upper-leaf rank bracketing grain filling are a structural minimum, not a sampling-adequacy guarantee. Synthetic values occur only inside software test fixtures and never enter the evidence or prediction tables.

The smallest useful next evidence step is recovery of French trait measurement methods/dates and Swiss disease dates and scale definitions, followed by acquisition of matched repeated functional-canopy observations, measured phenology, weather and harvest in independent environments. Grain-loss and feasible-adaptation claims additionally require comparable intervention/reference plots, disease observations in both arms, other-stress information and implementation constraints. The current supplementary prediction evidence is suitable for an explicitly exploratory SI comparison; it does not supply those missing observations.
