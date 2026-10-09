# Climate impacts on European wheat disease and production

Research code, numerical evidence and submission files for **Wheat phenology partly offsets projected increases in Septoria damage across Europe**, by Gang Zhao, Northwest A&F University.

The study examines climate and weather effects on crop development, Septoria-related canopy damage and conditional yield responses across European wheat-growing land-use grids. The completed experiment contains 810 annual jobs, 14,941 reference wheat cells and 14,932 cells with eligible winter-wheat calendars. It combines three climate models, SSP1–2.6, SSP2–4.5 and SSP5–8.5, and the periods 1991–2020, 2031–2060 and 2071–2100.

The scientific development criteria are documented in [the Nature Food review and research plan](docs/Nature_Food_Review_and_Research_Plan_20261009.md). The current release remains a conditional canopy-impact and production-exposure study; physiological grain-loss and intervention-effect validation are unresolved.

## Current manuscript and evidence

- [Manuscript Word](publication/european_wheat_stb/Manuscript.docx) and [PDF](publication/european_wheat_stb/Manuscript.pdf)
- [Supplementary Information Word](publication/european_wheat_stb/Supplementary_Information.docx) and [PDF](publication/european_wheat_stb/Supplementary_Information.pdf)
- [Source Data workbook](publication/european_wheat_stb/Source_Data.xlsx)
- [Supplementary model specification](publication/european_wheat_stb/Supplementary_Model_Specification.pdf)
- [Numerical verification receipt](publication/european_wheat_stb/verification_receipt.json)

The main results distinguish predictive support, climate-driven canopy damage, fixed-baseline production exposure, and crop–disease timing. Production exposure means SPAM2020 tonnes located in cells with projected damage changes; it is not an estimate of tonnes lost. Conditional HAD-to-yield conversions remain supplementary. Field evaluation records are pooled in manuscript statistics; original source identifiers remain in the numerical evidence for traceability. Model evaluation limitations and the conditional nature of the canopy–yield conversion are retained in the manuscript. Wheat phenology describes the timing of leaf appearance and unfolding, flowering and soft dough; it supplies leaf availability and seasonal timing to the disease model. Disease damage is mapped to assumed canopy-function loss, followed by a canopy-based estimate of the disease-related yield component. Disease-induced HAD loss integrates assumed functional green-leaf-area loss over time. Normalized HAD loss expresses this quantity as equivalent days of reference canopy function lost; measured lesion percentage and total grain yield are distinct quantities. Disease damage does not feed back into crop-development rates.

## Repository contents

| Path | Contents |
| --- | --- |
| `model/`, `process_model/` | Simulation runtime and process components |
| `calibration/`, `configurations/` | Calibration utilities and frozen model configurations |
| `examples/wheat_stb/` | Synthetic weather and runnable model examples |
| `tests/` | Existing numerical and behavioral checks |
| `analysis/paper_study/full_grid_climate_20261008/` | Full-grid calculation scripts, configuration, completed summaries and provenance |
| `analysis/paper_study/nature_food_impact_20261008/` | Current publication builder, pooled evaluation and verification |
| `analysis/paper_study/food_security_exposure_20261009/` | Production exposure, fractional source-country weights and coverage sensitivity |
| `analysis/paper_study/yield_transfer_audit_20261009/` | Additional field-source audit and yield-validation requirements |
| `analysis/paper_study/climate_robustness_20261009/` | Year-pairing and complete-population sensitivity; annual canopy-damage distributions |
| `analysis/paper_study/nordic_archive_validation_20261009/` | Older Nordic/Baltic management-grain contrasts and grouped benchmarks |
| `analysis/paper_study/mixture_adaptation_20261009/` | French mixture grain comparisons and Swiss control-comparability diagnostics |
| `analysis/paper_study/physiological_evidence_20261009/` | Physiological-source compatibility inventory and empty observation-intake schema |
| `publication/european_wheat_stb/` | Current manuscript, supplementary material, figures and source data |

The repository contains the current manuscript, maintained code and supporting numerical evidence. Superseded manuscript drafts, duplicated code snapshots and unused analysis outputs have been removed from the active tree. Dated folders that remain contain dependencies of the current workflow; their names preserve recorded source paths and provenance. The current submission is the publication folder linked above. Local environments, redundant backups, raw climate forcing and the complete archive of full-grid annual outputs remain outside Git. Their source paths and checksums remain in the recorded acquisition and completion receipts. Selected archived diagnostic and example outputs remain available as publication evidence. `repository_manifest.json` records the evidence and tracked-file scope. The software-and-evidence submission bundle contains maintained code, numerical evidence and source provenance; the corresponding files also remain directly inspectable in the repository.

## Clone and dependencies

Git LFS is required for the evidence archive, workbooks and binary numerical tables.

```sh
git lfs install
git clone https://github.com/SmartAG-Team/paper_cc_impact_wheat_disease.git
cd paper_cc_impact_wheat_disease
git lfs pull
python3.12 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m pip install -r publication/european_wheat_stb/Publication_requirements.txt
```

Node.js is required for the bundled citation renderer. The supplementary model specification is supplied as a rendered document. Its equation renderer has a locked npm dependency file under `analysis/paper_study/nature_food_fix_20261007/specification/.conversion_tools/`; installed npm dependencies are excluded from the repository.

## Regenerate and verify the manuscript

The output directory must be new.

```sh
.venv/bin/python -m analysis.paper_study.nature_food_impact_20261008.build \
  --output publication/european_wheat_stb_regenerated --documents-only
.venv/bin/python -m analysis.paper_study.nature_food_impact_20261008.verify \
  --output publication/european_wheat_stb_regenerated
```

Omit `--documents-only` to generate an optional local `Software_and_Evidence.zip` submission bundle.

Document regeneration uses bundled grid summaries and fixed evaluation predictions. Verification uses raw annual outputs when available and otherwise checks the bundled paired grid values and valid-year counts. A complete forcing-based rerun requires the externally stored NEX-GDDP-CMIP6v2 forcing, scenario-specific alignment files, land-use inputs and crop calendars identified in the configuration and provenance records. Numerical verification concerns implementation, calculations and reporting; it does not independently establish biological or absolute-yield predictive validity.

The daily model API and restart contract are documented in [WHEAT_STB_ENGINE.md](WHEAT_STB_ENGINE.md). A synthetic model example is described in [examples/wheat_stb/README.md](examples/wheat_stb/README.md).

## Access and third-party materials

The code, source-data workbook and selected numerical evidence are available in the public research repository at https://github.com/SmartAG-Team/paper_cc_impact_wheat_disease. Complete raw climate forcing and annual model outputs are not hosted in this repository; their provenance and rerun requirements are recorded in the evidence files. Third-party data, software and references retain their original provenance and applicable terms. No new open-source or data-redistribution licence is assigned by this repository.

### Expanded field-yield evidence

The field-yield companion preserves crop type, leaf scope, severity definition,
moisture basis and plot versus treatment-mean grain. Tunisia supplies 82
unprotected plot-season yields with two leaf assessments and 40 protected
references in 2019; the Nordic extension supplies 62 treatment contrasts in
five trials after source-specific exclusions. German original-site data add
3,264 cultivar–management comparisons across 16 site-years at five locations.
Repeated assessments and shared controls remain grouped during validation.

- [Executed validation notebook](analysis/paper_study/crop_damage_validation_20261009/Field_yield_validation.ipynb)
- [Readable validation HTML](analysis/paper_study/crop_damage_validation_20261009/Field_yield_validation.html)
- [Expanded field analysis](analysis/paper_study/crop_damage_validation_20261009/)
- [Nordic trial analysis](analysis/paper_study/nordic_yield_validation_20261009/)

The new comparisons do not validate a universal severity-to-yield coefficient.
No protected severity is imputed, and endpoint percentages do not supply HAD
trajectories. Crop-growth and physiological yield-loss validation remain
separate from the frozen climate simulations. The source collection and
correspondence drafts are local evidence; no correspondence has been sent.

### Production exposure and additional yield evidence

[Production exposure](analysis/paper_study/food_security_exposure_20261009/METHODS.md)
uses fixed SPAM2020 production weights, climate-model agreement and explicit
availability denominators. Its compact input bundle reproduces the analysis
without the original production rasters:

```sh
.venv/bin/python -m analysis.paper_study.food_security_exposure_20261009.run \
  --bundled --destination /tmp/exposure-reproduction
.venv/bin/python analysis/paper_study/yield_transfer_audit_20261009/verify.py
```

The [additional field evidence](analysis/paper_study/yield_transfer_audit_20261009/README.md)
reconciles 948 Swiss/French plot harvests and supplies exploratory whole-site
yield prediction, source attribution and a structural input contract for
future physiological validation. The compact sources retain CC BY 4.0
attribution. Neither production exposure nor those regressions establish
STB-attributable tonnes lost or adaptation efficacy. The [milestone table](analysis/paper_study/nature_food_impact_20261008/derived/research_milestones.csv)
records which research-plan requirements remain unresolved.

### Annual climate-impact robustness

The [annual sensitivity analysis](analysis/paper_study/climate_robustness_20261009/METHODS.md)
preserves exact canopy-damage values from all 810 annual files in a compact NPZ
bundle. It evaluates all 30 cyclic year pairings, separate-period and common-cell
means, rainfed-production weights, and annual regional damage distributions.
Common cells are complete in every year and all three climate models.

```sh
.venv/bin/python -m analysis.paper_study.climate_robustness_20261009.run \
  --output /tmp/climate-sensitivity-reproduction
```

These outputs quantify sensitivity of the existing conditional climate results.
High-damage-year frequencies concern canopy function; harvested-yield downside
risk remains a separate validation target. Model ranges are not confidence
intervals or calibrated probabilities.

### Observed management-grain comparisons

The [Nordic/Baltic archive analysis](analysis/paper_study/nordic_archive_validation_20261009/README.md)
retains 307 grain contrasts from 263 reported trial identifiers, with shared
controls and grouped validation. The [mixture analysis](analysis/paper_study/mixture_adaptation_20261009/README.md)
compares 195 French mixtures with constituent pure stands and checks Swiss
control comparability. These are observed management outcomes; they do not
calibrate STB-specific loss or future climate-adaptation benefits.

```sh
.venv/bin/python analysis/paper_study/nordic_archive_validation_20261009/verify_reproduction.py
.venv/bin/python analysis/paper_study/mixture_adaptation_20261009/verify.py
```

The [physiological evidence inventory](analysis/paper_study/physiological_evidence_20261009/COMPATIBILITY_REPORT.md)
records a bounded 16-source screen, exact missing measurements and concrete
experimental leads. It contains no new qualifying physiological observations.
Empty intake templates and structural checks support future observed inputs;
they are not fitted crop-growth or disease-loss models.
