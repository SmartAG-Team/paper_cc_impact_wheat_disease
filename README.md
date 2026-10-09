# Climate impacts on European wheat disease and production

Research code, numerical evidence and submission files for **Wheat phenology partly offsets projected increases in Septoria damage across Europe**, by Gang Zhao, Northwest A&F University.

The study examines climate and weather effects on crop development, Septoria-related canopy damage and conditional yield responses across European wheat-growing land-use grids. The completed experiment contains 810 annual jobs, 14,941 reference wheat cells and 14,932 cells with eligible winter-wheat calendars. It combines three climate models, SSP1–2.6, SSP2–4.5 and SSP5–8.5, and the periods 1991–2020, 2031–2060 and 2071–2100.

## Current manuscript and evidence

- [Manuscript Word](publication/european_wheat_stb/Manuscript.docx) and [PDF](publication/european_wheat_stb/Manuscript.pdf)
- [Supplementary Information Word](publication/european_wheat_stb/Supplementary_Information.docx) and [PDF](publication/european_wheat_stb/Supplementary_Information.pdf)
- [Source Data workbook](publication/european_wheat_stb/Source_Data.xlsx)
- [Supplementary model specification](publication/european_wheat_stb/Supplementary_Model_Specification.pdf)
- [Numerical verification receipt](publication/european_wheat_stb/verification_receipt.json)

The main results cover grid-scale climate impacts, environmental-region and country contrasts, disease occurrence and functional canopy damage, and conditional disease-related yield responses. Field evaluation records are pooled in manuscript statistics; original source identifiers remain in the numerical evidence for traceability. Model evaluation limitations and the conditional nature of the canopy–yield conversion are retained in the manuscript. Wheat phenology describes the timing of leaf appearance and unfolding, flowering and soft dough; it supplies leaf availability and seasonal timing to the disease model. Disease damage is mapped to assumed canopy-function loss, followed by a canopy-based estimate of the disease-related yield component. Disease-induced HAD loss integrates assumed functional green-leaf-area loss over time. Normalized HAD loss expresses this quantity as equivalent days of reference canopy function lost; measured lesion percentage and total grain yield are distinct quantities. Disease damage does not feed back into crop-development rates.

## Repository contents

| Path | Contents |
| --- | --- |
| `model/`, `process_model/` | Simulation runtime and process components |
| `calibration/`, `configurations/` | Calibration utilities and frozen model configurations |
| `examples/wheat_stb/` | Synthetic weather and runnable model examples |
| `tests/` | Existing numerical and behavioral checks |
| `analysis/paper_study/full_grid_climate_20261008/` | Full-grid calculation scripts, configuration, completed summaries and provenance |
| `analysis/paper_study/nature_food_impact_20261008/` | Current publication builder, pooled evaluation and verification |
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

The code, source-data workbook and selected numerical evidence are deposited in the private manuscript repository at https://github.com/SmartAG-Team/paper_cc_impact_wheat_disease. Complete raw climate forcing and annual model outputs are not hosted in this repository; their provenance and rerun requirements are recorded in the evidence files. Third-party data, software and references retain their original provenance and applicable terms. No new open-source or data-redistribution licence is assigned by this repository.

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
