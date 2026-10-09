# Climate impacts on European wheat disease and production

Research code, numerical evidence and submission files for **Wheat development partly offsets projected increases in Septoria damage across Europe**, by Gang Zhao, Northwest A&F University.

The study examines climate and weather effects on crop development, Septoria-related canopy damage and conditional yield responses across European wheat-growing land-use grids. The completed experiment contains 810 annual jobs, 14,941 reference wheat cells and 14,932 cells with eligible winter-wheat calendars. It combines three climate models, SSP1–2.6, SSP2–4.5 and SSP5–8.5, and the periods 1991–2020, 2031–2060 and 2071–2100.

## Current manuscript and evidence

- [Manuscript Word](publication/european_wheat_stb/Manuscript.docx) and [PDF](publication/european_wheat_stb/Manuscript.pdf)
- [Supplementary Information Word](publication/european_wheat_stb/Supplementary_Information.docx) and [PDF](publication/european_wheat_stb/Supplementary_Information.pdf)
- [Source Data workbook](publication/european_wheat_stb/Source_Data.xlsx)
- [Software and Evidence archive](publication/european_wheat_stb/Software_and_Evidence.zip)
- [Supplementary model specification](publication/european_wheat_stb/Supplementary_Model_Specification.pdf)
- [Numerical verification receipt](publication/european_wheat_stb/verification_receipt.json)

The main results cover grid-scale climate impacts, environmental-region and country contrasts, disease occurrence and functional canopy damage, and conditional disease-related yield responses. Field evaluation records are pooled in manuscript statistics; original source identifiers remain in the numerical evidence for traceability. Model evaluation limitations and the conditional nature of the canopy–yield conversion are retained in the manuscript. Wheat development supplies leaf availability and seasonal timing to the disease model. Disease damage is mapped to assumed canopy-function loss, followed by a canopy-based estimate of the disease-related yield component. Normalized HAD loss is reported in days; measured lesion percentage and total grain yield are distinct quantities. Disease damage does not feed back into crop-development rates.

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

Selected numerical evidence follows the verified publication archive. Earlier analysis code is retained where it supports provenance and reproducibility. The current submission is the publication folder linked above. Local environments, redundant backups, raw climate forcing and the complete archive of full-grid annual outputs remain outside Git. Their source paths and checksums remain in the recorded acquisition and completion receipts. Selected archived diagnostic and example outputs remain available as publication evidence. `repository_manifest.json` records the archived evidence and tracked-file scope.

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

Node.js is required for the bundled citation renderer. The equation renderer has a separately locked npm dependency file under `analysis/paper_study/nature_food_fix_20261007/specification/.conversion_tools/`; its installed dependencies are also retained in the evidence archive.

## Regenerate and verify the manuscript

The output directory must be new.

```sh
.venv/bin/python -m analysis.paper_study.nature_food_impact_20261008.build \
  --output publication/european_wheat_stb_regenerated --documents-only
.venv/bin/python -m analysis.paper_study.nature_food_impact_20261008.verify \
  --output publication/european_wheat_stb_regenerated
```

Document regeneration uses bundled grid summaries and fixed evaluation predictions. Verification uses raw annual outputs when available and otherwise checks the bundled paired grid values and valid-year counts. A complete forcing-based rerun requires the externally stored NEX-GDDP-CMIP6v2 forcing, scenario-specific alignment files, land-use inputs and crop calendars identified in the configuration and provenance records. Numerical verification concerns implementation, calculations and reporting; it does not independently establish biological or absolute-yield predictive validity.

The daily model API and restart contract are documented in [WHEAT_STB_ENGINE.md](WHEAT_STB_ENGINE.md). A synthetic model example is described in [examples/wheat_stb/README.md](examples/wheat_stb/README.md).

## Access and third-party materials

The repository is private pending publication. Third-party data, software and references retain their original provenance and applicable terms. No new open-source or data-redistribution licence is assigned by this repository.
