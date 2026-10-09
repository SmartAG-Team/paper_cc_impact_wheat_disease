**Priority 3 physiological evidence package — 2026-10-09**

No newly acquired dataset supports physiological fitting. Start with [COMPATIBILITY_REPORT.md](COMPATIBILITY_REPORT.md) and [source_inventory.csv](source_inventory.csv). The bounded acquisition audit separates metadata, source decisions and local literature caches. Downloaded documents and metadata are not raw field observations. Third-party article and report copies in `documentation/` remain local and are excluded from the public Git release; their source URLs, retrieval status and hashes remain in the acquisition records. The file manifest is the acquisition-time snapshot, including those local cache entries; it is not a current public-package checksum list. Current public-file hashes are recorded in the root `repository_manifest.json`.

| Artifact | Use |
| --- | --- |
| `source_inventory.csv` | Sixteen source decisions with exact URLs, units, license/access status and concrete missing inputs |
| `retrieval_log.jsonl`, `acquisition_summary.json` | Successful and failed retrievals, timestamps, file hashes and acquisition totals |
| `repository_file_inventory.csv` | Complete inspected GitHub trees and STICS deposit file listing |
| `api_search_coverage.csv`, `web_search_log.csv`, `web_evidence.json` | Search scope, returned/total results and source-located findings |
| `local_archive_review.csv`, `existing_rejection_audit.json` | In-place review of prior sources; no duplicated numerical acquisitions |
| `templates/`, `schema.json`, `data_dictionary.csv` | Empty CSV intake schema with keys, required fields, units and source locators |
| `input_audit.json` | Current no-observations result; no successful scientific validation implied |
| `comparison_protocol.json`, `parent_handoff.json` | Eligibility and model-comparison conditions; current decision against fitting |
| `verification.json`, `file_manifest.csv` | Acquisition-time checks and original file SHA-256 snapshot |

Run the following from this directory with Python 3; no package installation or network access is required:

```sh
python3 audit_existing_sources.py
python3 audit_inputs.py
```

The first command reads the original archive/follow-up locations from the local filesystem and refreshes the rejection audit. The second reads the header-only templates and returns exit code **2**, meaning **no observations**, not a passed join audit. For a populated intake directory, use `python3 audit_inputs.py PATH_TO_INPUT_DIRECTORY`; output remains in this package. Exit 1 indicates structural errors; exit 0 means only that the implemented structural checks passed. Scientific readiness always requires separate review of windows, measurement support, joins, references, environmental independence and model identifiability. Weather-series coverage and temporal canopy–disease alignment are specified in the protocol and are not fully automated by the structural audit.

All twelve tables are empty. `biomass_reserves.csv` is optional unless the intended model requires those constraints. Leaf rank 1 denotes the final flag leaf. Canopy area is expressed per ground area; unknown functional area remains blank. Preserve source values and source locators, document grain moisture, and retain natural senescence and distinct diseases separately. The sources table describes observation-bearing field datasets; externally sourced forcing requires separate provenance and scientific review. Soil and initial-state files for a particular crop model are additional inputs.

`build_inventory.py` regenerates the curated inventory from its recorded decisions and downloaded repository metadata. `build_templates.py` regenerates the empty schema and refuses to overwrite populated template CSVs. `acquire.py` is a bounded, logged downloader; rerunning it is unnecessary for this completed search. The downloaded Bancal manuscript retains the filename `documentation/bancal2022_author_manuscript.response`; its contents are a PDF, not a data table.

The original ZIP and follow-up directory are read-only inputs. The literal `/data/STB_paper_followup_20261009/` was absent; the inspected follow-up directory is `/Users/gangzhao/Documents/workspace/paper_cc_impact_wheat_disease/data/STB_paper_followup_20261009/`. The acquisition-time manifest remains unchanged as provenance. Subsequent public edits are covered by the root repository manifest, which excludes itself.
