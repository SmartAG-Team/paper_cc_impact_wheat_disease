# Phenology reproduction and field-transfer artifacts

The copied model and equation/input documentation are in [`../../process_model/README.md`](../../process_model/README.md). The model contains frozen empirical German winter-wheat parameters and discrete BBCH 0, 10, 31, 51 and 85 outputs. BBCH 85 denotes soft dough; intermediate leaf stages and anthesis are unavailable.

| Artifact | Evidence and interpretation |
| --- | --- |
| `source_copy_verification.json` | Source-versus-copy comparison on an existing 362-day station-season; daily features, daily stages and event CSVs are byte-identical, with zero numerical difference. |
| `example_source_station4/` | Archived weather, supplied sowing, observed-event reference, source provenance and executable copied-model outputs for station 4, 1991–1992. |
| `source_gdd_forcing_audit.json` | Audit of all 11,862,724 original daily forcing rows; the reconstructed temperature response matches within 1e−10. |
| `source_gdd_forcing_exceptions.csv` | The 48 hot days that differ from simple 0–20°C clipping and establish the source's declining response above 30°C. |
| `field_transfer_sowing_cases.csv` | Five published sowing cases, with site, season, wheat type, DOI and the explicit availability assumption. |
| `field_transfer_era5/case_metadata.csv` | Exact forcing location links, sowing dates, coordinate provenance and daily-coverage boundaries. |
| `field_transfer_era5/simulation/daily_features.csv` | 1,478 daily thermal, photoperiod and vernalization trajectories. |
| `field_transfer_era5/transfer_event_dates.csv` | Illustrative BBCH event dates from frozen parameters; no observed-validation statistics. |
| `field_transfer_era5/transfer_manifest.json` | Input and output hashes, transfer status and limitations. |

Source reproduction establishes implementation equivalence. External field outputs represent uncalibrated parameter transfer. Published sowing dates supply retrospective management anchors; independent known-at-sowing timestamps are unavailable. Cultivar effects, regional adaptation and the winter-wheat-to-durum transfer remain unvalidated. The French forcing coordinate represents the station; exact experimental-plot coordinates are unavailable. The reconstructed thermal decline is extrapolated for daily means above the source maximum of 30.867042°C. The BASF archive has no sowing or cultivar records and consequently has no field-specific phenology output in these artifacts.
