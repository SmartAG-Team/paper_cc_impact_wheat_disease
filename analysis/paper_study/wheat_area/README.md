The conserved European wheat registry is `data/paper_study/wheat_area/europe_wheat_cells_025.parquet`, with a CSV companion. It contains 14,941 positive-wheat cells and 54.405 million reference-year-2020 harvested hectares. Physical and harvested area, and irrigated and rainfed systems, remain separate. Winter/spring area allocation is unresolved.

The fixed target grid is EPSG:4326, transform `[0.25, 0, -180, 0, -0.25, 90]`. Cell IDs are `g025_r###_c####`; integer rows and columns use north/west origins. Latitude/longitude fields are exact target centres. Native SPAM grid identities determine the 3×3 footprint aggregation. Climate point sampling must use these declared centres and cell IDs.

Principal files:

| File | Meaning |
|---|---|
| `europe_wheat_cells_025.parquet` / `.csv` | Positive wheat cells, bounds/centres, six area bands, weights and total-source validity counts |
| `europe_source_wheat_pixels.parquet` | Included native pixel IDs, source administrative ownership, source values/validity and geographical fractions |
| `europe_cell_country_wheat_areas.parquet` | Country partitions within target cells, preserving boundary cells with multiple countries |
| `europe_country_wheat_area.csv` | Country totals under the declared geographical boundary |
| `europe_wheat_calendar_scenarios.parquet` / `.csv.gz` | Four winter/spring × rainfed/irrigated timing scenarios; no duplicated area weights |
| `trial_point_calendar_scenarios.csv` / `.parquet` | BASF and Corteva point calendar scenarios; planting/maturity DOY, missing flags and native-cell distance |
| `mirca_wheat_subcrop_country_totals.csv` | Supplemental MIRCA-OS Wheat1/Wheat2 provider-season areas; not matching SPAM type weights |
| `europe_operational_domain.geojson` | Supplementary cartographic geometry; ordinary-country area inclusion uses SPAM source ownership |

Source and license evidence is in `source_catalog.json`, `source_download_spec.json` and provider metadata under the data directory. SPAM2020 v2r2 is archived with IFPRI Terms of Use; section 4 supplies the default CC-BY-4.0 license unless a non-CC license is stated. Its structured Dataverse license field is null. GGCMI calendars and MIRCA-OS v2 are CC-BY-4.0. Natural Earth is public domain. The mirrored SPAM readme differs from the Dataverse readme checksum and is explicitly retained as a provider-mirror variant; the three SPAM data archives match repository checksums. Dataverse's stale citation text is retained in the raw metadata, while the delivered methods identify the actual version and release date.

Reproduction from archived source bytes:

```sh
uv run --no-project --python 3.12 --with numpy --with pandas --with pyarrow --with rasterio --with geopandas --with shapely --with pyproj --with xarray --with netcdf4 --with matplotlib python -B analysis/paper_study/wheat_area/run_all.py
```

`reproduce_sources.py` checks all archived downloads; `--download` acquires missing files only from the recorded advertised public URLs, with exact-byte verification and no silent replacement of changed source versions. `--extract` regenerates the wheat-only archive extracts and supplemental tables. Dynamic provider metadata can change independently of dataset bytes; a changed metadata snapshot fails explicitly. The existing Natural Earth country archive at `data/geography/ne_110m_admin_0_countries.zip` is a read-only dependency with prior acquisition provenance in `analysis/era5/natural_earth_provenance.json`.

`build_registry.py` performs the spatial aggregation. `sample_point_calendars.py` reads only trial-coordinate metadata columns. `verify_registry.py` independently checks source files, geometry, areas and calendar assignment. `plot_wheat_area.py` creates the standalone PNG/PDF. `run_all.py` runs these stages sequentially and records the software versions. All generated writes are confined to the wheat-area directories.

Verification receipts are `aggregation_receipt.json`, `independent_verification.json`, `source_verification.json` and `point_calendar_receipt.json`. The independent audit passes 112 checks. The principal registry SHA256 is `380833fc139e7c76adc274384f7175438b202b217cdb4194c439f4ea69af6c92`. The scientific assumptions, source disagreements and continental-boundary limits are stated in `METHODS.md`; boundary pixel alternatives are tabulated in `boundary_sensitivity.csv`.

A separate source-based season-area sensitivity is available in `europe_wheat_cells_025_seasonal_sensitivity.parquet` and its CSV companion. It retains every original registry column and provides autumn-window (August–December), spring-window (February–June), ambiguous and unallocated hectare partitions. Total and irrigated/rainfed denominators are conserved separately. `europe_wheat_seasonal_calendar_area_sensitivity.parquet` retains the source subcrop and monthly calendar labels. Genetic winter/spring type remains unidentified. The main unallocated registry and its original receipts are unchanged. Methods and reproduction commands are in `SEASONAL_ALLOCATION_METHODS.md`; source-member CRC/SHA provenance is in `mirca_monthly_wheat_download_manifest.json`, and the independent allocation audit passes 57 checks.
