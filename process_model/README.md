# T–P–V winter-wheat phenology process model

The runtime is the schema-2, known-sowing implementation from `/Users/gangzhao/Documents/workspace/AGC-Transformer/process_model/calibrate.py`. T, P and V denote thermal development, photoperiod and vernalization. The copied runtime retains every original function body. Runtime adaptations concern support imports only. `SOURCE_PROVENANCE.json` records source and destination SHA-256 hashes, and `reference/import_adaptations.patch` records the exact changes.

## Execution

From the project root, with NumPy and pandas installed:

```sh
python -m process_model.run_tpv \
  --weather analysis/phenology/example_source_station4/weather.csv \
  --sowing analysis/phenology/example_source_station4/sowing_records.csv \
  --output analysis/phenology/example_rerun
```

An existing local Python environment is available at `/Users/gangzhao/Documents/workspace/agri_water_management/.venv/bin/python`. The dependency list is `process_model/requirements.txt`. Output directories must be new.

Daily weather requires the following columns:

| Column | Definition and units |
| --- | --- |
| `PEP_ID` | Numeric station or explicitly assigned simulation-case identifier. |
| `DATE` | Daily calendar date; station/date keys must be unique. |
| `LAT` | Latitude in degrees, within −90 to 90. |
| `t_mean` | Daily mean air temperature, °C. |
| `t_max` | Daily maximum air temperature, °C. |
| `GDD` | Supplied daily effective thermal development, °C d. |

Management records require `PEP_ID`, `SOWING_DATE` and `SOWING_KNOWN_AT`. The availability date must be on or before the sowing date. A sowing record applies from its occurrence onward; later records do not change earlier daily states. Rows preceding the first supplied sowing retain missing process state. Each new sowing resets thermal and chilling accumulations.

The original process core accepts supplied `GDD`; it does not generate this field. Explicit forcing options are available when supplied GDD is absent:

```sh
python -m process_model.run_tpv \
  --weather canonical_weather.csv --sowing explicit_sowing.csv \
  --gdd-convention archived_temperature_response \
  --output analysis/phenology/new_simulation
```

`archived_temperature_response` is reconstructed from the source forcing table: GDD is zero below 0°C, equals mean temperature between 0 and 20°C, remains 20 between 20 and 30°C, and equals `20 − 2 × (t_mean − 30)` above 30°C. This relationship matches all 11,862,724 archived rows within 1e−10. The declining branch is directly represented by only 48 rows between 30.007650 and 30.867042°C. Warmer forcing extrapolates that slope; means above 40°C require supplied GDD. The original forcing-generator code is unavailable. The separate `clipped_mean_0_20` option omits the declining branch and therefore differs from those 48 source rows. The complete numerical audit is `analysis/phenology/source_gdd_forcing_audit.json`.

## Daily development equations

With latitude φ in radians and day of year n, the source daylength calculation is:

```text
δ = 0.4093 × sin[2π × (n − 81) / 365]
L = (24 / π) × arccos{clip[−tan(φ) × tan(δ), −1, 1]}
```

Daylength L has units of hours. The calculation retains the source's fixed 365-day denominator in leap years.

The thermal accumulation is `D_T(i) = Σ GDD(i)` within each supplied sowing cycle. Photoperiod adjustment is active when the previous day's `D_T` is at least 142.01 and below 1426.78°C d:

```text
f_P(i) = clip[1 − 0.09 × (16 − L(i)), 0, 1] during that interval
f_P(i) = 1 outside that interval
ΔD_TP(i) = GDD(i) × f_P(i)
```

Vernalization accounting starts when the previous day's `D_TP` reaches 142.01°C d. Its temperature response q is piecewise linear through `[-4, 0, 10, 16]°C → [0, 1, 1, 0]`, with zero response outside that temperature interval. While previous-day `D_TP` remains below 582.80°C d, `VERDAY = q`; subsequently `VERDAY = 0`. One day at full response contributes one equivalent vernalization day to `CUMVER`.

For an active day, the candidate chilling accumulation is `C* = C(i−1) + VERDAY(i)`. If `C* < 10` and daily maximum temperature exceeds 30°C, accumulation instead becomes `max[0, C(i−1) − 0.5 × (t_max − 30)]`. Otherwise accumulation is `C*`. The growth modifier is `f_V = 0.3 + 0.7 × min(CUMVER / 40, 1)` during the sensitive interval, and 1 outside it. The final daily development increment is:

```text
ΔD_TPV(i) = GDD(i) × f_P(i) × f_V(i)
```

The source retains vernalization-accounting activity after the sensitivity cutoff, while growth sensitivity has ended. No observed BBCH or event date controls either gate. Solar radiation, humidity, precipitation, soil conditions and stress responses are absent from this process core.

## Frozen parameters and outputs

The parameter file `parameters/calibration.json` contains pooled training-event medians from 187 German winter-wheat stations in the source's frozen partition. The source archive spans 1985–2015. These thresholds are empirical population references; the configuration contains no cultivar-specific parameter identity.

| BBCH | Stage interpretation | T threshold | T–P threshold | T–P–V threshold |
| --- | --- | ---: | ---: | ---: |
| 0 | Supplied sowing anchor | 11.12 | 11.12 | 11.12 |
| 10 | First leaf emerged | 142.01 | 142.01 | 142.01 |
| 31 | First node detectable | 962.43 | 582.80 | 542.30 |
| 51 | Beginning of heading | 1426.78 | 1025.05 | 986.65 |
| 85 | Soft dough | 2234.72 | 1809.45 | 1760.39 |

Thresholds have units of effective °C d. Daily stage output is the highest attained threshold among BBCH 0, 10, 31, 51 and 85. Stage zero also covers anchored pre-emergence days below the first threshold. Sowing is a supplied management event, not an independently predicted date. BBCH 85 represents soft dough, rather than physiological maturity or harvest. BBCH 39, anthesis and intermediate leaf stages are not outputs of this model.

Each run produces `daily_features.csv`, `daily_stages.csv`, `event_dates.csv`, a copy of the parameter file and `run_manifest.json`. Event dates use the first daily threshold attainment without interpolation. Incomplete daily coverage from sowing suppresses exact event dates. Observed stage labels in a weather file do not enter the command wrapper's simulation or event extraction.

## Numerical reproduction and external transfer

The retained source example contains 362 daily rows for source station 4, from 1991-10-05 to 1992-09-30, with its archived sowing record. Source and copied executions produce byte-identical daily features, daily stages and event-date CSVs; maximum numerical difference is zero. The verification receipt is `analysis/phenology/source_copy_verification.json`. The example establishes implementation reproduction and does not establish validation on the disease datasets.

```sh
python analysis/phenology/verify_tpv_copy.py \
  --source /Users/gangzhao/Documents/workspace/AGC-Transformer \
  --output analysis/phenology/new_copy_verification.json
python -m pytest -q
```

The external ERA5 transfer entry point is:

```sh
python analysis/phenology/run_field_transfer.py \
  --output analysis/phenology/new_field_transfer
```

Five field-season cases have published sowing dates: Lindau-Eschikon 2015-10-13 for the 2016 season ([Karisto et al., Phytopathology](https://doi.org/10.1094/PHYTO-04-17-0163-R)); Thiverval-Grignon 2017-10-17 and 2018-11-15 for the 2018 and 2019 seasons ([Orellana-Torrejon et al., Plant Pathology](https://doi.org/10.1111/ppa.13458)); and Kodia 2017-12-27 and 2018-11-08 for the 2018 and 2019 seasons ([Plant Pathology field-mixture experiment](https://doi.org/10.1111/ppa.13247)). The case table preserves these sources and an explicit known-at-sowing availability assumption. The Swiss weather link shares the same independently identified Lindau-Eschikon site and 2016 season with the retained ETHZ archive.

The current transfer output is `analysis/phenology/field_transfer_era5/`. It contains 1,478 daily forcing rows across the five complete sowing-to-record-end series. Parameters remain frozen. Cultivar, region and wheat-type transfer remain unvalidated, including transfer from German winter wheat to Tunisian durum wheat. Thiverval-Grignon weather uses a representative station coordinate rather than recovered experimental-plot coordinates. Realized ERA5 weather supports retrospective simulations; operational forecasts require weather forecasts or explicit future-weather scenarios.

The BASF disease archive lacks sowing dates and cultivar identities. It therefore has no defensible field-specific T–P–V phenology simulation under the supplied-management contract. Disease observation dates and BBCH records do not substitute for sowing or cultivar calibration.

## Provenance and verification scope

The source runtime and its support utilities were untracked working-tree files. Source Git HEAD alone does not identify their contents; file hashes provide the copied-source identity. Historical source README and forecast-input documentation remain unchanged in `reference/` and retain their original dates and archived context. The 20 transplanted process checks exclude two unrelated auxiliary-regressor tests; nine local checks cover command forcing and field-transfer behavior. NumPy 2.5.3 and pandas 3.0.6 were used for numerical reproduction. The unchanged source sowing helper emits a NumPy deprecation warning in the test environment.
