# Wheat STB daily model API

`model.wheat_stb` composes temperature–photoperiod–vernalization development, leaf appearance and unfolding, local overwintering source, imported pressure, canopy transmission and optional healthy-area-duration diagnostics. Runtime configuration contains explicit parameter values. Calibration, observations, fit selection and conversion of frozen fits to runtime cases remain outside `model/`.

The component interfaces follow the explicit state/rate separation described in the [PCSE reference guide](https://pcse.readthedocs.io/en/stable/reference_guide.html) and the composition and parameter metadata principles in [APSIM model design](https://docs.apsim.info/docs/development/software/modeldesign). These architectural references do not establish equivalent scientific scope, maturity or validation.

`WheatSTBSimulation(config, weather)` accepts an immutable `EngineConfig` and a consecutive `WeatherProvider` beginning at or before sowing. A zero-state run cannot omit earlier crop days. Its lifecycle is `initialize()`, repeated `calc_rates()` and `integrate(rates)` or `step()`, then `finalize()`. `run(days=None)` consumes available weather and preserves the same daily results as individual steps. Each rate proposal is bound to its owning component and day. Component states remain unchanged during calculation. A failed daily integration restores all components and leaves the weather cursor unchanged.

Phenology uses prior-day accumulated development for its photoperiod and vernalization gates. The day's proposed final accumulation determines the canopy's proposed state. Infection availability uses the canopy's prior-day appearance gate; leaf visibility and unfolding outputs describe the end of the day. Disease output follows the day's numerical integration. Yield diagnostics consume that same proposed end-of-day affected fraction. Weather before sowing or after an explicit crop end advances the forcing calendar while freezing crop development, disease tissue and residue. The crop-end date is inclusive.

The tissue compartments are susceptible, a configurable sequence of latent compartments, visibly affected but noninfectious tissue, infectious tissue, and retained damage. Each of the eight leaf slots has unit total normalized tissue. Slots F1–F7 denote the final-leaf hierarchy; the eighth is the juvenile canopy. `latent_fraction` sums latent compartments, `symptomatic_fraction` sums the final three disease compartments, and `affected_fraction` sums all nonsusceptible compartments. These are modeled tissue fractions rather than measured green-area-loss fractions. Residue compartments represent relative unready and competent local source potential. Daily pathway flows are the fractions newly infected through local primary, imported primary, splash and contact pathways; these are daily increments, not spore counts.

Summary event dates use the first included end-of-day affected or symptomatic fraction at or above the explicit `detection_fraction` for each leaf. The default 0.001 is an assumed observation/detection operator, not a biological infection threshold. Stage dates are the first included daily cumulative T-P-V crossing of each supplied threshold. Missing events are null. Latent infection dates are modeled events; field symptom assessments do not directly observe those infection dates.

Explicit stage thresholds define BBCH events without interpolation of numerical BBCH codes. Rank spacing has effective T-P-V units and represents a scenario rather than a measured phyllochron. The juvenile policy `handover_31_39` withdraws juvenile capacity between the declared BBCH31 and BBCH39 thresholds; `persistent` is an explicit alternative scenario. Fixed temperature response constants are intrinsic numerical assumptions. Stage thresholds and fitted disease coefficients are supplied by the caller.

Normalized final-leaf capacity has no natural senescence process and is not actual green leaf area after BBCH85. Nonpositive daily temperature stops the thermal disease-stage and residue clock; establishment also depends on the supplied weather exposure. Residue maturation and decay are effective source phenomenology, without measured inoculum or calibrated sexual-maturation rates.

The required CSV columns and their units are:

| Column | Unit and definition |
|---|---|
| `date` | Calendar day, exactly `YYYY-MM-DD`; unique, sorted and consecutive |
| `tmean_c`, `tmax_c` | °C; daily mean and maximum; the donor thermal response is supported for daily means ≤40 °C |
| `rh_mean_pct` | Daily mean relative humidity, percent in [0,100] |
| `precipitation_mm` | Nonnegative daily precipitation, mm |
| `imported_pressure` | Optional nonnegative relative imported source pressure; blank values inherit configured background, explicit zero overrides it |
| `exposure_override` | Optional dimensionless exposure in [0,1]; otherwise the configured rainfall-duration weather operator applies |
| `reference_lai_F1`–`F3` | Optional disease-free reference green lamina area, m² leaf m⁻² ground, each final leaf separately |
| `functional_loss_F1`–`F3` | Optional functional green-area-loss fraction in [0,1], each final leaf separately |

Reference LAI and functional-loss columns must be supplied together. Every included day in an enabled external-area window requires their daily values. Disease observations and unknown CSV columns are rejected. The weather object validates dates and physical bounds before simulation. T-P-V accumulation is expressed in effective °C d after photoperiod and vernalization response; cumulative vernalization is in effective days. Leaf area capacity is normalized to [0,1]. HAD is m² green lamina m⁻² ground d, equivalent to GLAI d. Yield-transfer slopes are t ha⁻¹ per GLAI d, and conditional losses are t ha⁻¹.

Yield diagnostics are disabled by default, and `actual_field_yield_forecast` remains null. `external_functional_loss` integrates explicitly supplied reference LAI multiplied by functional-loss fractions. `standardized_model_proxy` instead declares a nominal upper-three-leaf LAI, distributes it equally across the three final leaves, scales by their modeled capacity, and treats the modeled symptomatic/damage fraction as functional loss. Latent compartments are excluded from this conversion. The conversion is an unvalidated scenario. Binary infection status is never treated as complete leaf loss. An independent reference yield can be supplied for conditional percentage-loss reporting; no reference yield is inferred.

The optional inclusive flowering-proxy–BBCH85 window uses daily rectangular integration. The default transfer slopes 0.0141, 0.0180 and 0.0207 t ha⁻¹ per GLAI d are a cultivar-range scenario from [Parker et al. (2004)](https://bsppjournals.onlinelibrary.wiley.com/doi/10.1111/j.1365-3059.2004.00951.x). Their source TOP3 HAD used sequential green-area assessments starting at GS31, with trapezoidal integration and an unrecovered exact endpoint. Flowering–BBCH85 is a restricted-window adaptation. The slope range is not a confidence interval, and the transfer is not a locally yield-calibrated forecast. An uncompleted modeled window yields null conditional loss, not zero. Losses exceeding an independently supplied reference yield are flagged without automatic clipping.

The repository example uses a complete configuration converted outside the runtime from frozen original-field fits. The disease fit uses only observed signs and timing in the original 28 calibration fields; ecological rates are weakly identified and disease magnitudes were not fitted. Its imported background and weather operator are retained explicitly. The accompanying weather is synthetic and contains no field observations.

```sh
python -m model.wheat_stb \
  --config examples/wheat_stb/configuration.json \
  --weather examples/wheat_stb/weather.csv \
  --output /tmp/wheat_stb_example_result
```

The output directory must be new. Output files are `daily.csv`, `summary.json`, `provenance.json`, `checkpoint.json`, `effective_configuration.json` and a receipt containing output hashes. `configuration_standardized_had.json` enables the separately labelled per-unit upper-three-leaf LAI scenario. Installation from the repository uses `python -m pip install .`; the wheel contains model Python packages and runtime dependencies, without calibration files or analysis packages. The installed console command is `wheat-stb`.

```python
from model.wheat_stb import EngineConfig, WeatherProvider, WheatSTBSimulation

config = EngineConfig.from_json("explicit_case.json")
weather = WeatherProvider.from_csv("daily_weather.csv")
simulation = WheatSTBSimulation(config, weather)
simulation.run(days=100)
checkpoint = simulation.snapshot()

resumed = WheatSTBSimulation(config, weather).restore(checkpoint)
resumed.run()
results = resumed.finalize()
```

`snapshot()` is JSON serializable. Restore requires identical parameter values, model version, runtime source hashes and consumed historical weather. Future weather can be extended while retaining the historical prefix. Complete candidate components are validated before replacing the live engine, so a failed restore leaves it unchanged. Runtime provenance records parameter identity, weather hashes, code hashes, semantic version, requested numerical step and effective substep; disease substeps divide a day evenly using `ceil(1/requested_step)`.

The external conversion command is `python -m calibration.wheat_stb.configuration` with explicit `--tpv-fit`, `--stage-fit`, optional `--disease-fit`, `--latitude`, `--sowing-date` and `--output` paths. The runtime never searches for calibration files. `examples/wheat_stb/generate_example.py` records the fit input hashes and the synthetic weather formula; reproduction of those example inputs requires the explicitly named frozen fit files.
