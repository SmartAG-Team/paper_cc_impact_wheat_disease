# Explicit wheat STB example inputs

`configuration.json` contains complete phenology, leaf and disease parameters. `configuration_standardized_had.json` uses the same parameters and enables a conditional per-unit upper-three-leaf LAI scenario. No actual field yield is inferred.

`weather.csv` is synthetic consecutive daily weather from 2026-09-20 through 2027-09-30. The sowing date is 2026-10-01. `inputs.json` records the formula, frozen fit hashes and input file hashes. Original-field sign and timing fits are development evidence; disease magnitudes and local yields were not calibrated.

```sh
python -m model.wheat_stb \
  --config examples/wheat_stb/configuration.json \
  --weather examples/wheat_stb/weather.csv \
  --output /tmp/wheat_stb_example_result
```

The output directory must be new. Full API, daily boundary, units and restart specifications are in [WHEAT_STB_ENGINE.md](../../WHEAT_STB_ENGINE.md).
