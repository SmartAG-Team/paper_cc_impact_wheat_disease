# Climate pairing, population support and annual canopy damage

The primary quantity is normalized healthy-area-duration loss during flowering to soft dough, expressed in equivalent days. The input bundle contains exact float64 values from 810 completed annual files on 14,941 fixed wheat cells. Each original file is checked against the completed experiment's SHA-256 receipt. Invalid seasons remain unavailable. No crop or disease parameters are refitted and no climate simulation is rerun.

Historical and future year positions are not physically matched weather years. Thirty circular shifts of future-year positions assess the dependence of the reported mean change on the original positional pairing. Each shift retains the same year sets but changes which incomplete seasons intersect. The weighted mean divides the sum of valid paired changes by their summed fixed spatial weights. Full-support means are analytically invariant to a permutation; their numerical invariance is checked. The 30 shifts are a deterministic sensitivity set, not independent replicates, a probability distribution or all possible permutations.

Three support definitions are compared: original positional finite pairs; separately available historical and future seasons with separate denominators; and cells complete in every historical and future year in all three climate models. The third definition fixes the same spatial population across both periods and all climate models, separately for each scenario and future period. Its coverage uses the complete baseline population as denominator. Empty domains remain missing. Harvested-area, total-production and rainfed-production weights are analyzed separately; all weights retain the SPAM2020 baseline.

On the common complete population, each climate model supplies 30 spatially weighted annual values per period. Their mean, sample standard deviation, median, empirical 10th and 90th quantiles, and mean of the three largest values describe interannual variation in regional canopy damage. Empirical quantiles use linear interpolation. Future-year exceedance counts use that climate model's historical 90th quantile, with strict greater-than comparison. Climate-model summaries have equal weight; min–max ranges describe model spread. Annual observations from different climate models are not pooled into a single probability distribution. These statistics are conditional canopy-damage indicators, not grain-yield downside risk or future production forecasts.

Replacing total-production weights with rainfed-production weights tests aggregation sensitivity to irrigation coverage. The published total, rainfed and irrigated raster quantities remain separate; their small source residual is recorded rather than removed by rescaling. Rainfed reweighting neither simulates irrigation nor distinguishes winter and spring wheat. All underlying crop calendars still represent the imposed winter-wheat rainfed system.

## Reproduction

From the repository root, using the installed project Python dependencies:

```sh
python -m analysis.paper_study.climate_robustness_20261009.run --output /tmp/climate-sensitivity-reproduction
python -m pytest -q analysis/paper_study/climate_robustness_20261009/test_robustness.py
```

The compact NPZ input and registry CSV reproduce all numerical tables without raw annual files. Original extraction additionally requires the locally stored annual-output archive:

```sh
python -m analysis.paper_study.climate_robustness_20261009.run --prepare-inputs --source-root /path/to/original/repository
```

`input_receipt.json` records all 810 source hashes and the compact bundle identity. `receipt.json` records protocol, inputs, code and output hashes, as well as 198 independent comparisons with the previously reported regional aggregation. The protocol was fixed before these additional sensitivity calculations; earlier headline climate results were already known.
