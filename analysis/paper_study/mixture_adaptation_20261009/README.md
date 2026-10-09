# Empirical cultivar-mixture evidence

The French randomized single-site experiment supports a restricted observed harvest comparison. The Swiss dataset does not provide complete constituent controls within its identified mixture experiment; its matched-source contrasts are noncausal. Neither dataset validates future-climate adaptation or STB-mediated grain loss.

## Reproduction

From this directory, with the supplied environment:

```bash
/Users/gangzhao/Documents/workspace/paper_cc_impact_wheat_disease/.venv/bin/python -B run.py
/Users/gangzhao/Documents/workspace/paper_cc_impact_wheat_disease/.venv/bin/python -B verify.py
```

The runtime dependencies are Python, NumPy and pandas. Exact versions are recorded in [the analysis receipt](results/analysis_receipt.json). The run is offline and reads only this directory. It verifies the frozen protocol and every public-input hash before analysis. Output paths outside this module are rejected. `verify.py` runs the contract tests, reconstructs source calculations independently, checks grouping and claim restrictions, and performs a byte-identical replay inside a temporary subdirectory of this module. A successful verification replaces [verification_receipt.json](verification_receipt.json).

[public_inputs.zip](public_inputs.zip) contains 15 original public/provenance members in approximately 160 kB. [input_manifest.json](input_manifest.json) records exact hashes, source dataset DOIs, creators, licenses and parent-bundle lineage. The original public data are CC BY 4.0; archived local context retains its original provenance without a new license assignment. The associated Swiss paper is retained only as archived provenance. No excluded-journal literature is added.

`prepare_inputs.py` rebuilds the compact bundle from the specified prior audit bundle, after checking its frozen hash. It is unnecessary for ordinary reproduction. Older and newer Swiss releases describe overlapping observations; the earlier release supplies identifiers, not additional trials.

## Deliverables

- [PROTOCOL.md](PROTOCOL.md) and [protocol_lock.json](protocol_lock.json): pre-contrast specification, freeze time and hashes. Prior published findings were already visible; this is not preregistration.
- [RESULTS.md](RESULTS.md): numerical findings and their empirical scope.
- [CLAIMS.md](CLAIMS.md): manuscript-ready wording and the main-figure boundary.
- [data_dictionary.csv](data_dictionary.csv): source units, aggregation units, formulas and interpretation.
- [results/applicability.json](results/applicability.json): machine-readable distinctions among descriptive association, conditional randomized contrasts, hindsight selection, climate adaptation and STB-mediated loss.
- [French summary](results/french_summary.csv), [plot contrasts](results/french_contrasts.csv), [dependency groups](results/french_dependency_groups.csv), [influence diagnostics](results/french_influence.csv), [measured trade-offs](results/french_measured_tradeoffs.csv): primary numerical evidence. `french_plots.csv` links every harvest to original CSV line numbers; `french_exclusions.csv` documents all seven missing-harvest/control exclusions.
- [Swiss summary](results/swiss_summary.csv), [control coverage](results/swiss_constituent_coverage.csv), [same-trial gate](results/swiss_same_trial_constituent_gate.csv), [environment contrasts](results/swiss_environments.csv), [transfer scores](results/swiss_transfer_summary.csv): noncausal diagnostics. `swiss_records.csv` and `swiss_release_mapping.csv` retain source lines and quarantine ambiguous identity mappings.

`french_contrasts.csv` is suitable for a main empirical-management panel showing raw differences, observed downside and author-BLUP sensitivity on the identical 195 comparisons. The panel's domain is Mauguio 2018 durum inbred lines under the documented management, with alternating-row mixtures. Error bars implying independent environment or independent mixture replication are unsupported. Swiss diagnostics belong with supplementary control-comparability evidence. This module supplies no climate-impact rescaling, new production conversion, or modification to annual canopy-risk estimates.

The higher-observed-constituent columns are hindsight selection diagnostics. The mean paired percentage difference is a mean of 195 ratios, not the ratio of two grand means. Author spatial BLUPs are source-fitted full-trial estimates; they are not additional harvests, external predictions or independently validated effects. French thousand-kernel weight is a measured yield component, not a market-quality grade.

All implementation and outputs reside in this directory. No parent builder, previous module, root document, Git branch or index is modified by these scripts.
