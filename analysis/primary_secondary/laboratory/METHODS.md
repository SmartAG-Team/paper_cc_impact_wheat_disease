# Controlled-inoculation onset constraints

The pathology workbook contains three wheat-diversity panels, two transgenic tests and one EMS experiment. JIC assays used a 16-hour photoperiod at 18/12 °C day/night. The standard conidial concentration was 10⁶ spores mL⁻¹; transgenic IPO88004 assays used 10⁷. Leaves were spray-inoculated, incubated in darkness for 48 hours and kept under propagator lids through seven days. Damage denotes combined necrosis and chlorosis. The EMS experiment used second-leaf paintbrush inoculation at 10⁷ spores mL⁻¹ under 21/18 °C and 85% relative humidity. [Hafeez et al., Nature Plants (2025)](https://www.nature.com/articles/s41477-025-01920-2).

The source workbook is distributed under CC BY 4.0, with SHA256 `001426f9c85fcfd6c4f4d5769eba08f02977fffd53a60c08697c102d01e7cda3`. [Original pathology data](https://doi.org/10.5281/zenodo.14515753). Each scored source row identifies a seedling/leaf position. Assay, genotype, isolate, batch, experimental replicate, scorer, tray, box, block and plot identifiers are retained where available. Genotype labels remain unchanged, including replicate label 3.5. EMS Plant1–Plant6 identifiers remain distinct; their allocation to the two experimental trials is absent from the workbook. Source AUDPC, transformed scores and averages are excluded from the replicate count.

The extraction retains 60,828 score cells and 5,313 plant slots. Of these, 5,064 have at least one valid pycnidia assessment. The 249 entirely unassessed slots include 104 explicitly labelled Blank slots and 145 other unassessed positions. Unassessed positions contribute no right-censoring time and are excluded from timing denominators. Numeric percentages outside 0–100, uncertain text such as `5?`, and missing `-`, `*` or empty cells retain explicit quality states. No missing score is replaced by zero and no invalid percentage is clipped.

Additional source annotations retain accession identifiers, source genotype categories, mildew and undefined SN/sn labels alongside original derived summaries. Positive Mildew_Max annotations occur in 107 IPO323 plants; the annotation scale is unspecified in the workbook. Combined damage therefore has a potential co-disease component. Four of 26,837 valid paired assessments record pycnidia coverage above damage/necrosis, with a maximum excess of 35 percentage points. These observation conflicts remain explicit, and endpoint scores are fitted separately rather than converted into negative nonsporulating fractions.

Visible-pycnidia onset is the first assessment with a positive pycnidial score. The corresponding damage endpoint is the first positive combined necrosis/chlorosis score; EMS N scores measure necrosis alone. These endpoints are fitted separately. For the baseline threshold, any score above zero is positive. A second pycnidia analysis requires at least 5% coverage and represents detection-threshold sensitivity.

For onset time T measured from conidial application, U is the first positive assessment and L is the last valid negative assessment preceding U. A plant with both observations contributes T∈(L,U]. A plant positive at its first valid assessment contributes T∈(0,U], with zero denoting the challenge-time origin rather than an observed negative baseline. A plant with valid assessments but no positive score contributes T>L, where L is its final negative assessment. This last category is labelled right-censored late-or-never: resistant plants, failed establishment and onset after follow-up cannot be separated. Entirely unassessed plants remain unobserved.

Prior biological control roles remain separate from censoring states. Arina/ArinaLrFor challenged with IPO88004 retain their documented Stb15-resistant background annotation, and the EMS L6 row retains its explicit susceptible-control label. Other resistance classifications are not inferred from zero scores. A resistant-background annotation does not supply an observed event time or identify a statistical cure fraction.

The interval interpretation assumes that the detectable presence endpoint persists between inspections. Positive scores followed by negative scores flag departures from this assumption. Such reversals occur in 8 IPO323, 55 IPO88004 and 22 IPO90012 panel plants, plus three transgenic plants. Baseline results retain first detection, while a sensitivity analysis excludes each endpoint's presence-reversal records. Coverage decreases that remain positive do not change the first-presence endpoint. Damage and pycnidia reversals are assessed independently.

The onset distribution uses a nonparametric interval-censored model, fitted separately for each assay, isolate, endpoint and threshold. Scoring boundaries partition time into cells (aⱼ,bⱼ], followed by an unresolved tail. With cell probability pⱼ and membership Aᵢⱼ=1 when cell j lies within plant i's interval, the criterion is

ℓ(p)=Σᵢ log(Σⱼ Aᵢⱼpⱼ), with pⱼ≥0 and Σⱼpⱼ=1.

The EM update is pⱼ←pⱼΣᵢ[Aᵢⱼ/(ΣₖAᵢₖpₖ)]/n. Duplicate interval patterns are grouped by count without adding replicates. Convergence requires maximum probability change below 10⁻¹⁰ and a normalized likelihood optimality gap below 10⁻⁷. Every delivered fit satisfies these checks. Finite probabilities belong to intervals; no exact event date is assigned within a scoring interval. The median is consequently an interval, and an unresolved-tail median is reported as unreached during follow-up.

The fitting criterion describes the sampled plant mixture. Genotypes, trays and batches share environmental and genetic effects, so its independence likelihood is a working composite likelihood. Individual plants are not treated as independent units for standard errors or significance tests. Genotype summaries and batch-specific panel fits preserve the experimental structure. Missing assessments may be informative, and inspection/dropout independence is not established by the archive. Pooled model estimates therefore remain descriptive under this observation assumption.

Sample identification bounds provide a separate timing constraint. At time t, the fraction definitely showing onset is Σᵢ1(Uᵢ≤t)/n; the fraction that could show onset is Σᵢ1(Lᵢ<t)/n. The bounds refer to assessed plants and require the persistent-detection interval interpretation. They are neither confidence intervals nor population prediction intervals, and they do not rely on independent plant replication. Model CDF bounds also retain unidentified positions within finite cells.

| Assay/isolate | Assessed plants | Detected onsets | Median interval, dpi | Unresolved tail mass |
| --- | ---: | ---: | --- | ---: |
| Watkins IPO323 | 1,484 | 422 | >32 | 0.7130 |
| Watkins IPO88004 | 1,558 | 866 | (24,28] | 0.4419 |
| Watkins IPO90012 | 1,584 | 1,100 | (21,24] | 0.2896 |

At 28 dpi, sample identification bounds are 0.2412–0.2978 for IPO323, 0.4994–0.5051 for IPO88004 and 0.6667–0.6723 for IPO90012. The differing distributions combine host diversity, isolate specificity, censoring and assay conditions. Zero-only records do not identify a cure fraction. A common temperature response, successful-establishment probability or universal latent-stage transition rate is unsupported.

Conidial application supplies a known experimental exposure origin. Successful penetration time, airborne ascospore arrival, viable deposited inoculum per leaf and subsequent emitted infectious spores are unobserved. First visible pycnidia constrain an assay-specific detectable sporulation endpoint. They do not measure time of successful infection or infectious-spore flux. The CDF concerns plant/leaf presence and is distinct from infected-area or infectious-area fractions. The 18/12 °C assays contain no temperature-treatment contrast; the EMS experiment additionally changes genotype, dose, inoculation method and temperature, precluding a temperature-response estimate from its comparison. A separately specified Erlang/gamma progression kernel can be compared with these timing bounds only with explicit assumptions about establishment delay and observation mapping. Selecting only eventually pycnidia-positive plants would condition on a future endpoint and exclude resistant or slowly developing plants.

Source-cell values and quality flags agree exactly with the corresponding 60,828 harmonized observations. The audited harmonized snapshot mislabels 1,600 transgenic IPO92006 score cells as IPO88004 and labels 60 EMS cells with the generic seedling-leaf organ instead of second-leaf sections. Authoritative workbook isolates and assay-specific leaf identities define this analysis.

The original workbook, source-row/column keys, enriched scores, censoring intervals, model-cell probabilities, CDF constraints, genotype and batch summaries, reconciliation evidence and checksum receipts form the analysis record. Executable reproduction uses Python with NumPy, pandas, pyarrow and openpyxl:

```sh
/Users/gangzhao/Documents/workspace/agri_water_management/.venv/bin/python -B analysis/primary_secondary/laboratory/analyze_hafeez_onset.py
/Users/gangzhao/Documents/workspace/agri_water_management/.venv/bin/python -B analysis/primary_secondary/laboratory/verify_hafeez_onset.py
/Users/gangzhao/Documents/workspace/agri_water_management/.venv/bin/python -B analysis/primary_secondary/laboratory/extract_hafeez_annotations.py
```
