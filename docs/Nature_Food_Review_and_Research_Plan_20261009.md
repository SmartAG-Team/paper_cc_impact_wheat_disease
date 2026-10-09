# European wheat food security under climate change: manuscript review and research development plan

**Date:** 9 October 2026  
**Manuscript:** *Wheat phenology partly offsets projected increases in Septoria damage across Europe*  
**Version assessed:** `41e3f89`  
**Primary journal target:** Nature Food  
**Research focus:** Climate impacts on wheat production through crop phenology, disease damage and yield loss, with implications for food availability and adaptation.

## 1. Publication assessment

**Substantial scientific development is required before submission to Nature Food.** The study addresses an appropriate food-production question and contains a potentially valuable finding: changes in wheat phenology oppose part of the weather contribution to projected Septoria damage. Its principal strengths are spatially explicit simulations, regional comparisons and transparent evaluation against field observations. The principal limitations are disease prediction skill, the unvalidated relationship between simulated canopy damage and grain yield, and the absence of evaluated adaptation outcomes.

Nature Food includes agricultural sciences, crop sciences, pest control, food policy and food security within its scope. A European wheat study can therefore fit the journal without extending to every component of the food system. Its food-security contribution needs a defensible connection to production, production stability or a consequential management decision. National hunger, affordability and dietary outcomes require additional evidence beyond crop yield. [Nature Food aims and scope](https://www.nature.com/natfood/aims).

Nature is a more demanding and distinct target. Its criteria emphasize exceptional scientific importance and interest beyond a specialist readership. A larger simulation domain or more polished presentation alone would not establish that contribution. A Nature pathway would require a broadly consequential discovery about climate–crop–disease interactions, supported by independent evidence and a clear change in scientific understanding. Cross-system replication is one possible route to generality, rather than a formal requirement imposed by the journal. [Nature editorial criteria](https://www.nature.com/nature/for-authors/editorial-criteria-and-processes).

| Dimension | Current assessment | Requirement for a stronger submission |
| --- | --- | --- |
| Food-production relevance | Strong topic fit; actual production consequences remain unquantified | Validated disease-related yield responses and clearly defined regional production exposure |
| Scientific novelty | Potentially useful opposing weather and phenology effects | Evidence that the mechanism changes conclusions relative to simpler climate-risk and crop models |
| Disease prediction | Insufficient support for strong quantitative accuracy claims | Independent evaluation of timing, severity and their responses to weather |
| Crop and yield representation | Phenology and normalized canopy damage; conditional yield conversion | Evaluated crop growth, functional canopy loss and harvested grain yield |
| Spatial analysis | Complete simulation of the specified wheat mask | Robustness to crop type, calendar, management and regional coverage |
| Adaptation and policy | Monitoring and trial priorities; intervention benefits untested | Feasible options compared using validated production outcomes and trade-offs |
| Presentation | Concise abstract and readable maps; several overlapping indicators | A figure sequence that establishes credibility, impacts, mechanism and decisions |

## 2. Findings supported by the current evidence

The late-century SSP5–8.5 projections indicate a European increase of **2.66 equivalent days of normalized healthy-area-duration loss**, with increases across **77.3% of represented wheat area**. Regional contrasts are substantial: **+7.84 days in the Boreal region** and **−1.40 days in the Mediterranean region**. These are conditional canopy-damage projections under the specified model and management assumptions, rather than measured yield changes. [Regional results](../analysis/paper_study/full_grid_climate_20261008/results/environmental_region_changes.csv), [current manuscript](../publication/european_wheat_stb/Manuscript.txt).

The weather–phenology decomposition gives **+9.00 days** from weather and **−6.36 days** from wheat phenology, producing a net **+2.64 days** and a **70.6% offset**. This is the most promising scientific contribution. The estimate comes from a separate spatial sample, with 5,731 valid season pairs at 62 locations. Agreement with the full-grid net change supports the sampled estimate’s consistency, but does not independently validate the mechanism or its magnitude. [Decomposition results](../analysis/paper_study/nature_food_fix_20261007/climate/decomposition/supported_forcing_ensemble_decomposition.csv).

The collected field datasets are already used. Their results demonstrate the limits of static severity–yield relationships, rather than establish a validated replacement for the climate yield conversion.

| Evidence | Current result | Scientific implication |
| --- | --- | --- |
| Symptom timing | Mean error outside observed onset intervals: **5.32 days**, versus **2.79 days** for the benchmark; paired difference **+2.54 days**, 95% interval **1.15–3.93** | The disease model performs worse for this timing endpoint; this score is interval-excess error, not conventional date MAE |
| Disease severity | RMSE **31.19 percentage points**, versus **29.83**; bias **−17.56 points**; R² **−0.125** | Severity magnitude is poorly reproduced; the RMSE difference interval includes zero |
| Symptom detection | Sensitivity **97.16%**, specificity **45.06%**; balanced accuracy **71.11%**, versus **78.80%** for the benchmark | High sensitivity accompanies many false positives and does not establish high overall accuracy |
| Nordic severity–yield contrasts | **62 contrasts in five trials**; held-out RMSE **4.83**, versus **5.02 percentage points** | Small apparent gains are not conclusive; the stage-informed result uses four trials and has uncertain transfer |
| Tunisian plot yields | **82 unprotected plot-season yields**; cross-year RMSE **3.32**, versus **3.24 t ha⁻¹** | The tested severity predictors do not improve cross-year absolute-yield prediction |
| German protection comparisons | **3,264 comparisons**, 16 site-years, five locations; site-held-out RMSE **18.51**, versus **18.48 percentage points** | Endpoint severity adds little information; management explains more of the forward-transfer improvement |
| Climate yield estimate | **−47.8 kg ha⁻¹ per unit reference upper-canopy LAI** | A fixed rescaling of simulated HAD loss, rather than independently validated harvested yield |

Sources: [pooled disease metrics](../analysis/paper_study/nature_food_impact_20261008/derived/pooled_evaluation_metrics.csv), [Nordic analysis](../analysis/paper_study/nordic_yield_validation_20261009/analysis_receipt.json), [Tunisian analysis](../analysis/paper_study/crop_damage_validation_20261009/analysis_receipt.json), [German analysis](../analysis/paper_study/crop_damage_validation_20261009/briwecs_receipt.json), [Supplementary Information](../publication/european_wheat_stb/Supplementary_Information.txt).

## 3. Decisive scientific gaps

### 3.1 Disease prediction and observation compatibility

Poor timing and severity performance weakens confidence in the quantitative climate projections. Leaf numbering, percentage denominators, assessment stages and observation intervals differ among sources. Pooled reporting is appropriate when these differences remain explicit; merging incompatible measurements into a common physiological target is not.

The evaluation protocol needs whole-environment and forward-year holdouts, common benchmark comparisons and uncertainty that includes model fitting or selection where relevant. Repeated severity assessments sharing one treatment yield belong to the same statistical outcome and validation group. Predictive improvement must be judged on the endpoint supporting the principal impact claim, rather than the most favorable metric or subset.

### 3.2 Functional canopy damage and grain yield

The current model assumes a unit reference upper-canopy LAI, equal maximum areas for the upper three leaves, and a conversion from symptomatic tissue to functional leaf-area loss. Absolute green area and natural senescence are absent from the capacity formulation. These assumptions affect canopy-damage estimates before the yield coefficient is applied. The field and climate replays also use slightly different subthreshold-damage conventions. [Supplementary model specification](../publication/european_wheat_stb/Supplementary_Model_Specification.md).

The fixed HAD–yield coefficient was obtained for a wider developmental interval beginning at GS31, whereas the primary climate diagnostic covers GS65–85. A regional yield estimate derived from this conversion cannot establish actual grain production or food availability. Compatible measurements of leaf area, healthy canopy function, crop stages and grain yield are the critical missing evidence.

### 3.3 Mechanism, crop stages and uncertainty

The 70.6% offset is a model decomposition, not an observed adaptation benefit. The interaction is divided equally between weather and phenology by the Shapley convention. Alternative crop-stage parameters, canopy assumptions and weather representations may alter the magnitude.

The overall station-held-out phenology MAE of 8.02 days concerns BBCH 10, 31, 51 and 85. It does not by itself establish upper-leaf or flowering accuracy. In the disease-field evaluation, GS85 has only five two-sided observation intervals and a mean excess error of 10.50 days. Stage-specific uncertainty matters because it changes the interval over which disease damage is integrated. [Supplementary Tables S1–S4](../publication/european_wheat_stb/Supplementary_Information.txt).

### 3.4 The represented wheat-production population

The all-wheat land-use mask is combined with a winter-wheat rainfed calendar. Spring wheat, cultivar differences and actual calendar variation are not separately represented. This mismatch is especially important for interpreting the large Boreal response. Country assignments follow the dominant source-country label of each pixel and cover the study domain, rather than necessarily the entire national wheat-production area.

Production-focused analysis requires crop-type and management coverage, uncertainty in calendars and a distinction between the wider European study domain and any policy reporting region. Changing future wheat area should remain a separate scenario from impacts on the current fixed wheat area.

### 3.5 Food-security and adaptation outcomes

Food security is the motivating problem; the immediate modeled outcome is canopy damage. The missing analytical connection is from disease damage to credible yield loss, production exposure and feasible protection of production. Lower disease damage in the Mediterranean cannot establish higher total wheat yield when heat, drought and other stresses are omitted. Likewise, climate-induced earlier crop development cannot be equated with the yield benefit of changing sowing dates or cultivars.

## 4. Lessons from relevant published studies

| Study | Evidence and contribution | Implication for this study |
| --- | --- | --- |
| [Silva et al., Nature Food, 2026](https://www.nature.com/articles/s43016-025-01286-w) | Experimental data, crop modeling and observed yield trends distinguish genetic, climatic and agronomic contributions to northwest European wheat production | Connect the proposed mechanism to an evaluated production outcome and a specific agronomic constraint |
| [McDonald et al., Nature Food, 2022](https://www.nature.com/articles/s43016-022-00549-0) | Field and household observations, remote sensing and crop simulation connect planting calendars to yield opportunities and practical constraints in a coupled cropping system | Assess feasible adaptation and its trade-offs, rather than infer benefits from phenology shifts alone |
| [Vico et al., Nature Food, 2026](https://www.nature.com/articles/s43016-026-01293-5) | Long-term European experiments quantify food outputs from rotations and examine sensitivity to crop end use | Food-security indicators require explicit production and allocation assumptions; nutritional conversions are optional extensions, not substitutes for yield validation |
| [Chaloner et al., Nature Climate Change, 2021](https://www.nature.com/articles/s41558-021-01104-8) | Climate-driven pathogen risk is considered alongside projected crop yields across multiple crops | A new contribution must extend beyond the general observation that climate changes crop-disease pressure |
| [Pequeno et al., Nature Climate Change, 2024](https://www.nature.com/articles/s41558-023-01902-2) | Coupled wheat and disease simulations connect future disease vulnerability with production consequences | Crop–disease coupling alone is not sufficient novelty; outcome evaluation and the new mechanistic insight remain essential |

These precedents support an editorial inference: the most credible Nature Food pathway is a validated explanation of how wheat development modifies climate-related disease losses, followed by production consequences and an actionable adaptation comparison. None supplies a universal prediction-accuracy threshold or guarantees acceptance.

## 5. Research objective and testable propositions

**Primary objective:** Quantify how climate-driven changes in wheat development alter disease-related grain-yield loss across European wheat-growing environments, and identify feasible adaptation options that protect production under uncertainty.

Three propositions require evaluation:

1. Changes in wheat development modify the response of canopy damage to changing weather beyond the information supplied by weather-risk indicators alone.
2. Leaf position, the timing of damage and the crop’s physiological condition improve prediction of disease-related yield loss beyond endpoint severity and management-only benchmarks.
3. Adaptation choices differ among regions because protection of yield depends on disease pressure together with heat, water limitation and crop-development constraints.

Negative or mixed findings remain scientifically informative. The publication argument depends on the evidence, rather than a required direction of change.

## 6. Prioritized development plan

### Priority 1 — Establish the usable evidence and independent evaluation protocol

**Actions**

- Reconcile treatment yields, repeated assessment dates, final-leaf identity, severity denominators, cultivar, management, crop stages and grain-moisture conventions.
- Preserve a common field-evidence inventory and pooled reporting across the existing archives, with source identifiers retained for provenance and measurement compatibility.
- Update the readiness inventory: the German BRIWECS comparisons are already analyzed; the Nordic/Baltic 2012–2016 archive remains a harmonization opportunity.
- Identify existing datasets containing repeated leaf-specific green area, disease observations, weather, crop stages and harvested grain from the same experimental units. Published fitted curves and model-predicted means remain supporting evidence, not additional independent field observations.
- Specify primary endpoints, candidate models, benchmarks, eligibility, validation groups and evaluation criteria before further model selection. Keep independent evaluation environments fixed.

**Deliverable:** An outcome-level evidence table, a measurement-compatibility map and a fixed validation protocol.

**Completion criterion:** Every primary outcome has a traceable experimental unit, unit definition and independent evaluation assignment. Missing physiological measurements and disease attribution remain explicit.

### Priority 2 — Establish crop-stage and disease-response credibility

**Actions**

- Evaluate the stages relevant to damage and grain formation separately: stem elongation, upper-leaf emergence, flowering and soft dough.
- Assess symptom timing, severity progression and symptom-free seasons against the existing phenology and mean-severity benchmarks on identical records.
- Test temporal and geographic transfer; keep cultivar transfer distinct from environmental transfer when cultivars recur across folds.
- Evaluate whether observed responses to historical weather variability are reproduced across contrasting environments. Distinguish evidence for weather-response changes from accuracy of absolute disease levels.
- Examine the impact of uncertain stage dates and observation definitions on damage estimates. Retain unsuccessful candidate models and unfavorable validation outcomes.

**Deliverable:** Independent prediction plots, stage-specific errors, benchmark differences and uncertainty estimates.

**Completion criterion:** The endpoint supporting the principal impact claim has demonstrated predictive value, or independently supported weather-response behavior despite remaining level bias. A model that fails this condition supports a narrower conditional analysis rather than strong impact forecasts.

### Priority 3 — Validate the disease-related grain-yield response

**Actions**

- Compare a parsimonious stage- and leaf-aware yield-response model with a crop-growth formulation containing physiological damage. Select complexity using training evidence and parameter identifiability.
- Use a tested wheat-growth framework, such as a suitable WOFOST or DSSAT formulation, where required inputs and calibration evidence are available. Account for actual canopy area, natural senescence, light interception, grain growth and pre-anthesis reserves where justified by observations.
- Calibrate crop growth and healthy yield separately from the incremental loss associated with disease. Validate both outcomes on independent environments.
- Compare dynamic canopy information with endpoint severity, crop-management predictors and the current constant HAD conversion.
- Quantify how damage during stem elongation and grain filling affects the yield response. Compare GS31–85 and GS65–85 diagnostics using compatible observations rather than treating either window as universally sufficient.
- Keep mixed-disease protection responses distinct from cause-specific STB losses. Protected plots require observed disease information and treatment context; protection does not automatically establish a disease-free reference.

**Deliverable:** Evaluated yield and disease-loss predictions in consistent grain units, with uncertainty and domain limits.

**Completion criterion:** The selected approach predicts healthy yield and incremental disease-related loss across independent environments better than relevant simple benchmarks, with uncertainty and physically interpretable responses. If these conditions remain unmet, the fixed yield conversion stays in supplementary sensitivity evidence.

Physiological canopy and reserve-based approaches have precedent in [Bancal et al., Annals of Botany, 2007](https://doi.org/10.1093/aob/mcm163). Disease-damage formulations have also been compared across wheat-growth models in [Bregaglio et al., Field Crops Research, 2021](https://doi.org/10.1016/j.fcr.2021.108108). Those frameworks provide candidate structures; their publication does not establish validation for the present field environments.

### Priority 4 — Strengthen the climate-impact and attribution analysis

**Actions**

- Recalculate grid impacts after selection of an evaluated formulation. Preserve the current land-use-based grid calculations and harvested-area weighting.
- Partition or test winter/spring wheat and management-calendar assumptions using verified spatial evidence. Report actual coverage for environmental regions and country groups.
- Evaluate daily weather-exposure approximations against available higher-frequency weather or canopy observations. Examine whether adjustment preserves relevant joint weather behavior and seasonal variability.
- Extend weather–phenology attribution across environmental regions and emissions pathways. Prefer a full-grid calculation when feasible; otherwise quantify sampling precision and convergence in each reporting domain.
- Test sensitivity to historical–future year pairing and missing-season support. Pairing positions in two future and historical periods does not imply physically matched weather years.
- Propagate climate-model, crop-stage, disease, functional-canopy and crop-yield uncertainty. Three-model ranges remain model spread, rather than probability intervals.
- Investigate rare extreme timing projections against crop stages and model applicability. The previous −96-day appearance was a color-binning problem; a corrected legend alone does not validate the rare extreme predictions.

**Deliverable:** Robust spatial impacts, regional attribution and an uncertainty analysis identifying the assumptions that control conclusions.

**Completion criterion:** Principal regional contrasts and the opposing weather–phenology mechanism remain defensible across supported crop populations and plausible model assumptions. Unstable regions or magnitudes are reported as uncertain.

### Priority 5 — Quantify food-production consequences and feasible adaptation

**Actions**

- Report disease-related yield loss in t ha⁻¹ and percent of an independently evaluated reference yield. Separate direct climate impacts from the additional disease component and their interaction.
- Aggregate validated yield changes using crop-specific harvested area. Production tonnage equals harvested area multiplied by compatible yield change; no tonnage conversion is defensible from the current per-reference-LAI index alone.
- Examine annual production variability and downside risk alongside average losses. Relative sensitivity and absolute production exposure may identify different priority regions.
- Compare a small set of supported adaptation options, such as locally feasible cultivar or calendar choices, individually and together. Retain constraints from crop rotations, heat, water availability and establishment conditions.
- Evaluate adaptation using yield retained, production stability and measured or source-supported economic and resource trade-offs. Resistance attributes and management effects require observed evidence rather than arbitrary changes chosen to improve the projection.
- Develop a regional decision table connecting the magnitude and robustness of production exposure with evaluated options and remaining evidence gaps.
- Treat dietary energy, trade exposure or food-price analysis as optional extensions with their own allocation and economic assumptions. Crop-production results support the food-availability dimension of food security; broader dimensions need dedicated evidence.

**Deliverable:** Regional production exposure, evaluated adaptation outcomes and a concise decision table for agricultural planning.

**Completion criterion:** The policy conclusion follows from an explicit, evaluated comparison. A region has a defensible adaptation priority when both production exposure and the proposed response remain credible under uncertainty.

### Priority 6 — Align the manuscript and submission package with the final evidence

**Actions**

- Organize Results around independent credibility, spatial climate impacts, the crop–disease mechanism, production consequences and adaptation outcomes.
- Retain one primary symptom-timing reference in the main figures. First symptoms relative to flowering describe alignment with crop development; days after sowing remain a complementary supplementary diagnostic.
- Define healthy-area duration once, distinguish lesion severity from functional canopy damage, and use concrete terms for wheat stages and yield outcomes.
- Preserve benchmark failures, negative responses and uncertainty. Model selection based on held-out performance after repeated inspection requires explicit disclosure and fresh evaluation evidence.
- Keep essential validation visible in the main paper; move detailed source inventories, parameter diagnostics and repeated maps to the supplement.
- Provide a reviewer-accessible data and code release, input specifications, source licenses and a clear distinction between document regeneration and simulation reruns.

**Deliverable:** A coherent Article manuscript, concise supplementary evidence and a reproducible submission package.

**Completion criterion:** Every headline statement is supported by an evaluated outcome or is explicitly a conditional model result. Main figures supply distinct evidence rather than repeated transformations of the same variable.

## 7. Figure and table development

| Current item | Assessment | Recommended role |
| --- | --- | --- |
| Figure 1: scenario maps of normalized HAD loss | Strong spatial overview; no direct production outcome | Retain the climate maps. Add model agreement or robust-change information. Use evaluated yield-loss maps if Priority 3 succeeds |
| Figure 2a: symptom frequency | Symptoms occur in almost every modeled season, limiting discrimination | Move the frequency map to the supplement unless occurrence gains independent predictive or policy relevance |
| Figure 2b: timing relative to flowering | Relevant to alignment between disease damage and crop development | Retain and label **First symptoms relative to flowering** |
| Figure 2c: relative HAD loss | Useful functional-damage diagnostic under explicit assumptions | Retain alongside observed or evaluated canopy information; distinguish it from lesion percentage |
| Figure 2d: timing after sowing | Same symptom event as panel b, from a different time origin | Move to the supplement; retain its corrected continuous color scale |
| Figure 3a: environmental-region contrasts | Clear comparison of heterogeneous impacts | Retain with uncertainty and crop-population coverage |
| Figure 3b: country yield conversion | Deterministic rescaling of HAD with an unvalidated coefficient | Move to supplementary sensitivity evidence until a validated yield formulation is available |
| Figure 4: weather–phenology decomposition | Central mechanistic result | Strengthen with regional estimates, uncertainty and evidence of mechanism robustness |
| Table 1: regional indicators | Useful synthesis; current yield column is conditional | Use evaluated production metrics when available; otherwise retain canopy indicators and move the yield index to the supplement |

An informative final main-figure sequence contains: **(1)** independent prediction and benchmark comparisons; **(2)** spatial climate impacts; **(3)** weather and crop-development contributions; **(4)** regional yield or production consequences; and **(5)** feasible adaptation outcomes. One main table can summarize regional exposure, coverage and decisions. Production and adaptation figures depend on the relevant validation milestones; supplementary frequency and symptom-duration maps should not fill those roles merely because they are available.

## 8. Title, abstract and narrative

The current title identifies a result and is preferable to a generic climate-impact title. Its quantitative interpretation remains conditional on validation of disease damage and the attribution mechanism. Food-production wording becomes appropriate when actual yield or production outcomes are established.

| Evidence achieved | Appropriate title direction |
| --- | --- |
| Current conditional canopy projections | **Wheat phenology partly offsets projected increases in Septoria damage across Europe** |
| Independently evaluated disease-related yield impacts | A result-oriented title naming the verified effect of crop development on climate-related wheat yield loss |
| Evaluated adaptation with production benefits | A result-oriented title naming the supported adaptation and its contribution to European wheat-production resilience |

The current abstract is **144 words and unreferenced**; its length is already compatible with the Nature Food Article format. The substantive improvement is inclusion of the strongest independently supported production or adaptation finding when available. Its sequence remains: importance of wheat production; the unresolved climate–crop–disease relationship; study approach; essential quantitative findings; proportionate implications. Grid counts and auxiliary performance metrics belong in Methods or evaluation results.

Nature Food’s current Article guidance specifies **up to 3,000 main-text words**, excluding abstract, Methods, references and figure legends; **up to 150 abstract words**; **up to six figures/tables**; and an unheaded Introduction followed by Results, Discussion and Methods. The present package records 2,277 main-text words, four main figures, one main table and 28 references. Formatting is therefore not the primary publication barrier. The guidance recommends roughly 50 references rather than requiring that number. [Nature Food content types](https://www.nature.com/natfood/content).

## 9. Dependencies and completion milestones

| Milestone | Dependency | Evidence required for completion |
| --- | --- | --- |
| M1. Outcome and measurement definitions | Existing datasets and source documentation | Verified joins, final-leaf and stage definitions, consistent units, fixed independent validation groups |
| M2. Credible crop-stage and disease responses | M1; informative observations across environments | Endpoint-specific benchmark comparisons and uncertainty; limitations of transfer identified |
| M3. Evaluated yield-loss formulation | M1–M2; compatible canopy–yield evidence | Independent healthy-yield and incremental-loss evaluation; cause-specific scope established |
| M4. Robust European projections | M2–M3; supported crop calendars and forcing | Regionally defensible impacts, mechanism robustness and separated uncertainty components |
| M5. Production and adaptation evidence | M3–M4; feasible intervention evidence | Compatible production aggregation, downside-risk analysis and evaluated adaptation trade-offs |
| M6. Submission package | M1–M5 for the production-focused route | Distinct main figures, supported abstract/title, accessible evidence and complete declarations |

Presentation improvements and source-inventory corrections can proceed alongside M1. Expensive model expansion depends on M2–M3. A credible completion date depends primarily on access to compatible physiological observations; additional simulations cannot replace unavailable outcome-validation evidence.

## 10. Journal decision criteria

**Nature Food pathway:** A validated European food-production analysis is the recommended target. Submission becomes scientifically stronger when the opposing weather–phenology mechanism is robust, the disease-related yield response is evaluated independently, and regional production consequences support a specific adaptation or planning decision. A high-quality mechanism-focused study could remain relevant without a complete food-system model, but its implications must match its demonstrated outcomes.

**Nature pathway:** A separate expansion is justified only if the work establishes a broadly consequential principle with strong independent support. Replication across contrasting crop–disease systems, a decisive test of existing climate-impact assumptions or another demonstration of broad applicability could provide that support. Geographic expansion and larger sample counts alone are insufficient.

**Evidence-limited pathway:** If suitable canopy–yield observations remain unavailable or prediction skill remains inadequate, the scientifically defensible outcome is a narrower study of conditional canopy responses and model limitations. Strong food-production or adaptation claims require the missing evidence; stronger wording cannot supply it.

The key submission conditions are:

- Independent validation supports the principal modeled outcome and its weather response.
- Yield and production claims use evaluated crop and disease-loss formulations in compatible units.
- Climate-induced phenology changes remain distinct from achievable management benefits.
- Regional contrasts survive supported crop-population and uncertainty analyses.
- Each main figure contributes separate evidence; unfavorable benchmark comparisons remain visible.
- Policy implications identify a decision, an evaluated comparison, its constraints and its uncertainty.
- Data, code and source documentation are accessible under their applicable permissions and licenses.

