# Empirical management evidence from cultivar-mixture trials

The French randomized field experiment provides a restricted within-trial harvest comparison. The Swiss constituent-control comparison fails the exchangeability gate because the required pure stands originate from external trial populations. Neither dataset establishes climate adaptation, disease-mediated grain loss or a European production benefit.

## French observed harvest comparison

The author-retained population comprises 381 plots: 179 pure stands and 202 mixtures. Complete raw harvest measurements are available for all 179 pure stands and 200 mixtures. Five complete mixtures lack a constituent pure stand, leaving 195 eligible comparisons involving 156 unique pure-stand controls. These comparisons include mixtures without an STB score; disease-score availability is not an eligibility condition. The source population consists of durum-wheat experimental lines in Mauguio in 2018, with alternating-row mixtures and fungicide protection after early disease assessment. [French dataset](https://doi.org/10.5281/zenodo.5393959), [experimental methods](https://doi.org/10.1111/nph.17915).

| Endpoint on the same 195 comparisons | Mean difference (t ha⁻¹) | Mean paired relative difference (%) | Below constituent baseline | Empirical lower-decile difference (t ha⁻¹) |
| --- | ---: | ---: | ---: | ---: |
| Raw plot harvests | +0.378044 | +7.896553 | 79/195 (40.5128%) | −1.724580 |
| Author spatial BLUP sensitivity | +0.176144 | +3.520659 | 84/195 (43.0769%) | −0.966026 |

The constituent baseline is the arithmetic mean of the two pure-stand yields from the same experiment. Mean raw mixture yield is 7.221972 t ha⁻¹ and mean constituent baseline is 6.843928 t ha⁻¹. The reported percentage averages pair-specific ratios; it is not the percentage difference between these two grand means. Yields retain the source dried-grain basis, with unspecified moisture percentage. The author-adjusted estimates use the whole experiment and therefore constitute sensitivity evidence rather than independent validation. [Numerical summary](results/french_summary.csv), [individual comparisons](results/french_contrasts.csv).

Shared controls connect 189 of the 195 comparisons within one dependency group; three smaller groups contain one, four and one comparisons. Each genotype combination occupies one plot. A confidence interval treating the 195 differences as independent would ignore control reuse. Independent environmental uncertainty is unavailable because only one site-year is represented. Mean raw differences range from +0.322929 to +0.423331 t ha⁻¹ under deletion of each constituent genotype and all its comparisons, from +0.183003 to +0.506949 under deletion of each occupied grid row, and from +0.319542 to +0.468624 under deletion of each occupied grid column. These are influence ranges, not confidence bounds or environmental transfer tests. [Dependency groups](results/french_dependency_groups.csv), [influence diagnostics](results/french_influence.csv).

The worst observed raw comparison is −5.704832 t ha⁻¹. Mixtures exceed the higher observed constituent yield in 80/195 comparisons (41.0256%); the mean difference against that hindsight benchmark is −0.396193 t ha⁻¹. Selection of the higher-yielding pure stand uses the same observed outcomes, so this comparison does not evaluate a prospective cultivar-selection policy. [Downside summary](results/french_summary.csv), [hindsight comparison](results/french_hindsight_selection.csv).

The mean measured thousand-kernel-weight difference is +0.494657 g across 195 comparisons; 80/195 have lower thousand-kernel weight than their constituent baseline. Thirty-one comparisons combine a positive yield difference with a negative thousand-kernel-weight difference. The grain-mass/kernel-number aggregation preserves the measured harvest basis. Thousand-kernel weight is a yield component; milling quality, market acceptance, protein, net return and input savings are not established by this endpoint. [Measured trade-offs](results/french_measured_tradeoffs.csv).

Random allocation supports treatment-composition comparisons within the observed experiment. Complete-case exclusions, potential interplot interactions and unreplicated combinations limit causal precision. The difference is a total harvest response under the trial's management. It cannot be assigned to STB because the disease measurement preceded later fungicide protection and does not describe season-long disease damage.

## Swiss control comparability and diagnostic contrasts

The current release contains 3,881 rows and 3,813 unique shared-field payloads. Exact linkage to the earlier release resolves 3,780 payloads; 33 are ambiguous and excluded. The ambiguous payloads produce 69 candidate mapping rows, which are not 69 additional observations. Among 690 identified mixture plots, 637 binary mixtures have observed harvests, 23 binary mixtures lack harvests, and 30 four-way mixture plots lack documented composition weights for the specified binary baseline. [Release mapping](results/swiss_release_mapping.csv), [mixture eligibility](results/swiss_mixture_eligibility.csv).

The identified mixture experiment includes HANSWIN and MONTALBANO pure checks. None of the 637 eligible binary mixtures has both constituent controls within this experiment. Matching external pure stands by year, postcode and source trial code yields 147 plot–control-group comparisons for 117 distinct mixture plots, 13 compositions and eight environments at four sites. The external trial-code meanings and management regimes are unresolved; matching does not establish exchangeability. [Swiss public dataset](https://doi.org/10.5281/zenodo.17432866), [same-trial constituent gate](results/swiss_same_trial_constituent_gate.csv), [external control coverage](results/swiss_constituent_coverage.csv).

| External trial code | Mixture plots | Environments / sites | Equal-environment mean difference (t ha⁻¹) | Mean paired relative difference (%) | Site-bootstrap stability range (t ha⁻¹) |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 37 | 4 / 2 | −0.831042 | −8.389915 | −1.457167 to −0.204917 |
| 31 | 12 | 4 / 4 | +0.279583 | +4.183008 | −0.109583 to +0.715833 |
| 32 | 22 | 4 / 4 | −0.155000 | −1.610760 | −0.631042 to +0.248125 |
| 40 | 52 | 6 / 3 | −0.299083 | −3.871886 | −0.556000 to −0.019250 |
| 42 | 21 | 2 / 1 | −0.546667 | −7.035166 | Unavailable: one site |
| 43 | 3 | 1 / 1 | −0.300000 | −3.864319 | Unavailable: one site |

All six rows are noncausal diagnostics. Their cultivar and environment subsets differ, so between-row differences are not effects of the source trial codes. Replicate means receive equal composition weight within environments, followed by equal environment weight. The bootstrap resamples sites, retaining years and shared controls together; two to four represented sites provide limited stability information. These ranges do not quantify causal uncertainty or European transfer. Swiss yield retains the prior archive's stated 15% grain-moisture basis; the current README does not independently verify that convention. [Summary](results/swiss_summary.csv), [environment results](results/swiss_environments.csv).

The equal-trial fraction of composition means below the baseline ranges from 25% to 100% across source groups. Protein differences range from −0.680833 to +0.387500 percentage points of dry matter; specific-weight differences range from −0.132500 to +5.933333 kg hl⁻¹. These are measured complete-trait, matched-source differences without causal attribution. Same-block comparisons against HANSWIN and MONTALBANO average +0.007485 and +0.180303 t ha⁻¹, respectively, but compare different cultivar compositions and lack explicit documentary confirmation of pure-check co-randomization. [Measured trade-offs](results/swiss_measured_tradeoffs.csv), [check comparisons](results/swiss_check_summary.csv).

Leave-site-out prediction of noncausal environment differences using the training mean improves on a zero-difference reference only for source code 40 among the four codes with geographic holdouts. Forward-year diagnostics retain recurring sites and cultivars. Their scope is transfer of observed source differences, not validation of management adaptation. [Grouped transfer diagnostics](results/swiss_transfer_summary.csv).

## Empirical scope

The French comparison supports an observed, experiment-specific total-yield contrast with heterogeneous outcomes and sensitivity to source spatial adjustment. The Swiss data support provenance, comparability and matched-source diagnostics. Neither source provides replicated climate treatments, validated future-climate management responses, season-long physiological disease-loss measurements or evidence for annual production stability. The machine-readable applicability assessment preserves these distinctions. [Applicability](results/applicability.json).
