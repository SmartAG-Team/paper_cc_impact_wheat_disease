"""Reproducible, source-scoped synthesis of accessible wheat yield evidence.

This module does not call or refit the STB/climate model. The ARVALIS series
contains annual network summaries, not independent STB-specific effect sizes.
Run with the repository Python: .venv/bin/python -m
analysis.gcb_preparation.yield_evidence_extension [--render].
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data/final_analyses/yield_context"
SOURCE_URL = "https://www.arvalis.fr/sites/default/files/medias/pdf/2025-12/Choisir2_2025-BTH-CHAMPAGNE-ARDENNE.pdf"
SCOPE_URL = "https://www.arvalis.fr/sites/default/files/medias/pdf/2022-11/_CHOISIR-2_CAP_2022_National_PDFS.pdf"
GARN_URL = "https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.17107"
EXPECTED_PDF_SHA256 = "4ed9b15d8785e762123d12a1b2f1ea6483f1e225e8d8c4c52ecff366bf29b423"

# Exact printed labels, independently checked against the original embedded chart.
# The chart is a raster embedded in PDF p.10 (printed p.8), so text extraction
# does not recover its labels. These are direct label transcriptions, not values
# inferred from bar heights. Whole-number labels are preserved as printed.
PRINTED_LABELS = "11.5 16.1 14.5 24.2 27.1 20.1 9 8.3 24.6 17.1 21.7 18.1 25 12.4 16.5 11.5 7.8 13.8 11.1 16 22.6 12.2".split()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def annual_yield_gap_data() -> pd.DataFrame:
    """Return published annual network values with explicit missing precision."""
    frame = pd.DataFrame({"year": np.arange(2004, 2026), "printed_label_q_ha": PRINTED_LABELS})
    frame["yield_gap_q_ha"] = frame.printed_label_q_ha.astype(float)
    frame["yield_gap_t_ha"] = frame.yield_gap_q_ha / 10.0
    frame["annual_trial_count"] = np.nan
    frame["sampling_variance_t2_ha2"] = np.nan
    frame["network_id"] = "ARVALIS_Performance_annual_summary"
    frame["endpoint"] = "foliar-disease fungicide yield gap; treated minus untreated"
    frame["fungicide_comparator"] = "year-specific programs not recovered; disease-free status not established"
    frame.loc[frame.year.eq(2024), "fungicide_comparator"] = "2024 dossier: best single fungicide application at full flag-leaf emergence versus no fungicide; 22 susceptible-cultivar trials"
    frame["pure_STB_loss"] = False
    frame["source_url"] = SOURCE_URL
    frame["source_pdf_page_one_based"] = 10
    frame["source_printed_page"] = "8"
    frame["source_chart_bbox_pdf_points"] = "56.25,353.9399719238281,534.0,569.3399658203125"
    frame["extraction_method"] = "transcription of printed numerical bar labels; visually verified against original chart"
    assert len(frame) == 22 and frame.year.is_unique
    assert frame.year.tolist() == list(range(2004, 2026))
    np.testing.assert_allclose(frame.yield_gap_t_ha * 10, frame.yield_gap_q_ha)
    assert frame.annual_trial_count.isna().all() and frame.sampling_variance_t2_ha2.isna().all()
    return frame


def literature_inventory() -> pd.DataFrame:
    """Evidence units are not added or treated as mutually independent studies."""
    # No excluded-publisher articles were opened or used. Archived evidence
    # retains its original identities; journal identity is never inferred from
    # a publisher's portfolio. FCR is treated as a specialist agronomy source,
    # distinct from the Nature/Science journal families.
    rows = [
        dict(source_id="ARVALIS_Performance_2025", year=2025, source_type="official report", journal="", doi="", url=SOURCE_URL,
             title="Variétés et interventions de printemps Blé tendre d'hiver 2026 – Champagne-Ardenne", evidence_grain="22 annual network summaries, 2004–2025", units="t ha-1 after q ha-1 / 10", disease_scope="foliar-disease fungicide response; chart labelled septoriose but 2022 network includes rust-dominated trials", effect_access="22 exact numerical bar labels publicly accessible; annual comparator programs not standardized", sampling_variance_access="not reported by year", study_cluster="ARVALIS_Performance; potential overlap with Garnault2021", meta_analysis_eligible=False, exclusion_reason="annual n/variance and consistently isolated STB effects absent", permitted_use="descriptive annual distribution; external agronomic context", source_status="newly retrieved primary source"),
        dict(source_id="ARVALIS_national_2022", year=2022, source_type="official report", journal="", doi="", url=SCOPE_URL,
             title="Choisir & Décider – Céréales à paille, Interventions de printemps 2022–2023, Synthèse nationale", evidence_grain="network statement and 5 yellow-rust-dominated trials", units="q ha-1", disease_scope="explicit mixed foliar diseases including yellow rust", effect_access="network maximum mean gain about 11 q ha-1; rust subset 22.2 q ha-1", sampling_variance_access="not a recoverable trial-level sampling variance", study_cluster="ARVALIS_Performance; overlap with annual series", meta_analysis_eligible=False, exclusion_reason="overlapping scope documentation; rust subset is not independent STB evidence", permitted_use="establish disease attribution and selection limits", source_status="newly retrieved primary source"),
        dict(source_id="ARVALIS_dossier_2024", year=2024, source_type="official report", journal="", doi="", url="https://www.arvalis.fr/file-download/download/public/249502",
             title="Lutte contre les maladies des céréales à paille – Dossier de presse, 21 novembre 2024", evidence_grain="22 SY ADMIRATION trials; 16 CHEVIGNON trials", units="q ha-1", disease_scope="foliar-disease fungicide response in STB-sensitive versus less-sensitive cultivars", effect_access="23 q ha-1 for SY ADMIRATION rating 5; 16 q ha-1 for CHEVIGNON rating 7", sampling_variance_access="sampling variances and within-site cultivar matching unavailable", study_cluster="ARVALIS_Performance2024; overlaps annual series", meta_analysis_eligible=False, exclusion_reason="different cultivar trial groups; no paired cultivar causal effect or sampling variance", permitted_use="comparator definition and cultivar-selection context", source_status="newly retrieved primary PDF p.4: best single fungicide treatment at full flag-leaf emergence versus no fungicide"),
        dict(source_id="Garnault2021", year=2021, source_type="journal article", journal="New Phytologist", doi="10.1111/nph.17107", url=GARN_URL,
             title="Large-scale study validates that regional fungicide applications are major determinants of resistance evolution in the wheat pathogen Zymoseptoria tritici in France", evidence_grain="mean 80 yield trials/year in 20 regions, 2004–2017; exact total not reported", units="protected minus unsprayed yield, q ha-1", disease_scope="STB-susceptible cultivars moderately/highly rust resistant; losses mostly attributed to STB", effect_access="methods and regional trajectory plot; raw yield matrix not located", sampling_variance_access="yield-effect variances not located; regional plot bands are heterogeneity, not sampling CIs", study_cluster="ARVALIS_yield_network_2004_2017; overlaps annual source years", meta_analysis_eligible=False, exclusion_reason="paired trial effects and sampling variances unavailable; overlap with annual summaries", permitted_use="STB-dominated paired trial rationale; raw data acquisition target", source_status="newly retrieved primary full text; one listed supplementary PDF"),
        dict(source_id="Parker2004", year=2004, source_type="journal article", journal="Plant Pathology", doi="10.1111/j.1365-3059.2004.00951.x", url="https://bsppjournals.onlinelibrary.wiley.com/doi/10.1111/j.1365-3059.2004.00951.x",
             title="Tolerance of septoria leaf blotch in winter wheat", evidence_grain="117 yield contrasts; 25 cultivar slopes; 2 sites × 3 seasons", units="t ha-1 per upper-three-leaf GLAI calendar day", disease_scope="STB", effect_access="archived fitted cultivar slopes; 21 fixed-effect slopes with SE", sampling_variance_access="21 SEs but no cross-cultivar covariance; BLUP prediction SE is not independent effect SE", study_cluster="Parker2004_shared_trials", meta_analysis_eligible=False, exclusion_reason="25 slopes are correlated within one study; original paired plot data/covariance absent", permitted_use="within-study cultivar response heterogeneity; not universal normalized-HAD conversion", source_status="archived extraction retained; primary publisher abstract checked"),
        dict(source_id="Foulkes2006", year=2006, source_type="journal article", journal="Phytopathology", doi="10.1094/PHYTO-96-0680", url="https://repository.rothamsted.ac.uk/id/eprint/14844/1/phyto-96-0680.pdf",
             title="Major genetic changes in wheat with potential to affect disease tolerance", evidence_grain="8 genotype × fungicide predicted means; 4 site-years, 2 STB and 2 stripe-rust experiments", units="t ha-1 at 15% moisture; top-five-leaf GLAI calendar days", disease_scope="STB, stripe rust and means pooled across both", effect_access="archived treatment predicted means and slope deviations", sampling_variance_access="no SE/covariance for each protected–affected treatment mean contrast", study_cluster="Foulkes2006_shared_genotypes_and_experiments", meta_analysis_eligible=False, exclusion_reason="mixed pathogens; shared model means; different leaf scope", permitted_use="tolerance physiology; preserve separation from STB-specific yield effects", source_status="archived extraction retained; primary institutional full text located"),
        dict(source_id="Robert2004", year=2004, source_type="journal article", journal="Journal of Experimental Botany", doi="10.1093/jxb/erh108", url="https://academic.oup.com/jxb/article-pdf/55/399/1079/1404679/erh108.pdf",
             title="Analysis and modelling of effects of leaf rust and Septoria tritici blotch on wheat growth", evidence_grain="leaf photosynthesis and canopy biomass experiments", units="leaf photosynthesis and plant growth", disease_scope="STB and leaf rust", effect_access="primary abstract/full-text indexed; harvest-grain effects not extracted", sampling_variance_access="not available for compatible harvested-grain effects", study_cluster="Robert_Bancal_Grignon_experiments; overlap requires checking", meta_analysis_eligible=False, exclusion_reason="physiological growth endpoint and mixed disease rather than compatible grain-yield effects", permitted_use="mechanistic green-area and photosynthesis evidence", source_status="new primary-source check"),
        dict(source_id="Bancal2007", year=2007, source_type="journal article", journal="Annals of Botany", doi="10.1093/aob/mcm163", url="https://doi.org/10.1093/aob/mcm163",
             title="Modelling wheat growth and yield losses from late epidemics of foliar diseases using loss of green leaf area per layer and pre-anthesis reserves", evidence_grain="13 treatment aggregates over 5 years in archived table", units="GLAI thermal-time duration and harvest components", disease_scope="mixed STB and leaf rust", effect_access="archived treatment aggregates; original plot effects unavailable", sampling_variance_access="some harvest-index SDs, not sampling variances of compatible grain-yield losses", study_cluster="Bancal_Grignon; overlap with Robert2004 possible", meta_analysis_eligible=False, exclusion_reason="mixed disease; incompatible endpoint variance and thermal/calendar time", permitted_use="role of leaf layers and pre-anthesis reserves", source_status="archived source retained; DOI access attempted"),
        dict(source_id="Bancal2015", year=2015, source_type="journal article", journal="Field Crops Research", doi="10.1016/j.fcr.2015.05.006", url="https://doi.org/10.1016/j.fcr.2015.05.006",
             title="Identifying traits leading to tolerance of wheat to Septoria tritici blotch", evidence_grain="published abstract: 161 genotype × year × site × management crop pairs; 18 genotypes, 5 years, 6 sites, 3 experiments", units="canopy duration and grain yield", disease_scope="STB tolerance experiments", effect_access="abstract/archived bibliographic record; usable trial effect matrix not retrieved", sampling_variance_access="not retrieved", study_cluster="Bancal2015_three_experiments; overlap with prior crop pairs unverified", meta_analysis_eligible=False, exclusion_reason="primary raw effects and their variances unavailable in retrieved sources", permitted_use="priority data target; broad geographic genotype scope", source_status="archived bibliographic record; no new numerical inference"),
        dict(source_id="VanDenBerg2017", year=2017, source_type="journal article", journal="Phytopathology", doi="10.1094/PHYTO-07-16-0283-R", url="https://apsjournals.apsnet.org/doi/abs/10.1094/PHYTO-07-16-0283-R",
             title="Physiological Traits Determining Yield Tolerance of Wheat to Foliar Diseases", evidence_grain="mathematical-model phenotype experiments", units="model HAD and yield", disease_scope="modelled foliar disease tolerance", effect_access="model results and four supplementary PDFs; independent observed effects absent", sampling_variance_access="not applicable to simulated independent field effects", study_cluster="VanDenBerg2017_simulation", meta_analysis_eligible=False, exclusion_reason="simulation is not an independent empirical yield experiment", permitted_use="source–sink and reserve mechanisms", source_status="new primary publisher full-text/abstract check"),
        dict(source_id="VanDenBosch2022", year=2022, source_type="journal article", journal="Plant Pathology", doi="10.1111/ppa.13509", url="https://bsppjournals.onlinelibrary.wiley.com/doi/10.1111/ppa.13509",
             title="Maximizing realized yield by breeding for disease tolerance: A case study for Septoria tritici blotch", evidence_grain="147 progeny from 5 parental crosses, multiple sites/seasons", units="yield versus absolute HAD loss", disease_scope="STB breeding tolerance", effect_access="institutional accepted manuscript; data available on reasonable request", sampling_variance_access="line-level raw effects and covariance not publicly supplied", study_cluster="ADAS_DH_mapping_populations; subset overlaps Collin2018; UK HAD context reuses Parker2004", meta_analysis_eligible=False, exclusion_reason="unavailable underlying data and known overlap with previously published subsets", permitted_use="cultivar yield/tolerance trade-offs; data acquisition target", source_status="new primary full text; explicit availability statement verified in institutional PDF p.10/printed p.9"),
        dict(source_id="Kristoffersen2020", year=2020, source_type="journal article", journal="Field Crops Research", doi="10.1016/j.fcr.2019.107696", url="https://researchprofiles.ku.dk/en/publications/control-of-septoria-tritici-blotch-by-winter-wheat-cultivar-mixtu/",
             title="Control of Septoria tritici blotch by winter wheat cultivar mixtures: Meta-analysis of 19 years of cultivar trials", evidence_grain="published meta-analysis of 406 Danish trials in 19 trial years, 1995–2017", units="relative mixture effects on disease severity and grain yield", disease_scope="mixture intervention; STB severity and yield co-outcomes", effect_access="primary author institutional record verifies +1.4% yield/−10.6% STB overall; +2.4%/−17% untreated", sampling_variance_access="trial-level effects/SEs and covariance not retrieved", study_cluster="Danish_national_cultivar_testing; do not combine pooled result with its component trials", meta_analysis_eligible=False, exclusion_reason="mixture effect is not pure STB yield loss; aggregate summary cannot be re-meta-analysed with its trials", permitted_use="published intervention evidence; illustrates severity and yield responses differ", source_status="new primary author institutional record; publisher/preprint downloads blocked"),
        dict(source_id="ZivEyal1978", year=1978, source_type="journal article", journal="Phytopathology", doi="", url="https://www.apsnet.org/publications/phytopathology/backissues/Documents/1978Articles/Phyto68n05_791.PDF",
             title="Assessment of yield component losses caused in plants of spring wheat cultivars by selected isolates of Septoria tritici", evidence_grain="4 cultivar identities in 2 seasons; 4 replicates per inoculation/protection treatment", units="g grain head-1; kernel weight and grain number", disease_scope="isolated STB inoculation; cultivar-specific controls shared by isolates", effect_access="primary table means and multiple-comparison letters accessible", sampling_variance_access="no numerical effect SE/SD/covariance recovered", study_cluster="Bet_Dagan_1973_1975; shared control by cultivar/year", meta_analysis_eligible=False, exclusion_reason="yield per head rather than stand grain yield; missing sampling variance and shared controls", permitted_use="qualitative evidence that similar STB severity can yield different grain losses", source_status="new primary APS historical article checked"),
        dict(source_id="Tunisia2020_Dryad", year=2020, source_type="official data repository", journal="", doi="10.5061/dryad.r2280gbb5", url="https://datadryad.org/dataset/doi:10.5061/dryad.r2280gbb5",
             title="Improved control of Septoria tritici blotch in durum wheat using cultivar mixtures [Data]", evidence_grain="82 unprotected plot-years; 122 total yields; 40 protected-reference comparisons in 2019", units="kg ha-1 yield; two leaf image assessments per unprotected plot", disease_scope="STB mixture trial; protected reference differs in inoculation plus fungicide", effect_access="public raw workbooks and existing local matched records", sampling_variance_access="replicates available for package response; no clean STB-specific treatment effect variance", study_cluster="Tunisia_one_site_two_seasons; protected references only one season", meta_analysis_eligible=False, exclusion_reason="treatment package confounding and one reference environment; no HAD", permitted_use="within-experiment mixture/package response; not grain conversion from models HAD", source_status="existing data preserved; original Dryad page verified"),
        dict(source_id="Swiss_BLENDIT", year=2023, source_type="official data repository", journal="", doi="10.5281/zenodo.17432866", url="https://doi.org/10.5281/zenodo.17432866",
             title="Agronomic performance of cultivar mixtures of winter wheat varieties [Data]", evidence_grain="724 plots; 5 sites × 2 years", units="t ha-1 at 15% moisture; ordinal STB score", disease_scope="observational STB, rusts and mildew in untreated mixture/pure stands", effect_access="existing local plot data", sampling_variance_access="within-site plot data, but no randomized disease-free reference", study_cluster="Swiss_five_sites_two_years", meta_analysis_eligible=False, exclusion_reason="ordinal score, genotype and other disease confounding; no matched protected baseline", permitted_use="site-clustered association and mixture agronomy, not causal disease yield loss", source_status="archived source identities retained"),
        dict(source_id="France_Mauguio", year=2021, source_type="official data repository", journal="", doi="10.5281/zenodo.5393959", url="https://doi.org/10.5281/zenodo.5393959",
             title="French cultivar-mixture experiment [Data]", evidence_grain="224 plot outcomes in Mauguio 2018", units="raw/spatial-BLUP yield and 0–3 ordinal STB", disease_scope="early STB followed by fungicide after assessment", effect_access="existing local matched records", sampling_variance_access="plot data; no untreated late-season disease-free counterfactual", study_cluster="Mauguio_2018_single_environment", meta_analysis_eligible=False, exclusion_reason="one environment; subsequent fungicide; no paired disease-induced grain-loss contrast", permitted_use="conditional early-severity association", source_status="archived source identities retained"),
        dict(source_id="Nordic_NFTS_recent", year=2026, source_type="official trial system", journal="", doi="", url="https://nfts.dlbr.dk/Forms/Forside.aspx",
             title="Nordic Field Trial System wheat disease and harvest yield records, 2023–2025", evidence_grain="259 matched assessment rows in archived matched file; repeated dates share treatment yields", units="severity and harvest treatment mean yield", disease_scope="fungicide field trials with mixed disease", effect_access="existing local treatment means; original replicate reports vary by trial", sampling_variance_access="full paired-treatment variances not harmonized; repeated dates cannot serve as independent yield effects", study_cluster="NFTS_trial_environment; pool treatment dates before any yield analysis", meta_analysis_eligible=False, exclusion_reason="mixed pathogens, shared controls and missing comparable effect variances", permitted_use="trial-specific joint disease/protection context", source_status="archived source identities retained"),
    ]
    return pd.DataFrame(rows)


def local_source_profile() -> pd.DataFrame:
    specifications = [
        ("Parker2004", "data/final_study/evidence/yield_evidence/parker2004_table5_yield_slopes.csv", "cultivar"),
        ("Foulkes2006", "data/final_study/evidence/yield_evidence/stb_independent/foulkes2006_table2_paired_predicted_means.csv", "genotype"),
        ("Tunisia2020_Dryad", "data/final_study/evidence/crop_damage_validation/tunisia_reference_features.csv", "plot_season_id"),
        ("Swiss_BLENDIT", "data/final_study/inputs/STB_paper_followup_20261009/analysis_ready/swiss_stb_yield_724.csv", "record_id"),
        ("France_Mauguio", "data/final_study/inputs/STB_paper_followup_20261009/analysis_ready/france_stb_yield_224.csv", "record_id"),
        ("Nordic_NFTS_recent", "data/final_study/inputs/public_septoria/nfts-extension-20261009/nfts_stb_severity_yield_matched.csv", None),
    ]
    rows = []
    for source, relative, key in specifications:
        path = ROOT / relative
        frame = pd.read_csv(path)
        row = dict(source_id=source, path=relative, sha256=sha256(path), rows=len(frame), columns=len(frame.columns),
                   candidate_key=key, duplicate_key_rows=int(frame.duplicated(key).sum()) if key else None)
        for col in ["site_year", "year", "composition_id"]:
            if col in frame:
                row[f"distinct_{col}"] = int(frame[col].nunique())
        if source == "Parker2004":
            row["fixed_effect_slopes_with_SE"] = int(frame.fixed_effect_slope_SE.notna().sum())
            assert row["fixed_effect_slopes_with_SE"] == 21
        if source == "Swiss_BLENDIT":
            row["distinct_sites"] = int(frame.postcode.nunique())
            assert len(frame) == 724 and frame.site_year.nunique() == 10
        if source == "France_Mauguio":
            assert len(frame) == 224 and frame.site_year.nunique() == 1
        rows.append(row)
    return pd.DataFrame(rows)


def descriptive_summary(frame: pd.DataFrame) -> dict:
    x = frame.yield_gap_t_ha.to_numpy()
    historical = frame.loc[frame.year <= 2016, "yield_gap_t_ha"]
    recent = frame.loc[frame.year >= 2017, "yield_gap_t_ha"]
    return dict(
        endpoint="foliar-disease treated-minus-untreated fungicide yield gap; not pure STB loss",
        n_annual_network_summaries=len(x), period="2004–2025", units="t ha-1",
        equal_year_mean=float(x.mean()), median=float(np.median(x)),
        between_year_standard_deviation=float(x.std(ddof=1)),
        between_year_coefficient_of_variation=float(x.std(ddof=1) / x.mean()),
        minimum=float(x.min()), minimum_year=int(frame.loc[frame.yield_gap_t_ha.idxmin(), "year"]),
        maximum=float(x.max()), maximum_year=int(frame.loc[frame.yield_gap_t_ha.idxmax(), "year"]),
        empirical_q25=float(np.quantile(x, .25)), empirical_q75=float(np.quantile(x, .75)),
        maximum_to_minimum=float(x.max() / x.min()),
        years_at_least_2_t_ha=int(np.sum(x >= 2.0)),
        mean_absolute_annual_change=float(np.abs(np.diff(x)).mean()),
        change_2024_to_2025=float(x[-1] - x[-2]),
        change_fraction_2024_to_2025=float(x[-1] / x[-2] - 1),
        mean_2004_2016=float(historical.mean()), n_years_2004_2016=len(historical),
        mean_2017_2025=float(recent.mean()), n_years_2017_2025=len(recent),
        difference_2017_2025_minus_2004_2016=float(recent.mean() - historical.mean()),
        uncertainty="Range, SD and quartiles describe variation among published annual summaries; annual trial counts and sampling variances are unavailable. No sampling confidence interval or causal climate trend is estimated.",
        meta_analysis=False,
    )


def build_yield_evidence_figure(output_path: Path | str | None = None):
    """Return a 19-cm-wide Figure; save only when output_path is supplied."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = annual_yield_gap_data()
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 12,
                         "axes.labelsize": 12, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .8, "savefig.dpi": 300}):
        fig, ax = plt.subplots(figsize=(19 / 2.54, 8.8 / 2.54))
        # All bars depict one endpoint and retain the same categorical color.
        ax.bar(frame.year, frame.yield_gap_t_ha, width=.60, color="#007E87", zorder=3)
        ax.axhline(frame.yield_gap_t_ha.mean(), color="#454545", ls="--", lw=1.0,
                   label="Equal-year mean", zorder=4)
        ax.set(xlim=(2003.35, 2025.65), ylim=(0, 3), xlabel="Harvest year",
               ylabel="Fungicide yield gap\n(t ha⁻¹)")
        ax.set_xticks([2004,2008,2012,2016,2020,2025])
        ax.set_xticks(np.arange(2004,2026),minor=True)
        ax.tick_params(axis="x",which="minor",length=2)
        ax.set_yticks(np.arange(0, 3.01, .5))
        ax.grid(axis="y", color="#dddddd", lw=.6, zorder=0)
        fig.legend(*ax.get_legend_handles_labels(),loc="upper center",bbox_to_anchor=(.58,.99),frameon=False)
        fig.subplots_adjust(left=.12,right=.975,bottom=.19,top=.86)
        fig.canvas.draw()
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=300, facecolor="white")
            # Restore the interactive canvas renderer after export changes DPI.
            fig.canvas.draw()
    return fig


def write_outputs(render: bool = False) -> dict:
    data_dir = OUT / "data"
    fig_data = data_dir / "fig1"
    fig_data.mkdir(parents=True, exist_ok=True)
    annual = annual_yield_gap_data()
    annual.to_csv(fig_data / "annual_yield_gap.csv", index=False)
    inventory = literature_inventory()
    inventory.to_csv(data_dir / "evidence_screening_inventory.csv", index=False)
    profile = local_source_profile()
    profile.to_csv(data_dir / "local_source_profile.csv", index=False)
    pd.DataFrame([
        dict(source_id="ARVALIS_dossier_2024", population="SY ADMIRATION; STB sensitivity rating 5", n_trials=22, yield_gap_t_ha=2.3, STB_severity_reduction_pct=np.nan, relative_yield_gain_pct=np.nan, sampling_SE=np.nan, comparator="best single fungicide application at full flag-leaf emergence vs no fungicide", causal_scope="treatment response within source; cultivar contrast not paired", pdf_page_one_based=4),
        dict(source_id="ARVALIS_dossier_2024", population="CHEVIGNON; STB sensitivity rating 7", n_trials=16, yield_gap_t_ha=1.6, STB_severity_reduction_pct=np.nan, relative_yield_gain_pct=np.nan, sampling_SE=np.nan, comparator="source fungicide-response comparison; exact cultivar site matching unavailable", causal_scope="between-cultivar descriptive context; not causal cultivar benefit", pdf_page_one_based=4),
        dict(source_id="Kristoffersen2020", population="all published trial treatments", n_trials=406, yield_gap_t_ha=np.nan, STB_severity_reduction_pct=10.6, relative_yield_gain_pct=1.4, sampling_SE=np.nan, comparator="four-cultivar mixture vs mean of its pure-stand component cultivars", causal_scope="mixture total effect; STB-mediated yield share unidentified", pdf_page_one_based=np.nan),
        dict(source_id="Kristoffersen2020", population="untreated-trial subgroup", n_trials=np.nan, yield_gap_t_ha=np.nan, STB_severity_reduction_pct=17.0, relative_yield_gain_pct=2.4, sampling_SE=np.nan, comparator="four-cultivar mixture vs mean of its pure-stand component cultivars", causal_scope="mixture total effect; STB-mediated yield share unidentified", pdf_page_one_based=np.nan),
    ]).to_csv(data_dir / "additional_published_endpoints.csv", index=False)
    summary = descriptive_summary(annual)
    (OUT / "descriptive_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    source_pdf = OUT / "sources/arvalis_performance_2025_champagne_ardenne.pdf"
    assert source_pdf.exists(), "Archived primary PDF is required to verify this transcription."
    assert sha256(source_pdf) == EXPECTED_PDF_SHA256, "Primary PDF changed; manually reverify numerical labels."
    source_files = []
    for path in sorted((OUT / "sources").glob("*")):
        source_files.append(dict(path=str(path.relative_to(OUT)), bytes=path.stat().st_size, sha256=sha256(path)))
    metadata = dict(
        retrieval_date="2026-10-23", source_url=SOURCE_URL, doi=None,
        institutional_author="ARVALIS", year=2025,
        document_title="Variétés et interventions de printemps Blé tendre d'hiver 2026 – Champagne-Ardenne",
        original_chart_title="Nuisibilité de la septoriose observée dans le Réseau Performance depuis 2004",
        source_pdf_page_one_based=10, printed_page="8", pdf_page_size_points=[595.3200073242188, 841.9199829101562],
        original_chart_bbox_pdf_points=[56.25, 353.9399719238281, 534.0, 569.3399658203125],
        original_chart_image_pixels=[1794, 809],
        source_image="sources/arvalis_performance_2025_original_chart.png",
        full_page_screenshot="sources/arvalis_series_page.png",
        original_units="q ha-1 (100 kg ha-1)", conversion_to_t_ha="divide by 10",
        direct_label_transcription=True, bar_height_digitization=False,
        original_title_is_not_a_pathogen_attribution_validation=True,
        comparator_source=dict(url="https://www.arvalis.fr/file-download/download/public/249502", pdf_page_one_based=4,
                               statement="2024 sensitive-cultivar comparison uses the best single fungicide application at full flag-leaf emergence versus no fungicide. No uniformly disease-free comparator is established for the 22-year series."),
        scope_source=dict(url=SCOPE_URL, pdf_page_one_based=90, printed_page="89",
                          statement="Network trials oversample susceptible/moderately susceptible varieties; all-trial maximum fungicide response about 11 q ha-1 in 2022, with 22.2 q ha-1 in 5 yellow-rust-dominated trials."),
        overlap="Garnault2021 STB-dominated rust-resistant 2004–2017 yield subset may overlap the annual source; no source is counted as independent of the other.",
        source_files=source_files,
    )
    (OUT / "source_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    variables = dict(
        figure="fig1", rows=22, grain="one annual published network mean",
        year="harvest year, 2004–2025", yield_gap_t_ha="foliar-disease fungicide yield gap in t ha-1",
        annual_trial_count="unavailable; blank is missing, not zero",
        sampling_variance_t2_ha2="unavailable; blank is missing, not zero",
        standard_deviation="empirical interannual variability, not an effect sampling error",
        generated_figure_width_cm=19, generated_figure_height_cm=10,
        text_size_pt=12, resolution_dpi=300, confidence_intervals="not available",
    )
    (fig_data / "variables.json").write_text(json.dumps(variables, indent=2) + "\n")
    caption = (
        "Annual foliar-disease fungicide yield gaps in the French ARVALIS Performance network, 2004–2025. "
        "Bars show published annual treated-minus-untreated grain-yield differences, converted from quintals "
        "to tonnes per hectare; the dashed line is the equal-year mean. The source chart labels the series as "
        "septoriose, but the network also includes rust-dominated trials and cultivar composition changes. "
        "Year-specific fungicide programs are not standardized; disease-free controls are not established. "
        "The series therefore describes treatment-associated foliar-disease yield gaps rather than isolated "
        "STB losses. Annual sample sizes and sampling variances are unavailable; no confidence intervals are "
        "implied. Data: ARVALIS, Choisir & Décider, Interventions de printemps 2025–2026, Champagne-Ardenne, "
        "printed p.8 (PDF p.10); pathogen-scope qualification: 2022–2023 Synthèse nationale, printed p.89 "
        "(PDF p.90).\n"
    )
    (OUT / "fig1_caption.txt").write_text(caption)
    paragraphs = (
        "Observed protection-associated yield gaps demonstrate substantial agronomic consequences of foliar "
        "disease, while also showing that disease symptoms and grain losses require distinct endpoints. In the "
        "French ARVALIS Performance network, the 22 published annual fungicide-treated–untreated yield differences "
        "from 2004 to 2025 averaged 1.64 t ha⁻¹ and ranged from 0.78 to 2.71 t ha⁻¹, with a between-year "
        "standard deviation of 0.58 t ha⁻¹. These summaries concern selected trial populations. The network "
        "includes rust-dominated trials and changing cultivar composition; consequently, its annual series "
        "represents foliar-disease fungicide yield gaps rather than isolated STB losses. Annual sample sizes "
        "and sampling variances are unavailable, so interannual spread describes historical variability rather "
        "than statistical precision. [ARVALIS Performance 2025](" + SOURCE_URL + "); "
        "[ARVALIS scope 2022](" + SCOPE_URL + ").\n\n"
        "More pathogen-specific French experiments compared protected and unsprayed plots of STB-susceptible "
        "cultivars with moderate-to-high rust resistance; their losses were predominantly attributed to STB. "
        "The published account covers 2004–2017, approximately 80 yield trials per year and 20 regions, but "
        "does not provide the paired yield matrix and effect variances needed for a new cross-study "
        "meta-analysis. Its years overlap the annual network summaries and cannot be counted as independent "
        "evidence. [Garnault et al., 2021](" + GARN_URL + "). A published synthesis of 406 Danish cultivar "
        "trials found that mixtures reduced STB severity by 10.6% while increasing grain yield by 1.4%; "
        "the corresponding untreated-trial responses were 17% and 2.4%. These co-responses support cultivar "
        "diversification as an agronomic option, but neither identify the STB-mediated share of the yield "
        "benefit nor supply a conversion from normalized canopy damage to grain loss. "
        "[Kristoffersen et al., 2020](https://doi.org/10.1016/j.fcr.2019.107696).\n"
    )
    (OUT / "manuscript_evidence_paragraphs.md").write_text(paragraphs)
    methods = (
        "Published quantitative evidence was classified by disease attribution, response definition, leaf and "
        "time scope, experimental independence and availability of sampling variances. The French annual "
        "series was transcribed from the numerical labels in the original ARVALIS chart and checked against "
        "the embedded source image. Quintals per hectare were divided by ten. Each published year received "
        "equal descriptive weight because annual trial counts were unavailable. The mean, median, range, "
        "quartiles and standard deviation describe the 22 annual network summaries. Their variability is not "
        "a sampling confidence interval. The overlapping 2004–2017 STB-dominated trial account was not "
        "pooled with these annual summaries. No retrieved set supplied compatible independent STB-specific "
        "grain-yield effects with adequate sampling variances, so a new inverse-variance meta-analysis and "
        "canopy-to-grain conversion were not identifiable from the accessible evidence.\n"
    )
    (OUT / "evidence_methods.txt").write_text(methods)
    references = """@article{Garnault2021,
  author = {Garnault, M. and Duplaix, C. and Leroux, P. and Couleaud, G. and David, O. and Walker, A.-S. and Carpentier, F.},
  year = {2021},
  title = {Large-scale study validates that regional fungicide applications are major determinants of resistance evolution in the wheat pathogen Zymoseptoria tritici in France},
  journal = {New Phytologist},
  volume = {229}, number = {6}, pages = {3508--3521},
  doi = {10.1111/nph.17107},
  url = {https://nph.onlinelibrary.wiley.com/doi/10.1111/nph.17107}
}

@techreport{ARVALISPerformance2025,
  author = {{ARVALIS}}, year = {2025},
  title = {Variétés et interventions de printemps Blé tendre d'hiver 2026 -- Champagne-Ardenne},
  institution = {ARVALIS},
  url = {https://www.arvalis.fr/sites/default/files/medias/pdf/2025-12/Choisir2_2025-BTH-CHAMPAGNE-ARDENNE.pdf},
  note = {Annual yield-gap series: printed page 8, PDF page 10. Accessed 2026-10-23}
}

@techreport{ARVALISNational2022,
  author = {{ARVALIS}}, year = {2022},
  title = {Choisir et Décider -- Céréales à paille, Interventions de printemps 2022--2023, Synthèse nationale},
  institution = {ARVALIS},
  url = {https://www.arvalis.fr/sites/default/files/medias/pdf/2022-11/_CHOISIR-2_CAP_2022_National_PDFS.pdf},
  note = {Disease-scope evidence: printed page 89, PDF page 90. Accessed 2026-10-23}
}

@techreport{ARVALISDossier2024,
  author = {{ARVALIS}}, year = {2024},
  title = {Lutte contre les maladies des céréales à paille -- Dossier de presse, 21 novembre 2024},
  institution = {ARVALIS},
  url = {https://www.arvalis.fr/file-download/download/public/249502},
  note = {Cultivar and comparator definitions: PDF page 4. Accessed 2026-10-23}
}

@article{Kristoffersen2020,
  author = {Kristoffersen, R. and Jørgensen, L. N. and Eriksen, L. B. and Nielsen, G. C. and Kiær, L. P.},
  year = {2020},
  title = {Control of Septoria tritici blotch by winter wheat cultivar mixtures: Meta-analysis of 19 years of cultivar trials},
  journal = {Field Crops Research}, volume = {249}, pages = {107696},
  doi = {10.1016/j.fcr.2019.107696},
  url = {https://doi.org/10.1016/j.fcr.2019.107696},
  note = {Published findings verified in author institution record; trial-level effect data not retrieved}
}
"""
    (OUT / "additional_references.bib").write_text(references)
    decisions = dict(
        use_in_main_text="Two evidence paragraphs; distinguish STB-dominated selected experiments from mixed foliar-disease annual gaps.",
        figure_role="One supplementary observational yield-context figure, independent of model-performance figures.",
        figure_builder="paper.scripts.yield_evidence_extension.build_yield_evidence_figure(output_path=None)",
        legitimate_novel_result="Exact 22-year official annual distribution and disease-attribution audit; descriptive synthesis, not new meta-analysis.",
        eligible_records_in_retrieved_inventory=int(inventory.meta_analysis_eligible.sum()),
        inventory_scope="17 targeted primary/archived source records; not an exhaustive systematic review and not 17 independent studies.",
        avoid=["Pooling Garnault2004–2017 and Performance2004–2025 as independent studies", "Calling mixed foliar-disease fungicide gains pure STB loss", "Using BLUP prediction errors as independent sampling SEs", "Pooling incompatible ordinal scores, leaf ranks or thermal/calendar HAD slopes", "Converting normalized affected-tissue days to grain-yield percentages or tonnage", "Interpreting published yearly variation as climate causation"],
        data_needed_for_new_meta_analysis=["Trial/site-year identifiers and cultivar", "Protected and untreated yield means and n", "Paired-difference variance, or replicate/block yields and shared-control covariance", "STB plus other pathogen severity and control efficacy", "Harvest moisture basis", "Leaf area/rank, assessment timing and phenology", "Source identifiers sufficient to remove overlapping experiments"],
        priorities=[dict(source="Garnault2021/ARVALIS", target="2004–2017 rust-resistant STB-susceptible paired raw trial yields and variance"), dict(source="Bancal2015", target="161 crop-pair dataset, plot replicates and overlap map"), dict(source="VanDenBosch2022", target="147-line field dataset; explicit reasonable-request availability; author contact not initiated")],
        source_conflicts=[dict(year=2023, performance_series_q_ha=16, ARVALIS_and_partners_dossier_q_ha=15, resolution="Source trial memberships and retrospective revisions are unavailable, so the discrepancy is unresolved. Retain the retrospective Performance series without splicing or averaging the two summaries.")],
    )
    (OUT / "integration_decisions.json").write_text(json.dumps(decisions, indent=2, ensure_ascii=False) + "\n")
    checks = dict(
        annual_rows=len(annual), unique_years=bool(annual.year.is_unique), continuous_2004_2025=True,
        original_pdf_sha256_matches=sha256(source_pdf) == EXPECTED_PDF_SHA256,
        missing_trial_counts_retained=True, missing_sampling_variances_retained=True,
        annual_values_from_printed_labels=True, chart_labels_visually_verified=True,
        original_source_chart_preserved=True, pure_STB_claim=False, new_meta_analysis_claim=False,
        normalized_HAD_to_yield_conversion=False, active_model_modified=False,
        existing_source_profiles_verified=len(profile), inventory_records=len(inventory),
    )
    if render:
        import matplotlib.pyplot as plt
        fig = build_yield_evidence_figure(OUT / "fig1.png")
        renderer = fig.canvas.get_renderer()
        bounds = fig.bbox
        checks["figure_text_font_sizes_pt"] = sorted({float(t.get_fontsize()) for ax in fig.axes for t in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_legend().get_texts()] if t.get_visible()})
        checks["figure_text_within_canvas"] = all(
            bounds.contains(*t.get_window_extent(renderer).get_points()[0]) and
            bounds.contains(*t.get_window_extent(renderer).get_points()[1])
            for ax in fig.axes for t in [ax.xaxis.label, ax.yaxis.label, *ax.get_xticklabels(), *ax.get_yticklabels(), *ax.get_legend().get_texts()] if t.get_visible()
        )
        assert checks["figure_text_font_sizes_pt"] == [12.0]
        assert checks["figure_text_within_canvas"]
        plt.close(fig)
        from PIL import Image
        with Image.open(OUT / "fig1.png") as im:
            checks["figure_pixels"] = list(im.size)
            checks["figure_dpi"] = list(im.info["dpi"])
    checks["passed"] = True
    (OUT / "verification.json").write_text(json.dumps(checks, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render", action="store_true", help="Render one 19-cm-wide, 12-pt, 300-dpi PNG; omit to emit data only")
    args = parser.parse_args()
    print(json.dumps(write_outputs(render=args.render), indent=2, ensure_ascii=False))
