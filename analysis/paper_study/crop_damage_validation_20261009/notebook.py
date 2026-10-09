"""Build and execute the inspectable scientific analysis companion."""
from pathlib import Path
import sys
import nbformat as nbf
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from .run import HERE,ROOT


def main():
    nb=nbf.v4.new_notebook()
    md=nbf.v4.new_markdown_cell;code=nbf.v4.new_code_cell
    nb.cells=[
      md('''# Field evidence for wheat disease-related yield responses

The expanded collection contains 423 Tunisia/Nordic assessments sharing 272 source yield keys. Exclusion of post-harvest assessments and a pooled-control alias leaves 412 assessments and 263 yield outcomes. German source harmonization adds 3,264 cultivar–management comparisons in 16 site-years at five locations. These populations retain distinct measurement definitions.

Endpoint severity does not establish a transferable physiological yield-loss coefficient. Nordic stage regressions have unstable conditional severity effects, Tunisian leaf-based models fail cross-year yield transfer, and German endpoint severity adds little to management descriptors. Climate-derived yield indices consequently remain conditional on the published canopy conversion.

Data: 9 October 2026 collection. All yield responses retain their signs; repeated leaf measurements share an outcome. Whole trials, years or geographic sites define held-out groups. Protected severity remains missing where unobserved.'''),
      code('''from pathlib import Path
import sys, json
import numpy as np
import pandas as pd
from IPython.display import display, Image
ROOT = Path.cwd()
assert (ROOT / "analysis/paper_study").is_dir()
from analysis.paper_study.crop_damage_validation_20261009 import run, briwecs, verify, figures
from analysis.paper_study.nordic_yield_validation_20261009 import run as nordic, figures as nordic_figures
HERE = run.HERE
print("Python", sys.version.split()[0], "NumPy", np.__version__, "pandas", pd.__version__)'''),
      md('''## Assessment and yield hierarchy

Tunisia has 82 unprotected plot-season yields, each linked to two leaf-rank assessments, and 40 protected reference yields in 2019. Nordic observations are published treatment means rather than independent plots. The Tunisian moisture basis is unspecified; Nordic yield is at 15% moisture and German yield is dry mass. Absolute yields and source-specific severity percentages are not fitted as exchangeable observations.'''),
      code('''plots, reference, quality = run.prepare()
clean, endpoints, contrasts, nordic_quality = nordic.prepare()
display(pd.DataFrame({
    "Population": ["Tunisia leaf assessments", "Tunisia plot-season yields", "Tunisia reference contrasts", "Eligible Nordic assessments", "Eligible Nordic yield outcomes", "Nordic treatment contrasts"],
    "Records": [164, len(plots), len(reference), len(clean), len(endpoints), len(contrasts)],
    "Environment groups": [2, 2, 1, 11, 11, 5]
}))
assert not plots.plot_season_id.duplicated().any()
assert quality["protected_severity_observations"] == 0
assert len(clean) + 2*len(plots) == 412'''),
      md('''## Nordic prediction and assessment stage

The response is 1 − untreated yield / treated yield. Same-date treatment differences in observed severity are predictors. Stage zero is missing, and control-only stem-extension assessments do not supply a treated-plot measurement. The primary 62-contrast population spans five trials. Assessment stage varies mainly between trials, so its interaction cannot isolate stage effects from other environmental differences. Nested model choice uses training trials only. The forward test fits 2024 and evaluates 2025. Assessment dates are symptom-measurement dates, rather than infection dates.'''),
      code('''predictions = nordic.leave_trial_out(contrasts, "affine_ridge", ["delta_severity"])
benchmark = nordic.leave_trial_out(contrasts, "training_mean", [])
display(pd.DataFrame([{"Model": "Severity + intercept", **nordic.score(predictions)}, {"Model": "Training mean", **nordic.score(benchmark)}])[["Model", "contrasts", "trials", "RMSE_pp", "R2"]].round(3))
coefficients = pd.read_csv(nordic.HERE / "stage_coefficient_diagnostics.csv")
display(coefficients)
image_path = nordic_figures.validation_figure()
display(Image(filename=str(image_path), width=1000))'''),
      md('''The stage model’s RMSE is 3.99 percentage points versus 5.79 for its matched benchmark, but the paired trial-bootstrap interval for the difference spans −3.38 to +1.09 points. Three of four fitted conditional severity slopes have the opposite sign to a monotone physiological damage response. The secondary sign-constrained comparison has RMSE 7.55 points. Its growth-stage basis is ordinal, rather than elapsed time. These diagnostics do not validate a stage-dependent yield-loss function.'''),
      md('''## Tunisian leaf-rank evidence

Flag leaves were assessed at GS75 in both years; the immediately lower leaf was assessed at GS61 in 2018 and GS73 in 2019. Leaf identity, assessment timing and annual environment therefore cannot be separately estimated. One measurement per leaf does not define a same-leaf time integral. Protected-reference gaps describe a management-package comparison. Mixture and replicate holdouts in 2019 remain within the same environment.'''),
      code('''fresh = run.grouped_predictions(plots, "year", "two_leaf_ridge", ["severity_flag", "severity_lower"], "yield_t_ha")
mean = run.grouped_predictions(plots, "year", "training_mean", [], "yield_t_ha")
display(pd.DataFrame([{"Model": "Both leaf severities", **run.score(fresh, "year", 1)}, {"Model": "Training mean", **run.score(mean, "year", 1)}]).round(3))
display(Image(filename=str(figures.tunisia_figure()), width=1000))'''),
      md('''Two-leaf severity does not improve cross-year absolute-yield prediction: RMSE is 3.32 t ha⁻¹ versus 3.24 for the benchmark. Within 2019, flag-leaf severity has mixture-held-out response error of 9.92 points versus 10.06 for the benchmark. The leaf-based relationship remains an association at one site.'''),
      md('''## German multi-environment comparisons

Original BRIWECS site files retain fractional disease scores and missing values. Same location, year, cultivar, nitrogen level and source-coded water regime define paired treatment means. Only unsuffixed HN/LN × NF/WF treatments enter; drought-suffixed references are excluded. Cultivar identities can occur on both sides of an environmental holdout; prediction for unseen cultivars is a separate task. No protected severity is supplied for missing observations. Published rounded outputs and duplicate archive copies do not create additional observations.'''),
      code('''german, german_quality = briwecs.prepare()
display(german.groupby("Location").agg(Comparisons=("contrast_id", "size"), Site_years=("site_year", "nunique"), Negative_responses=("response_fraction", lambda x: (x < 0).sum())))
metric = pd.read_csv(HERE / "briwecs_model_comparison_metrics.csv")
display(metric[metric.domain.eq("unprotected_endpoint") & metric.validation.eq("leave_location_out")][["model", "n", "groups", "RMSE_pp", "baseline_RMSE_pp", "skill"]].round(3))
display(Image(filename=str(figures.briwecs_figure()), width=1000))'''),
      md('''Unprotected endpoint severity has location-held-out error of 18.51 points, versus 18.48 for the training mean. A nitrogen-and-water model improves forward-transfer error from 19.56 to 17.55 points; adding severity changes it to 17.51. Multiple diseases, incomplete replicate coverage and undocumented assessment dates limit interpretation as Septoria-specific damage.'''),
      md('''## Physiological model requirements

A daily crop-growth calculation can connect layer-specific healthy leaf area to intercepted radiation, biomass accumulation, pre-anthesis reserves, grain number and grain filling. Disease damage and natural senescence require separate observations. Stem extension and upper-leaf emergence matter before flowering, whereas flowering is a reference for grain filling. The current climate model predicts crop timing and assumed functional damage; it does not calculate the full crop carbon balance.

The WHEATPEST damage framework has been compared across HERMES, WOFOST_GT, SSM_WHEAT and DSSAT-Nwheat ([Bregaglio et al., 2021](https://doi.org/10.1016/j.fcr.2021.108108)). Layer-specific green area and pre-anthesis reserves provide physiological structure ([Bancal et al., 2007](https://doi.org/10.1093/aob/mcm163)), while cultivar-dependent canopy interception and source–sink relationships preclude a universal severity–yield slope ([Bancal et al., 2015](https://doi.org/10.1016/j.fcr.2015.05.006)). These references support model architecture; their reported fits are not substitutes for independent validation of this study.

Independent crop calibration needs protected-plot phenology, canopy area, biomass and harvest data. Disease-damage evaluation needs matched untreated/protected trajectories by final-leaf rank, natural senescence, intercepted radiation and yield components. Entire sites and years remain outside fitting and parameter selection. Shared controls and cultivar identities remain grouped. Infection dates cannot be validated using dated symptom assessments alone.'''),
      md('''## Journal precedents and outcome definitions

Nature Food climate-impact papers connect a defined production problem to measured or modelled yield outcomes. [Silva et al. (2026)](https://www.nature.com/articles/s43016-025-01286-w) distinguish potential, water-limited and actual wheat yield and quantify contributions to a regional yield plateau. [McDonald et al. (2022)](https://www.nature.com/articles/s43016-022-00549-0) combine field observations and crop simulations to evaluate feasible cropping-calendar changes and production trade-offs. These precedents support result-led regional comparisons and specific adaptation mechanisms; projected disease damage alone does not establish intervention benefits.

Prediction metrics require the target and holdout task. In Nature Communications, [de los Campos et al. (2020)](https://www.nature.com/articles/s41467-020-18480-y) assign entire year–location trials to cross-validation folds. Mean within-trial correlation of 0.58 describes cultivar-yield prediction, rather than 58% classification accuracy. Prediction for previously untested cultivars has lower correlation. Severity RMSE, symptom-detection sensitivity, relative-yield response error and total-yield prediction therefore remain separate quantities.

The closest crop–disease climate-coupling example, [Pequeno et al. (2024)](https://www.nature.com/articles/s41558-023-01902-2), is in Nature Climate Change. It separates disease-related production vulnerability from direct climate effects on wheat yield. Crop calibration uses treated plots from the same field experiments, so its field comparison is not equivalent to fully independent calibration and validation. The distinction between calibration fit, held-out prediction and conditional climate projection remains essential.'''),
      md('''## Source and arithmetic verification

Native Nordic numeric cells, Tunisian source workbooks and German treatment means are reconciled independently. Grouped scores are recalculated from prediction tables rather than the model-scoring helpers. The complete local collection manifest checks source copies against their original paths.'''),
      code('''native_nordic = verify.native_nordic_cells()
native_tunisia = verify.native_tunisia()
source_means = verify.briwecs_source_means()
metric_checks = verify.grouped_scores()
display(pd.DataFrame({"Check": ["Native Nordic numeric cells", "Tunisian leaf records", "German source treatment means", "Grouped score comparisons"], "Verified": [native_nordic, native_tunisia["leaf_records"], source_means, metric_checks]}))'''),
      md('''## Source records

[Combined assessment CSV](../../../data/stb_collection_20261009/analysis_ready/combined_severity_yield_assessments.csv); [collection workbook](../../../data/stb_collection_20261009/STB_data_collection.xlsx); [source catalogue](../../../data/stb_collection_20261009/dataset_catalogue.csv).

Primary data sources: [Tunisia Dryad archive](https://doi.org/10.5061/dryad.r2280gbb5), [Nordic Field Trial System](https://nfts.dlbr.dk/Forms/Forside.aspx), [BRIWECS archive](https://doi.org/10.6084/m9.figshare.27910269.v1). Published treatment aggregates, fitted response grids and model inputs retain their separate evidence status. Email drafts are correspondence material and supply no observations.''')
    ]
    nb.metadata.kernelspec={'display_name':'Python 3','language':'python','name':'python3'}
    client=NotebookClient(nb,timeout=180,kernel_name='python3',resources={'metadata':{'path':str(ROOT)}})
    client.execute()
    nbf.validate(nb)
    path=HERE/'Field_yield_validation.ipynb';nbf.write(nb,path)
    body,_=HTMLExporter().from_notebook_node(nb)
    (HERE/'Field_yield_validation.html').write_text(body)
    assert all(cell.get('execution_count') is not None for cell in nb.cells if cell.cell_type=='code')
    assert not any(output.output_type=='error' for cell in nb.cells if cell.cell_type=='code' for output in cell.outputs)
    print(str(path));print('Executed code cells:',sum(cell.cell_type=='code' for cell in nb.cells))


if __name__=='__main__':main()
