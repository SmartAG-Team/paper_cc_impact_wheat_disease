# Climate change, host development and Septoria in European wheat

Research code, retained observations, fitted monitoring models and publication data for **Host exposure windows shape projected Septoria damage and regional sowing adaptation in European wheat**.

**Author:** Gang Zhao · Northwest A&F University · [ORCID](https://orcid.org/0009-0003-0619-5849) · [Contact](mailto:gang.zhao@nwafu.edu.cn)

## Paper

Climate change alters both pathogen-favourable weather and the development of susceptible host tissue. This study combines wheat–Septoria process simulations with field monitoring, weather–host attribution, spatial analysis, fixed-production exposure and bounded sowing-date experiments. Continental simulations use daily temperature, humidity and precipitation, location and crop calendars. Their domain is restricted to cells containing harvested wheat; aggregation uses fixed SPAM2020 wheat-area or production weights.

The principal findings are:

- Late-century SSP5–8.5 increases the European harvested-area-weighted normalized affected-tissue integral by **2.66 days** relative to 1991–2020. The **50–60°N** belt contains **44.3%** of fixed wheat area and contributes **68.6%** of the positive change.
- Host attribution depends on temporal support: the host contribution changes from **−5.30 days** over whole developmental windows to **+0.57 days** over their shared elapsed time, using identical season pairs.
- Regional sowing choices selected on two climate models and evaluated on the third reduce the future own-window integral by **1.89 days** on complete-season support. A uniform 14-day delay reduces the disease proxy but increases exposure to maximum temperatures above 30°C by **2.05 days**.

The affected-tissue integral is a normalized symptom-tissue proxy. It does not measure physiological canopy-function loss or grain-yield loss. Sowing responses are bounded model sensitivities; their local feasibility and yield benefits remain unverified.

The separate observation-conditioned monitoring evaluation contains **2,017 assessments per design**, from **192 field-seasons at 121 sites**. Temporal RMSE is **17.15 percentage points** with predictive R² **0.625**; chronology-plus-50-km spatial RMSE is **17.31 percentage points** with R² **0.618**. Query context admits only strictly earlier disease observations from the same field and weather through the preceding day. These are adaptive retrospective comparisons; their skill does not validate autonomous climate projections.

## Repository structure

The repository has one model-code tree, one data tree and one paper workflow.

| Directory | Contents |
| --- | --- |
| `models/` | Executable crop, disease, monitoring and climate model components |
| `models/seasonal_septoria/`, `models/wheat_stb/` | Disease compartments, host development and coupled daily simulation |
| `models/phenology/`, `models/calibration/` | Inherited crop-development model and calibration routines |
| `models/monitoring/`, `models/learned/` | Selected monitoring implementation and inherited fitted feature-model interfaces |
| `models/climate/`, `models/methods/` | Climate simulation/aggregation and retained numerical methods |
| `models/reference_snapshot/` | Frozen source reference for archived numerical checks and portable export |
| `data/` | Observations, weather, fitted model assets, parameter values, numerical results and provenance |
| `data/final_monitoring/` | Twenty fitted monitoring folds, fixed memberships, features and original forecasts |
| `data/final_climate/` | Climate registry, parameters, alignment coefficients and forcing identities |
| `data/final_study/`, `data/final_analyses/` | Source observations, retained census and season-level analysis records |
| `data/paper/`, `data/paper_parameters/` | Numbered figure data, endpoint checks and manuscript parameter records |
| `data/phenology_parameters/`, `data/configurations/` | Crop-development parameters and model configurations |
| `data/derived/` | Recomputed analyses supporting the current paper |
| `data/reproductions/` | Fresh, separately named workflow outputs |
| `paper/` | Manuscript, figures, executable workflow, analyses and verification |
| `paper/scripts/`, `paper/analyses/` | Figure renderers, document generation and numerical analyses |
| `tests/`, `examples/` | Regression checks and a synthetic daily-engine example |

`paper/data` and `paper/parameters` link to their authoritative locations under `data/`; they are not additional data copies. Historical module aliases preserve the identity of model classes used by frozen recipes. The active workflow imports the shared `models/` implementation. [Daily engine documentation](docs/model_engine.md) describes state variables and input units.

## Paper files

| Material | Files |
| --- | --- |
| Main manuscript | [Word](paper/Manuscript.docx) · [PDF](paper/Manuscript.pdf) · [source](paper/Manuscript.md) |
| Supplementary information | [Word](paper/Supplementary_Information.docx) · [PDF](paper/Supplementary_Information.pdf) · [source](paper/Supplementary_Information.md) |
| Mathematical specification | [Word](paper/Mathematical_Model_Specification.docx) · [source](paper/Mathematical_Model_Specification.md) |
| Editorial cover letter | [Word](paper/Cover_Letter.docx) · [source](paper/Cover_Letter.md) |
| Figure index | [Data/script mapping](paper/figure_index.csv) · [machine-readable index](paper/figure_index.json) |
| Final figures and plotting data | [300-dpi PNGs](paper/figures/) · [numerical data](data/paper/) |
| File integrity | [Paper manifest](paper/publication_manifest.json) · [repository manifest](repository_manifest.json) |
| Submission bundle | [ZIP](paper/exports/GCB_Submission_20261024.zip) |

The manuscript is prepared for *Global Change Biology*; it has not been submitted or accepted.

## Installation

Use Python 3.12 from the repository root. Binary model and data files use Git LFS.

```sh
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Reproduce the paper with one command

```sh
python paper/run.py --output data/reproductions/my_run
```

The output directory must be new. The workflow runs the twenty saved monitoring models, reconstructs F1 states and onset histories, replays the 62-cell climate reference, recomputes attribution, production exposure, climate-response statistics, sowing-policy analyses and annual yield context, then regenerates all fourteen figure PNGs, Word/PDF documents and scientific checks.

The standard run uses the retained continental census and season-level experiment records. Their statistical analyses and all figure inputs are regenerated. It does not refit models or repeat the 810 continental simulations. The complete output package is written to `data/reproductions/my_run/package/`; progress, per-stage logs, input hashes and the completion receipt accompany it.

To rebuild the continental census and sampled process experiments from daily forcing before continuing through the analyses and figures:

```sh
python paper/run.py --full-climate --deep-process --output data/reproductions/full_model_run
```

This mode executes 810 annual climate jobs and reruns the weather–host attribution and sowing experiments. It requires the 687 daily forcing files listed in [the climate manifest](data/final_climate/scientific_manifest.json), and substantially more computation. Large forcing parquet files are excluded from Git; source receipts, hashes, alignment tables and monthly reference statistics are retained. The current workstation has the required forcing at its existing data path.

Print the execution plan without creating outputs, or regenerate figures and their analysis prerequisites:

```sh
python paper/run.py --dry-run
python paper/run.py --stage figures --output data/reproductions/figure_run
```

`--stage` accepts individual stages and includes their prerequisites. `--verify-forcing-hashes` additionally rereads all forcing contents. The workflow stops on a failed stage, writes its failure record and leaves the original inputs unchanged.

## Figure-to-data and script links

Every figure has one 300-dpi PNG at 19 cm width with 12 pt text. The index records the source data, rendering function, upstream analysis stage and executable reproduction command. Numbered data folders retain variables, units, panel identities and displayed intervals.

| Figure | Numerical data | Rendering script |
| --- | --- | --- |
| [fig1](paper/figures/fig1.png) | [data](data/paper/fig1/) | [figure_methods.py](paper/scripts/figure_methods.py) |
| [fig2](paper/figures/fig2.png) | [data](data/paper/fig2/) | [figure_data_overview.py](paper/scripts/figure_data_overview.py) |
| [fig3](paper/figures/fig3.png) | [data](data/paper/fig3/) | [figure_methods.py](paper/scripts/figure_methods.py) |
| [fig4](paper/figures/fig4.png) | [data](data/paper/fig4/) | [figure_validation_onset.py](paper/scripts/figure_validation_onset.py) |
| [fig5](paper/figures/fig5.png) | [data](data/paper/fig5/) | [figure_maps_v2.py](paper/scripts/figure_maps_v2.py) |
| [fig6](paper/figures/fig6.png) | [data](data/paper/fig6/) | [climate_response_insights.py](paper/scripts/climate_response_insights.py) |
| [fig7](paper/figures/fig7.png) | [data](data/paper/fig7/) | [climate_response_insights.py](paper/scripts/climate_response_insights.py) |
| [fig8](paper/figures/fig8.png) | [data](data/paper/fig8/) | [figure_results.py](paper/scripts/figure_results.py) |
| [fig9](paper/figures/fig9.png) | [data](data/paper/fig9/) | [figure_results.py](paper/scripts/figure_results.py) |
| [fig10](paper/figures/fig10.png) | [data](data/paper/fig10/) | [adaptation_results_figure.py](paper/scripts/adaptation_results_figure.py) |
| [figs1](paper/figures/figs1.png) | [data](data/paper/figs1/) | [figure_validation_v2.py](paper/scripts/figure_validation_v2.py) |
| [figs2](paper/figures/figs2.png) | [data](data/paper/figs2/) | [figure_results.py](paper/scripts/figure_results.py) |
| [figs3](paper/figures/figs3.png) | [data](data/paper/figs3/) | [yield_evidence_extension.py](paper/scripts/yield_evidence_extension.py) |
| [graphical_abstract](paper/figures/graphical_abstract.png) | [data](data/paper/graphical_abstract/) | [figure_results.py](paper/scripts/figure_results.py) |

## Verification

```sh
python -m pytest -q
python -m models.learned.replay_monitoring --output /tmp/wheat_monitoring_replay
python -m models.climate --check-inputs
python -m models.climate --replay-sample
python -m paper.analyses.source_support.verify --metrics --inventory
```

The portable F1/onset checks remain available under `paper/code/`. Repository-wide climate/source checks require the registered forcing. Export the verified manuscript package with `python -m paper.export`. Refresh the repository inventory after intentional changes with `python -m paper.analyses.source_support.inventory --write`.

## Provenance and use

Disease sources retain original BASF and Corteva record identities, treatment status and measurement definitions. Station weather retains source qualification and reanalysis fallbacks. Missing cultivar, replicate and anatomical information is not inferred. Climate outputs describe a normalized affected-tissue proxy under fixed land use and an imposed rainfed winter-wheat calendar; they do not establish physiological or grain-yield losses.

Third-party data and model components retain their recorded terms. Repository: [SmartAG-Team/paper_cc_impact_wheat_disease](https://github.com/SmartAG-Team/paper_cc_impact_wheat_disease). Public deposition of the current paper package and a persistent data/code identifier remain pending.
