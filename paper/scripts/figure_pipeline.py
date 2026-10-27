"""Build source-backed Methods, validation and climate displays from fixed inputs."""
import json
from pathlib import Path
import re
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.monitoring.inputs import sha
from models.monitoring.innovation_common import ROOT
from .figure_registry import MAIN_ROLES, SUPPLEMENT_ROLES, ROLE_NAMES, figure_number
from .figure_methods import build_workflow, build_attribution_design
from .figure_data_overview import prepare_cropland,prepare_overview,overview_figure
from .figure_validation_v2 import (prepare_validation, density_figure,
    diagnostic_figure, STYLE)
from .figure_validation_onset import validation_onset_figure,onset_eligibility
from .figure_maps_v2 import build_climate_map
from .climate_response_insights import build_latitude_environment, build_climate_phenology
from .adaptation_results_figure import build_adaptation_figure
from .yield_evidence_extension import build_yield_evidence_figure
from .figure_results import attribution_figure, exposure_figure, support_figure, graphical_figure

OUT=ROOT/'paper'
EXT=ROOT/'data/final_analyses/reference'
OLD=ROOT/'data/final_monitoring/reference'
CHRONOLOGY=ROOT/'data/final_monitoring/cohort/training_observation_ledger.csv'

CAPTIONS={
 'data_overview': 'Retained field observations and sampling characteristics. (a) All 128 source locations '
    'over SPAM2020 physical wheat area, expressed as a percentage of native 0.25° cell area (IFPRI, 2026). '
    'Circles denote BASF and triangles denote Corteva. Grey land is outside mapped wheat support; pale blue denotes water. '
    '(b) Assessment counts by season and source. '
    '(c) Recorded disease-score distributions in 10-percentage-point bins; the final bin includes 100%. '
    '(d) Unique assessment dates per field-season. The 2,304 assessments represent 218 field-seasons '
    'during 2014–2019 and explicitly untreated, zero-spray controls. Each assessment aggregates one '
    'field, date and source-native leaf category. Native score and anatomical definitions remain qualified. '
    'Natural Earth outlines provide geographic context (Natural Earth, n.d.). '
    'Map lines delineate study areas and do not necessarily depict accepted national boundaries.',
 'workflow': 'Study workflow and information settings. (a) Historical observation-conditioned monitoring uses dated '
    'untreated BASF/Corteva assessments and qualified station/reanalysis weather. Outer statistical fits use earlier '
    'seasons; query context admits strictly earlier disease from the same field and weather through the preceding day. '
    'Query-day disease is withheld across all native codes. Inherited crop-feature ancestry remains qualified. '
    '(b) Autonomous climate simulations use daily mean/maximum temperature, humidity and precipitation with imposed '
    'sowing and site coordinates. Temperature, daylength and vernalization determine host development; host tissue '
    'availability affects disease progression and normalized affected-tissue duration. Disease does not feed back '
    'on host development. The dashed historical flag-leaf replay branch compares retained states with reused '
    'observations (115 records: 45 calibration and 70 original-2019 evaluation records); it supplies no new fit or independent physiological validation. The monitoring evaluation comprises five temporal and 15 spatial folds, 2,017 forecasts per design, 192 field-seasons and 121 sites. Query-day, later and other-field disease is withheld. The two pathways have distinct '
    'inputs and endpoints; monitoring accuracy is not transferred to continental projections.',
 'attribution_design': 'Factorial attribution and integration support. (a) Historical/future disease weather '
    '(first subscript) and historical/future host trajectories (second subscript) define four loss terms. Host '
    'replacement includes tissue availability, area, renewal and window timing while disease parameters remain '
    'fixed. (b) Whole-window losses use each host trajectory’s own interval; shared-time losses use the intersection '
    'of those intervals in elapsed days since sowing for all four combinations. Shading indicates common time. '
    'Positions are illustrative and unitless rather than observed dates. Both integration definitions answer '
    'conditional biological questions; restricting support does not identify a uniquely correct physiological '
    'definition or an adaptation benefit.',
 'monitoring': 'Monitoring forecasts and weighted observation density. (a) Five historical temporal '
    'splits; (b) 15 chronology-plus-50 km spatial splits. Both retain 2,017 assessments from 192 field-seasons at '
    '121 sites. Each cell spans 2.5 percentage points on both axes. Its colour represents the sum of original '
    'evaluation weights in the cell, divided by total weight for that design and expressed as a percentage. '
    'Cells sum to 100% per design; marginal histograms use the same weights. A shared logarithmic colour '
    'scale depicts weighted density, not prediction error or confidence. Dashed lines indicate 1:1 agreement. '
    'RMSE, MAE and predictive R² use all individual forecasts, without fitting or averaging the plotted cells. '
    'Parentheses give descriptive 95% percentile intervals from 2,000 paired site resamples of fixed forecasts, with the weighted '
    'target mean recomputed in every resample. They exclude adaptive model-selection uncertainty. All query-day '
    'disease is withheld; earlier same-field disease and previous-day weather are admitted. Native score '
    'definitions and inherited crop-feature ancestry remain qualified. These results do not validate autonomous '
    'climate projections.',
 'diagnostics': 'Error concentration and field-scale heterogeneity. (a) Prediction bias within observed-disease bands, positioned at their weighted mean observed value; this outcome-conditioned diagnostic differs from forecast calibration. (b) Cumulative percentage of 192 field-seasons per design at or below each RMSE. Each field-season has one vote and retains original relative assessment weights internally; pooled RMSE is not their arithmetic mean. (c) Paired temporal and spatial RMSE by visit context; the focused 14–22-pp axis includes all estimates. Labels give assessment counts. First and last categories also include only-visit fields. Blue circles/solid lines and orange squares/dashed lines denote temporal and spatial designs; grey segments join estimates for the same context. Visit groups differ in case mix and do not identify the causal value of an extra visit.',
 'climate_change': 'Climate-response magnitude and direction agreement. (a) SSP1–2.6; (b) SSP2–4.5; '
    '(c) SSP5–8.5: equal-three-model mean change in normalized affected-tissue duration, 2071–2100 relative to '
    '1991–2020. (d) Three-model direction agreement under SSP5–8.5. Coloured cells represent wheat within '
    '14,941 native 0.25° footprints, with common EPSG:3035 extent. Grey land is outside mapped wheat '
    'support; pale blue denotes water. Dark grey denotes unavailable responses. In (a)–(c), white is zero '
    'disease change within wheat cells. The shared colour bar retains changes beyond ±12 days through its '
    'triangular ends. Model agreement is not a probability; affected-tissue duration is a symptom-tissue '
    'proxy. Natural Earth outlines provide geographic context (Natural Earth, n.d.). Map lines delineate study areas and do not '
    'necessarily depict accepted national boundaries.',
 'attribution_results': 'Weather, host and net attribution under eight settings on exactly 4,117 common '
    'model–draw–year pairs. Small points represent three climate models, diamonds their equally weighted mean, '
    'and horizontal lines their minimum–maximum spread, not confidence intervals. Host replacement includes '
    'tissue availability, area, renewal and window timing. Threshold and accounting perturbations are fixed '
    'structural diagnostics. Whole-window and shared-time integration answer distinct conditional questions. '
    'Retained weighted support is 72.02% and differs among models; its distribution is shown in Figure S1.',
 'production_exposure': 'Magnitude thresholds and production geography define different aspects of exposure. '
    '(a) Percentage of fixed European baseline production where every climate model strictly exceeds each '
    'specified increase in normalized affected-tissue days by 2071–2100. Lines connect evaluated thresholds '
    'without implying a fitted response. (b) Baseline production partitioned by direction agreement under '
    'SSP5–8.5, for the five largest production regions and all remaining regions combined. The complete '
    'denominator is 241.845 million tonnes, including unavailable outcomes. These are baseline amounts '
    'located in affected cells, not future production or tonnes of grain lost. The imposed scenario is '
    'rainfed winter wheat, while production weights describe all reference wheat.',
 'support': 'Design-weighted area–time support for the identical eight-setting comparison, by climate model. Blue segments show retained support; light grey completes the fixed requested population to 100%. Labels give retained percentages. The denominator retains original design weights and differs from the unweighted count of pairs. Model-specific support qualifies the attribution contrasts.',
 'graphical_abstract': 'Host exposure windows change climate–disease attribution in European wheat. On the '
    'same 4,117 model–draw–year pairs, host contributions are −5.30 normalized affected-tissue days over '
    'whole host windows and +0.57 days over shared elapsed time. Intervals are schematic, without a date scale. '
    'The sign change concerns conditional model attribution of a symptom-tissue proxy.'}


# Compact publication captions retain definitions; design inventories remain in Methods/SI/data.
CAPTIONS.update({
 'workflow': 'Study workflow and information settings. (a) Retrospective monitoring uses dated untreated BASF/Corteva assessments and qualified station/reanalysis weather. Historical fitting precedes the evaluated season; query context admits strictly earlier own-field disease and weather through the preceding day. Query-day, later and other-field disease is withheld. Five temporal and 15 spatial folds retain 2,017 forecasts per design, 192 field-seasons and 121 sites. Inherited crop-feature ancestry remains qualified. (b) Autonomous simulations use daily mean/maximum temperature, humidity and precipitation, imposed sowing and coordinates. Host development determines tissue availability and exposure windows; disease does not feed back on development. The normalized symptom-tissue integral is not measured physiological loss. The dashed branch is a retrospective flag-leaf replay: 45 calibration and 70 original-2019 evaluation records, without new fitting or independent physiological validation. Monitoring skill is not transferred to climate projections.',
 'latitude_environment': '(a, b) Latitude profiles of normalized affected-tissue change for 2031–2060 (a) and 2071–2100 (b), relative to 1991–2020. Lines and markers are equal-three-GCM means of area/valid-pair-weighted one-degree bins; ribbons span GCMs. Display requires at least 0.1% of fixed wheat area and 90% paired-area-time coverage in every GCM; unsupported bins break curves. (c) Fixed wheat-area shares and positive/absolute-negative contribution shares by environmental zone, SSP5–8.5, 2071–2100. Whiskers span GCMs. Each directional denominator sums harvested area × valid-pair fraction × positive or absolute-negative cell change across the domain; each set sums to 100%. Outside EEA and Unassigned cells remain included. Affected-tissue days integrate symptom-gated damage over modelled GS65–85, weighted by reference upper-three-leaf area and normalized by nominal canopy area; they are a symptom-tissue proxy.',
 'climate_phenology': 'Co-changes under SSP5–8.5, 2071–2100 relative to 1991–2020. (a) Area/valid-pair-weighted affected-tissue change in two-day bins of F1 symptom-timing shift relative to modelled GS65; negative shifts indicate earlier symptoms. Equal timing/complete-season valid-pair counts identify identical paired years. Displayed bins represent at least 0.5% of each GCM’s full-domain paired-area weight. (b–d) GCM-specific weighted changes in GS65–85 mean temperature (b), accumulated rainfall (c) and duration (d), grouped by unanimous increase, mixed/zero signs or unanimous decrease in affected-tissue damage. Membership is shared across GCMs; unavailable cells remain in source data. Window climate incorporates shifted phenology. Timing is an internal state entering the tissue endpoint, so associations establish neither independent validation nor causal mediation. GS65 is a modelled gate, not independently validated anatomical anthesis. Affected-tissue days are the normalized symptom-tissue proxy defined in Figure 5.',
 'adaptation': '(a) Fixed sowing shifts under unchanged weather, SSP5–8.5, 2071–2100: own-window affected-tissue changes on complete-season and common-calendar supports. (b) Leave-one-GCM-out regional choices trained separately on each support; choices agree across three folds. (c) The complete-season-trained policy evaluated on complete-season support: affected-tissue versus GS65–85 Tmax >30°C exposure changes. Lines/points are equal-GCM means; bands/bars span three GCMs, not confidence intervals. Complete-season support requires all five options complete in both periods/all GCMs: 1,902/1,920 draw-years, 62 cells, 99.64% design weight. Adding nonempty all-option calendar overlap retains 1,656 draw-years, 57 cells, 77.46%. Complete-season sensitivity is post-audit; calendar support was preregistered. Proxy responses do not establish agronomic or yield optima.'
})

CAPTIONS['workflow'] += ' Icons are conceptual; solid arrows show dependencies and the dashed arrow marks retrospective checking.'

# Published cross-references follow scientific roles when display order changes.
CAPTIONS['attribution_results']=re.sub(r'Figure S\d+',
    'Figure '+figure_number('support'),CAPTIONS['attribution_results'])
CAPTIONS['climate_phenology']=re.sub(r'defined in Figure \d+',
    'defined in Figure '+figure_number('latitude_environment'),CAPTIONS['climate_phenology'])

CAPTIONS['climate_phenology'] += ' Timing associations describe the matched symptom-event subset; geographic retention is reported in Section S3.'
CAPTIONS['monitoring']='(a) Temporal monitoring: 2,017 assessments from 192 field-seasons, 2015–2019; cells span 2.5 percentage points on both axes. Colour shows evaluation weight, not prediction error or confidence. (b) Forecast calibration: temporal (blue circles) and 50 km spatial (orange squares). Fixed 10-pp bins; bars: 95% site-bootstrap intervals (1,000 resamples). Grey lines indicate 1:1. (c) Cumulative evaluation weight with absolute error at or below each threshold; bands: 95% paired-site intervals (2,000 resamples). (d) Signed onset interval-excess errors for the separate frozen process: 2017–2018 calibration (hollow, n=3) and 2019 evaluation (filled, n=4), the bracketed histories among 55 flag-leaf field-seasons. Zero is within the interval; negative is early and positive late. Bracket widths are 12–29 and 8–18 days, respectively. Onset calibration cases are development data. Scores retain individual forecasts and original weights. Intervals condition on fixed forecasts.'


def save(fig,role,sources,description=None):
    from matplotlib.text import Text
    name=ROLE_NAMES[role];fig.canvas.draw()
    for ax in fig.axes:
        if ax.axison and ax.get_label()!='<colorbar>':
            for spine in ax.spines.values():
                spine.set_visible(True);spine.set_linewidth(.8);spine.set_edgecolor('#353B40')
    for item in fig.findobj(Text):
        if item.get_text():assert item.get_fontsize()==12,(name,item.get_text(),item.get_fontsize())
    renderer=fig.canvas.get_renderer()
    labels=list(fig.texts)+[t for ax in fig.axes for t in [ax.xaxis.label,ax.yaxis.label,*ax.texts]]
    labels += [t for legend in fig.legends for t in legend.get_texts()]
    for item in labels:
        if item.get_text() and item.get_visible():
            box=item.get_window_extent(renderer)
            assert box.x0>=-1 and box.y0>=-1 and box.x1<=fig.bbox.width+1 and box.y1<=fig.bbox.height+1, (name,item.get_text(),'clipped text')
    fig.savefig(OUT/'figures'/f'{name}.png',dpi=300,facecolor='white')
    width,height=fig.get_size_inches()*2.54;plt.close(fig)
    description=CAPTIONS[role] if description is None else description
    caption=description if role=='graphical_abstract' else f'Figure {figure_number(role)}. '+description
    (OUT/'figures'/f'{name}_caption.txt').write_text(caption+'\n')
    return dict(name=name,role=role,width_cm=float(width),height_cm=float(height),font_pt=12,
        png_dpi=300,formats=['png'],axis_labels_and_annotations_within_canvas=True,
        source_sha256={str(path.relative_to(ROOT)):sha(path) for path in sources})


def build(output=None, inputs=None):
    global OUT
    if output is not None:OUT=Path(output).resolve()
    if inputs is None and (OUT/'figure_inputs.json').is_file():
        inputs={key:ROOT/value for key,value in json.loads((OUT/'figure_inputs.json').read_text()).items()}
    inputs={} if inputs is None else {key:Path(value) for key,value in inputs.items()}
    OLD=inputs.get('monitoring',globals()['OLD'])
    CHRONOLOGY=inputs.get('chronology',globals()['CHRONOLOGY'])
    attribution_inputs=inputs.get('attribution',EXT/'data/attribution')
    exposure_inputs=inputs.get('exposure',EXT/'data/exposure')
    from paper.analyses.food_security.exposure import write_economic_sensitivity
    write_economic_sensitivity(exposure_inputs/'europe_thresholds.csv',OUT/'data/economic_sensitivity')
    plt.rcParams.update(STYLE)
    (OUT/'figure_inputs.json').write_text(json.dumps({key:str(value.relative_to(ROOT)) for key,value in inputs.items()},indent=2)+'\n')
    (OUT/'data').mkdir(exist_ok=True);(OUT/'figures').mkdir(exist_ok=True)
    # Inputs are outside these generated folders; prior review files were archived.
    for path in (OUT/'data').iterdir():
        if path.is_dir() and re.fullmatch(r'figs?\d+',path.name):shutil.rmtree(path)
    for path in (OUT/'figures').iterdir():
        if path.is_file():path.unlink()
    records=[]
    def folder(role):
        path=OUT/'data'/ROLE_NAMES[role];path.mkdir(exist_ok=True);return path
    def frame(role,name,data):data.to_csv(folder(role)/f'{name}.csv',index=False)
    def payload(role,name,value):
        (folder(role)/f'{name}.json').write_text(json.dumps(value,indent=2,ensure_ascii=False)+'\n')
    def emit(role,fig,sources,units,description=None):
        description=CAPTIONS[role] if description is None else description
        payload(role,'variables',dict(role=role,units=units,caption=description))
        records.append(save(fig,role,sources,description))
    parameter_sources=[ROOT/'data/final_climate'/f'{name}_parameters.json' for name in ['phenology','disease']]
    for role,builder in [('workflow',build_workflow),('attribution_design',build_attribution_design)]:
        fig,spec=builder();payload(role,'schematic',spec)
        emit(role,fig,[Path(__file__).with_name('figure_methods.py'),CHRONOLOGY,*parameter_sources],
            'Diagram positions are unitless; cohort numbers are design counts, not model performance.')

    basf_source=ROOT/'data/gcb_preparation_20261023/source_snapshots/basf-wheat-diseases.txt'
    corteva_source=ROOT/'data/final_study/inputs/stb_collection_20261009/sources/other_field_sources/corteva-2014-2018/corteva-wheat-diseases.txt'
    overview=prepare_overview(pd.read_csv(CHRONOLOGY),pd.read_csv(basf_source,sep='\t'),
        pd.read_csv(corteva_source,sep='\t',low_memory=False))
    landuse_source=ROOT/'data/final_study/evidence/full_grid_climate/results/full_landuse_cell_registry.csv'
    overview['cropland']=prepare_cropland(pd.read_csv(landuse_source))
    for name,table in overview.items():frame('data_overview',name,table)
    fig,geometry=overview_figure(overview);payload('data_overview','map_geometry',geometry)
    emit('data_overview',fig,[CHRONOLOGY,basf_source,corteva_source,landuse_source,
        Path(__file__).with_name('figure_data_overview.py'),Path(__file__).with_name('figure_maps_v2.py')],
        dict(assessments='count of aggregated field-date-native-code records',field_seasons='distinct field-season identities',
        visits='unique assessment dates per field-season',recorded_disease='native recorded disease percent',
        raw_leaf_INFECT_rows='original source rows; not independent biological replicates',num_sprays='source reported fungicide applications',
        treatment_eligibility='explicit Untreated label AND zero sprays; treated/unknown labels excluded',
        wheat_cropland_percent='physical wheat area / geometric cell area × 100; percent; panel (a)',
        physical_total_ha='SPAM2020 physical wheat area, hectares; panel (a)',
        grid_cell_geometric_area_ha='geometric native cell area, hectares; panel (a)'))
    validation=prepare_validation(pd.read_csv(OLD/'data/fig1/predictions.csv'),pd.read_csv(CHRONOLOGY))
    v_sources=[Path(__file__).with_name('figure_validation_v2.py'),Path(__file__).with_name('figure_validation.py'),
        OLD/'figure_manifest.json',CHRONOLOGY]
    for name in ['points','density','scores','metric_resamples','r2_audit']:
        frame('monitoring','predictions' if name=='points' else name,validation[name])
    coverage=validation['points'].groupby(['evaluation','season_year']).agg(assessments=('global_target_index','size'),field_seasons=('field_id','nunique'),sites=('site_id','nunique')).reset_index()
    frame('monitoring','severity_year_coverage',coverage)
    performance=[]
    for design,label in [('temporal','Temporal'),('spatial_50km','Spatial')]:
        row=validation['scores'].set_index('evaluation').loc[design]
        performance.append(f'{label}: RMSE {row.rmse:.2f} ({row.rmse_lower95:.2f}–{row.rmse_upper95:.2f}) '
            f'pp; predictive R² {row.r2:.3f} '
            f'({row.r2_lower95:.3f}–{row.r2_upper95:.3f}).')
    onset_source=OUT/'data/onset_evaluation/onset_records.csv'
    onset_metric_source=OUT/'data/onset_evaluation/onset_metrics.csv'
    cohort_source=OUT/'data/onset_cohorts'
    cohorts={name:pd.read_csv(cohort_source/f'{name}.csv') for name in
        ['field_histories','coverage_by_year','bracketed_histories','phase_metrics']}
    fig,onset_tables,onset_spec=validation_onset_figure(validation,cohorts)
    for name,table in onset_tables.items():frame('monitoring',name,table)
    frame('monitoring','onset_records',pd.read_csv(onset_source))
    frame('monitoring','onset_metrics',pd.read_csv(onset_metric_source))
    flag_source=OUT/'data/process_f1/evaluation70_F1_records_with_replay.csv'
    eligibility=onset_eligibility(overview['assessments'],pd.read_csv(flag_source))
    frame('monitoring','onset_eligibility_2019',eligibility)
    summary=eligibility.groupby('onset_eligibility').agg(fields=('field_id','size'),F1_assessments=('qualified_F1_assessments','sum')).reset_index()
    frame('monitoring','onset_eligibility_summary',summary)
    onset_spec['cohort_counts']=dict(all_2019_fields=len(eligibility),all_2019_sites=eligibility.site_id.nunique(),
        qualified_F1_fields=int(eligibility.qualified_F1_assessments.gt(0).sum()),
        eligibility_status_counts=eligibility.onset_eligibility.value_counts().to_dict())
    payload('monitoring','figure_layout',onset_spec)
    emit('monitoring',fig,[*v_sources,onset_source,onset_metric_source,
        Path(__file__).with_name('figure_validation_onset.py'),Path(__file__).with_name('validation_error_coverage.py'),flag_source,
        *[cohort_source/f'{name}.csv' for name in cohorts],Path(__file__).with_name('replay_onset_cohorts.py')],
        dict(observed_percent='native disease score, percent',prediction_percent='forecast native score, percent',
        scientific_weight='original dimensionless hierarchy mass',scientific_mass_percent='percent of evaluation mass per2.5×2.5ppcell',
        assessments='raw record count',rmse='percentage points',mae='percentage points',bias='percentage points',
        onset_dates='calibration2017–2018 and evaluation2019 separated; counts55 F1 field-seasons',
        onset_intervals='day after last zero through first positive; observed admissible daily dates',
        onset_error='signed distance outside daily interval for3 calibration and4 evaluation histories; not exact-date MAE',
        calibration_panel='b; fixed10pp prediction bins; original weights; observed mean interval from1000site resamples',
        error_coverage_panel='c; threshold_pp in percentage points; covered_weight_percent/lower95/upper95 in percent of evaluation weight;2000pairedsite resamples',
        r2='1−weightedSSE/weightedSST,allassessments',intervals='descriptive95%pairedsitepercentiles;2000replicates;seed20261024'),
        description=CAPTIONS['monitoring']+' '+' '.join(performance))
    for name in ['bias','contexts','fields']:
        frame('diagnostics',dict(bias='severity_bias',contexts='visit_context',fields='field_metrics').get(name,name),validation[name])
    emit('diagnostics',diagnostic_figure(validation),v_sources,
        dict(observed_mean='percent',
        rmse='percentage points',bias='percentage points,forecastminusobserved',n='assessmentcount',
        ECDF='display percent;one vote per field-season;192 per design;original relative weights within field',
        panels={'a':'outcome-band prediction bias','b':'field-season RMSE distribution','c':'visit-context RMSE'}))
    auxiliary=OUT/'data/monitoring_diagnostics';auxiliary.mkdir(exist_ok=True)
    validation['sites'].to_csv(auxiliary/'site_metrics.csv',index=False)

    grid_source=inputs.get('climate_summary',ROOT/'data/final_study/evidence/full_grid_climate/results/full_grid_ensemble_paired_changes.parquet')
    cell_source=inputs.get('climate_cells',ROOT/'data/final_climate/cells.parquet')
    grid=pd.read_parquet(grid_source,filters=[('metric','==','GS65_85_lost_had3'),('period','==','2071-2100')])
    cells=pd.read_parquet(cell_source)
    grid=grid.merge(cells[['cell_id','west','east','south','north']],on='cell_id',validate='many_to_one')
    fig,agreement,geometry=build_climate_map(grid)
    frame('climate_change','grid',grid);frame('climate_change','direction_agreement',agreement)
    payload('climate_change','map_geometry',geometry)
    emit('climate_change',fig,[grid_source,cell_source,Path(__file__).with_name('figure_maps_v2.py'),
        ROOT/'data/gcb_preparation_20261023/natural_earth_source.json'],
        dict(panels={'a':'SSP126change','b':'SSP245change','c':'SSP585change','d':'SSP585directionagreement'},
        mean_change='normalized affected-tissue days,futureminushistorical',direction_agreement='categoricalthree-memberdirectionagreement',
        longitude='degreeseast',latitude='degreesnorth',colour_limits_days=[-12,12],
        projection='EPSG3035',geographical_scale=False,missing='unavailableisnotzero'))

    a_source=attribution_inputs/'attribution_summary.csv';all_attribution=pd.read_csv(a_source)
    attribution=all_attribution.loc[all_attribution.support.eq('across_settings')].copy()
    frame('attribution_results','attribution',attribution)
    emit('attribution_results',attribution_figure(attribution),[a_source,
        attribution_inputs/'attribution_pairs.parquet',Path(__file__).with_name('figure_results.py')],
        dict(weather_had_days='normalizedaffected-tissuedays',host_had_days='normalizedaffected-tissuedays',
        net_had_days='normalizedaffected-tissuedays',support='4117identicalpairsacross8settings',
        ranges='min–maxofthreemodels,notconfidenceintervals'))
    t_source=exposure_inputs/'europe_thresholds.csv';r_source=exposure_inputs/'production_exposure.csv'
    thresholds=pd.read_csv(t_source).query("period=='2071-2100' and weight_basis=='total' and minimum_paired_seasons==1")
    regional=pd.read_csv(r_source).query("domain_type=='environment_region' and scenario=='ssp585' and period=='2071-2100' and weight_basis=='total' and threshold_days==0 and member_count==3 and minimum_paired_seasons==1")
    assert regional.domain.is_unique
    np.testing.assert_allclose(regional.baseline_production_tonnes.sum(),241844854.561714,rtol=1e-12)
    fig,display=exposure_figure(thresholds,regional)
    frame('production_exposure','thresholds',thresholds);frame('production_exposure','all_regions',regional)
    frame('production_exposure','display_regions',display)
    emit('production_exposure',fig,[t_source,r_source,Path(__file__).with_name('figure_results.py')],
        dict(threshold_days='normalizedaffected-tissuedays',shares='percentofcompletebaselineproduction',
        production_tonnes='tonnes,plottedinmilliontonnes',minimum_paired_seasons=1,
        denominator_tonnes=241844854.561714,meaning='baselineproductionlocatedinaffectedcells,notgrainloss'))
    fig,support=support_figure(all_attribution);frame('support','support',support)
    emit('support',fig,[a_source,Path(__file__).with_name('figure_results.py')],
        'Design-weighted area–time support fraction, plotted as percent; distinct from attribution effect sizes.')
    insight=inputs.get('insights',ROOT/'data/final_analyses/climate_response')
    captions=json.loads((insight/'figure_captions.json').read_text())
    climate_evidence=OUT/'data/climate_response_analysis';climate_evidence.mkdir(exist_ok=True)
    for path in (insight/'data').iterdir():
        if path.is_file():shutil.copy2(path,climate_evidence/path.name)
    for name in ['analysis_audit.json','numeric_findings.json','run_receipt.json']:
        shutil.copy2(insight/name,climate_evidence/name)
    for role,source_name,builder in [('latitude_environment','fig3',build_latitude_environment),
                                   ('climate_phenology','fig4',build_climate_phenology)]:
        destination=folder(role)
        for path in (insight/'data'/source_name).iterdir():
            if path.is_file():shutil.copy2(path,destination/path.name)
        sources=[*sorted((insight/'data'/source_name).glob('*')),insight/'analysis_audit.json',
            Path(__file__).with_name('climate_response_insights.py'),grid_source,
            inputs.get('climate_members',ROOT/'data/final_study/evidence/full_grid_climate/results/full_grid_paired_changes_by_gcm.parquet'),cell_source]
        emit(role,builder(destination),sources,
             'Normalized affected-tissue days; latitude degrees north; window temperature degrees Celsius, rainfall mm and duration days.',
             description=CAPTIONS[role])
    adaptation=inputs.get('adaptation',ROOT/'data/final_analyses/adaptation')
    for path in adaptation.iterdir():
        if path.is_file() and path.suffix in ['.csv','.json','.parquet','.gz','.txt']:
            shutil.copy2(path,folder('adaptation')/path.name)
    payload('adaptation','archived_receipt_scope',dict(original_output_directory=str(adaptation.relative_to(ROOT)),
        original_receipts='Preserved provenance of original analysis outputs; delivery manifest refers to the original preview, not the compact publication caption.',
        published_png='figures/'+ROLE_NAMES['adaptation']+'.png'))
    if (adaptation/'figure_spec.json').is_file():
        shutil.copy2(adaptation/'figure_spec.json',folder('adaptation')/'source_analysis_figure_spec.json')
    else:
        payload('adaptation','source_analysis_figure_spec',dict(source='No archived preview specification provided',registration='registration.json'))
    fig,spec=build_adaptation_figure(adaptation);payload('adaptation','figure_spec',spec)
    adaptation_caption=spec.get('caption')
    assert adaptation_caption, 'Final support-sensitive adaptation caption is required.'
    emit('adaptation',fig,[Path(__file__).with_name('adaptation_analysis.py'),
        Path(__file__).with_name('adaptation_results_figure.py'),Path(__file__).with_name('adaptation_support_sensitivity.py'),
        *[adaptation/name for name in spec['source_files']]],
        'Affected-tissue proxy days; sowing shift days; temperature degrees Celsius; hot exposure days; GCM ranges not confidence intervals.',
        description=adaptation_caption)
    yield_source=inputs.get('yield',ROOT/'data/final_analyses/yield_context')
    for path in (yield_source/'data/fig1').iterdir():
        if path.is_file():shutil.copy2(path,folder('yield_context')/path.name)
    for name in ['descriptive_summary.json','source_metadata.json','verification.json']:
        shutil.copy2(yield_source/name,folder('yield_context')/name)
    for name in ['evidence_screening_inventory.csv','additional_published_endpoints.csv']:
        shutil.copy2(yield_source/'data'/name,folder('yield_context')/name)
    yield_evidence=OUT/'data/yield_source_evidence';yield_evidence.mkdir(exist_ok=True)
    for name in ['arvalis_performance_2025_champagne_ardenne.pdf','arvalis_national_2022.pdf','arvalis_2024_dossier.pdf',
                 'arvalis_performance_2025_original_chart.png','arvalis_2022_scope_page.txt']:
        shutil.copy2(yield_source/'sources'/name,yield_evidence/name)
    yield_caption=(yield_source/'fig1_caption.txt').read_text().strip()
    yield_caption=re.sub(r'^Figure\s+\S+\.\s*','',yield_caption)
    emit('yield_context',build_yield_evidence_figure(),[Path(__file__).with_name('yield_evidence_extension.py'),
        yield_source/'data/fig1/annual_yield_gap.csv',yield_source/'source_metadata.json',
        yield_source/'sources/arvalis_performance_2025_champagne_ardenne.pdf'],
        'Published fungicide yield gap, tonnes per hectare; annual summaries, not pure STB or independently pooled study effects.',
        description=yield_caption)
    graphical_data=attribution.loc[attribution.model.eq('ensemble') &
        attribution.setting.isin(['baseline','fixed_common_overlap'])].copy()
    frame('graphical_abstract','attribution_summary',graphical_data)
    emit('graphical_abstract',graphical_figure(attribution),[a_source,
        Path(__file__).with_name('figure_results.py')],
        'Illustrative interval positions; host contribution in normalized affected-tissue days on identical season pairs.')

    mapping={ROLE_NAMES[r]:sorted(q.name for q in folder(r).iterdir() if q.is_file()) for r in [*MAIN_ROLES,*SUPPLEMENT_ROLES]}
    (OUT/'verification/figure_data_cleanup.json').write_text(json.dumps(dict(
        prior_layout_archived='/Users/gangzhao/Documents/workspace/paper_cc_impact_wheat_disease_intermediates_20261023/publication/gcb_before_methods_validation_visuals',
        role_to_name=ROLE_NAMES,allowed_files_by_figure=mapping),indent=2)+'\n')
    roots=[OUT/'figures',auxiliary,climate_evidence,yield_evidence,OUT/'data/economic_sensitivity',
        *[folder(r) for r in ROLE_NAMES]]
    manifest=dict(figures=records,main_figures=[ROLE_NAMES[r] for r in MAIN_ROLES],
        supplementary_figures=[ROLE_NAMES[r] for r in SUPPLEMENT_ROLES],role_to_name=ROLE_NAMES,
        builder_path=str(Path(__file__).relative_to(ROOT)),code_sha256=sha(Path(__file__)),
        entrypoint_sha256=sha(Path(__file__).with_name('figures.py')),
        registry_sha256=sha(Path(__file__).with_name('figure_registry.py')),
        artifacts_sha256={str(q.relative_to(OUT)):sha(q) for directory in roots for q in directory.rglob('*') if q.is_file()})
    (OUT/'figure_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('figures',len(records),'main',len(MAIN_ROLES),'supplementary',len(SUPPLEMENT_ROLES),flush=True)
    return OUT
