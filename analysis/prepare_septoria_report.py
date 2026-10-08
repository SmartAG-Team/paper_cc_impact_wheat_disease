"""Create the bounded reviewed snapshot and academic narrative for the report."""
from pathlib import Path
from datetime import datetime,timezone
import json
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'analysis'
summary=json.loads((OUT/'septoria/summary.json').read_text())
quality=json.loads((ROOT/'data/harmonized/validation.json').read_text())
weather=json.loads((OUT/'era5/validation.json').read_text())
catalog=[
 dict(dataset='BASF wheat trials',grain='Trial × date × organ × treatment',records=2276,role='Field disease and yield',source='https://doi.org/10.5281/zenodo.6521175'),
 dict(dataset='Karisto Swiss field',grain='Scanned leaf',records=21420,role='Necrosis and pycnidia; two collections',source='https://zenodo.org/records/4933099'),
 dict(dataset='Hafeez isolate panel',grain='Seedling record with repeated scores',records=4684,role='Three-isolate pycnidia and damage panel; additional transgenic/mutant sheets retained',source='https://doi.org/10.5281/zenodo.14515753'),
 dict(dataset='Orellana French field',grain='Plant assessment',records=2400,role='7774 numeric leaf scores and2366 senescent leaf markers',source='https://doi.org/10.15454/4MAAI0'),
 dict(dataset='Orellana greenhouse assays',grain='Inoculated plant–leaf score',records=39024,role='3035 field-origin isolates; source information retained',source='https://doi.org/10.15454/4MAAI0'),
 dict(dataset='Ben M’Barek Tunisian field',grain='Scanned, pycnidia-positive leaf',records=3004,role='Absolute leaf measurements; relative views archived separately',source='https://doi.org/10.5281/zenodo.3979476'),
 dict(dataset='Chaloner process experiments',grain='Biological/technical replicate',records=3141,role='Germination rows; mortality and growth experiments also retained',source='https://doi.org/10.6084/m9.figshare.7937516'),
 dict(dataset='Nordic–Baltic ancillary data',grain='Yield record and station-day',records=662,role='Mixed leaf-blotch pathogens; 36089 decoded daily weather records',source='https://doi.org/10.6084/m9.figshare.19203377.v4')]
links=[dict(label=x['dataset'],href=x['source']) for x in catalog]
queries={}

def query(name,frame,label,files,source_links,caveats,method):
    rows=json.loads(frame.to_json(orient='records')) if isinstance(frame,pd.DataFrame) else frame
    queries[name]={'rows':rows,'source':{'label':label,'files':[{'label':f} for f in files],
        'links':source_links,'caveats':caveats,'evidenceFlow':[{'title':'Source acquisition','detail':label},
        {'title':'Reproducible transformation','detail':method}]},'methods':[{'language':'python','code':method}]}

query('catalog',catalog,'Repository downloads and original BASF archive',
    ['data/public_septoria/download-manifest.json','data/harmonized/source_table_inventory.csv'],links,
    ['Different designs and measurement denominators; record counts cannot be added as independent replicates.','Analysis code alone does not establish access to raw rain-splash observations.'],
    "Run analysis/gather_public_septoria.py, analysis/inspect_public_septoria.py and analysis/harmonize_septoria.py; preserve source table and row keys.")
query('coverage',pd.read_csv(OUT/'septoria/mapped_trial_locations.csv'),'BASF trial coordinates and public field sites',
    ['data/basf-wheat-diseases.txt','analysis/septoria/mapped_trial_locations.csv'],links[:6],
    ['BASF locations are rounded to0.1degree; French point is station-representative.','Controlled inoculation assays have no assigned field location.'],
    "Group untreated SEPTTR by source latitude and longitude; count trials with at least three dates at the same explicit leaf rank. Map in EPSG:3035 with archived Natural Earth boundaries.")
query('controlled',pd.read_csv(OUT/'septoria/controlled_infection_progress.csv'),'Nature Plants controlled inoculation data',
    ['data/public_septoria/hafeez-2025-infection/Hafeez et al. Stb15, Nature Plants - pathology data.xlsx'],
    [links[2]],['Changing numbers of scored records; one thermal regime.','Visible pycnidia are not emitted spores or known penetration times.'],
    "Select the three main isolate sheets; retain numeric scores in0–100; summarize pycnidia-positive fractions and medians by isolate and days after inoculation.")
query('censoring',pd.read_csv(OUT/'septoria/pycnidia_censoring_counts.csv'),'First-visible-pycnidia observation intervals',
    ['analysis/septoria/first_pycnidia_intervals.csv'],[links[2]],
    ['Right-censored records may reflect resistance, failed establishment or delayed reproduction.','The interval starts at the last scored zero and ends at the first positive score.'],
    "Classify each main-panel source record as interval-, left-, right-censored or unobserved using only its valid pycnidia scores.")
query('relationships',pd.read_csv(OUT/'septoria/plot_necrosis_pycnidia.csv'),'Within-plot necrosis and pycnidia summaries',
    ['analysis/septoria/plot_necrosis_pycnidia.csv'],[links[1],links[5]],
    ['Each point is a plot–collection mean.','Tunisian scanned leaves condition on pycnidia presence; source populations are not pooled.'],
    "Use absolute PLACL and leaf-area pycnidia density; average leaves within plot and date; preserve study and year. Relative-index tables and repeated plot means are not biological replicates.")
query('french',pd.read_csv(OUT/'septoria/french_plot_leaf_progress.csv'),'French field pycnidia-covered leaf area',
    ['data/public_septoria/orellana-torrejon-2022-field/F1_Field_disease_severity_rawdata.csv'],[links[3]],
    ['S denotes senescent; leaf survival and assessed plots change across dates.','Plant labels do not establish longitudinal physical-leaf identity.'],
    "Calculate plot–date–cultivar–rank means from numeric source scores; then average available plot means without interpreting senescent leaves as zero disease.")
query('weather',pd.read_csv(OUT/'era5/basf_assessment_weather_30d_stats.csv'),'ERA5 served through Open-Meteo; completed30-day assessment exposures',
    ['data/era5/retrieval_inventory.csv','analysis/era5/basf_assessment_weather_30d_stats.csv'],
    [{'label':'Open-Meteo historical API','href':'https://open-meteo.com/en/docs/historical-weather-api'},
     {'label':'ERA5 single levels','href':'https://doi.org/10.24381/cds.adbb2d47'}],
    ['Humidity exposure is not measured leaf wetness.','Figures count distinct eligible ERA5 grid-cell/date windows; full September–September summaries are descriptive coverage intervals.'],
    "Force models=era5, elevation=nan, timezone=UTC and nearest grid cell. Calculate exposures from the30 complete days strictly before each assessment, with720 hourly records per window.")
query('seasonal',pd.read_csv(ROOT/'data/era5/seasonal_weather.csv'),'ERA5 fixed seasonal coverage summaries',
    ['data/era5/seasonal_weather.csv'],queries['weather']['source']['links'],
    ['Fixed September1 of prior year to September30; this is a395/396-day interval.','This interval contains future weather relative to some disease observations and is not a forecasting predictor.','ETH2023/2024 coverage is weather-only because disease data were not downloaded.'],
    "Aggregate each unique requested coordinate-season once; map BASF2017–2019 precipitation with a common colour scale.")

for name,path,label in [('disease_evaluation','analysis/disease_evaluation/fold_metrics.csv','Conditional disease predictive benchmark'),
                        ('phenology_validation','analysis/phenology/heldout_validation/heldout_test_figure_data.csv','German station-held-out phenology evaluation')]:
    p=ROOT/path
    query(name,pd.read_csv(p) if p.exists() else [],label,[path],[],
        ['Model and evaluation scope must match the corresponding formal methods.'],
        'Read the source-backed metrics and split manifest; aggregate held-out predictions at their documented series and station grain.')

p=ROOT/'data/era5/europe/era5_fixed_intervals_2017_2019_native025_grid.parquet'
if p.exists():
    grid=pd.read_parquet(p)
    query('continental',grid[['latitude','longitude','country','grid_cell_id','tmean_2019_c','precipitation_2019_mm','selected_grid_cell_area_km2']],
        'ERA5 monthly climate context supplied through Google Earth Engine',
        ['data/era5/europe/era5_fixed_intervals_2017_2019_native025_grid.parquet','data/era5/europe/geographic_scope.json'],
        [{'label':'ERA5 Monthly catalogue','href':'https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_MONTHLY'}],
        ['Monthly climate context; not daily epidemic forcing or wheat-specific disease risk.','Initial geographic mask excludes European Turkey and some small islands/states.','Area represents selected complete grid cells, not exact land or wheat area.'],
        'Calendar-day-weight monthly2mtemperature; sum monthly precipitation for September2018–September2019; preserve native monthly registration and country-mask centre selection.')
    queries['continental']['source']['metricDefinitions']=[
        {'label':'Temperature','definition':'Calendar-day-weighted mean 2m temperature for the13-month fixed interval, degreesCelsius.'},
        {'label':'Precipitation','definition':'Sum of monthly total precipitation, including liquid and frozen precipitation, millimetres.'}]

snapshot={'surface':'report','title':'Septoria model evaluation and European scaling',
    'generatedAt':datetime.now(timezone.utc).isoformat(),'status':'reviewed','buildStatus':'creating',
    'report':{'asOf':'2026-10-04'},'filters':[],'queries':queries,
    'academic':{'summary':summary,'harmonization':quality,'weather':weather}}
(OUT/'septoria-reviewed.json').write_text(json.dumps(snapshot,indent=2)+'\n')
print('Reviewed snapshot: analysis/septoria-reviewed.json')
