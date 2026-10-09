"""Create header-only input contracts. No example or synthetic measurements."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Column order is fixed. Required means necessary for the corresponding table,
# not an assertion that the source observations exist.
TABLES = {
    'sources': ('source_id', 'source_id,source_url,source_file_sha256,raw_data_license,evidence_type'),
    'environments': ('environment_id', 'environment_id,site_id,harvest_year,independent_environment_group,weather_series_id,latitude_deg,longitude_deg,source_id,source_locator'),
    'treatments': ('treatment_id', 'treatment_id,protection_role,management_package_id,treatment_description,source_id,source_locator'),
    'plots': ('plot_id', 'plot_id,environment_id,treatment_id,cultivar_id,block_id,replicate_id,plot_area_m2,harvested_area_m2,source_id,source_locator'),
    'canopy': ('plot_id,observation_date,leaf_rank_from_flag', 'plot_id,observation_date,leaf_rank_from_flag,total_lai_m2_m2,green_lai_m2_m2,natural_senescent_lai_m2_m2,functional_lai_m2_m2,area_basis,measurement_method,sampling_support,n_sampled_leaves,source_id,source_locator'),
    'disease': ('plot_id,observation_date,leaf_rank_from_flag,disease_identity', 'plot_id,observation_date,leaf_rank_from_flag,disease_identity,severity_pct,symptom_area_index_m2_m2,severity_denominator,assessment_method,n_sampled_leaves,source_id,source_locator'),
    'stages': ('plot_id,stage_code', 'plot_id,stage_code,stage_scale,stage_date,date_status,source_id,source_locator'),
    'harvest': ('plot_id', 'plot_id,harvest_date,grain_yield_reported,grain_yield_unit,grain_moisture_pct,moisture_basis,harvest_method,source_id,source_locator'),
    'weather': ('weather_series_id,date', 'weather_series_id,date,tmin_degC,tmax_degC,solar_radiation_MJ_m2_day,precipitation_mm_day,weather_origin,station_id,source_id,source_locator'),
    'reference_pairs': ('comparison_id', 'comparison_id,affected_plot_id,reference_plot_id,matching_basis,attribution_scope,source_id,source_locator'),
    'management': ('event_id', 'event_id,plot_id,event_date,event_type,operation_description,input_value,input_unit,source_id,source_locator'),
    'biomass_reserves': ('plot_id,observation_date,organ', 'plot_id,observation_date,organ,dry_biomass_g_m2,water_soluble_carbohydrate_g_m2,nitrogen_g_m2,measurement_method,source_id,source_locator'),
}
OPTIONAL = {
    'latitude_deg','longitude_deg','total_lai_m2_m2','natural_senescent_lai_m2_m2',
    'functional_lai_m2_m2','green_lai_m2_m2','severity_pct','n_sampled_leaves','symptom_area_index_m2_m2',
    'station_id','input_value','input_unit','water_soluble_carbohydrate_g_m2','nitrogen_g_m2',
}
DEFINITIONS = {
    'source_id': ('text', '', 'Stable source accession, not a paper-level replacement for plot identity.'),
    'source_url': ('url', '', 'Exact public source URL or DOI landing page.'),
    'source_file_sha256': ('text', '', 'SHA-256 of the original numerical observation file.'),
    'raw_data_license': ('text', '', 'License/permission applying to raw observations; manuscript license is insufficient.'),
    'evidence_type': ('enum', '', 'field_plot_observation only; model outputs, curves and summary means are ineligible.'),
    'source_locator': ('text', '', 'Original file, sheet/table, row and source key; supports reversal of every transformation.'),
    'environment_id': ('text', '', 'Unique site and harvest-year environment; separate management trials remain linked.'),
    'site_id': ('text', '', 'Stable site identity across years and publications.'),
    'harvest_year': ('integer', 'year', 'Actual harvest year; no inferred calendar year from publication date.'),
    'independent_environment_group': ('text', '', 'Holdout group for shared site-year/weather and overlapping publications.'),
    'weather_series_id': ('text', '', 'Foreign key for the meteorological time series used by this environment.'),
    'latitude_deg': ('number', 'decimal degrees north', 'Documented site coordinate; missing is not zero.'),
    'longitude_deg': ('number', 'decimal degrees east', 'Documented site coordinate; missing is not zero.'),
    'treatment_id': ('text', '', 'Stable treatment identifier; must include experiment context if labels recur.'),
    'protection_role': ('enum', '', 'protected, affected, or other; protected is not a measured disease-free status.'),
    'management_package_id': ('text', '', 'Identify management bundle so fungicide, N, water and source-sink changes remain separable.'),
    'treatment_description': ('text', '', 'Source-supported treatment description including non-target disease control.'),
    'plot_id': ('text', '', 'Globally unique harvested experimental unit across site, season and archive versions.'),
    'cultivar_id': ('text', '', 'Original cultivar or mixture identity with component proportions retained separately if needed.'),
    'block_id': ('text', '', 'Original randomized block; not interchangeable with an independent environment.'),
    'replicate_id': ('text', '', 'Biological plot replication; leaf subsamples are not harvest replicates.'),
    'plot_area_m2': ('number', 'm2 ground', 'Actual plot ground area.'),
    'harvested_area_m2': ('number', 'm2 ground', 'Area used to calculate grain yield; exclude borders/destructive samples as documented.'),
    'observation_date': ('date', 'YYYY-MM-DD', 'Actual field date; thermal time must not replace the calendar date.'),
    'leaf_rank_from_flag': ('integer', 'rank; flag=1', 'Final leaf rank: 1 flag, 2 flag-minus-one; do not combine numbering systems.'),
    'total_lai_m2_m2': ('number', 'm2 lamina m-2 ground', 'Area for this leaf layer. Whole-canopy values must not be duplicated over ranks.'),
    'green_lai_m2_m2': ('number', 'm2 green lamina m-2 ground', 'Observed green area for this leaf layer, with measured ground-area normalization.'),
    'natural_senescent_lai_m2_m2': ('number', 'm2 lamina m-2 ground', 'Separately attributed natural senescence; not all non-green area.'),
    'functional_lai_m2_m2': ('number', 'm2 equivalent functional leaf m-2 ground', 'Only with an independently justified functional measurement; greenness alone does not establish function.'),
    'area_basis': ('text', '', 'Lamina-only and ground-area normalization; explain any stem/ear contributions separately.'),
    'measurement_method': ('text', '', 'Method and normalization; flag sensor-derived estimates versus direct observations.'),
    'sampling_support': ('text', '', 'Parent plot and destructive microplot/leaf subsample relationship; do not invent repeated identical leaves.'),
    'n_sampled_leaves': ('integer', 'leaves', 'Original sampling count; does not increase independent harvest n.'),
    'disease_identity': ('text', '', 'Observed disease identity; separate STB, rust, mildew and mixed/unidentified symptoms.'),
    'severity_pct': ('number', '%', 'Measured symptom fraction; ordinal scores cannot be silently converted to percent.'),
    'symptom_area_index_m2_m2': ('number', 'm2 symptomatic lamina m-2 ground', 'Separate observed symptom area when available, with overlap conventions.'),
    'severity_denominator': ('text', '', 'All leaves versus selected infected leaves, original lamina versus surviving green area.'),
    'assessment_method': ('text', '', 'Scoring method and diagnosis basis; retain incidence separately when severity is conditional.'),
    'stage_code': ('integer', 'BBCH/Zadoks code', 'Actual stage observation; do not substitute GS59/83 for GS65/85.'),
    'stage_scale': ('enum', '', 'BBCH or Zadoks; source mapping required before harmonization.'),
    'stage_date': ('date', 'YYYY-MM-DD', 'Date stage was reached in the corresponding plot.'),
    'date_status': ('enum', '', 'observed or interpolated; source-supported interpolation and uncertainty required.'),
    'harvest_date': ('date', 'YYYY-MM-DD', 'Date of the grain measurement.'),
    'grain_yield_reported': ('number', 'see grain_yield_unit', 'Original measured grain yield, not BLUP/predicted mean or reconstructed loss.'),
    'grain_yield_unit': ('enum', '', 'kg_ha, t_ha, dt_ha, or g_m2; convert mass area first, then moisture basis.'),
    'grain_moisture_pct': ('number', '% wet mass', 'Moisture of reported grain yield; 0 only if dry-matter basis explicitly documented.'),
    'moisture_basis': ('text', '', 'Reported moisture correction or dry-matter method; unknown remains missing.'),
    'harvest_method': ('text', '', 'Combine/sample method; distinguish harvest samples from pooled/duplicated row representations.'),
    'date': ('date', 'YYYY-MM-DD', 'Meteorological local calendar day.'),
    'tmin_degC': ('number', 'degC', 'Daily minimum air temperature.'),
    'tmax_degC': ('number', 'degC', 'Daily maximum air temperature.'),
    'solar_radiation_MJ_m2_day': ('number', 'MJ m-2 day-1', 'Daily global solar radiation, not PAR; any conversion requires a documented method.'),
    'precipitation_mm_day': ('number', 'mm day-1', 'Daily precipitation; irrigation is a separate management event.'),
    'weather_origin': ('text', '', 'Station or named reanalysis; gap-filling flagged explicitly, never treated as a field measurement.'),
    'station_id': ('text', '', 'Documented meteorological station identifier.'),
    'comparison_id': ('text', '', 'Unique predefined affected/reference contrast.'),
    'affected_plot_id': ('text', '', 'Plot receiving the affected management condition; reference uses a separate plot key.'),
    'reference_plot_id': ('text', '', 'Observed protected comparator from same site-year and cultivar/design context.'),
    'matching_basis': ('text', '', 'Source-supported design pairing; same replicate number alone does not prove matching.'),
    'attribution_scope': ('enum', '', 'STB_specific, mixed_disease_protection, or management_package; STB attribution needs evidence.'),
    'event_id': ('text', '', 'Unique management event identifier.'),
    'event_date': ('date', 'YYYY-MM-DD', 'Actual operation date; no inferred dates for missing records.'),
    'event_type': ('text', '', 'Sowing, fertilization, irrigation, crop protection, or other operation.'),
    'operation_description': ('text', '', 'Original operation description; preserve co-interventions.'),
    'input_value': ('number', 'see input_unit', 'Original operation amount if reported.'),
    'input_unit': ('text', '', 'Unit of the reported operation amount.'),
    'organ': ('text', '', 'Grain, stem, leaf, ear or total; totals cannot be added to constituent organs.'),
    'dry_biomass_g_m2': ('number', 'g dry matter m-2 ground', 'Observed dry biomass; optional table for crop-growth identification.'),
    'water_soluble_carbohydrate_g_m2': ('number', 'g carbohydrate m-2 ground', 'Measured reserve pool; absent data cannot justify fitted reserve-remobilization parameters.'),
    'nitrogen_g_m2': ('number', 'g N m-2 ground', 'Measured organ nitrogen, if available.'),
}


def main():
    target=ROOT/'templates';target.mkdir(exist_ok=True)
    schema={};dictionary=[]
    for name,(key,columns) in TABLES.items():
        columns=columns.split(',')
        path=target/(name+'.csv')
        if path.exists() and len(path.read_text().splitlines())>1:
            raise RuntimeError(f'Refusing to overwrite populated template {path}')
        with path.open('w',newline='') as f:csv.writer(f).writerow(columns)
        schema[name]={'primary_key':key.split(','),'columns':columns,
                      'required_columns':[c for c in columns if c not in OPTIONAL],
                      'optional_table':name=='biomass_reserves'}
        for column in columns:
            datatype,unit,meaning=DEFINITIONS[column]
            dictionary.append(dict(table=name,column=column,data_type=datatype,unit=unit,
                                   required_if_table_populated=column not in OPTIONAL,meaning=meaning))
    (ROOT/'schema.json').write_text(json.dumps(schema,indent=2)+'\n')
    with (ROOT/'data_dictionary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(dictionary[0]));writer.writeheader();writer.writerows(dictionary)
    print(json.dumps({'empty_templates':len(schema),'dictionary_fields':len(dictionary),'observation_rows':0}))


if __name__=='__main__':main()
