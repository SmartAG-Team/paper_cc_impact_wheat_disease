"""Create the source-resolved frozen parameter inventory for the specification."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[4]
DEST = Path(__file__).resolve().parent
TPV = 'configurations/wheat_stb/tpv_calibration.json'
STAGE = 'configurations/wheat_stb/calibrated_stage_thresholds.json'
FIT = 'analysis/paper_study/overwinter_leaf_model_20261006/disease/overwinter_source_model/frozen_selected_fit.json'
ANTHESIS = 'analysis/paper_study/infection_priority_20261006/anthesis_clock/threshold_selection.json'
PHENOLOGY = 'model/wheat_stb/phenology.py'
LEAF = 'model/seasonal_septoria/leaf_phenology.py'
SOURCE = 'model/seasonal_septoria/overwinter.py'
WETNESS = 'model/seasonal_septoria/wetness.py'
CONFIG = 'examples/wheat_stb/configuration_standardized_had.json'
CLIMATE = 'analysis/paper_study/overwinter_leaf_model_20261006/climate/configuration_before_climate_results.json'
SAMPLING = 'data/paper_study/regional_parameter_uncertainty/sampling_receipt.json'
OBSERVATION = 'calibration/seasonal_septoria/infection_events.py'


def sha(relative):
    return hashlib.sha256((ROOT/relative).read_bytes()).hexdigest()


def read(relative):
    return json.loads((ROOT/relative).read_text())


def main():
    tpv, stage, fit, example = map(read, [TPV, STAGE, FIT, CONFIG])
    rows = []

    def add(symbol, name, value, unit, role, provenance, uncertainty, interpretation=''):
        rows.append(dict(symbol=symbol, parameter=name, value=json.dumps(value, ensure_ascii=False),
            unit=unit, role=role, provenance=provenance, source_sha256=sha(provenance),
            uncertainty=uncertainty, interpretation=interpretation))

    fixed_donor = 'Fixed in current study; inherited calibration; no current parameter interval.'
    for symbol, key, unit in [('G_P_on','photoperiod_onset_gdd','effective degC d'),
            ('G_P_off','photoperiod_stop_gdd','effective degC d'),
            ('U_V_on','vernalization_onset_tpp','effective degC d'),
            ('U_V_off','vernalization_stop_tpp','effective degC d')]:
        add(symbol,key,tpv[key],unit,'fixed',TPV,fixed_donor,'Prior-day clock gate.')
    for symbol, name, value, unit in [
            ('T_base','thermal_zero_bound',0,'degC'),('T_cap','thermal_plateau_start',20,'degC'),
            ('T_decline','thermal_decline_start',30,'degC'),('T_supported_max','maximum_supported_daily_mean',40,'degC'),
            ('b_hot','thermal_decline_slope',2,'effective degC per degC'),
            ('delta_amplitude','solar_declination_amplitude',0.4093,'radian'),
            ('j_equ','solar_phase_day',81,'day of year'),('n_solar','astronomical_period',365,'d'),
            ('ell_ref','photoperiod_reference_daylength',16,'h'),('b_P','photoperiod_response_slope',0.09,'h^-1'),
            ('v_knots_T','vernalization_temperature_knots',[-4,0,10,16],'degC'),
            ('v_knots_R','vernalization_response_at_knots',[0,1,1,0],'effective vernalization d per d'),
            ('V_early','early_vernalization_hot_loss_bound',10,'effective vernalization d'),
            ('T_devern','maximum_temperature_hot_loss_bound',30,'degC'),
            ('b_devern','hot_loss_slope',0.5,'effective vernalization d per degC per d'),
            ('V_complete','vernalization_response_saturation',40,'effective vernalization d'),
            ('f_V_min','minimum_vernalization_multiplier',0.3,'1'),
            ('f_V_span','vernalization_multiplier_increment',0.7,'1')]:
        add(symbol,name,value,unit,'fixed',PHENOLOGY,'Intrinsic retained donor assumption; no uncertainty distribution.')
    add('q','development_multiplier',1,'1','fixed',STAGE,'Retained q=1; no rescaling in current runtime.')
    for k, value in stage['all_stage_thresholds'].items():
        if int(k) in (10,31,51,85):
            role, uncertainty = 'fixed', fixed_donor
        elif int(k) == 65:
            role = 'fitted'
            uncertainty = 'Inherited stage-only estimate; exact minimizing range 1152.462482–1177.144788; 1-d loss tolerance 1069.44018–1264.654782; not confidence intervals.'
        else:
            role = 'fitted'
            d = next(x for x in stage['diagnostics'] if x['event'] == int(k))
            uncertainty = '; '.join(f"{x['joint_loss_tolerance_days']:g}-d loss tolerance {x['threshold_low']:.12g}–{x['threshold_high']:.12g}"
                for x in d['joint_profile_loss_tolerance_sets']) + '; weak identification; not confidence intervals.'
        add(f'C_{k}',f'BBCH{k}_threshold',value,'effective degC d',role,STAGE,uncertainty,'First included end-day TPV crossing.')
    add('f_65','flowering_threshold_fraction',read(ANTHESIS)['fraction'],'1','fitted',ANTHESIS,
        'Stage-only grid [0.02,0.60]; exact minimum fractions [0.2143,0.2462]; not a confidence interval.',
        'C65 = C51 + f65*(C85-C51); no numerical BBCH interpolation.')
    fitted = fit['fitted']
    add('Delta','rank_spacing_units',fitted['rank_spacing_units'],'effective degC d','fitted',FIT,
        'Selected from 80,120,160 with disease signs/timing; not leaf-count or measured-phyllochron calibration.',
        'Effective structural setting held fixed in current summaries.')
    mapping = [
        ('alpha','primary_scale','d^-1 per relative-source unit','fitted','Grid 0.0001,0.001,0.01,0.1; source-scale dependent; no confidence interval.'),
        ('beta','secondary_scale','d^-1','fitted','Grid 0.1,1,10; selected upper grid boundary; weak identification; no confidence interval.'),
        ('tau_L','latent_reference_days','reference d at 18 degC','fitted','Grid 20,30 d; effective latent-chain mean; finite-substep timing differs from continuous-time mean.'),
        ('m','latent_stages','count','fixed','Fixed effective chain architecture; no uncertainty distribution.'),
        ('tau_N','nonsporulating_reference_days','reference d at 18 degC','fixed','Effective assumption; not calibrated to disease magnitude.'),
        ('tau_I','infectious_reference_days','reference d at 18 degC','fixed','Effective assumption; not calibrated to disease magnitude.'),
        ('tau_M','residue_maturation_reference_days','reference d at 18 degC and exposure=1','fixed','Effective source-competence assumption; no measured reproductive-maturation calibration.'),
        ('tau_Q','residue_decay_reference_days','reference d at 18 degC','fixed','Effective common source-ageing assumption; same-fit half/double lifetime sensitivities.'),
        ('f_0','initial_ready_fraction','1','fixed','Initial relative-source partition assumption; no measured readiness.'),
        ('lambda','rank_distance_scale','rank intervals','fixed','Effective canopy geometry assumption; no calibrated physical transport distance.'),
        ('p_star','rain_scale_mm','mm','fixed','Effective splash-response assumption; no uncertainty distribution.'),
        ('omega','local_airborne_fraction','1','fixed','Relative local establishment mixture; not a measured spore-source share.'),
        ('kappa','contact_fraction','1','fixed','Effective neighboring-leaf contact multiplier; contact-off structural sensitivity.')]
    for symbol,key,unit,role,uncertainty in mapping:
        add(symbol,key,fitted['parameters'][key],unit,role,FIT,uncertainty)
    add('M','constant_imported_pressure',fitted['constant_imported_pressure'],'relative-source units','fitted',FIT,
        'Grid 0,0.1; no observed imported source; source-scale dependent; no confidence interval.')
    add('r_star','rain_rate_mm_hour',fitted['weather_preprocessing']['rain_rate_mm_hour'],'mm h^-1','fitted',FIT,
        'Training-only meteorological median; 2695 eligible rain days at 6432 unique location-days; no sampling interval supplied.',
        'Meteorology-only estimate; distinct from disease-sign fitting.')
    add('J_policy','juvenile_policy',fitted['juvenile_policy'],'policy','scenario',FIT,
        'Primary handover_31_39; persistent alternative tested without fitting new ecological rates.')
    add('D_J','juvenile_renewal_degree_day_scale',300,'positive degC d','fixed',LEAF,
        'Retained juvenile renewal assumption; no measured turnover calibration.')
    add('T_ref','disease_thermal_reference',18,'degC','fixed',SOURCE,
        'Defines relative stage/source clock; no fitted temperature-response uncertainty.')
    for symbol, name, value, unit in [
            ('H_RH','humid_bin_cutoff',90,'percent RH'),('eps_RH','humidity_cutoff_tolerance',1e-7,'percentage points'),
            ('n_bin','synthetic_hourly_bins',24,'count'),('phi_bin','hourly_bin_phase_offset',0.5,'h'),
            ('b_sat','saturation_exponent_coefficient',17.625,'1'),('T_sat','saturation_temperature_constant',243.04,'degC'),
            ('n_bisect','humidity_bisection_iterations',30,'count'),('T_exp','exposure_temperature_optimum',18,'degC'),
            ('sigma_exp','exposure_temperature_width',8,'degC')]:
        add(symbol,name,value,unit,'fixed',WETNESS,'Daily exposure reconstruction assumption; no measured canopy-wetness error distribution.')
    add('Q_0','initial_local_source',example['initial_local_source'],'relative-source units','scenario',CONFIG,
        'Conditional unit initial source; local-source-off intervention does not identify causal source effects.')
    add('Delta_t_req','requested_time_step',example['time_step'],'d','fixed',CONFIG,
        'Frozen numerical setting; convergence checks assess numerical behavior rather than biological uncertainty.')
    add('n_sub','daily_substeps',4,'count','fixed',SOURCE,'n=ceil(1/requested_step); current effective substep is 0.25 d.')
    add('epsilon','detection_fraction',example['detection_fraction'],'normalized tissue fraction','scenario',CONFIG,
        'Fixed detection operator; post-fit cutoffs 0.0001 and 0.01; not biological infection threshold.')
    add('D_miss','missing_positive_minimum_penalty',30,'d','fixed',OBSERVATION,
        'Scoring convention; missing-positive loss=max(30,forcing_end+1-upper_bound).')
    add('n_slot','leaf_slots',8,'count','fixed',LEAF,'F1–F7 plus aggregate juvenile; does not estimate actual main-stem leaf number.')
    add('needs_V','vernalization_required',True,'boolean','scenario',CONFIG,
        'Winter-wheat requirement assumed; vernalization-off is a structural sensitivity.')
    add('L_star','standardized_upper3_lai',example['yield_model']['standardized_upper3_lai'],'m2 lamina m^-2 ground','scenario',CONFIG,
        'Nominal scenario maximum; actual disease-free reference area and natural senescence unmeasured.')
    for i,b in enumerate(example['yield_model']['coefficients_t_ha_per_glai_day']):
        add(f'b_{i+1}','yield_transfer_slope',b,'t ha^-1 per GLAI d','scenario',CONFIG,
            'Published cultivar-range scenario; not confidence or prediction interval; no local yield calibration.',
            'Source DOI 10.1111/j.1365-3059.2004.00951.x retained in runtime documentation; GS65–85 is a restricted-window adaptation.')
    add('Y_ref','reference_yield_t_ha',None,'t ha^-1','scenario',CONFIG,
        'Absent by default; independently supplied positive value required for percentage diagnostic; not inferred.')
    add('f_proxy','functional_loss_conversion','N+I+D','normalized functional fraction','scenario',CONFIG,
        'Unmeasured model proxy; latent tissue excluded; field/public runtime integrates pre-detection values.')
    add('gate_climate','climate_functional_detection_gate','zero until first symptom fraction >= 0.001','policy','scenario',CLIMATE,
        'Climate reporting convention only; differs from current field/public-runtime functional postprocessing.')
    add('W_window','yield_window','inclusive first BBCH65 to first BBCH85','calendar window','scenario',CLIMATE,
        'Development-window adaptation; GS31–85 sensitivity separately reported; no measured physiological yield window.')
    add('H','spatial_strata',16,'count','fixed',SAMPLING,'Geographical stratification conditional on area/calendar registry.')
    add('n_h','draws_per_stratum',4,'count','fixed',SAMPLING,'Independent area-proportional replacement draws; MCSE separately estimated.')
    add('seed_spatial','sampling_seed',read(SAMPLING)['seed'],'integer','fixed',SAMPLING,'Exact draw identities supplied; 64 draws at 62 unique cells.')
    add('phi_breaks','latitude_strata_breaks',read(SAMPLING)['latitude_breaks'],'degrees north','fixed',SAMPLING,'Design constants; no uncertainty distribution.')
    add('lon_breaks','longitude_strata_breaks',read(SAMPLING)['longitude_breaks'],'degrees east','fixed',SAMPLING,'Design constants; no uncertainty distribution.')
    add('n_year','period_year_count',30,'yr','fixed',CLIMATE,'Historical 1991–2020; future 2031–2060 and 2071–2100; relative-year pairing.')
    add('n_GCM','climate_model_count',3,'count','scenario',CLIMATE,'ACCESS-CM2, MPI-ESM1-2-HR, MRI-ESM2-0; range is not a confidence interval.')
    add('calendar_type','regional_crop_calendar_type','winter wheat/rainfed on all-wheat registered area','policy','scenario',CLIMATE,
        'Genetic wheat type not verified; calendar and area mismatch is a structural limitation.')
    with (DEST/'Parameter_inventory.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    sources=sorted({r['provenance'] for r in rows} | {
        'WHEAT_STB_ENGINE.md', 'model/wheat_stb/config.py', 'model/wheat_stb/engine.py',
        'model/wheat_stb/weather.py', 'model/wheat_stb/yield_model.py', 'model/wheat_stb/_version.py',
        'calibration/wheat_stb/configuration.py', 'calibration/seasonal_septoria/spatial_estimation.py',
        'analysis/paper_study/overwinter_leaf_model_20261006/manuscript/full_season_replay.py',
        'analysis/paper_study/overwinter_leaf_model_20261006/climate/run.py',
        'analysis/paper_study/overwinter_leaf_model_20261006/disease/run.py',
        'configurations/wheat_stb/legacy/infection_priority_model.json',
        'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json',
        'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv',
        'analysis/paper_study/overwinter_leaf_model_20261006/climate/weather_cache_receipt.json'})
    manifest=dict(runtime_version='0.1.0', specification_date='2026-10-07',
        current_disease_model='overwinter_source_model/a0.001_b10_L20_s120_i0.1',
        current_fit_sha256=sha(FIT), stage_fit_sha256=sha(STAGE), donor_tpv_sha256=sha(TPV),
        archived_models=['seasonal_calibration_v1','infection_and_upper3_yield_v1_20261006'],
        files=[dict(path=p,bytes=(ROOT/p).stat().st_size,sha256=sha(p)) for p in sources],
        inventory_rows=len(rows), roles=['fitted','fixed','scenario'])
    (DEST/'source_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(inventory_rows=len(rows), source_files=len(sources))))


if __name__ == '__main__':
    main()
