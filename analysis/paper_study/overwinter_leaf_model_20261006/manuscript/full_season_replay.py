"""Frozen seasonal source model replay with complete, archived crop weather."""
from pathlib import Path
from copy import copy
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.paper_study.structural_evaluation.run import load_inputs, DEFAULT_PATHS
from analysis.paper_study.infection_priority_20261006.build_priority_package import extend_forcing
from analysis.paper_study.overwinter_leaf_model_20261006.disease.run import predict_record

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent


def main():
    dest = HERE/'replay'; dest.mkdir(exist_ok=True)
    fitted_path = STUDY/'disease/overwinter_source_model/frozen_selected_fit.json'
    stage_path = STUDY/'phenology/calibrated_stage_thresholds.json'
    fitted = json.loads(fitted_path.read_text())['fitted']
    thresholds = {int(k): float(v) for k,v in json.loads(stage_path.read_text())['all_stage_thresholds'].items()}
    data, weather, original_accumulation, _ = load_inputs(DEFAULT_PATHS)
    forcing, accumulation = extend_forcing(data, weather)
    forcing.targets = data.targets.copy(); forcing.targets['value'] = np.nan
    trajectory, host = predict_record(forcing, accumulation, thresholds, fitted)
    original_forcing=copy(data);original_forcing.targets=data.targets.copy()
    original_forcing.targets['value']=np.nan
    original, original_host=predict_record(original_forcing,original_accumulation,thresholds,fitted)
    strict = set(pd.read_csv(STUDY.parents[2]/'data/paper_study/observations/corteva_location_disjoint_external_units.csv',
        usecols=['source_unit'], dtype={'source_unit':str}).source_unit)
    rows, daily = [], []
    for meta in forcing.metadata.itertuples():
        field, length = int(meta.field_index), int(meta.forcing_days)
        dates = pd.date_range(meta.sowing_date, periods=length)
        source = data.targets.loc[data.targets.field_index.eq(field), 'source'].iloc[0]
        complete = all(host.stage_day_index[stage][field] >= 0 for stage in (65,85))
        start, end = (int(host.stage_day_index[stage][field]) for stage in (65,85))
        record = dict(field_id=meta.field_id, source=source, season_year=int(meta.season_year),
            reused_validation_status=('original_calibration' if source=='BASF' and meta.season_year in (2017,2018)
                else 'reused_BASF2019' if source=='BASF' else 'reused_strict_Corteva' if str(meta.source_unit) in strict
                else 'overlapping_location_descriptive_only'),
            sowing_date=meta.sowing_date, forcing_end_date=str(dates[-1].date()), forcing_days=length,
            grain_fill_complete=complete, grain_fill_days=None if not complete else end-start+1,
            reference_lai_scenario=1., locally_yield_calibrated=False)
        for stage, indices in host.stage_day_index.items():
            index=int(indices[field]); record[f'BBCH{stage}_date']=None if index<0 else str(dates[index].date())
        for leaf in range(3):
            infection, symptom = int(trajectory.infection_day[field,leaf]), int(trajectory.symptom_day[field,leaf])
            record[f'F{leaf+1}_infection_date'] = None if infection<0 else str(dates[infection-1].date())
            record[f'F{leaf+1}_symptom_date'] = None if symptom<0 else str(dates[symptom-1].date())
            record[f'F{leaf+1}_infection_relative_anthesis_days'] = None if infection<0 or start<0 else infection-1-start
            record[f'F{leaf+1}_symptom_relative_anthesis_days'] = None if symptom<0 or start<0 else symptom-1-start
        # Equal mature F1--F3 reference LAI is a specified scenario. Actual
        # reference area/senescence and functional green loss are unmeasured.
        lai = host.area[field,:length,:3]/3.
        damage = trajectory.damage[field,1:length+1,:3]
        if complete:
            reference_had = float(lai[start:end+1].sum())
            lost_had = float((lai[start:end+1]*damage[start:end+1]).sum())
            assert -1e-12 <= lost_had <= reference_had+1e-12
            record.update(reference_had3=reference_had, model_proxy_lost_had3=lost_had,
                model_proxy_had_loss_fraction=lost_had/reference_had if reference_had else None)
            for slope in (.0141,.018,.0207):
                record[f'conditional_yield_loss_b{slope:g}_t_ha_per_unit_lai']=lost_had*slope
        else:
            record.update(reference_had3=None,model_proxy_lost_had3=None,model_proxy_had_loss_fraction=None)
            for slope in (.0141,.018,.0207):record[f'conditional_yield_loss_b{slope:g}_t_ha_per_unit_lai']=None
        rows.append(record)
        for leaf in range(3):
            daily.append(pd.DataFrame(dict(field_id=meta.field_id,leaf_index=leaf,date=dates,
                reference_lai_scenario=lai[:,leaf],model_affected_fraction=damage[:,leaf],
                included_grain_fill=np.arange(length)>=start if complete else False)))
            if complete:daily[-1]['included_grain_fill'] &= np.arange(length)<=end
        # Extended simulations must retain every original observed-horizon state.
        original_length=int(data.metadata.loc[data.metadata.field_index.eq(field),'forcing_days'].iloc[0])
        assert dates[original_length-1] == pd.Timestamp(data.metadata.loc[data.metadata.field_index.eq(field),'last_forcing_date'].iloc[0])
        np.testing.assert_array_equal(trajectory.state[field,:original_length+1],original.state[field,:original_length+1])
        np.testing.assert_array_equal(trajectory.residue[field,:original_length+1],original.residue[field,:original_length+1])
        np.testing.assert_array_equal(host.area[field,:original_length],original_host.area[field,:original_length])
    result=pd.DataFrame(rows); result.to_csv(dest/'full_season_top3_timing_and_conditional_yield.csv',index=False)
    pd.concat(daily,ignore_index=True).to_parquet(dest/'daily_top3_conditional_replay.parquet',index=False)
    receipt=dict(status='complete',field_seasons=len(result),complete_grain_fill_windows=int(result.grain_fill_complete.sum()),
        model_affected_fraction_is_measured_functional_loss=False,actual_yield_predictions=False,
        direct_infection_dates_validated=False,reference_lai_scenario=1.,equal_mature_top3_lai=True,natural_senescence_unmeasured=True,
        grain_fill_window='inclusive first predicted65 to first85',
        coefficients_t_ha_per_glai_day=[.0141,.018,.0207],cultivar_range_is_confidence_interval=False,
        all_original_tissue_residue_and_host_area_prefixes_identical=True,
        fitted_model_sha256=hashlib.sha256(fitted_path.read_bytes()).hexdigest(),
        stage_threshold_sha256=hashlib.sha256(stage_path.read_bytes()).hexdigest(),
        weather_sha256=hashlib.sha256(Path(DEFAULT_PATHS['weather']).read_bytes()).hexdigest())
    (dest/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
