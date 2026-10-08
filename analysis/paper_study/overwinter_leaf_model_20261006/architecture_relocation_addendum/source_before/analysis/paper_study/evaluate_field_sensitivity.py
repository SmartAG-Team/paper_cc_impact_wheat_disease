"""Frozen field sensitivity to calendar sowing and phenology transfer checks."""

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from model.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.calibrate import predict
from model.seasonal_septoria.core import Parameters
from model.seasonal_septoria.endpoints import onset_distance,chronological_partition
from process_model.calibrate import transform
from analysis.paper_study.calibrate_seasonal import metrics,sha

DEST = ROOT/'analysis/paper_study/field_sensitivity'


def events_from_features(data,weather,calibration):
    thresholds = {row['BBCH']:row['Cumulative_t_pp_v_GDD'] for row in calibration['thresholds']}
    events = {}
    for meta in data.metadata.itertuples():
        frame = weather.loc[weather.location_id.eq(meta.site_id)&weather.date.between(meta.first_forcing_date,meta.last_forcing_date)].copy()
        sowing = pd.Timestamp(meta.sowing_date)
        frame = frame.rename(columns={'date':'DATE','tmean_c':'t_mean','tmax_c':'t_max'})
        frame['PEP_ID'],frame['LAT'] = meta.field_index+1,meta.latitude
        frame['SOWING_DATE'],frame['SOWING_KNOWN_AT'] = sowing,sowing
        frame['GDD'] = np.where(frame.t_mean>30,20-2*(frame.t_mean-30),np.clip(frame.t_mean,0,20))
        features = transform(frame,calibration)
        for stage in [10,31,51,85]:
            dates = features.loc[features.Cumulative_t_pp_v_GDD.ge(thresholds[stage]),'DATE']
            events[meta.field_id,stage] = None if dates.empty else pd.Timestamp(dates.iloc[0])
    return events


def main():
    if DEST.exists():
        raise FileExistsError('Field sensitivity archive already exists.')
    DEST.mkdir(parents=True)
    frozen = ROOT/'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json'
    digest = sha(frozen)
    p = Parameters(**json.loads(frozen.read_text())['parameters'])
    calibration = json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    calendar_path = ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv'
    calendars = pd.read_csv(calendar_path)
    weather_path = ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet'
    weather = pd.read_parquet(weather_path)
    full = pd.read_csv(ROOT/'analysis/paper_study/seasonal_calibration_v1/basf_source_assessments_snapshot.csv')
    external = pd.read_csv(ROOT/'data/paper_study/observations/corteva_external_assessments.csv')
    external = external[external.assessment_eligible.eq(True)].copy()
    external['endpoint_series'] = external.physical_unit.astype(str)+'|'+external.organ.astype(str)
    strict = set(pd.read_csv(ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv').source_unit.astype(str))
    scores,onsets,phenology_rows = [],[],[]
    for source_name,source in [('BASF',full),('Corteva',external)]:
        for offset in [-14,0,14]:
            shifted = calendars.copy()
            shifted['planting_doy'] = shifted.planting_doy+offset
            data = prepare_fields(source,shifted,weather)
            data.metadata['sowing_basis'] = 'shifted_calendar_scenario_not_observed'
            targets = data.targets.copy()
            targets['partition'] = chronological_partition(targets,[2019]) if source_name=='BASF' else 'external'
            targets['strict_location_disjoint'] = targets.physical_unit.astype(str).isin(strict) if source_name=='Corteva' else True
            forecast,trajectory = predict(data,p)
            targets['predicted_percent'] = forecast
            targets['model'],targets['source'],targets['sowing_offset_days'] = 'frozen_seasonal_seir',source_name,offset
            targets.to_parquet(DEST/f'{source_name.lower()}_sow{offset:+d}_predictions.parquet',index=False)
            data.metadata.to_csv(DEST/f'{source_name.lower()}_sow{offset:+d}_inputs.csv',index=False)
            if offset==0:
                archived_path = ROOT/('analysis/paper_study/seasonal_calibration_v1/basf_frozen_predictions.parquet'
                    if source_name=='BASF' else 'analysis/paper_study/corteva_external_seasonal_v1/frozen_external_predictions.parquet')
                archived = pd.read_parquet(archived_path)
                archived = archived[archived.model.eq('seasonal_seir')]
                if len(archived)!=len(targets):
                    raise ValueError('Unshifted source membership changed.')
                keys = ['field_id','endpoint_series','date']
                reconciled = targets[keys+['predicted_percent']].merge(
                    archived[keys+['predicted_percent']],on=keys,how='outer',
                    validate='one_to_one',indicator=True,suffixes=('_new','_frozen'))
                if not reconciled['_merge'].eq('both').all():
                    raise ValueError('Unshifted assessment membership changed.')
                np.testing.assert_allclose(reconciled.predicted_percent_new,
                    reconciled.predicted_percent_frozen,rtol=0,atol=1e-10)
            for partition,group in targets.groupby('partition'):
                scopes = [('all_source',group)] if source_name=='BASF' else [('all_source',group),('strict_locations',group[group.strict_location_disjoint])]
                for geography,selected in scopes:
                    for leaf_scope,leaf_data in [('all_ordinal',selected),('upper_three',selected[selected.leaf_index.lt(3)])]:
                        final = leaf_data.sort_values('date').groupby(['field_id','endpoint_series']).tail(1).copy()
                        final['weight'] = 1/final.groupby('coordinate_year').field_id.transform('nunique')/final.groupby('field_id').endpoint_series.transform('nunique')
                        scores.append(dict(source=source_name,partition=partition,geography=geography,sowing_offset_days=offset,
                            leaf_scope=leaf_scope,endpoint='last_numeric_leaf_assessment',**metrics(final)))
            for (field,series),group in targets.groupby(['field_index','endpoint_series']):
                row = group.iloc[0];meta = data.metadata.loc[data.metadata.field_index.eq(field)].iloc[0]
                length,rank = int(meta.forcing_days),int(row.leaf_index)
                for cutoff in [.1,1.,5.]:
                    ordered = group.sort_values('date')
                    positive = ordered[ordered.value.ge(cutoff)]
                    upper = None if positive.empty else positive.date.min()
                    negative = ordered[ordered.value.lt(cutoff)]
                    lower = negative.date.max() if upper is None else negative.loc[negative.date.lt(upper),'date'].max()
                    lower = None if pd.isna(lower) else lower
                    found = np.flatnonzero(trajectory.damage[field,1:length+1,rank]*100>=cutoff)
                    predicted = None if not len(found) else pd.Timestamp(meta.sowing_date)+pd.Timedelta(days=int(found[0]))
                    onsets.append(dict(source=source_name,field_id=row.field_id,coordinate_year=row.coordinate_year,
                        series=series,partition=row.partition,strict_location_disjoint=bool(row.strict_location_disjoint),
                        sowing_offset_days=offset,cutoff_percent=cutoff,
                        censoring='right' if upper is None else 'left' if lower is None else 'interval',
                        **onset_distance(predicted,lower,upper)))
            if offset==0:
                events = events_from_features(data,weather,calibration)
                stages = targets.copy()
                stages['stage_from'] = pd.to_numeric(stages.stage_from,errors='coerce')
                stages['stage_to'] = pd.to_numeric(stages.stage_to,errors='coerce')
                stages = stages[stages.stage_from.between(0,99)&stages.stage_to.between(0,99)
                    &stages.stage_from.le(stages.stage_to)].drop_duplicates(['field_id','date','stage_from','stage_to'])
                for field,group in stages.groupby('field_id'):
                    row = group.iloc[0]
                    for stage in [10,31,51,85]:
                        positive = group[group.stage_from.ge(stage)]
                        upper = None if positive.empty else positive.date.min()
                        negative = group[group.stage_to.lt(stage)]
                        lower = negative.date.max() if upper is None else negative.loc[negative.date.lt(upper),'date'].max()
                        lower = None if pd.isna(lower) else lower
                        if lower is None and upper is None:
                            continue
                        predicted = events[field,stage]
                        phenology_rows.append(dict(source=source_name,field_id=field,coordinate_year=row.coordinate_year,
                            partition=row.partition,strict_location_disjoint=bool(row.strict_location_disjoint),stage=stage,
                            censoring='right' if upper is None else 'left' if lower is None else 'interval',
                            predicted_date=None if predicted is None else str(predicted.date()),
                            last_negative=None if lower is None else str(lower.date()),first_positive=None if upper is None else str(upper.date()),
                            observed_stage_assimilated=False,**onset_distance(predicted,lower,upper)))
    pd.DataFrame(scores).to_csv(DEST/'final_severity_sowing_sensitivity.csv',index=False)
    pd.DataFrame(onsets).to_csv(DEST/'onset_sowing_sensitivity.csv',index=False)
    pd.DataFrame(phenology_rows).to_csv(DEST/'independent_stage_transfer_intervals.csv',index=False)
    if sha(frozen)!=digest:
        raise ValueError('Frozen fit changed.')
    receipt = dict(status='complete',sowing_offsets_days=[-14,0,14],parameters_refitted=False,
        offsets_selected_using_validation=False,true_sowing_dates_observed=False,
        frozen_fit_sha256=digest,weather_sha256=sha(weather_path),calendar_sha256=sha(calendar_path),
        unshifted_predictions_reconcile_with_frozen_archives=True,
        source_code_sha256=sha(Path(__file__)),phenology_intervals=len(phenology_rows))
    (DEST/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)
    print(pd.DataFrame(scores).to_string(index=False),flush=True)


if __name__=='__main__':
    main()
