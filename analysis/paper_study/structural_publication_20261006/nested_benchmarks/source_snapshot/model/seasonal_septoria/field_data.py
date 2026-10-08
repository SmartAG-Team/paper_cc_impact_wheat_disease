"""Season-start input assembly with explicit calendar and observation operators."""

from dataclasses import dataclass
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

from process_model.calibrate import transform
from .host import cohort_inputs, sowing_date_from_calendar

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class FieldData:
    temperature: np.ndarray
    humidity: np.ndarray
    rain: np.ndarray
    host_active: np.ndarray
    host_renewal: np.ndarray
    metadata: pd.DataFrame
    targets: pd.DataFrame
    excluded: pd.DataFrame


def _leaf_index(organ):
    label = str(organ).upper()
    match = re.search(r'(?:LEAF[, ]+|F)([1-7])(?![0-9])',label)
    if match:
        return int(match.group(1))-1
    return 0 if label == 'FLAG LEAF' else None


def prepare_fields(assessments, calendars, weather):
    """Construct weather/phenology inputs independently of observed severity.

    Only supported ordinal leaf percentages enter targets. Missing calendars
    are explicit field exclusions; an incomplete daily forcing series is an
    error. The copied winter phenology parameters and rainfed winter calendar
    are a declared crop scenario where the source omits wheat type or sowing.
    """
    source = assessments.copy()
    source['date'] = pd.to_datetime(source.date)
    source['leaf_index'] = source.organ.map(_leaf_index)
    source['value'] = pd.to_numeric(source.value,errors='coerce')
    source['exclusion_reason'] = ''
    source.loc[source.leaf_index.isna(),'exclusion_reason'] = 'unsupported_organ_definition'
    supported_metrics = {'infection_percent_unspecified_basis','necrotic_leaf_area_percent',
                         'pycnidial_coverage_percent'}
    if 'unit' not in source:
        raise ValueError('Source measurement units are required.')
    valid_measurement = source.metric.isin(supported_metrics) & source.unit.astype(str).str.strip().str.lower().isin(['percent','%'])
    source.loc[~valid_measurement,'exclusion_reason'] = 'unsupported_metric_or_unit'
    source.loc[source.value.isna(),'exclusion_reason'] = 'no_numeric_assessment'
    source.loc[source.value.notna() & ~source.value.between(0,100),'exclusion_reason'] = 'outside_percent_range'
    excluded = source.loc[source.exclusion_reason.ne('')].rename(columns={'exclusion_reason':'reason'}).copy()
    source = source.loc[source.exclusion_reason.eq('')].copy()
    if source.empty:
        raise ValueError('No eligible leaf assessments.')
    forcing = weather.copy()
    forcing['date'] = pd.to_datetime(forcing.date)
    if forcing.duplicated(['location_id','date']).any():
        raise ValueError('Duplicate weather location/date keys.')
    calendar_rows = calendars.loc[calendars.crop_season.eq('winter_wheat')
        & calendars.water_system.eq('rainfed') & calendars.calendar_valid.eq(True)].copy()
    if calendar_rows.duplicated(['dataset_id','point_id']).any():
        raise ValueError('Duplicate crop-calendar point keys.')
    parameters = json.loads((ROOT/'process_model/parameters/calibration.json').read_text())
    thresholds = {row['BBCH']:row['Cumulative_t_pp_v_GDD'] for row in parameters['thresholds']}
    pieces, metadata, targets = [], [], []
    for (dataset,unit,year), group in source.groupby(['dataset_id','physical_unit','season_year'],sort=True):
        identities = group[['site_id','latitude','longitude','country']].drop_duplicates()
        if len(identities) != 1:
            raise ValueError('Field-season has conflicting spatial metadata.')
        row = identities.iloc[0]
        known = group.get('sowing_date',pd.Series(dtype=object)).dropna().unique()
        if len(known) > 1:
            raise ValueError('Field-season has conflicting sowing dates.')
        if len(known):
            sowing = pd.Timestamp(known[0])
            sowing_basis = 'observed_source_sowing'
        else:
            match = calendar_rows.loc[calendar_rows.dataset_id.eq(dataset)
                                      & calendar_rows.point_id.eq(row.site_id)]
            if match.empty:
                quarantine = group.copy()
                quarantine['reason'] = 'missing_winter_rainfed_calendar'
                excluded = pd.concat([excluded,quarantine],ignore_index=True)
                continue
            calendar = match.iloc[0]
            sowing = sowing_date_from_calendar(int(year),calendar.planting_doy,calendar.maturity_doy)
            sowing_basis = 'calendar_scenario_not_observed'
        dates = pd.date_range(sowing,group.date.max())
        if dates.empty or (group.date < sowing).any():
            raise ValueError('Assessment precedes seasonal sowing scenario.')
        frame = forcing.loc[forcing.location_id.eq(row.site_id)
                            & forcing.date.between(dates[0],dates[-1])].sort_values('date').copy()
        if not pd.DatetimeIndex(frame.date).equals(dates):
            raise ValueError(f'Missing daily weather for {dataset}|{unit}|{year}.')
        weather_columns = ['tmean_c','tmax_c','rh_mean_pct','precipitation_mm']
        if not np.isfinite(frame[weather_columns]).all().all() or (frame.tmean_c > 40).any():
            raise ValueError('Invalid or unsupported daily weather values.')
        field_index = len(pieces)
        raw = frame.rename(columns={'date':'DATE','tmean_c':'t_mean','tmax_c':'t_max'})
        raw['PEP_ID'],raw['LAT'] = field_index+1,row.latitude
        raw['SOWING_DATE'],raw['SOWING_KNOWN_AT'] = sowing,sowing
        raw['GDD'] = np.where(raw.t_mean > 30,20-2*(raw.t_mean-30),np.clip(raw.t_mean,0,20))
        features = transform(raw,parameters)
        accumulated = features.Cumulative_t_pp_v_GDD.to_numpy()[None,:]
        t = features.t_mean.to_numpy()[None,:]
        active,renewal = cohort_inputs(accumulated,t,thresholds)
        pieces.append((frame,active[0],renewal[0]))
        field_id = f'{dataset}|{unit}|{int(year)}'
        metadata.append(dict(field_index=field_index,field_id=field_id,dataset_id=dataset,
            source_unit=unit,season_year=int(year),site_id=row.site_id,country=row.country,
            latitude=row.latitude,longitude=row.longitude,sowing_date=str(sowing.date()),
            sowing_basis=sowing_basis,wheat_type='winter_wheat_scenario',
            first_forcing_date=str(dates[0].date()),last_forcing_date=str(dates[-1].date()),
            forcing_days=len(dates),calendar_known_at_sowing_assumed=sowing_basis!='observed_source_sowing'))
        field_targets = group.drop(columns='exclusion_reason').copy()
        field_targets['field_index'],field_targets['field_id'] = field_index,field_id
        field_targets['day_index'] = (field_targets.date-sowing).dt.days+1
        field_targets['leaf_index'] = field_targets.leaf_index.astype(int)
        field_targets['observation_operator'] = np.where(field_targets.metric.eq('pycnidial_coverage_percent'),
                                                        'pycnidia','damage_proxy')
        targets.append(field_targets)
    if not pieces:
        raise ValueError('No complete field-season inputs.')
    days = max(len(x[0]) for x in pieces)
    n = len(pieces)
    temperature,humidity,rain = [np.zeros((n,days)) for _ in range(3)]
    active = np.zeros((n,days,8),bool)
    renewal = np.zeros(active.shape)
    for i,(frame,a,g) in enumerate(pieces):
        length = len(frame)
        temperature[i,:length] = frame.tmean_c
        humidity[i,:length] = frame.rh_mean_pct
        rain[i,:length] = frame.precipitation_mm
        active[i,:length],renewal[i,:length] = a,g
    metadata = pd.DataFrame(metadata)
    targets = pd.concat(targets,ignore_index=True)
    targets['coordinate_year'] = targets.site_id.astype(str)+'|'+targets.season_year.astype(int).astype(str)
    counts = targets.groupby('coordinate_year').field_id.transform('nunique')
    series = targets.groupby('field_id').endpoint_series.transform('nunique')
    dates_per_series = targets.groupby(['field_id','endpoint_series']).value.transform('size')
    targets['weight'] = 1/counts/series/dates_per_series
    return FieldData(temperature,humidity,rain,active,renewal,metadata,targets,excluded)
