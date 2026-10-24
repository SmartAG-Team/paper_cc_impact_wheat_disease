"""Replay the bundled F1 records with unchanged equations, without fitting.

The identical script is copied to the review package's code directory. Run
that copy with Python from any working directory; only bundled inputs enter.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


def replay(package, output):
    package = Path(package).resolve()
    checkout=next((parent for parent in [package,*package.parents] if (parent/'models/compat.py').is_file()),None)
    if checkout is not None:
        sys.path.insert(0,str(checkout))
        from models.monitoring.climate_prefix import simulate_prefix
    else:
        sys.path.insert(0,str(package/'code'))
        from analysis.disease_improvement.climate_prefix import simulate_prefix

    data = package / 'data/process_f1'
    records = pd.read_csv(data / 'all115_F1_records_with_replay.csv')
    fields = pd.read_csv(data / 'field55_calendar_membership.csv')
    weather = pd.read_csv(data / 'four_weather_forcing.csv', parse_dates=['date'])
    phenology = json.loads((package / 'parameters/phenology_parameters.json').read_text())
    fit = json.loads((package / 'parameters/disease_parameters.json').read_text())['fitted']
    thresholds = {int(k): v for k, v in json.loads(
        (package / 'parameters/stage_thresholds.json').read_text())['all_stage_thresholds'].items()}
    names = ('tmean_c', 'tmax_c', 'rh_mean_pct', 'precipitation_mm')
    predictions = []
    state_error = 0.
    for meta in fields.itertuples():
        part = weather[weather.field_id.eq(meta.field_id)].sort_values('date')
        dates = pd.DatetimeIndex(part.date)
        assert dates.equals(pd.date_range(meta.sowing_date, meta.last_forcing_date))
        assert len(dates) == meta.forcing_days
        forcing = {k: part[k].to_numpy(float)[None, :] for k in names}
        sowing = np.array([meta.sowing_date], dtype='datetime64[D]')
        latitude = np.array([meta.latitude])
        full = simulate_prefix(dates, forcing, sowing, latitude, phenology, fit, thresholds)
        for record in records[records.field_id.eq(meta.field_id)].itertuples():
            q = (pd.Timestamp(record.date) - dates[0]).days + 1
            result = dict(field_id=meta.field_id, date=record.date)
            for label, days in [('day_end', q), ('previous_day', q-1)]:
                prefix = simulate_prefix(dates[:days], {k: v[:, :days] for k, v in forcing.items()},
                    sowing, latitude, phenology, fit, thresholds)
                value = float(100 * prefix['trajectory'].damage[0, -1, 0])
                expected = getattr(record, 'climate_prefix_' + label + '_percent')
                assert abs(value - expected) < 1e-10
                error = np.max(abs(prefix['trajectory'].state - full['trajectory'].state[:, :days+1]))
                state_error = max(state_error, float(error))
                result[label + '_percent'] = value
            predictions.append(result)
    predictions = pd.DataFrame(predictions)
    records = records.merge(predictions, on=['field_id', 'date'], validate='one_to_one')
    evaluation = records[records.partition.eq('reused_BASF2019')].copy()
    assert len(records) == 115 and len(evaluation) == 70 and state_error == 0
    # Rebuild the F1-only hierarchy; retain separately the archived parent mass.
    nfields = evaluation.groupby('coordinate_year').field_id.transform('nunique')
    nrecords = evaluation.groupby(['coordinate_year', 'field_id']).field_id.transform('size')
    evaluation['conditional_weight_rebuilt'] = 1 / (nfields * nrecords)
    assert np.allclose(evaluation.conditional_weight_rebuilt, evaluation.F1_partition_hierarchy_weight)
    def rmse(frame, column, weight, truth='observed_percent'):
        return float(np.sqrt(np.average((frame[column]-frame[truth])**2, weights=frame[weight])))
    scores = []
    for label, weight in [('F1 conditional', 'conditional_weight_rebuilt'),
                          ('parent-relative', 'original_evaluation_hierarchy_weight')]:
        for column in ['day_end_percent', 'previous_day_percent', 'leaf_mean']:
            scores.append(dict(endpoint='severity', population=label, model=column,
                records=70, rmse=rmse(evaluation, column, weight), units='percentage points'))
    windows = pd.read_csv(data / 'evaluation23_F1_windows.csv')
    integrated = []
    for row in windows.itertuples():
        part = evaluation[evaluation.field_id.eq(row.field_id)].sort_values('date')
        days = (pd.to_datetime(part.date)-pd.to_datetime(part.date.iloc[0])).dt.days.to_numpy()
        item = dict(field_id=row.field_id)
        for column in ['observed_percent', 'day_end_percent', 'previous_day_percent', 'leaf_mean']:
            item[column] = float(np.trapezoid(part[column], days))
        integrated.append(item)
    integrated = windows[['field_id','coordinate_year','original_evaluation_window_hierarchy_weight']].merge(
        pd.DataFrame(integrated), on='field_id', validate='one_to_one')
    nfields = integrated.groupby('coordinate_year').field_id.transform('nunique')
    integrated['conditional'] = 1 / nfields
    for label, weight in [('F1-window conditional','conditional'),
                          ('parent-window relative','original_evaluation_window_hierarchy_weight')]:
        for column in ['day_end_percent', 'previous_day_percent', 'leaf_mean']:
            scores.append(dict(endpoint='observed-window integral', population=label, model=column,
                records=23, rmse=rmse(integrated, column, weight), units='percentage-point days'))
    output = Path(output).resolve()
    if output == data or data in output.parents:
        raise ValueError('Replay receipts must not overwrite the archived input directory')
    output.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(output/'predictions.csv', index=False)
    pd.DataFrame(scores).to_csv(output/'scores.csv', index=False)
    checks = dict(status='passed', records=115, fields=len(fields), evaluation_records=70,
        calibration_records=45, evaluation_windows=23, maximum_state_error=state_error,
        fitted=False, weather_endpoints_separate=True, disease_parameters_modified=False,
        retrospective_reuse=True)
    (output/'checks.json').write_text(json.dumps(checks, indent=2)+'\n')
    print(json.dumps(checks, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    bundled = Path(__file__).resolve().parent.parent
    parser.add_argument('--package', type=Path, default=bundled)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    replay(args.package, args.output)
