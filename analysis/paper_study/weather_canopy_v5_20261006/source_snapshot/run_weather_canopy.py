"""Compare daily climate-compatible duration proxies on calibration sites."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from analysis.paper_study.calibrate_seasonal import sha, write_json
from analysis.paper_study.refine_seasonal_accuracy import score
from analysis.paper_study.run_empirical_benchmarks import endpoint, paired_intervals
from analysis.paper_study.run_structural_canopy import development_inputs, onset_predictions
from model.seasonal_septoria.field_data import prepare_fields
from model.seasonal_septoria.structural import fit_canopy, predict_canopy
from model.seasonal_septoria.wetness import duration_exposure

V1 = ROOT/'analysis/paper_study/seasonal_calibration_v1'
V4 = ROOT/'analysis/paper_study/structural_canopy_v4_20261006'
DEFAULT = ROOT/'analysis/paper_study/weather_canopy_v5_20261006'
KEYS = ['field_id', 'endpoint_series', 'date']


def training_weather(data, weather, fields):
    forcing = weather.copy(); forcing['date'] = pd.to_datetime(forcing.date)
    rows = []
    for meta in data.metadata[data.metadata.field_index.isin(fields)].itertuples():
        rows.append(forcing[forcing.location_id.eq(meta.site_id)&
            forcing.date.between(meta.first_forcing_date, meta.last_forcing_date)])
    if not rows:
        raise ValueError('No calibration weather fields.')
    return pd.concat(rows).drop_duplicates(['location_id', 'date'])


def training_rain_rate(data, weather, fields):
    selected = training_weather(data, weather, fields)
    wet = selected[(selected.rain_hours_gt_0_1mm > 0)&(selected.tmean_c > 2)
                   &(selected.precipitation_mm > 0)]
    if wet.empty:
        raise ValueError('No rainy calibration meteorology for duration scaling.')
    return float(np.median(wet.precipitation_mm/wet.rain_hours_gt_0_1mm))


def exposure_for(data, rate):
    return duration_exposure(data.temperature, data.maximum_temperature,
                             data.humidity, data.rain, rate)['exposure']


def meteorological_checks(data, weather, rate):
    rows = []
    for partition, selected in data.targets.groupby('partition'):
        w = training_weather(data, weather, selected.field_index.unique())
        proxy = duration_exposure(w.tmean_c.to_numpy(), w.tmax_c.to_numpy(),
                                 w.rh_mean_pct.to_numpy(), w.precipitation_mm.to_numpy(), rate)
        for label, observed, forecast in [('humid_hours', w.rh_hours_ge_90pct.to_numpy(), proxy['humid_hours']),
                                          ('rain_hours', w.rain_hours_gt_0_1mm.to_numpy(), proxy['rain_hours'])]:
            error = forecast-observed
            variance = np.var(observed)
            rows.append(dict(partition=partition, variable=label, days=len(w),
                rmse_hours=float(np.sqrt(np.mean(error**2))), mae_hours=float(np.mean(abs(error))),
                bias_hours=float(error.mean()), r2=None if not variance else float(1-np.mean(error**2)/variance)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT)
    args = parser.parse_args(); dest = args.output.resolve()
    if dest.exists():
        raise FileExistsError('Use a new weather-operator archive.')
    dest.mkdir(parents=True)
    paths = dict(source=V1/'basf_source_assessments_snapshot.csv',
        weather=ROOT/'data/paper_study/field_weather_completed/daily_weather.parquet',
        calendars=ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv',
        phenology=ROOT/'process_model/parameters/calibration.json')
    protocol = json.loads((V1/'configuration_before_fitting.json').read_text())
    old = json.loads((V4/'calibration_only_selection.json').read_text())
    controls = [row for row in old if not row['amplification_free']
                and row['host_parameters']['leaf_interval'] in [80., 120., 160.]]
    candidates = [dict(name='duration_'+row['name'], host_parameters=row['host_parameters'],
        amplification_free=False, latent_days=row['latent_days'], weather_operator='duration_proxy') for row in controls]
    dependencies = [Path(__file__), *paths.values(), V4/'calibration_only_selection.json',
        V4/'frozen_selected_fit.json', V4/'frozen_evaluation_predictions.parquet',
        ROOT/'analysis/paper_study/run_structural_canopy.py',
        ROOT/'analysis/paper_study/calibrate_seasonal.py',
        ROOT/'analysis/paper_study/refine_seasonal_accuracy.py',
        ROOT/'analysis/paper_study/run_empirical_benchmarks.py',
        *sorted((ROOT/'model/seasonal_septoria').glob('*.py')),
        *sorted((ROOT/'process_model').glob('*.py')),
        *sorted((ROOT/'process_model/_support').glob('*.py'))]
    config = dict(registered_utc=datetime.now(timezone.utc).isoformat(), candidates=candidates,
        control_candidates_source='v4 original daily-OR out-of-fold predictions',
        calibration_years=[2017, 2018], site_fold=protocol['site_fold'],
        final_loss_weight=.5, initiation_upper_bound_duration=1., initiation_upper_bound_control=.1,
        selection='minimum calibration-only pooled final severity RMSE',
        amplification_absent_after_v4_calibration_selection=True,
        weather_assumptions=['symmetric temperature cycle', 'constant within-day vapour pressure',
            'synthetic RH constrained to supplied daily mean', 'independent rain/humidity overlap',
            'rain intensity calibrated on training meteorology only'],
        inferred_duration_is_not_measured_leaf_wetness=True,
        all_evaluation_sources_previously_examined=True, untouched_test=False,
        input_and_source_sha256={str(path.relative_to(ROOT)):sha(path) for path in dependencies})
    write_json(dest/'configuration_before_fitting.json', config)
    weather, calendars, phenology = pd.read_parquet(paths['weather']), pd.read_csv(paths['calendars']), json.loads(paths['phenology'].read_text())
    data = prepare_fields(pd.read_csv(paths['source']), calendars, weather)
    data.targets['partition'] = np.where(data.targets.season_year.lt(2019), 'calibration', 'reused_development_2019')
    accumulation, thresholds = development_inputs(data, weather, phenology)
    training = np.flatnonzero(data.targets.partition.eq('calibration'))
    folds = data.targets.site_id.map(protocol['site_fold']).to_numpy()
    drivers, rates = {}, {}
    for fold in range(3):
        indices = training[folds[training] != fold]
        rates[fold] = training_rain_rate(data, weather, data.targets.iloc[indices].field_index.unique())
        drivers[fold] = exposure_for(data, rates[fold])
    selection = [dict(**row, weather_operator='daily_or') for row in controls]
    for index, candidate in enumerate(candidates):
        pieces = []
        for fold in range(3):
            fitted_rows, heldout_rows = training[folds[training] != fold], training[folds[training] == fold]
            if set(data.targets.iloc[fitted_rows].site_id)&set(data.targets.iloc[heldout_rows].site_id):
                raise ValueError('Location leakage.')
            fitted = fit_canopy(data, accumulation, thresholds, fitted_rows,
                candidate['host_parameters'], False, candidate['latent_days'],
                environmental_response=drivers[fold], initiation_max=1.)
            fitted['weather_operator'], fitted['rain_rate_mm_hour'] = 'duration_proxy', rates[fold]
            write_json(dest/f"inner_{candidate['name']}_fold{fold}.json", fitted)
            forecast, _ = predict_canopy(data, accumulation, thresholds, fitted, drivers[fold])
            scored = data.targets.iloc[heldout_rows].copy()
            scored['predicted_percent'], scored['inner_fold'] = forecast[heldout_rows], fold
            scored.to_parquet(dest/f"inner_{candidate['name']}_fold{fold}_predictions.parquet", index=False)
            pieces.append(scored)
        pooled = pd.concat(pieces, ignore_index=True)
        result = dict(**candidate, **score(endpoint(pooled, True, False)),
            all_assessment_rmse=score(endpoint(pooled, False, False))['rmse'])
        selection.append(result)
        write_json(dest/'calibration_only_selection.json', selection)
        print(f"duration CV {index+1}/{len(candidates)} {candidate['name']}: final RMSE {result['rmse']:.4f}", flush=True)
    chosen = min(selection, key=lambda row:row['rmse'])
    rate = training_rain_rate(data, weather, data.targets.iloc[training].field_index.unique())
    driver = exposure_for(data, rate) if chosen['weather_operator'] == 'duration_proxy' else None
    fitted = fit_canopy(data, accumulation, thresholds, training, chosen['host_parameters'],
        False, chosen['latent_days'], environmental_response=driver,
        initiation_max=1. if driver is not None else .1)
    fitted['selected_candidate'], fitted['selection_cv_rmse'] = chosen['name'], chosen['rmse']
    fitted['weather_operator'], fitted['rain_rate_mm_hour'] = chosen['weather_operator'], rate
    write_json(dest/'frozen_selected_fit.json', fitted)
    freeze = dict(frozen_utc=datetime.now(timezone.utc).isoformat(), fit_sha256=sha(dest/'frozen_selected_fit.json'),
                  selection=chosen, external_or_2019_disease_used=False, prior_evaluation_exposure=True)
    write_json(dest/'frozen_before_evaluation.json', freeze)
    print(f"FROZEN weather comparison: {chosen['name']} RMSE {chosen['rmse']:.4f}", flush=True)
    external = pd.read_csv(ROOT/'data/paper_study/observations/corteva_external_assessments.csv')
    external = external.loc[external.assessment_eligible.eq(True)].copy()
    external['endpoint_series'] = external.physical_unit.astype(str)+'|'+external.organ.astype(str)
    transfer = prepare_fields(external, calendars, weather)
    strict = set(pd.read_csv(ROOT/'data/paper_study/observations/corteva_location_disjoint_external_units.csv').source_unit.astype(str))
    transfer.targets['partition'] = np.where(transfer.targets.physical_unit.astype(str).isin(strict), 'reused_external_strict', 'reused_external_overlap')
    external_accumulation, _ = development_inputs(transfer, weather, phenology)
    reference = pd.read_parquet(V4/'frozen_evaluation_predictions.parquet')
    predictions, onsets, meteo = [], [], []
    for source, inputs, accumulated in [('BASF', data, accumulation), ('Corteva', transfer, external_accumulation)]:
        actual_driver = exposure_for(inputs, rate) if chosen['weather_operator'] == 'duration_proxy' else None
        forecast, trajectory = predict_canopy(inputs, accumulated, thresholds, fitted, actual_driver)
        scored = inputs.targets.copy(); scored['model'], scored['source'], scored['predicted_percent'] = 'weather_canopy_v5', source, forecast
        predictions.append(scored)
        for model in ['reference_v1', 'structural_canopy_v4']:
            values = reference[reference.source.eq(source)&reference.model.eq(model)][KEYS+['predicted_percent']]
            baseline = inputs.targets.merge(values, on=KEYS, how='left', validate='one_to_one')
            if baseline.predicted_percent.isna().any():
                raise ValueError('Comparator membership differs.')
            baseline['model'], baseline['source'] = model, source; predictions.append(baseline)
        onsets.append(onset_predictions(inputs, trajectory.expressed, 'weather_canopy_v5'))
        meteo.extend(meteorological_checks(inputs, weather, rate))
    predictions = pd.concat(predictions, ignore_index=True)
    predictions['physical_unit'] = predictions.physical_unit.astype(str)
    for column in predictions.select_dtypes(include=['object', 'str']).columns:
        nonmissing = predictions[column].dropna()
        if nonmissing.map(type).nunique() > 1:
            predictions[column] = predictions[column].map(lambda value: None if pd.isna(value) else str(value)).astype('string')
    predictions.to_parquet(dest/'frozen_evaluation_predictions.parquet', index=False)
    pd.concat(onsets).to_csv(dest/'onset_interval_predictions.csv', index=False)
    pd.DataFrame(meteo).to_csv(dest/'duration_proxy_weather_validation.csv', index=False)
    rows, paired = [], []; generator = np.random.default_rng(20261006)
    for partition, group in predictions.groupby('partition'):
        for final in [False, True]:
            for upper in [False, True]:
                name, scope = ('final_numeric_assessment' if final else 'all_assessments'), ('upper_three' if upper else 'all_ordinal_leaves')
                for model, frame in group.groupby('model'):
                    rows.append(dict(model=model, partition=partition, endpoint=name, leaf_scope=scope,
                                     **score(endpoint(frame, final, upper))))
                selected = endpoint(group[group.model.eq('weather_canopy_v5')], final, upper)
                baseline = endpoint(group[group.model.eq('reference_v1')], final, upper)
                matched = selected.merge(baseline[KEYS+['predicted_percent']], on=KEYS,
                    validate='one_to_one', suffixes=('', '_reference'))
                matched['comparator_percent'] = matched.predicted_percent_reference
                clusters = np.array(sorted(matched.coordinate_year.unique()))
                draws = generator.integers(0, len(clusters), (2000, len(clusters)))
                paired.append(dict(partition=partition, endpoint=name, leaf_scope=scope,
                    **paired_intervals(matched, clusters, draws)))
    pd.DataFrame(rows).to_csv(dest/'severity_metrics.csv', index=False)
    pd.DataFrame(paired).to_csv(dest/'paired_improvement_intervals.csv', index=False)
    if sha(dest/'frozen_selected_fit.json') != freeze['fit_sha256']:
        raise ValueError('Fit changed during evaluation.')
    snapshot = dest/'source_snapshot'; snapshot.mkdir()
    for path in [Path(__file__), ROOT/'model/seasonal_septoria/structural.py', ROOT/'model/seasonal_septoria/wetness.py']:
        (snapshot/path.name).write_bytes(path.read_bytes())
    write_json(dest/'receipt.json', dict(status='complete', selection=chosen, fit_sha256=freeze['fit_sha256'],
        prior_evaluation_exposure=True, untouched_test=False, original_manuscript_and_climate_unchanged=True))
    print(pd.DataFrame(rows).query("endpoint == 'final_numeric_assessment' and leaf_scope == 'all_ordinal_leaves'").to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
