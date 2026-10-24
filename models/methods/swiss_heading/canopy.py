"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/fip_heading_adaptation_20261011')
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from scipy.special import expit
from analysis.methods.crop_canopy.models import thermal_basis
from analysis.methods.crop_canopy.inputs import weather_prefix
from analysis.methods.swiss_heading import inputs, run
SCOPE = 'Post hoc development diagnostic on 2016–2018; not held-year canopy validation'

def marginal_value(cdf, temperature, assessment, parameters):
    exposure, mass = thermal_basis(cdf, temperature, assessment, 'marginal')
    return float(mass @ expit((parameters['midpoint'] - exposure) / parameters['scale']))

def independent_value(cdf, temperature, assessment, parameters):
    prefix = np.maximum.accumulate(np.asarray(cdf, float)[:assessment + 1])
    event_mass = np.diff(np.r_[0.0, prefix])
    positive = np.maximum(0.0, temperature[:assessment + 1])
    probability = (1 - prefix[-1]) * expit(parameters['midpoint'] / parameters['scale'])
    for day, mass in enumerate(event_mass):
        exposure = positive[day + 1:].sum()
        probability += mass * expit((parameters['midpoint'] - exposure) / parameters['scale'])
    return float(probability)

def summarize(predictions, windows):
    rows = []
    for model, part in predictions.groupby('model'):
        for year, group in [('pooled', part), *list(part.groupby('year'))]:
            error = group.prediction - group.green_fraction
            point = pd.DataFrame(dict(year=group.year, plot_id=group.plot_id, squared=error ** 2, absolute=abs(error), bias=error))
            value = point.groupby(['year', 'plot_id']).mean(numeric_only=True).groupby('year').mean().mean()
            integral = windows[windows.model.eq(model)]
            if year != 'pooled':
                integral = integral[integral.year.eq(year)]
            rows.append(dict(scope=SCOPE, model=model, year=str(year), observations=len(group), plots=group.plot_id.nunique(), rmse=float(np.sqrt(value.squared)), mae=float(value.absolute), bias=float(value.bias), integral_plots=len(integral), integral_mae_score_days=float(integral.groupby('year').absolute_error.mean().mean()), integral_bias_score_days=float(integral.groupby('year').signed_error.mean().mean())))
    return pd.DataFrame(rows)

def diagnose(output=inputs.RESULTS):
    output = Path(output)
    inputs.check_sources()
    run.check_lock(output / 'development_lock.json')
    paths = [inputs.SOURCE / 'results/predictions.csv.gz', inputs.SOURCE / 'results/fixed_parameters.json', inputs.SOURCE / 'results/metrics.csv']
    with zipfile.ZipFile(inputs.ARCHIVE) as archive:
        for path in paths:
            assert inputs.sha(path) == hashlib.sha256(archive.read(str(path.relative_to(inputs.ROOT)))).hexdigest()
    archived = pd.read_csv(paths[0])
    frame = archived[archived.scope.eq('coupled_all') & archived.model.eq('marginal')].copy()
    assert len(frame) == 22688 and frame.plot_id.nunique() == 2215 and (set(frame.year) == {2016, 2017, 2018})
    assert not frame.observation_id.duplicated().any()
    fixed = json.loads(paths[1].read_text())
    parameters = next((x for x in fixed['coupled'] if x['model'] == 'marginal'))
    constant = next((x for x in fixed['coupled'] if x['model'] == 'training_mean'))['mean']
    selected = json.loads((output / 'selection.json').read_text())['selected_model']
    names = json.loads((output / 'prediction_models.json').read_text())
    arrays = np.load(output / 'prediction_cdfs.npy')
    registry = pd.read_csv(output / 'registry.csv')
    weather = pd.read_csv(inputs.SOURCE / 'results/weather_daily.csv')
    cycles = {int(row.year): (i, row.sowing_date, weather_prefix(weather, row.sowing_date, 300)[:, 0]) for i, row in enumerate(registry.itertuples()) if row.year in inputs.TRAIN}
    predictions = []
    max_independent_difference = 0.0
    max_original_difference = 0.0
    for kind, model in [('frozen', 'frozen_heading'), (selected, 'adapted_heading'), ('constant', 'training_mean')]:
        values = {}
        for year, date in frame[['year', 'date']].drop_duplicates().itertuples(index=False, name=None):
            if kind == 'constant':
                value = constant
            else:
                i, sowing, temperature = cycles[year]
                day = (pd.Timestamp(date) - pd.Timestamp(sowing)).days
                assert 0 <= day < 300
                cdf = arrays[names.index(kind), i]
                value = marginal_value(cdf, temperature, day, parameters)
                independent = independent_value(cdf, temperature, day, parameters)
                max_independent_difference = max(max_independent_difference, abs(value - independent))
                assert abs(value - independent) < 1e-10
            values[year, date] = value
        part = frame.copy()
        part['prediction'] = [values[r.year, r.date] for r in part.itertuples()]
        part['model'] = model
        part['scope'] = SCOPE
        if kind == 'frozen':
            max_original_difference = float(np.abs(part.prediction.to_numpy() - frame.prediction.to_numpy()).max())
            assert max_original_difference < 1e-06
        predictions.append(part)
    predictions = pd.concat(predictions, ignore_index=True)
    windows, exclusions = ([], [])
    for (model, year, plot), part in predictions.groupby(['model', 'year', 'plot_id']):
        part = part.sort_values('date')
        assert not part.date.duplicated().any()
        exclusions.append(dict(model=model, year=year, plot_id=plot, observations=len(part), integral_eligible=len(part) >= 2, reason='eligible' if len(part) >= 2 else 'single_visit'))
        if len(part) < 2:
            continue
        times = (pd.to_datetime(part.date) - pd.Timestamp(part.date.iloc[0])).dt.days.to_numpy()
        dt = np.diff(times)
        error = (part.prediction - part.green_fraction).to_numpy()
        signed = float(np.sum(dt * (error[1:] + error[:-1]) / 2))
        windows.append(dict(model=model, year=year, plot_id=plot, observations=len(part), first_date=part.date.iloc[0], last_date=part.date.iloc[-1], duration_days=int(times[-1]), signed_error=signed, absolute_error=abs(signed), scope=SCOPE))
    windows = pd.DataFrame(windows)
    metrics = summarize(predictions, windows)
    assert len(windows) == 3 * 2211 and len(metrics) == 12
    original = pd.read_csv(paths[2])
    original = original[original.scope.eq('coupled_all') & original.model.eq('marginal')].set_index('year')
    for row in metrics[metrics.model.eq('frozen_heading')].itertuples():
        old = original.loc[row.year]
        np.testing.assert_allclose([row.rmse, row.mae, row.bias, row.integral_mae_score_days], [old.rmse, old.mae, old.bias, old.integral_mae_green_fraction_days], atol=1e-06, rtol=1e-06)
    predictions.to_csv(output / 'canopy_diagnostic_predictions.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    windows.to_csv(output / 'canopy_diagnostic_integrals.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    pd.DataFrame(exclusions).to_csv(output / 'canopy_diagnostic_eligibility.csv', index=False)
    metrics.to_csv(output / 'canopy_diagnostic_metrics.csv', index=False)
    receipt = dict(scope=SCOPE, observations=22688, plots=2215, integral_plots=2211, single_visit_plots=4, selected_heading_model=selected, canopy_parameters=parameters, canopy_parameters_refitted=False, independent_formula_max_abs_difference=max_independent_difference, archived_frozen_prediction_max_abs_difference=max_original_difference, independent_later_year_canopy_evidence=False, prior_evidence_changed=False)
    inputs.save(output / 'canopy_diagnostic.json', receipt)
    return receipt
