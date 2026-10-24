"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/phenology_canopy_coupling_20261011')
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.methods.crop_canopy import inputs, models, evaluation
HERE = evidence_dir('analysis/paper_study/phenology_canopy_coupling_20261011')
CODE = ('inputs.py', 'models.py', 'evaluation.py', 'run.py')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def dump(path, value, exclusive=False):
    with Path(path).open('x' if exclusive else 'w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')

def freeze():
    if (HERE / 'results').exists():
        raise ValueError('Protocol must precede inference and fitting')
    protocol = dict(schema_version=1, endpoint='Whole-canopy green fraction and observed-date integral', study_status='Retrospective development; source and 2018 outcomes previously accessed', years=dict(training=[2015, 2016, 2017], evaluation=2018), policies=list(inputs.POLICIES), primary_policy='trial_doy', horizon_days=330, source59_definition='day when 75% of the ears are visible', source51_substitution=False, parent='Frozen three-member wheat ensemble; no retraining', fractions=models.FRACTIONS.tolist(), bridge_objective='Equal-year/equal-plot restricted CRPS', models=list(models.KINDS), bounds=models.BOUNDS, optimizer=dict(method='least_squares_dogbox', log_parameters=True, midpoint_start_fractions=[0.25, 0.5, 0.75], calendar_width_starts=[10.0, 30.0], thermal_width_starts=[150.0, 450.0], ftol=1e-10, xtol=1e-10, gtol=1e-10, max_nfev=2000, jacobian='analytic', x_scale='jac'), numerical_amendment='Both finite-difference TRF and analytic-Jacobian TRF exhausted all six starts in the first marginal fold, approaching a lower-bound width solution. Analytic derivatives, Jacobian scaling and dogbox converge on this bounded objective within the original budget. This numerical amendment precedes later-year scoring; objectives, bounds, starts, budgets, data and selection remain fixed. Both failed attempts and diagnosis are preserved.', selection='Mean whole-held-year MSE, 2015–2017; calendar then median then marginal ties', fit_weights='Equal year, plot within year and visit within plot', prediction='Positive mean temperature strictly after source59; prefix-only; retain unreached mass', bootstrap=dict(replicates=2000, seed=20261011, unit='Whole cultivar', scope='Conditional on fixed fits and one site-year; excludes selection, refitting and date-source uncertainty'), expected_support=dict(historical_records=5059, historical_plots=1320, missing_heading_records=2420, evaluation_records=728, evaluation_plots=104, original_comparison_records=724), exclusions='Nonmeasurement anchors, missing green fraction, non-HN_WF; no missing-heading exclusion', climate_adoption=False, functional_HAD_validation=False, independent_location_validation=False)
    dump(HERE / 'protocol.json', protocol, exclusive=True)
    sources = {str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / 'inputs').iterdir()) if p.is_file()}
    sources['source_manifest.json'] = sha(HERE / 'source_manifest.json')
    dump(HERE / 'fitting_lock.json', dict(created_utc=datetime.now(timezone.utc).isoformat(), protocol_sha256=sha(HERE / 'protocol.json'), sources=sources, implementation={name: sha(HERE / name) for name in CODE}, results_absent_at_freeze=True), exclusive=True)

def check_lock():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def folds(frame, labels):
    if set(frame.year.unique()) != {2015, 2016, 2017, 2018}:
        raise ValueError('Exactly prescribed study years required')
    result = []
    for year in (*evaluation.YEARS, 2018):
        training_years = [y for y in evaluation.YEARS if y != year]
        result.append(('evaluation' if year == 2018 else 'historical_cv', year, frame.loc[frame.year.isin(training_years)].copy(), frame.loc[frame.year.eq(year)].copy(), labels.loc[labels.year.isin(training_years)].copy()))
    return result

def stage_diagnostics(labels, registry, parent, fraction):
    rows = []
    for row in labels.itertuples():
        i = np.flatnonzero(registry.year.to_numpy() == row.year)[0]
        curve = models.interpolate(parent[i, 2], parent[i, 3], fraction)
        observed = (pd.Timestamp(row.heading_date) - pd.Timestamp(registry.iloc[i].sowing_date)).days
        quantiles = []
        for level in (0.1, 0.5, 0.9):
            hit = np.flatnonzero(curve >= level)
            quantiles.append(float(hit[0]) if len(hit) else np.nan)
        lo, median, hi = quantiles
        crps = float(np.sum((curve - (np.arange(len(curve)) >= observed)) ** 2))
        rows.append(dict(plot_id=row.plot_id, cultivar_id=row.cultivar_id, year=row.year, source_stage=59, observed_day=observed, q10=lo, q50=median, q90=hi, error_days=median - observed, interval80_reached=bool(np.isfinite(lo) and np.isfinite(hi)), covered80=bool(lo <= observed <= hi), tail_mass=float(1 - curve[-1]), crps=crps))
    return pd.DataFrame(rows)
