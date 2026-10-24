"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/fip_heading_adaptation_20261011')
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.methods.swiss_canopy.inputs import source_registry
from analysis.methods.crop_canopy.inputs import parent_predictions, weather_prefix
HERE = evidence_dir('analysis/paper_study/fip_heading_adaptation_20261011')
SOURCE = HERE.parent / 'fip_canopy_transfer'
RESULTS = HERE / 'results'
TRAIN = (2016, 2017, 2018)
EVALUATION = (2019, 2021, 2022)
HORIZON = 300

def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')

def check_sources():
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def qualify_labels(plots, *, years, horizon=HORIZON):
    if plots.plot_uid.isna().any() or plots.plot_uid.duplicated().any():
        raise ValueError('Unique source plot identities required')
    part = plots[plots.harvest_year.isin(years)].copy()
    if part.empty or part.genotype_id.isna().any():
        raise ValueError('Nonempty source population and genotype identities required')
    labels, ledger = ([], [])
    for row in part.itertuples():
        common = dict(plot_id=row.plot_uid, year=int(row.harvest_year), genotype_id=str(row.genotype_id))
        if pd.isna(row.heading_date):
            ledger.append(dict(common, eligible=False, reason='missing_heading', event_after_horizon=False))
            continue
        date, sowing, harvest = map(pd.Timestamp, [row.heading_date, row.sowing_date, row.harvest_date])
        if any((pd.isna(x) for x in [date, sowing, harvest])) or not sowing <= date <= harvest:
            raise ValueError('Heading outside documented crop cycle: ' + row.plot_uid)
        elapsed = (date - sowing).days
        if date != date.normalize():
            raise ValueError('Native calendar dates required')
        labels.append(dict(common, heading_date=date.strftime('%Y-%m-%d'), elapsed_day=int(elapsed)))
        ledger.append(dict(common, eligible=True, reason='eligible', event_after_horizon=elapsed >= horizon))
    return (pd.DataFrame(labels), pd.DataFrame(ledger))

def read_labels(years):
    columns = ['plot_uid', 'harvest_year', 'genotype_id', 'sowing_date', 'harvest_date', 'heading_date']
    plots = pd.read_csv(SOURCE / 'results/plots.csv', usecols=columns, dtype={'genotype_id': str})
    return qualify_labels(plots, years=years)

def training_labels():
    return read_labels(TRAIN)

def prepare_predictors(output=RESULTS):
    check_sources()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    plots = pd.read_csv(SOURCE / 'results/plots.csv', usecols=['plot_uid', 'harvest_year', 'sowing_date'])
    registry = source_registry(plots, json.loads((SOURCE / 'sources/site_location.json').read_text()))
    if registry.year.tolist() != list(TRAIN + EVALUATION):
        raise ValueError('Changed source-year population')
    registry['horizon_days'] = HORIZON
    weather = pd.read_csv(SOURCE / 'results/weather_daily.csv')
    eligibility = []
    for row in registry.itertuples():
        matrix = weather_prefix(weather, row.sowing_date, HORIZON)
        eligibility.append(dict(year=int(row.year), sowing_date=row.sowing_date, horizon_days=HORIZON, days=len(matrix), eligible=True, imputed_days=0))
    parent = parent_predictions(registry, weather)
    historical = np.load(SOURCE / 'results/parent_cdf.npy', allow_pickle=False)
    difference = float(np.max(np.abs(parent[:3] - historical[:, :, :HORIZON])))
    if difference > 1e-06:
        raise ValueError(f'Historical parent prefix changed: {difference}')
    registry.to_csv(output / 'registry.csv', index=False)
    pd.DataFrame(eligibility).to_csv(output / 'weather_eligibility.csv', index=False)
    np.save(output / 'parent_cdf.npy', parent, allow_pickle=False)
    save(output / 'prefix_verification.json', dict(status='passed', horizon_days=HORIZON, source_horizon_days=330, historical_max_abs_difference=difference, cycles=len(registry), heading_labels_used=False, later_year_performance_scored=False))
    check_sources()
    return (registry, weather, parent)
