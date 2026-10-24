"""Fit spatially and chronologically disjoint trait priors and freeze disease inputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from .innovation_common import ROOT, DATA, PARENT
from .inputs import sha
from .phenology_transfer import fit_development_prior, development_increments, summarize_development, STAGES, MODELS

EJA = ROOT.parent/'EJA-PGN-Wheat'
TRAIT = EJA/'data/dataset/external/dwd_independent_prepared_20261006'


def distances(a, b):
    a, b = np.radians(a), np.radians(b)
    value = (np.sin((a[:, None, 0]-b[None, :, 0])/2)**2+
        np.cos(a[:, None, 0])*np.cos(b[None, :, 0])*np.sin((a[:, None, 1]-b[None, :, 1])/2)**2)
    return 6371*2*np.arcsin(np.sqrt(np.clip(value, 0, 1)))


def build(output=None):
    out = Path(output or ROOT/'data/disease_improvement/phenology_transfer_20261022').resolve()
    if out.exists():
        raise FileExistsError('Fresh trait-transfer destination required')
    out.mkdir(parents=True)
    sources = [TRAIT/name for name in ['records.json', 'daily_inputs.npz', 'manifest.json', 'protocol.json']]
    sources += [EJA/'code/research_validation/prepare.py', DATA/'fields.csv', DATA/'daily_weather.parquet',
        PARENT/'training_observation_ledger.csv']
    codes = [Path(__file__), Path(__file__).with_name('phenology_transfer.py')]
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p): sha(p) for p in sources}, code_sha256={str(p.relative_to(ROOT)): sha(p) for p in codes},
        trait_dates='Complete cycles ending before2013, no later-stage targets or preprocessing',
        trait_space='Every admitted station location morethan50km from EVERY diseasecohort location, notonlycurrentfold',
        trait_weighting='Equal station mass perstage, divided among its observed historical cycles',
        algorithms='Reproduce EJA thermal/photoperiod/vernalization increments, refit allfour gates and eventthreshold distributions from admitted raw weather',
        prohibited_reuse='Prepared process array and checkpoints use later phenotype data and are not read or reused',
        features='57developmentprobability/relativeclock/entropy/7and21daychange/dailyincrement plusnearesttraitdistance/latitudeextrapolation',
        information='Queryday-minus-one weather only, current retained estimated sowing scenario; no observed disease or stage labels used in transformation',
        interpretation='Crop-development priors, not observed cultivar, leaf anatomy or EPPO leaf-scoring qualification',
        current_disease_targets_unchanged=True, independent_disease_validation=False)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in codes:
        destination = out/'code_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
    shutil.copyfile(EJA/'code/research_validation/prepare.py', out/'eja_prepare_source_snapshot.py')
    started = time.monotonic()
    source_manifest = json.loads((TRAIT/'manifest.json').read_text())
    for name in ['records.json', 'daily_inputs.npz']:
        assert sha(TRAIT/name) == source_manifest['artifacts'][name]
    original_records = json.loads((TRAIT/'records.json').read_text())
    historical = [r for r in original_records if r['END_DATE'] < '2013-01-01' and
        r['eligible'] and not r['observation_order_conflict']]
    fields = pd.read_csv(DATA/'fields.csv')
    disease_coordinates = fields[['latitude', 'longitude']].drop_duplicates().to_numpy(float)
    trait_coordinates = pd.DataFrame([dict(station=r['PEP_ID'], latitude=r['lat'], longitude=r['lon'])
        for r in historical]).drop_duplicates()
    matrix = distances(trait_coordinates[['latitude', 'longitude']].to_numpy(), disease_coordinates)
    trait_coordinates['minimum_disease_distance_km'] = matrix.min(axis=1)
    # A station with any historical location inside the buffer is excluded in full.
    eligible_stations = trait_coordinates.groupby('station').minimum_disease_distance_km.min().gt(50.)
    admitted = set(eligible_stations.index[eligible_stations].astype(int))
    records = [r for r in historical if r['PEP_ID'] in admitted]
    trait_coordinates['admitted'] = trait_coordinates.station.isin(admitted)
    trait_coordinates.to_csv(out/'trait_station_qualification.csv', index=False)
    assert len(records) > 100 and all(r['END_DATE'] < '2013-01-01' for r in records)
    assert trait_coordinates.loc[trait_coordinates.admitted, 'minimum_disease_distance_km'].min() > 50.
    (out/'admitted_trait_cycles.json').write_text(json.dumps(records, indent=2)+'\n')
    print('fit trait cycles', len(records), 'stations', len(admitted), flush=True)
    with np.load(TRAIT/'daily_inputs.npz', allow_pickle=False) as arrays:
        weather = arrays['weather']  # Deliberately do not read the source's calibrated process array.
    prior = fit_development_prior(records, weather)
    prior['trait_coordinates'] = trait_coordinates.loc[trait_coordinates.admitted].copy()
    joblib.dump(prior, out/'development_prior.joblib')
    print('trait fit seconds', round(time.monotonic()-started, 1), 'gates', prior['parameters'], flush=True)
    summary = dict(trait_cycles=len(records), trait_stations=len(admitted),
        minimum_disease_distance_km=float(trait_coordinates.loc[trait_coordinates.admitted, 'minimum_disease_distance_km'].min()),
        maximum_trait_date=prior['maximum_trait_date'], parameters=prior['parameters'],
        event_count={str(stage): len(prior['distributions']['gdd'][stage]['values']) for stage in STAGES},
        median_thresholds={model: {str(stage): float(d['values'][np.searchsorted(d['cumulative_weight'], .5)])
            for stage, d in stages.items()} for model, stages in prior['distributions'].items()})
    (out/'calibration_summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    obs = pd.read_csv(PARENT/'training_observation_ledger.csv', float_precision='round_trip')
    daily = pd.read_parquet(DATA/'daily_weather.parquet')
    daily['date'] = pd.to_datetime(daily.date).dt.normalize()
    weather_by_site = {key: group.set_index('date').sort_index() for key, group in daily.groupby('location_id')}
    field_by_id = fields.set_index('field_id')
    features = np.zeros((len(obs), 59)); probabilities = np.zeros((len(obs), 3, 4)); provenance = []
    for field_id, group in obs.groupby('field_id', sort=False):
        field = field_by_id.loc[field_id]
        start, end = pd.Timestamp(field.sowing_date), pd.to_datetime(group.date).max()-pd.Timedelta(days=1)
        dates = pd.date_range(start, end)
        forcing = weather_by_site[field.site_id].reindex(dates)
        values = forcing[['tmean_c', 'tmax_c', 'tmin_c', 'precipitation_mm']].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError('Complete retained causal weather required')
        inc = development_increments(values, dates, field.latitude, prior['parameters'])
        full, names, stage_prob = summarize_development(np.cumsum(inc, axis=0), inc, prior['distributions'])
        near = float(distances(np.array([[field.latitude, field.longitude]]),
            prior['trait_coordinates'][['latitude', 'longitude']].to_numpy()).min())
        outside = float(not prior['trait_coordinates'].latitude.min() <= field.latitude <= prior['trait_coordinates'].latitude.max())
        for row in group.itertuples():
            query = pd.Timestamp(row.date)
            day = (query-pd.Timedelta(days=1)-start).days
            if not 0 <= day < len(full):
                raise ValueError('Query must follow the declared sowing scenario')
            features[row.Index] = np.r_[full[day], near/100., outside]
            probabilities[row.Index] = stage_prob[day]
            provenance.append(dict(global_target_index=int(row.global_target_index),
                field_id=field_id, query_date=str(query.date()), weather_start=str(start.date()),
                weather_end=str((query-pd.Timedelta(days=1)).date()), nearest_trait_station_km=near,
                observed_sowing_used=False, estimated_sowing_date=str(start.date())))
    names += ['phenology_nearest_trait_station_distance_100km', 'phenology_latitude_outside_training_support']
    np.savez_compressed(out/'predictors.npz', phenology=features, stage_probability=probabilities,
        target_ids=obs.global_target_index.to_numpy(int))
    pd.DataFrame(provenance).to_csv(out/'query_provenance.csv', index=False)
    (out/'feature_schema.json').write_text(json.dumps(dict(feature_names=names,
        stages=STAGES, models=MODELS, units='Probability, index, entropy, capped relative accumulation, scaled daily increment, distance/100km and indicator; specified in each variable name',
        gate='T-P-V stage probabilities define fourphaseweights:1-P31,P31-P51,P51-P87,P87'), indent=2)+'\n')
    assert features.shape == (2304, 59) and np.isfinite(features).all()
    assert np.all(np.diff(probabilities, axis=2) <= 1e-12)
    verification = dict(passed=True, trait_fit_chronology_before2013=True,
        all_trait_stations_beyond50km=True, disease_targets_unchanged=True,
        features_shape=list(features.shape), source_unchanged=all(sha(Path(p)) == h for p,h in registration['source_sha256'].items()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}, elapsed_seconds=time.monotonic()-started)
    (out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print('complete', summary, 'features', features.shape, 'seconds', round(time.monotonic()-started, 1), flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    build(args.output)
