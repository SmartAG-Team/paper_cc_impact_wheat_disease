"""Register and rebuild four-weather-field autonomous disease inputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from .autonomous_weather import build_autonomous_features, WEATHER
from .innovation_common import ROOT, DATA, PARENT
from .inputs import sha
from .sowing_distribution_data import _calendar_date

PHENO = ROOT/'data/disease_improvement/phenology_transfer_20261022'
SOWING = ROOT/'data/disease_improvement/sowing_distribution_v2_20261022'
EXTENSION = ROOT/'data/disease_improvement/sowing_weather_extension_20261022/daily_weather_extension.parquet'
AUTONOMOUS = ROOT/'data/disease_improvement/autonomous_weather_20261022'


def build(output=None):
    out = Path(output or AUTONOMOUS).resolve()
    if out.exists():
        raise FileExistsError('Fresh autonomous data destination required')
    sources = [DATA/'daily_weather.parquet', DATA/'fields.csv', PARENT/'training_observation_ledger.csv',
        EXTENSION, SOWING/'field_sowing_prior.csv', SOWING/'conditional_sowing_prior.joblib',
        SOWING/'fitted_sowing_rf.joblib', SOWING/'admitted_sowing_training.csv', SOWING/'registration.json',
        SOWING/'verification.json', PHENO/'development_prior.joblib', PHENO/'registration.json',
        PHENO/'verification.json', PHENO/'admitted_trait_cycles.json', PHENO/'trait_station_qualification.csv',
        ROOT/'analysis/disease_improvement/development_constraints.json']
    codes = [Path(__file__), Path(__file__).with_name('autonomous_weather.py'),
        Path(__file__).with_name('phenology_phase_growth.py'),
        Path(__file__).with_name('phenology_transfer.py'), Path(__file__).with_name('phenology_transfer_data.py'),
        Path(__file__).with_name('sowing_distribution_data.py'), ROOT/'model/seasonal_septoria/core.py',
        ROOT/'model/seasonal_septoria/wetness.py']
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p): sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)): sha(p) for p in codes},
        runtime_weather=WEATHER, satellite_used=False, history_used=False, observed_bbch_used=False,
        sowing='Existing disjoint pre2013 RF conditional mean: previous-calendar annualmeanT and latitude; declared calendar fallback only for incomplete four-variable prefix',
        weather='Station-preferred total precipitation and qualified retained air variables; no liquid-phase rain or measured subdaily quantities',
        development='Historical qualified DWD prior; only Tmean/Tmax enter development increments; no later calibrated phenotype arrays',
        disease_targets_unchanged=True, adaptive_internal_development=True,
        endpoint='Source-native nominal code, not an anatomical F1/F2/F3 mapping',
        climate_integration_status='Not integrated; previous annual mean temperature requires the extra preceding year in scenario forcing')
    out.mkdir(parents=True)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in codes:
        dest = out/'code_snapshot'/path.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, dest)
    started = time.monotonic()
    for root in [PHENO, SOWING]:
        receipt = json.loads((root/'verification.json').read_text())
        assert receipt['passed'] and receipt['source_unchanged']
        for name, digest in receipt['artifacts_sha256'].items():
            assert sha(root/name) == digest, name
    observations = pd.read_csv(PARENT/'training_observation_ledger.csv', float_precision='round_trip')
    fields = pd.read_csv(DATA/'fields.csv')
    sowing = pd.read_csv(SOWING/'field_sowing_prior.csv')
    assert sowing.annual_temperature_year.eq(sowing.season_year-2).all()
    dates = [_calendar_date(day, year) for day, year in zip(sowing.conditional_empirical_mean_day, sowing.season_year)]
    fields = fields.merge(pd.DataFrame(dict(field_id=sowing.field_id, estimated_sowing_date=dates)),
        on='field_id', validate='one_to_one')
    assert fields.estimated_sowing_date.notna().all()
    weather = pd.read_parquet(DATA/'daily_weather.parquet', columns=['location_id', 'date', *WEATHER])
    extension = pd.read_parquet(EXTENSION, columns=['location_id', 'date', *WEATHER])
    weather.date, extension.date = pd.to_datetime(weather.date), pd.to_datetime(extension.date)
    existing = pd.MultiIndex.from_frame(weather[['location_id', 'date']])
    extension = extension.loc[~pd.MultiIndex.from_frame(extension[['location_id', 'date']]).isin(existing)]
    weather = pd.concat([weather, extension], ignore_index=True)
    prior = joblib.load(PHENO/'development_prior.joblib')
    values = build_autonomous_features(weather, fields, observations, prior)
    assert values['baseline'].shape == (2304, 34) and values['augmented'].shape == (2304, 129)
    np.savez_compressed(out/'predictors.npz', **{key: values[key] for key in ['baseline', 'augmented', 'phase']},
        target_ids=observations.global_target_index.to_numpy(int))
    values['provenance'].to_csv(out/'query_provenance.csv', index=False)
    fields[['field_id', 'site_id', 'latitude', 'longitude', 'season_year', 'sowing_date', 'estimated_sowing_date']].to_csv(
        out/'field_sowing_scenarios.csv', index=False)
    (out/'feature_schema.json').write_text(json.dumps({key: values[key] for key in ['baseline_names', 'augmented_names']}, indent=2)+'\n')
    audit = dict(conclusion='No archived ML predictor qualifies as the matched four-weather autonomous control across all20splits',
        candidate_process='Saved learner72inputs=process37+observed_history35',
        history_blind_first='Zero disease history but nine observed-BBCH columns and Tmin-derived frost counts remain',
        current='Allfive archived temporal choices history RF/HGB; temporalRMSE19.408544408244754; monitoring',
        archive_weather_rf=dict(temporal_rmse=26.3857443, severe_rmse=49.1445819,
            history_conditioned=False, queryday_weather=True, later_trained_phenology=True,
            crop_representation_requires_tmin=True, matching_spatial_fits_available=False,
            exact2017currentassessment_ids_after_excluding_target1036=True,
            current_training_memberships_match=False, saved_models_replay_exact=True),
        climate_kernel='Frozen kernel uses exactly four weather variables and endogenous development; unchanged',
        interpretation='Historical weather RF is context only; monitoring gains cannot establish autonomous climate accuracy')
    (out/'deployment_input_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    verification = dict(passed=True, source_unchanged=all(sha(Path(p)) == h for p, h in registration['source_sha256'].items()),
        baseline_shape=list(values['baseline'].shape), augmented_shape=list(values['augmented'].shape),
        fallback_queries=int(values['provenance'].sowing_fallback.sum()),
        fallback_fields=int(values['provenance'].loc[values['provenance'].sowing_fallback, 'field_id'].nunique()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}, elapsed_seconds=time.monotonic()-started)
    assert verification['source_unchanged']
    (out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps({k: v for k, v in verification.items() if k != 'artifacts_sha256'}, indent=2), flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    build(parser.parse_args().output)
