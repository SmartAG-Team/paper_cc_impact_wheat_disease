"""Build a label-free bank of shared-equation disease process trajectories."""
from dataclasses import replace
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd

from model.seasonal_septoria.overwinter import OverwinterParameters
from model.seasonal_septoria.leaf_phenology import leaf_host
from model.seasonal_septoria.wetness import duration_exposure
from .autonomous_weather import WEATHER
from .autonomous_weather_data import AUTONOMOUS, PHENO, EXTENSION
from .phenology_transfer import development_increments
from .process_establishment import simulate_establishment
from .innovation_common import ROOT, DATA, PARENT
from .inputs import sha

PROCESS_DATA = ROOT/'data/disease_improvement/process_establishment_20261022'
PHYSICS = list(itertools.product([.001, .01, .1], [.1, 1., 10.], [10., 20., 30.]))
INTRODUCTION = [0., 2., 6.]


def _forcing(weather, fields, queries, prior):
    """Assemble only four-weather and fixed public metadata; targets ignored."""
    queries = queries[['global_target_index', 'field_id', 'date']].copy()
    queries['date'] = pd.to_datetime(queries.date)
    fields = fields[['field_id', 'site_id', 'latitude', 'estimated_sowing_date']].copy()
    fields['estimated_sowing_date'] = pd.to_datetime(fields.estimated_sowing_date)
    weather = weather[['location_id', 'date', *WEATHER]].copy()
    weather['date'] = pd.to_datetime(weather.date)
    if weather.duplicated(['location_id', 'date']).any() or fields.field_id.duplicated().any():
        raise ValueError('Unique physical weather and field identities required')
    sites = {key: group.set_index('date') for key, group in weather.groupby('location_id')}
    fields = fields.set_index('field_id').loc[queries.field_id.drop_duplicates()].reset_index()
    length = [(queries.loc[queries.field_id.eq(row.field_id), 'date'].max()-row.estimated_sowing_date).days
        for row in fields.itertuples()]
    if min(length) < 1:
        raise ValueError('Every query must follow estimated sowing')
    shape = (len(fields), max(length))
    temperature, maximum, humidity, rain, accumulation = [np.zeros(shape) for _ in range(5)]
    mask = np.zeros(shape, bool); index_field = np.zeros(len(queries), int); index_day = np.zeros(len(queries), int)
    stage = {key: float(prior['distributions']['tpv'][key]['values'][np.searchsorted(
        prior['distributions']['tpv'][key]['cumulative_weight'], .5)]) for key in [10, 31, 51, 87]}
    span = stage[51]-stage[31]
    thresholds = {10: stage[10], 31: stage[31], 37: stage[31]+.35*span,
        39: stage[31]+.60*span, 51: stage[51], 85: stage[87]}
    provenance = []
    for i, field in enumerate(fields.itertuples()):
        start = field.estimated_sowing_date
        dates = pd.date_range(start, periods=length[i])
        values = sites[field.site_id].reindex(dates)[WEATHER].to_numpy(float)
        if (not np.isfinite(values).all() or (values[:, 1] < values[:, 0]-.001).any()
                or ((values[:, 2] < 0)|(values[:, 2] > 100)).any() or (values[:, 3] < 0).any()
                or (values[:, 0] > 40).any()):
            raise ValueError('Complete supported four-variable sowing-to-query weather required')
        development = development_increments(values[:, :2], dates, field.latitude, prior['parameters'])[:, 2]
        acc = np.cumsum(development); season = np.ones(len(dates), bool)
        terminal = np.flatnonzero(acc >= stage[87])
        if len(terminal):
            season[terminal[0]+1:] = False
        mask[i, :len(dates)] = season
        for matrix, source in zip([temperature, maximum, humidity, rain], values.T):
            matrix[i, :len(dates)] = np.where(season, source, 0.)
        accumulation[i, :len(dates)] = acc
        accumulation[i, len(dates):] = acc[-1]
        use = np.flatnonzero(queries.field_id.eq(field.field_id))
        index_field[use] = i
        index_day[use] = (queries.iloc[use].date-start).dt.days.to_numpy()
        for row in queries.iloc[use].itertuples():
            provenance.append(dict(global_target_index=int(row.global_target_index), field_id=field.field_id,
                sowing_date=str(start.date()), query_date=str(row.date.date()),
                weather_end=str((row.date-pd.Timedelta(days=1)).date()), actual_sowing_used=False,
                observed_bbch_used=False, disease_history_used=False, satellite_used=False))
    host = leaf_host(accumulation, temperature, thresholds, rank_spacing_units=120., forcing_mask=mask,
        juvenile_policy='handover_31_39')
    exposure = duration_exposure(temperature, maximum, humidity, rain, .6)['exposure']
    return dict(temperature=temperature, exposure=exposure, rain=rain, active=host.active, renewal=host.renewal,
        area=host.area, valid=mask, field_index=index_field, day_index=index_day, host_thresholds=thresholds,
        provenance=pd.DataFrame(provenance), field_ids=fields.field_id.tolist())


def build():
    out = PROCESS_DATA
    if out.exists():
        raise FileExistsError('Fresh process-bank destination required')
    sources = [DATA/'daily_weather.parquet', EXTENSION, AUTONOMOUS/'field_sowing_scenarios.csv',
        AUTONOMOUS/'query_provenance.csv', AUTONOMOUS/'registration.json', AUTONOMOUS/'verification.json',
        PHENO/'development_prior.joblib', PHENO/'registration.json', PARENT/'training_observation_ledger.csv']
    codes = [Path(__file__)] + [Path(__file__).with_name(name+'.py') for name in ['process_establishment',
        'phenology_transfer', 'autonomous_weather', 'autonomous_weather_data']]
    codes += [ROOT/'model/seasonal_septoria'/f'{name}.py' for name in ['overwinter', 'leaf_phenology', 'wetness']]
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        physics=PHYSICS, introduction_thresholds=INTRODUCTION, labels_used=False,
        host='Disjointpre2013DWDTPV10/31/51/87 medians;37/39clockfractions.35/.60 assumed, notmeasured;internal85aliasofdonor87',
        information='QuerydayminusoneTmean/Tmax/RH/totalprecip;qualifiedRFpoint sowing; noobservedstage/history/satellite',
        parameter_scope='Unfitted generic bank; outer-fold fitting/selecting follows after this label-free artifact',
        source_sha256={str(p): sha(p) for p in sources}, code_sha256={str(p.relative_to(ROOT)): sha(p) for p in codes})
    for parent in [AUTONOMOUS, PHENO]:
        registration['source_sha256'].update({path: digest for path, digest in json.loads((parent/'registration.json').read_text())['source_sha256'].items() if not path.endswith('.md')})
    out.mkdir(parents=True)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for source in codes:
        dest = out/'code_snapshot'/source.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    started = time.monotonic()
    query = pd.read_csv(PARENT/'training_observation_ledger.csv')[['global_target_index', 'field_id', 'date']]
    fields = pd.read_csv(AUTONOMOUS/'field_sowing_scenarios.csv')
    assert not pd.read_csv(AUTONOMOUS/'query_provenance.csv').sowing_fallback.any()
    weather = pd.read_parquet(DATA/'daily_weather.parquet', columns=['location_id', 'date', *WEATHER])
    extra = pd.read_parquet(EXTENSION, columns=['location_id', 'date', *WEATHER])
    weather.date, extra.date = pd.to_datetime(weather.date), pd.to_datetime(extra.date)
    existing = pd.MultiIndex.from_frame(weather[['location_id', 'date']])
    extra = extra.loc[~pd.MultiIndex.from_frame(extra[['location_id', 'date']]).isin(existing)]
    forcing = _forcing(pd.concat([weather, extra], ignore_index=True), fields, query, joblib.load(PHENO/'development_prior.joblib'))
    query_states = np.zeros((len(PHYSICS), len(query), 3, 8))
    largest_mass_error, smallest_state = 0., 1.
    arguments = [forcing[k] for k in ['temperature', 'exposure', 'rain', 'active', 'renewal', 'area', 'valid']]
    for j, (alpha, beta, latency) in enumerate(PHYSICS):
        parameters = replace(OverwinterParameters(), primary_scale=alpha, secondary_scale=beta, latent_reference_days=latency)
        for k, threshold in enumerate(INTRODUCTION):
            state = simulate_establishment(*arguments, parameters, threshold)
            largest_mass_error = max(largest_mass_error, float(np.max(np.abs(state.sum(-1)-1.))))
            smallest_state = min(smallest_state, float(state.min()))
            query_states[j, :, k] = state[forcing['field_index'], forcing['day_index'], :, -3:].sum(axis=-1)
        print('physics', j+1, 'of27', 'seconds', round(time.monotonic()-started, 1), flush=True)
    assert largest_mass_error < 1e-10 and smallest_state >= -1e-12
    np.savez_compressed(out/'query_states.npz', damage=query_states, target_ids=query.global_target_index.to_numpy(int),
        physics=np.asarray(PHYSICS), introduction_thresholds=np.asarray(INTRODUCTION))
    np.savez_compressed(out/'forcing.npz', **{k: forcing[k] for k in ['temperature', 'exposure', 'rain',
        'active', 'renewal', 'area', 'valid', 'field_index', 'day_index']})
    forcing['provenance'].to_csv(out/'query_provenance.csv', index=False)
    (out/'host_scenario.json').write_text(json.dumps(dict(thresholds=forcing['host_thresholds'],
        field_ids=forcing['field_ids'], internal_terminal85_means='Donor observedBBCH87; not observedsoftdough85',
        flag_visibility_and_unfolding='Assumed clock fractions; not individually stage-calibrated',
        severity_denominator='Normalized modeled tissue proxy; no healthy-area/yield qualification'), indent=2)+'\n')
    verification = dict(passed=True, shape=list(query_states.shape), label_free=True,
        largest_mass_error=largest_mass_error, smallest_state=smallest_state,
        source_unchanged=all(sha(Path(path)) == digest for path, digest in registration['source_sha256'].items()),
        artifacts_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()}, seconds=time.monotonic()-started)
    assert verification['source_unchanged']
    (out/'verification.json').write_text(json.dumps(verification, indent=2)+'\n')
    print(json.dumps({k: v for k, v in verification.items() if k != 'artifacts_sha256'}, indent=2), flush=True)
    return out


if __name__ == '__main__':
    build()
