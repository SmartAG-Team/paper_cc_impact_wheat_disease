"""Propagate disjoint historical RF sowing priors into crop-development inputs."""
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
from .sowing_rf import fit_sowing_rf, calendar_day, FEATURES
from .sowing_distribution import conditional_sowing_prior
from .phenology_transfer import development_increments, summarize_development
from .phenology_transfer_data import distances

PHENO = ROOT/'data/disease_improvement/phenology_transfer_20261022'
SOW_TRAIN = ROOT/'data/disease_improvement/sowing_rf_source_20261022/sowing_training_table.csv'
SOW_FIELDS = ROOT/'outputs/disease_improvement/sowing_rf_20261022/field_estimated_sowing.csv'
EXTENSION = ROOT/'data/disease_improvement/sowing_weather_extension_20261022/daily_weather_extension.parquet'


def _calendar_date(day, season):
    day = int(np.clip(np.rint(day), 1, 365))
    canonical = pd.Timestamp('2001-01-01')+pd.Timedelta(days=day-1)
    return pd.Timestamp(int(season)-1, canonical.month, canonical.day)


def prior_weather_coverage(weather, starts, query_dates):
    """Check every prior start through each query's preceding day separately."""
    query_dates = pd.DatetimeIndex(pd.to_datetime(query_dates))
    if not len(query_dates) or query_dates.max() <= min(starts):
        return np.zeros(len(query_dates), dtype=bool)
    dates = pd.date_range(min(starts), query_dates.max()-pd.Timedelta(days=1))
    values = weather.reindex(dates)[['tmean_c','tmax_c','tmin_c','precipitation_mm']].to_numpy(float)
    missing = np.cumsum(~np.isfinite(values).all(axis=1))
    position = np.asarray((query_dates-min(starts)).days)-1
    result = np.zeros(len(query_dates), dtype=bool)
    valid = (position >= 0) & (position < len(missing)) & (query_dates > max(starts))
    result[valid] = missing[position[valid]] == 0
    return result


def build(output=None):
    out = Path(output or ROOT/'data/disease_improvement/sowing_distribution_v2_20261022').resolve()
    if out.exists():
        raise FileExistsError('Fresh sowing-distribution destination required')
    out.mkdir(parents=True)
    sources = [SOW_TRAIN, SOW_FIELDS, EXTENSION, DATA/'fields.csv', DATA/'daily_weather.parquet',
        PARENT/'training_observation_ledger.csv', *[PHENO/name for name in [
            'registration.json', 'verification.json', 'development_prior.joblib', 'predictors.npz',
            'feature_schema.json', 'trait_support_interpretation_amendment.json']]]
    codes = [Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'sowing_distribution', 'sowing_rf', 'phenology_transfer', 'phenology_transfer_data']]
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        sowing_fit='Original qualified sowingtable eligibility, observed sowing through2012, every documented pre2013stationcoordinate >50km fromeverydiseasecohortlocation',
        sowing_model='RF400trees,maxdepth12,minleaf20, exact previous-calendar annualmeanT+latitude; equalstationbootstrapmassinexpectation under sklearn1.9.1; seed20261022',
        empirical_prior='Equal mass perforesttree; alladmittedobservations assignedtoqueryleaf weightedbyoriginalscientificstationmass insideleaf; bootstrapmultiplicity not reused',
        nodes='Nine equalprobability quantilemidpoints ofempiricalconditional sowingprior; equalnodeweights; dates mappedfromnonleap2001 toautumnprecedingharvest',
        point_control='Single date atconditionalempiricalmean, notmeanofbootstrap-treepredictivevalues; both estimators saved separately',
        coverage='AtEACHquery,bothdesignsfallbacktooriginalsowingscenario if anypriornode OR pointmean lackscompleteweatherthroughthatquerydayminus1; lateravailabilityneverchangesearlierdecisions; no node truncation/imputation/targetdeletion',
        weather='Originalqualifiedstation-preferredweather plus previouslyqualified17dayextension; nonewdatafillorweatherselection',
        transformation='Integrate stagefeaturemeans andstandarddeviations acrosssowingnodes, not a calibrated full diseasepredictivedistribution',
        features='45mean development summaries excluding7daychanges,12stagepassageSDs,2traitgeography,6sowing/support/coverage=65',
        interpretation='Empiricalconditionalprior, not observed sowing or calibrated predictioninterval; DWDannualT→ERA5queryproducttransfer remains recorded',
        disease_targets_unchanged=True, independent_disease_validation=False)
    (out/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
    for path in codes:
        destination = out/'code_snapshot'/path.relative_to(ROOT)
        destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(path, destination)
    started = time.monotonic()
    fields = pd.read_csv(DATA/'fields.csv')
    all_training = pd.read_csv(SOW_TRAIN)
    historical = all_training.loc[pd.to_datetime(all_training.sowing_date).lt('2013-01-01')]
    coordinates = historical[['station_id', 'lat', 'lon']].dropna().drop_duplicates()
    coordinates['minimum_disease_distance_km'] = distances(coordinates[['lat', 'lon']].to_numpy(),
        fields[['latitude', 'longitude']].drop_duplicates().to_numpy()).min(axis=1)
    station_eligible = coordinates.groupby('station_id').minimum_disease_distance_km.min().gt(50.)
    admitted = set(station_eligible.index[station_eligible].astype(int))
    coordinates['admitted'] = coordinates.station_id.isin(admitted)
    coordinates.to_csv(out/'sowing_station_qualification.csv', index=False)
    training = all_training.loc[all_training.rf_common_training_cutoff_eligible.eq(True) &
        all_training.station_id.isin(admitted)].rename(columns={'lat':'latitude'}).copy()
    assert len(training) > 100 and not training.duplicated(['station_id', 'sowing_date']).any()
    assert pd.to_datetime(training.sowing_date).max() < pd.Timestamp('2013-01-01')
    assert np.isfinite(training[FEATURES].to_numpy(float)).all()
    assert training.annual_temperature_year.eq(pd.to_datetime(training.sowing_date).dt.year-1).all()
    training.to_csv(out/'admitted_sowing_training.csv', index=False)
    print('fit sowing rows', len(training), 'stations', training.station_id.nunique(), flush=True)
    forest = fit_sowing_rf(training)
    assert forest['training_rows'] == len(training)
    joblib.dump(forest, out/'fitted_sowing_rf.joblib')
    # Read only the earlier annual-temperature predictors from the old field
    # table; its previously fitted dates play no role in this new estimator.
    field_temperature = pd.read_csv(SOW_FIELDS)
    query_names = ['field_id','season_year','annual_mean_temperature_c','annual_temperature_year','latitude',
        'annual_temperature_source','temperature_product_transfer']
    query = fields[['field_id']].merge(field_temperature[query_names], on='field_id', validate='one_to_one')
    assert query.field_id.tolist() == fields.field_id.tolist()
    assert query.annual_temperature_year.eq(query.season_year-2).all()
    x_training, x_query = training[FEATURES].to_numpy(float), query[FEATURES].to_numpy(float)
    weight = 1/training.groupby('station_id').station_id.transform('size').to_numpy(float)
    prior = conditional_sowing_prior(forest['estimator'], x_training, calendar_day(training.sowing_date), weight, x_query)
    prior['field_ids'] = query.field_id.tolist()
    prior['training_station_ids'] = training.station_id.astype(int).to_numpy()
    prior['training_sowing_dates'] = training.sowing_date.astype(str).to_numpy()
    prior['training_weight'] = weight
    joblib.dump(prior, out/'conditional_sowing_prior.joblib')
    query['rf_tree_mean_day'] = forest['estimator'].predict(x_query)
    query['conditional_empirical_mean_day'] = prior['mean']
    query['conditional_empirical_sd_days'] = prior['standard_deviation']
    query['conditional_effective_records'] = prior['effective_records']
    query['outside_temperature_support'] = (x_query[:,0] < forest['lower'][0]) | (x_query[:,0] > forest['upper'][0])
    query['outside_latitude_support'] = (x_query[:,1] < forest['lower'][1]) | (x_query[:,1] > forest['upper'][1])
    query.to_csv(out/'field_sowing_prior.csv', index=False)
    node_rows = []
    for i, row in query.iterrows():
        for k, day in enumerate(prior['days'][i]):
            node_rows.append(dict(field_id=row.field_id, node=k, quantile=float(prior['quantiles'][k]),
                weight=1/9, sowing_day_nonleap=float(day), estimated_sowing_date=str(_calendar_date(day, row.season_year).date())))
    pd.DataFrame(node_rows).to_csv(out/'sowing_prior_nodes.csv', index=False)
    development = joblib.load(PHENO/'development_prior.joblib')
    with np.load(PHENO/'predictors.npz') as arrays:
        original_features, original_probability = arrays['phenology'], arrays['stage_probability']
        original_target_ids = arrays['target_ids']
    obs = pd.read_csv(PARENT/'training_observation_ledger.csv', float_precision='round_trip')
    np.testing.assert_array_equal(original_target_ids, obs.global_target_index)
    schema = json.loads((PHENO/'feature_schema.json').read_text())
    original_names = schema['feature_names'][:57]
    selected = [i for i, name in enumerate(original_names) if '_probability_change_7d' not in name]
    passage = [i for i, name in enumerate(original_names) if name.endswith('_passed_probability')]
    assert len(selected) == 45 and len(passage) == 12
    names = ['mean_'+original_names[i] for i in selected]+['sd_'+original_names[i] for i in passage]
    names += schema['feature_names'][57:]+['sowing_prior_sd_30d', 'sowing_prior_effective_records_1000',
        'sowing_prior_weather_complete', 'sowing_temperature_outside_support', 'sowing_latitude_outside_support',
        'sowing_prior_mean_nonleap_day_365']
    assert len(names) == 65
    point, integrated = np.zeros((len(obs), 65)), np.zeros((len(obs), 65))
    point_probability, integrated_probability = np.zeros((len(obs),3,4)), np.zeros((len(obs),3,4))
    weather = pd.read_parquet(DATA/'daily_weather.parquet')
    extension = pd.read_parquet(EXTENSION)
    weather['date'], extension['date'] = pd.to_datetime(weather.date), pd.to_datetime(extension.date)
    existing = pd.MultiIndex.from_frame(weather[['location_id','date']])
    extension = extension.loc[~pd.MultiIndex.from_frame(extension[['location_id','date']]).isin(existing)]
    weather = pd.concat([weather, extension], ignore_index=True)
    assert not weather.duplicated(['location_id','date']).any()
    by_site = {key: group.set_index('date').sort_index() for key, group in weather.groupby('location_id')}
    field_metadata, sowing_metadata = fields.set_index('field_id'), query.set_index('field_id')
    field_position = pd.Series(np.arange(len(query)), index=query.field_id)
    coverage_rows, provenance = [], []
    for field_id, group in obs.groupby('field_id', sort=False):
        field, sowing = field_metadata.loc[field_id], sowing_metadata.loc[field_id]
        fp = int(field_position.loc[field_id]); ix = group.index.to_numpy()
        point_start = _calendar_date(prior['mean'][fp], field.season_year)
        starts = [_calendar_date(day, field.season_year) for day in prior['days'][fp]]
        first = min([point_start, *starts])
        covered = prior_weather_coverage(by_site[field.site_id], [point_start, *starts], group.date)
        pm = original_features[ix, :57].copy(); mean = pm.copy(); sd = np.zeros_like(mean)
        pp = original_probability[ix].copy(); ip = pp.copy()
        if covered.any():
            active = np.flatnonzero(covered)
            active_queries = pd.to_datetime(group.date.iloc[active])
            end = active_queries.max()-pd.Timedelta(days=1)
            daily = by_site[field.site_id].reindex(pd.date_range(first, end))
            cache = {}
            for start in set([point_start, *starts]):
                dates = pd.date_range(start, end)
                forcing = daily.loc[dates, ['tmean_c','tmax_c','tmin_c','precipitation_mm']].to_numpy(float)
                inc = development_increments(forcing, dates, field.latitude, development['parameters'])
                full, _, probability = summarize_development(np.cumsum(inc, axis=0), inc, development['distributions'])
                days = (active_queries-pd.Timedelta(days=1)-start).dt.days.to_numpy()
                assert np.all(days >= 0)
                cache[start] = (full[days], probability[days])
            pm[active], pp[active] = cache[point_start]
            samples = np.stack([cache[start][0] for start in starts])
            mean[active], sd[active] = samples.mean(axis=0), samples.std(axis=0)
            ip[active] = np.mean([cache[start][1] for start in starts], axis=0)
        common = np.column_stack([original_features[ix, 57:],
            np.full(len(ix), prior['standard_deviation'][fp]/30), np.full(len(ix), prior['effective_records'][fp]/1000),
            covered.astype(float), np.full(len(ix), int(sowing.outside_temperature_support)),
            np.full(len(ix), int(sowing.outside_latitude_support)), np.full(len(ix), prior['mean'][fp]/365)])
        point[ix] = np.column_stack([pm[:, selected], np.zeros((len(ix),12)), common])
        integrated[ix] = np.column_stack([mean[:, selected], sd[:, passage], common])
        point_probability[ix], integrated_probability[ix] = pp, ip
        coverage_rows.append(dict(field_id=field_id, complete_all_nodes_and_point_weather=bool(covered.all()),
            earliest_prior_sowing=str(first.date()), point_sowing=str(point_start.date()),
            fixed_scenario_sowing=str(pd.Timestamp(field.sowing_date).date()), targets=len(ix),covered_queries=int(covered.sum())))
        for local, row in enumerate(group.itertuples()):
            provenance.append(dict(global_target_index=int(row.global_target_index), field_id=field_id,
                weather_end=str((pd.Timestamp(row.date)-pd.Timedelta(days=1)).date()),
                prior_weather_complete=bool(covered[local]), actual_sowing_observed=False))
    np.savez_compressed(out/'predictors.npz', point=point, integrated=integrated,
        point_stage_probability=point_probability, integrated_stage_probability=integrated_probability,
        target_ids=original_target_ids)
    pd.DataFrame(coverage_rows).to_csv(out/'weather_coverage.csv', index=False)
    pd.DataFrame(provenance).to_csv(out/'query_provenance.csv', index=False)
    (out/'feature_schema.json').write_text(json.dumps(dict(feature_names=names,
        stages=schema['stages'], models=schema['models'], units='Explicit in each variable name',
        variance_interpretation='Prior spread in development features, not new measurement precision or calibrated disease prediction uncertainty'), indent=2)+'\n')
    assert np.isfinite(point).all() and np.isfinite(integrated).all()
    assert np.all(np.diff(integrated_probability, axis=2) <= 1e-12)
    summary = dict(passed=True, training_rows=len(training), training_stations=training.station_id.nunique(),
        latest_training_sowing_date=forest['maximum_training_date'],
        minimum_documented_source_distance_km=float(coordinates.loc[coordinates.admitted,'minimum_disease_distance_km'].min()),
        fields_with_complete_prior_weather=sum(r['complete_all_nodes_and_point_weather'] for r in coverage_rows),
        cohort_rows=len(obs), features_per_variant=65, point_and_prior_coverage_matched=True,
        disease_targets_unchanged=True, source_unchanged=all(sha(Path(p)) == h for p,h in registration['source_sha256'].items()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}, elapsed_seconds=time.monotonic()-started)
    (out/'verification.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'artifacts_sha256'}, indent=2), flush=True)
    return out


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    build(args.output)
