"""Four-variable weather predictors and autonomous delayed-exposure regressors.

Historical crop-development priors are auxiliary supervision. Delays and generic
host gates are structural representations, not verified native-leaf anatomy.
No disease or observed-BBCH values enter the forecast input interface.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from model.seasonal_septoria.core import Parameters, simulate_season
from model.seasonal_septoria.wetness import duration_exposure
from .phenology_transfer import development_increments, summarize_development
from .phenology_transfer_data import distances
from .phenology_phase_growth import fit_phase_growth, predict_phase_growth, phase_weights

WEATHER = ['tmean_c', 'tmax_c', 'rh_mean_pct', 'precipitation_mm']
ENDPOINTS = [f'{source}:code{leaf}' for source in ['BASF', 'Corteva'] for leaf in range(7)]
OUTPUTS = ['climate_reference', 'climate_renewal_shared', 'climate_renewal_phase']


def delayed_exposure(dose, thermal, stage_probability, end):
    """Express past dose with an assumed three-stage thermal delay kernel.

    Thermal units are days at 18 C. The gamma CDF is a continuous three-stage
    approximation, rather than the discrete epidemic simulator's mean delay.
    State at the query follows all weather through ``end``, with dose arising
    at the end of its own weather day. Only subsequent thermal increments age
    that dose. A 21-calendar-day decay defines a recent-expression descriptor.
    """
    dose = np.asarray(dose, float)[:end+1]
    thermal = np.asarray(thermal, float)[:end+1]
    stage = np.asarray(stage_probability, float)[:end+1]
    if (not len(dose) or dose.shape != thermal.shape or stage.shape != (len(dose), 4)
            or not all(np.isfinite(a).all() for a in [dose, thermal, stage])
            or (dose < 0).any() or (thermal < 0).any() or ((stage < 0) | (stage > 1)).any()):
        raise ValueError('Finite nonnegative dose/thermal and four passage probabilities required')
    accumulation = np.cumsum(thermal)
    age = accumulation[-1]-accumulation
    recent = np.exp(-(len(dose)-1-np.arange(len(dose)))/21.)
    result = []
    for latent in [10., 20., 30.]:
        z = 3.*age/latent
        passed = np.clip(1-np.exp(-z)*(1+z+z*z/2), 0, 1)
        for host in [np.ones(len(dose)), stage[:, 1], stage[:, 2]]:
            weighted = dose*host
            expressed = np.sum(weighted*passed)
            result.extend([np.sum(weighted*(1-passed)), expressed,
                np.sum(weighted*passed*recent), -np.expm1(-.01*expressed)])
    return np.asarray(result)


def _valid(a):
    return (np.isfinite(a).all(axis=1) & (a[:, 1] >= a[:, 0]-.001)
        & (a[:, 0] >= -80.) & (a[:, 1] <= 80.)
        & (a[:, 2] >= 0) & (a[:, 2] <= 100) & (a[:, 3] >= 0))


def build_autonomous_features(weather, fields, queries, prior):
    """Read only the explicit four-field weather and public metadata whitelist.

    Each sowing choice is checked against its query prefix independently.
    Additional input columns, including disease, stage or remote observations,
    have no effect. Invalid future weather cannot reject an earlier query.
    """
    w = weather[['location_id', 'date', *WEATHER]].copy()
    w['date'] = pd.to_datetime(w.date).dt.normalize()
    f = fields[['field_id', 'site_id', 'latitude', 'longitude', 'sowing_date', 'estimated_sowing_date']].copy()
    q = queries[['global_target_index', 'field_id', 'source', 'leaf_index', 'date']].copy()
    q['date'] = pd.to_datetime(q.date).dt.normalize()
    if (w.duplicated(['location_id', 'date']).any() or f.field_id.duplicated().any()
            or q.global_target_index.duplicated().any()):
        raise ValueError('Unique weather, field and target identities required')
    by_site = {key: group.set_index('date').sort_index() for key, group in w.groupby('location_id')}
    by_field = f.set_index('field_id')
    baseline, augmented, phase, provenance, cache = [], [], [], [], {}
    coordinates = prior['trait_coordinates'][['latitude', 'longitude']].to_numpy(float)
    names = [f'weather_{name}_mean_{width}d' for width in [7, 21, 42]
        for name in ['temperature_c', 'maximum_c', 'humidity_pct', 'log1p_precipitation_mm']]
    names += ['calendar_sin', 'calendar_cos', 'elapsed_days_330', 'latitude_90', 'longitude_180']
    names += ENDPOINTS + ['generic_process_damage', 'generic_process_infectious', 'generic_process_latent']
    for row in q.itertuples(index=False):
        field = by_field.loc[row.field_id]
        if not np.isfinite([field.latitude, field.longitude]).all():
            raise ValueError('Finite field coordinates required')
        end = row.date-pd.Timedelta(days=1)
        site = by_site[field.site_id]

        def forcing(start):
            dates = pd.date_range(start, end)
            values = site.reindex(dates)[WEATHER].to_numpy(float)
            return dates, values, bool(len(dates) and _valid(values).all())

        start = pd.Timestamp(field.estimated_sowing_date)
        dates, values, complete = forcing(start)
        fallback = not complete
        if fallback:
            start = pd.Timestamp(field.sowing_date)
            dates, values, complete = forcing(start)
        if not complete:
            raise ValueError('Complete four-variable query weather prefix required')
        key = (row.field_id, str(start.date()), str(end.date()))
        if key not in cache:
            t, tx, rh, rain = values.T
            inc = development_increments(values[:, :2], dates, field.latitude, prior['parameters'])
            dev, dev_names, probabilities = summarize_development(np.cumsum(inc, axis=0), inc, prior['distributions'])
            near = float(distances(np.array([[field.latitude, field.longitude]]), coordinates).min())/100.
            outside = float(not coordinates[:, 0].min() <= field.latitude <= coordinates[:, 0].max())
            phenotype = np.r_[dev[-1], near, outside]
            dose = duration_exposure(t, tx, rh, rain)['exposure']
            kernels = delayed_exposure(dose, np.maximum(t, 0)/18., probabilities[:, 2], len(t)-1)
            active = np.ones((1, len(t), 1), bool)
            trajectory = simulate_season(t[None], rh[None], rain[None], active, np.zeros(active.shape), Parameters())
            process = [trajectory.damage[0, -1, 0], trajectory.infectious[0, -1, 0],
                trajectory.state[0, -1, 0, 1:4].sum()+trajectory.state[0, -1, 0, 7:10].sum()]
            summaries = []
            for width in [7, 21, 42]:
                recent = values[-width:]
                summaries.extend([*recent[:, :3].mean(axis=0), np.log1p(recent[:, 3].mean())])
            calendar = 2*np.pi*(row.date.dayofyear-1)/365.2425
            common = [*summaries, np.sin(calendar), np.cos(calendar), (row.date-start).days/330.,
                field.latitude/90., field.longitude/180.]
            cache[key] = (common, process, phenotype, kernels, probabilities[-1, 2])
        common, process, phenotype, kernels, probability = cache[key]
        endpoint = f'{row.source}:code{row.leaf_index}'
        if endpoint not in ENDPOINTS:
            raise ValueError('Declared supported source-native endpoint required')
        onehot = np.array([float(endpoint == name) for name in ENDPOINTS])
        base = np.r_[common, onehot, process]
        baseline.append(base); augmented.append(np.r_[base, phenotype, kernels])
        phase.append(phase_weights(probability[None])[0])
        provenance.append(dict(global_target_index=int(row.global_target_index), field_id=row.field_id,
            query_date=str(row.date.date()), weather_start=str(start.date()), weather_end=str(end.date()),
            estimated_sowing_date=str(pd.Timestamp(field.estimated_sowing_date).date()),
            sowing_fallback=fallback, observed_sowing_used=False, disease_history_used=False,
            observed_bbch_used=False, satellite_used=False, endpoint=endpoint))
    extra_names = dev_names+['nearest_trait_station_100km', 'latitude_outside_trait_support']
    extra_names += [f'renewal_latency{latent}_{host}_{stat}' for latent in [10, 20, 30]
        for host in ['continuous', 'past31', 'past51'] for stat in ['pending', 'expressed', 'recent', 'bounded']]
    result = dict(baseline=np.asarray(baseline), augmented=np.asarray(augmented), phase=np.asarray(phase),
        baseline_names=names, augmented_names=names+extra_names, provenance=pd.DataFrame(provenance))
    for key in ['baseline', 'augmented', 'phase']:
        if not np.isfinite(result[key]).all():
            raise ValueError('Finite autonomous predictor matrix required')
    return result


def fit_autonomous(baseline, augmented, frame, weights, phase, members=20, iterations=200):
    baseline, augmented = np.asarray(baseline, float), np.asarray(augmented, float)
    y = frame.observed_percent.to_numpy(float)
    if (baseline.shape[0] != len(frame) or augmented.shape[0] != len(frame)
            or not np.isfinite(baseline).all() or not np.isfinite(y).all() or ((y < 0) | (y > 100)).any()):
        raise ValueError('Aligned weather predictors and bounded training percentages required')
    growth = fit_phase_growth(augmented, frame, np.zeros(len(frame)), weights, phase, members, iterations)
    models = []
    for index, member in enumerate(growth['members']):
        ix = np.flatnonzero(frame.coordinate_year.astype(str).isin(member['clusters']))
        mass = np.asarray(weights)[ix]; mass = mass/mass.mean()
        model = HistGradientBoostingRegressor(max_iter=iterations, learning_rate=.05, max_depth=3,
            max_leaf_nodes=15, min_samples_leaf=10, l2_regularization=3., early_stopping=False,
            random_state=20261022+index)
        model.fit(baseline[ix], y[ix], sample_weight=mass)
        models.append(model)
    return dict(baseline=models, growth=growth, baseline_features=baseline.shape[1],
        augmented_features=augmented.shape[1], history_conditioned=False, observed_bbch_used=False,
        satellite_used=False, runtime_weather=WEATHER.copy(), training_ids=frame.global_target_index.astype(int).tolist())


def predict_autonomous(fitted, baseline, augmented, phase):
    baseline, augmented = np.asarray(baseline, float), np.asarray(augmented, float)
    if (baseline.shape != (len(augmented), fitted['baseline_features'])
            or augmented.shape[1] != fitted['augmented_features']
            or not np.isfinite(baseline).all() or not np.isfinite(augmented).all()):
        raise ValueError('Schema-aligned autonomous features required')
    values = predict_phase_growth(fitted['growth'], augmented, np.zeros(len(augmented)), phase)
    reference = np.mean([np.clip(model.predict(baseline), 0, 100) for model in fitted['baseline']], axis=0)
    return dict(zip(OUTPUTS, [reference, values['shared'], values['stage']]))
