"""Historical DWD crop-development priors, separate from disease leaf endpoints.

Thermal/photoperiod/vernalization increments reproduce EJA-PGN-Wheat's
research_validation/prepare.py. Gates and threshold distributions are refitted
only on the explicitly admitted historical trait observations.
"""
from collections import Counter

import numpy as np
import pandas as pd

STAGES = [10, 31, 51, 87]
MODELS = ['gdd', 'tp', 'tpv']


def development_increments(weather, dates, latitude, parameters=None):
    temp, maximum = np.asarray(weather, float)[:, :2].T
    gdd = np.interp(temp, [0, 20, 30, 40], [0, 20, 20, 0], left=0, right=0)
    if parameters is None:
        return gdd
    doy = pd.DatetimeIndex(dates).dayofyear.to_numpy()
    declination = .4093*np.sin(2*np.pi*(doy-81)/365)
    daylength = 24/np.pi*np.arccos(np.clip(-np.tan(np.radians(latitude))*np.tan(declination), -1, 1))
    previous = np.cumsum(gdd)-gdd
    active = (previous >= parameters['photoperiod_onset_gdd']) & (previous < parameters['photoperiod_stop_gdd'])
    tp = gdd*np.where(active, np.clip(1-.09*(16-daylength), 0, 1), 1.)
    if 'vernalization_onset_tpp' not in parameters:
        return np.column_stack([gdd, tp])
    previous = np.cumsum(tp)-tp
    cold, factor = 0., np.ones(len(tp))
    response = np.interp(temp, [-4, 0, 10, 16], [0, 1, 1, 0])
    for i in range(len(tp)):
        if previous[i] >= parameters['vernalization_onset_tpp']:
            sensitive = previous[i] < parameters['vernalization_stop_tpp']
            updated = cold+(response[i] if sensitive else 0)
            if updated < 10 and maximum[i] > 30:
                updated = max(0, cold-.5*(maximum[i]-30))
            cold = updated
            if sensitive:
                factor[i] = .3+.7*min(cold/40, 1)
    return np.column_stack([gdd, tp, tp*factor])


def _distribution(values, stations):
    values = np.asarray(values, float)
    count = Counter(stations)
    weight = np.array([1./count[s] for s in stations])
    weight /= weight.sum()
    order = np.argsort(values, kind='stable')
    cumulative = np.cumsum(weight[order]); cumulative[-1] = 1.
    return dict(values=values[order], cumulative_weight=cumulative,
        station=np.asarray(stations)[order], weight=weight[order])


def _median(distribution):
    return float(distribution['values'][np.searchsorted(distribution['cumulative_weight'], .5)])


def fit_development_prior(records, weather):
    """Refit process gates and event distributions on admitted trait cycles."""
    if not records:
        raise ValueError('Historical trait cycles required')
    if any(r['END_DATE'] >= '2013-01-01' or not r['eligible'] or r['observation_order_conflict'] for r in records):
        raise ValueError('Only weather-qualified, ordered trait cycles ending before 2013 can fit')
    prepared = []
    for record in records:
        daily = np.asarray(weather[record['row_start']:record['row_stop']], float)
        dates = pd.date_range(record['SOWING_DATE'], periods=record['length'])
        if len(daily) != len(dates) or not np.isfinite(daily).all():
            raise ValueError('Complete trait weather required')
        events = {}
        for stage, date in zip(STAGES, record['true_dates']):
            if date is not None:
                lead = int((pd.Timestamp(date)-dates[0]).days)
                if not 0 <= lead < len(dates) or pd.Timestamp(date) > pd.Timestamp(record['END_DATE']):
                    raise ValueError('Trait event outside declared historical cycle')
                events[stage] = lead
        prepared.append((record, daily, dates, events))

    def samples(parameters=None):
        result = {model: {stage: ([], []) for stage in STAGES} for model in MODELS}
        for record, daily, dates, events in prepared:
            inc = development_increments(daily, dates, record['lat'], parameters)
            inc = inc[:, None] if inc.ndim == 1 else inc
            cumulative = np.cumsum(inc, axis=0)
            for column, model in enumerate(MODELS[:inc.shape[1]]):
                for stage, lead in events.items():
                    value, station = result[model][stage]
                    value.append(float(cumulative[lead, column])); station.append(int(record['PEP_ID']))
        return {model: {stage: _distribution(*value) for stage, value in stages.items() if len(value[0])}
            for model, stages in result.items() if any(len(v[0]) for v in stages.values())}

    initial = samples()
    parameters = dict(photoperiod_onset_gdd=_median(initial['gdd'][10]),
        photoperiod_stop_gdd=_median(initial['gdd'][51]))
    intermediate = samples(parameters)
    parameters.update(vernalization_onset_tpp=_median(intermediate['tp'][10]),
        vernalization_stop_tpp=_median(intermediate['tp'][31]))
    distributions = samples(parameters)
    for model in MODELS:
        if np.any(np.diff([_median(distributions[model][s]) for s in STAGES]) <= 0):
            raise ValueError('Historical median stage thresholds must increase')
    return dict(parameters=parameters, distributions=distributions,
        trait_cycle_ids=[f'{r["PEP_ID"]}|{r["SOWING_DATE"]}' for r in records],
        maximum_trait_date=max(r['END_DATE'] for r in records),
        trait_stations=sorted({int(r['PEP_ID']) for r in records}),
        threshold_weighting='Equal station mass per stage, divided among its observed cycles')


def summarize_development(cumulative, increments, distributions):
    cumulative, increments = np.asarray(cumulative, float), np.asarray(increments, float)
    features, names, stage_probabilities = [], [], []
    for column, model in enumerate(MODELS):
        probability, ratio = [], []
        for stage in STAGES:
            distribution = distributions[model][stage]
            values, weight = distribution['values'], distribution['cumulative_weight']
            position = np.searchsorted(values, cumulative[:, column], side='right')
            probability.append(np.where(position > 0, weight[np.maximum(0, position-1)], 0.))
            ratio.append(np.clip(cumulative[:, column]/max(_median(distribution), .001), 0, 4))
        probability = np.minimum.accumulate(np.column_stack(probability), axis=1)
        stage_probabilities.append(probability)
        category_probability = np.column_stack([1-probability[:, 0],
            probability[:, :-1]-probability[:, 1:], probability[:, -1]])
        entropy = -np.sum(category_probability*np.log(np.maximum(category_probability, 1e-12)), axis=1)/np.log(5)
        features.extend([probability, probability.sum(axis=1)[:, None]/4, entropy[:, None], np.column_stack(ratio)])
        names.extend([f'phenology_{model}_BBCH{s}_passed_probability' for s in STAGES])
        names.extend([f'phenology_{model}_expected_stage_index_0to1', f'phenology_{model}_stage_entropy_0to1'])
        names.extend([f'phenology_{model}_relative_accumulation_BBCH{s}_capped4' for s in STAGES])
        for lag in [7, 21]:
            before = np.zeros_like(probability)
            if len(before) > lag:
                before[lag:] = probability[:-lag]
            features.append(probability-before)
            names.extend([f'phenology_{model}_BBCH{s}_probability_change_{lag}d' for s in STAGES])
        features.append(increments[:, column, None]/20)
        names.append(f'phenology_{model}_daily_increment_scaled20')
    result = np.column_stack(features)
    if not np.isfinite(result).all() or (result < -1e-12).any():
        raise ValueError('Finite nonnegative causal development summaries required')
    return result, names, np.stack(stage_probabilities, axis=1)
