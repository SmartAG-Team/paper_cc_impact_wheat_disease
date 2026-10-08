"""Chronological field initialization and independently grouped observations."""
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

LEAF_RANK = {'LEAF, 1ST / FLAG LEAF':1, 'LEAF, FLAG':1, 'LEAF, 2ND':2,
    'LEAF, 3RD':3, 'LEAF, 4TH':4, 'LEAF, 5TH':5, 'LEAF, 6ST':6,
    'LEAF, 6TH':6, 'LEAF, 7ST':7, 'LEAF, 7TH':7}


def canopy_snapshot(records, trial, cutoff, ranks):
    frame = records.loc[records.TrialId.eq(trial)].copy()
    if frame.duplicated(['leaf_rank','Date']).any():
        raise ValueError('Duplicate trial/leaf/date disease records.')
    if not frame.Value.between(0,100).all():
        raise ValueError('Canopy disease percentages are missing or outside[0,100].')
    frame = frame.loc[frame.Date.le(cutoff)].sort_values('Date')
    latest = frame.groupby('leaf_rank').tail(1).set_index('leaf_rank')
    visible = np.zeros(len(ranks))
    active = np.zeros(len(ranks), dtype=bool)
    for i, rank in enumerate(ranks):
        if rank in latest.index:
            visible[i] = float(latest.loc[rank,'Value'])/100
            active[i] = True
    return visible, active


def weather_window(weather, location, start, end):
    """Daily sums/means are used on[start,end), before assessment at end."""
    source = weather.loc[weather.location_id.eq(location)].copy()
    if source.date.duplicated().any():
        raise ValueError('Duplicate location/day weather records.')
    days = pd.date_range(start,end-pd.Timedelta(days=1))
    out = source.set_index('date').reindex(days)
    columns = ['tmean_c','rh_hours_ge_90pct','rain_mm','hour_count']
    if out[columns].isna().any().any() or not out.hour_count.eq(24).all():
        raise ValueError(f'Incomplete daily weather for{location}.')
    if not out.rh_hours_ge_90pct.between(0,24).all() or not out.rain_mm.ge(0).all():
        raise ValueError('Invalid weather exposure range.')
    return out


@dataclass
class FieldBatch:
    episodes: pd.DataFrame
    targets: pd.DataFrame
    initial_visible: np.ndarray
    active: np.ndarray
    temperature: np.ndarray
    humidity_hours: np.ndarray
    rain: np.ndarray
    ranks: np.ndarray

    def subset(self, ids):
        positions = np.flatnonzero(self.episodes.series_id.isin(ids).to_numpy())
        mapping = {old:new for new,old in enumerate(positions)}
        targets = self.targets.loc[self.targets.episode_index.isin(positions)].copy()
        targets['episode_index'] = targets.episode_index.map(mapping).astype(int)
        return FieldBatch(self.episodes.iloc[positions].reset_index(drop=True),
            targets.reset_index(drop=True), self.initial_visible[positions], self.active[positions],
            self.temperature[positions], self.humidity_hours[positions], self.rain[positions], self.ranks)


def prepare_basf(root):
    root = Path(root)
    raw = pd.read_csv(root/'data/basf-wheat-diseases.txt',sep='\t')
    raw['source_row'] = np.arange(2,len(raw)+2)
    raw['Date'] = pd.to_datetime(raw.Date, errors='raise')
    raw['Value'] = pd.to_numeric(raw.Value,errors='coerce')
    valid = raw.loc[raw.Treatment.eq('Untreated') & raw.Organism.eq('SEPTTR') &
        raw.Parameter.eq('INFECT') & raw.Method.eq('P%INF') &
        ~raw.Clarifier.eq('CROP INJURY') & raw.PlantPart.isin(LEAF_RANK) &
        raw.Value.between(0,100)].copy()
    valid['leaf_rank'] = valid.PlantPart.map(LEAF_RANK).astype(int)
    if valid.duplicated(['TrialId','leaf_rank','Date']).any():
        raise ValueError('Duplicate disease source keys.')
    valid['series_id'] = valid.TrialId.astype(str)+'|leaf'+valid.leaf_rank.astype(str)
    counts = valid.groupby('series_id').Date.transform('nunique')
    selected = valid.loc[counts.ge(3)].copy()
    links = pd.read_csv(root/'data/era5/basf_trial_weather_links.csv')
    selected = selected.merge(links[['TrialId','location_id','season_year']],on='TrialId',
                              validate='many_to_one')
    selected['year'] = selected.Date.dt.year
    selected['coordinate_year'] = selected.location_id+'|'+selected.year.astype(str)
    selected = selected.sort_values(['series_id','Date']).reset_index(drop=True)
    episodes = selected.groupby('series_id',sort=True).first().reset_index()
    episodes = episodes.rename(columns={'Date':'start','Value':'initial_percent'})
    episodes['end'] = selected.groupby('series_id').Date.max().reindex(episodes.series_id).to_numpy()
    episodes['target_leaf_index'] = episodes.leaf_rank-1
    index = dict(zip(episodes.series_id,episodes.index))
    selected['episode_index'] = selected.series_id.map(index).astype(int)
    start_map = episodes.set_index('series_id').start
    selected['start'] = selected.series_id.map(start_map)
    selected['day'] = (selected.Date-selected.start).dt.days
    selected['fraction'] = selected.Value/100
    selected['conditioning'] = selected.day.eq(0)
    selected['target_leaf_index'] = selected.leaf_rank-1
    ranks = np.arange(1,8)
    max_days = int(selected.day.max())
    n = len(episodes)
    initial = np.zeros((n,7))
    active = np.zeros_like(initial,dtype=bool)
    temperature = np.full((n,max_days),18.0)
    humidity = np.zeros_like(temperature)
    rain = np.zeros_like(temperature)
    weather = pd.read_parquet(root/'data/era5/daily_weather.parquet')
    weather['date'] = pd.to_datetime(weather.date)
    for i, episode in episodes.iterrows():
        initial[i],active[i] = canopy_snapshot(valid,episode.TrialId,episode.start,ranks)
        frame = weather_window(weather,episode.location_id,episode.start,episode.end)
        length = len(frame)
        temperature[i,:length] = frame.tmean_c.to_numpy()
        humidity[i,:length] = frame.rh_hours_ge_90pct.to_numpy()
        rain[i,:length] = frame.rain_mm.to_numpy()
        if not active[i,episode.target_leaf_index] or not np.isclose(
                initial[i,episode.target_leaf_index],episode.initial_percent/100):
            raise ValueError('Target initial condition does not match chronological canopy state.')
    return FieldBatch(episodes,selected,initial,active,temperature,humidity,rain,ranks)


def hierarchical_weights(targets):
    if len(targets)==0:
        raise ValueError('No scored observations.')
    groups = targets.coordinate_year.nunique()
    series_per_group = targets.groupby('coordinate_year').series_id.transform('nunique')
    observations_per_series = targets.groupby('series_id').series_id.transform('size')
    weights = 1/(groups*series_per_group*observations_per_series)
    if not np.isclose(weights.sum(),1):
        raise ValueError('Hierarchical weights do not sum to one.')
    return weights.to_numpy()
