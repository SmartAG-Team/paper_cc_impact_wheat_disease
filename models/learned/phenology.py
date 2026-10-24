"""Frozen established residual crop CDFs with known sowing/calendar inputs."""
from pathlib import Path
from types import SimpleNamespace
import json
import joblib
import numpy as np
import pandas as pd
import torch
from .integrity import DATA, ROOT, verify
from . import phenology_math as m
from .phenology_math import geo_features, correct

HERE = DATA / 'current'
r = SimpleNamespace(daily_features=m.daily_features)
t = SimpleNamespace(epoch_features=m.epoch_features, predict=m.predict_heads)
a = SimpleNamespace(age_features=m.age_features)

def apply(base,arrays,cycles,selection):
    if not selection['adopted']:return base.copy(),None
    path=Path(selection['model']['path'])
    if not path.is_absolute():path=ROOT/path
    package=joblib.load(path)
    known=cycles.rename(columns={'latitude':'lat','longitude':'lon'}) if 'lat' not in cycles else cycles
    features=t.epoch_features(geo_features(r.daily_features(arrays['x'],arrays['p'],arrays['inc'],base),known),known)
    features=a.age_features(features,base,arrays['p'])
    recipe=selection['recipe'];delta=t.predict(package,features,coupled=recipe['coupled'])
    return correct(base,delta,arrays['inc'],recipe['strength'],recipe['mask'],recipe['variant']),delta


def selected_cdf(base_cdf,features,progress,increments,latitudes,longitudes,*,
                 known_sowing_dates=None,known_sowing_year=None,known_sowing_day_of_year=None,registry_path=None):
    """Return prefix-causal stage CDFs; no observation or identity input exists.

    Sowing dates, or known sowing years plus sowing days-of-year, are management
    inputs. All weather/crop predictors use the same prefix as ``base_cdf``.
    The registry explicitly records the selected model's performance limits.
    """
    base=np.asarray(base_cdf);n=len(base)
    if known_sowing_dates is not None:
        sowing=pd.DatetimeIndex(pd.to_datetime(known_sowing_dates,errors='raise'))
        if sowing.isna().any():raise ValueError('Complete known sowing dates are required')
    else:
        if known_sowing_year is None or known_sowing_day_of_year is None:
            raise ValueError('Known sowing date, or known sowing year plus day-of-year, is required')
        years=np.asarray(known_sowing_year);days=np.asarray(known_sowing_day_of_year)
        if years.shape!=(n,) or days.shape!=(n,) or np.any(days<1) or np.any(days>366) or np.any(days!=np.floor(days)):
            raise ValueError('Known sowing calendar inputs must have one valid value per cycle')
        sowing=pd.to_datetime([f'{int(year):04d}-01-01' for year in years])+pd.to_timedelta(days-1,unit='D')
        if np.any(sowing.year!=years):raise ValueError('Sowing day-of-year falls outside supplied year')
    if len(sowing)!=n or len(latitudes)!=n or len(longitudes)!=n:raise ValueError('Complete known sowing/coordinate inputs are required')
    cycles=pd.DataFrame(dict(SOWING_DATE=sowing.strftime('%Y-%m-%d'),lat=latitudes,lon=longitudes))
    path=Path(registry_path) if registry_path is not None else HERE/'model_registry.json'
    registry=json.loads(path.read_text());selected_path=Path(registry['selection']['path'])
    if not selected_path.is_absolute():selected_path=ROOT/selected_path
    selection=json.loads(selected_path.read_text())
    arrays=dict(x=np.asarray(features),p=np.asarray(progress),inc=np.asarray(increments))
    if registry['kind']=='temporal':raise ValueError('Only the frozen primary phase-age model is packaged')
    elif registry['kind']=='phase_age':
        from .phenology import apply
        prediction,_=apply(base,arrays,cycles,selection)
    else:raise ValueError('Unsupported registered residual kind')
    return prediction

_selected_cdf = selected_cdf

def selected_cdf(*args, **kwargs):
    verify('phenology')
    return _selected_cdf(*args, **kwargs)


def predict_cdf(weather, known_sowing_dates, latitudes, longitudes):
    verify('phenology')
    weather = np.asarray(weather)
    cycles = pd.DataFrame(dict(SOWING_DATE=known_sowing_dates, lat=latitudes, lon=longitudes))
    if weather.ndim != 3 or weather.shape[-1] != 4 or not 0 < weather.shape[1] <= 330 or len(cycles) != len(weather):
        raise ValueError('Weather must contain one known sowing/coordinate record and 1..330 days per cycle')
    package = torch.load(DATA / 'current/base_ensemble.pt', map_location='cpu', weights_only=False)
    x, p, inc = [], [], []
    for i, row in enumerate(cycles.itertuples(index=False)):
        dates = pd.date_range(row.SOWING_DATE, periods=weather.shape[1])
        feature, progress, increment = m.build_features(weather[i], dates, row.lat, package['members'][0]['metadata']['gates'])
        x.append(feature); p.append(progress); inc.append(increment)
    x, p, inc = np.asarray(x), np.asarray(p), np.asarray(inc)
    members = []
    for member in package['members']:
        model = m.WheatPhenology(member['metadata'], member['mode'])
        model.load_state_dict(member['state_dict'])
        members.append(m.predict_batches(model, x, p, inc, np.arange(len(cycles))))
    base = np.mean(members, axis=0, dtype=np.float64).astype(np.float32)
    return selected_cdf(base, x, p, inc, latitudes, longitudes, known_sowing_dates=known_sowing_dates)
