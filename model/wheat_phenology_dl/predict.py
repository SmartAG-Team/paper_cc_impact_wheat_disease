"""Native-stage wheat DL inference with explicit sowing and causal daily weather."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from process_model.calibrate import transform
from analysis.paper_study.wheat_dl_phenology_20261009.prepare import pack_cycle, FEATURES, HORIZON
from .network import WheatDevelopmentModel, STAGES, quantile_days

ROOT=Path(__file__).resolve().parents[2]
CALIBRATION=ROOT/'process_model/parameters/calibration.json'


def weather_features(weather,sowing,latitude):
    frame=weather.rename(columns={'date':'DATE','tmean_c':'t_mean','tmin_c':'t_min','tmax_c':'t_max'}).copy()
    required={'DATE','t_mean','t_min','t_max'}
    if not required.issubset(frame):
        raise ValueError('Required weather columns: date, tmean_c, tmin_c, tmax_c; daily mean is not imputed')
    if not np.isfinite(latitude) or abs(latitude)>90:raise ValueError('Invalid latitude')
    frame['DATE']=pd.to_datetime(frame.DATE);sowing=pd.Timestamp(sowing)
    frame=frame.loc[frame.DATE>=sowing].sort_values('DATE').reset_index(drop=True)
    if frame.DATE.duplicated().any():raise ValueError('Duplicate weather dates')
    n=min(len(frame),HORIZON)
    valid=np.isfinite(frame[['t_mean','t_min','t_max']].iloc[:n].to_numpy()).all(1)
    valid&=pd.DatetimeIndex(frame.DATE.iloc[:n])==pd.date_range(sowing,periods=n)
    valid&=(frame.t_min.iloc[:n]<=frame.t_mean.iloc[:n])&(frame.t_mean.iloc[:n]<=frame.t_max.iloc[:n])
    stop=int(np.flatnonzero(~valid)[0]) if (~valid).any() else n
    if not stop:return np.zeros((HORIZON,len(FEATURES)),np.float32),np.zeros(HORIZON,np.float32),0
    frame=frame.iloc[:stop].copy()
    if (frame.t_mean>40).any():raise ValueError('Frozen wheat thermal response is unsupported above 40 C daily mean')
    frame['GDD']=np.where(frame.t_mean>30,20-2*(frame.t_mean-30),np.clip(frame.t_mean,0,20))
    frame['PEP_ID']=0;frame['LAT']=float(latitude)
    frame['SOWING_DATE']=sowing;frame['SOWING_KNOWN_AT']=sowing
    parameters=json.loads(CALIBRATION.read_text())
    features=transform(frame,parameters)
    return pack_cycle(features,sowing)


def predict(weather,sowing,latitude,checkpoints):
    raw,g,length=weather_features(weather,sowing,latitude)
    if not length:raise ValueError('No contiguous valid weather beginning on sowing day')
    calibration_sha=hashlib.sha256(CALIBRATION.read_bytes()).hexdigest()
    cdfs=[];models=[]
    for path in checkpoints:
        saved=torch.load(path,map_location='cpu',weights_only=True)
        m=saved['metadata'];c=saved['config']
        if m['stages']!=list(STAGES) or m['features']!=FEATURES:
            raise ValueError('Checkpoint does not use the native wheat feature/stage contract')
        if m['source_hashes']['process_model/parameters/calibration.json']!=calibration_sha:
            raise ValueError('Checkpoint and wheat forcing calibration differ')
        model=WheatDevelopmentModel(m['thresholds'],len(FEATURES),encoder=c['encoder'],width=c['width'])
        model.load_state_dict(saved['state_dict']);model.eval()
        x=(raw-np.asarray(m['mean'],np.float32))/np.asarray(m['std'],np.float32)
        with torch.no_grad():_,cdf=model(torch.tensor(x[None]),torch.tensor(g[None]),m['thermal_scale'])
        cdfs.append(cdf.numpy());models.append(dict(file=Path(path).name,sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),encoder=c['encoder'],seed=c['seed']))
    if not cdfs:raise ValueError('At least one wheat-fitted checkpoint required')
    if len({r['encoder'] for r in models})!=1:raise ValueError('Use the selected family, not a post-hoc mixture of families')
    ensemble=np.mean(cdfs,0);rows=[]
    for j,stage in enumerate(STAGES):
        row={'BBCH':stage}
        for label,q in [('lower90',.05),('median',.5),('upper90',.95)]:
            day=quantile_days(ensemble,q,np.array([length]))[0,j]
            row[label+'_date']=None if not np.isfinite(day) else str((pd.Timestamp(sowing)+pd.Timedelta(days=int(day)-1)).date())
        row['probability_reached_by_last_weather_day']=float(ensemble[0,j,length-1]);rows.append(row)
    return dict(stages=rows,weather_days=length,sowing=str(pd.Timestamp(sowing).date()),latitude=float(latitude),checkpoints=models,
                scope='Native wheat stage hindcast conditional on supplied weather and fitted weights; GS39/65 are not native outputs; no fixed-date fallback')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--weather',required=True,type=Path);p.add_argument('--sowing',required=True)
    p.add_argument('--latitude',required=True,type=float);p.add_argument('--checkpoints',required=True,nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();torch.set_num_threads(2);result=predict(pd.read_csv(a.weather),a.sowing,a.latitude,a.checkpoints)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
