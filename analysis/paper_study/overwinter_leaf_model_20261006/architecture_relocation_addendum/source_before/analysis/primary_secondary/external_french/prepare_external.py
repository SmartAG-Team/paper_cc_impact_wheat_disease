"""External French adult-leaf pycnidia records, without field parameter fitting."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from model.primary_secondary.field_data import FieldBatch,canopy_snapshot,weather_window

ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/'data/public_septoria/orellana-torrejon-2022-field/F1_Field_disease_severity_rawdata.csv'


def prepare():
    raw=pd.read_csv(SOURCE,sep=';',comment='#')
    raw['Date']=pd.to_datetime(raw.date,dayfirst=True,errors='raise')
    raw['year']=raw.Date.dt.year
    raw['TrialId']=raw.year.astype(str)+'|'+raw.mixture+'|'+raw.rep+'|'+raw.var_origin
    groups=['TrialId','year','Date','mixture','rep','var_origin']
    tables=[]
    for rank in (1,2,3):
        column=f'%spor_F{rank}'
        frame=raw[groups+['id',column]].copy()
        text=frame[column].astype(str).str.strip()
        frame['numeric']=pd.to_numeric(frame[column],errors='coerce')
        frame['senescent']=text.eq('S')
        frame['unassessed']=frame[column].isna()|text.isin(['NA','nan',''])
        frame['unresolved']=frame.numeric.isna() & ~frame.senescent & ~frame.unassessed
        if frame.unresolved.any() or not frame.numeric.dropna().between(0,100).all():
            raise ValueError('Adult-leaf scores require source interpretation.')
        grouped=frame.groupby(groups,dropna=False).agg(Value=('numeric','mean'),
            n_numeric=('numeric','count'),n_senescent=('senescent','sum'),
            n_unassessed=('unassessed','sum'),source_rows=('id',lambda x:','.join(x.astype(str))))
        grouped=grouped.reset_index()
        grouped['leaf_rank']=rank
        tables.append(grouped)
    observations=pd.concat(tables,ignore_index=True)
    observations.to_csv(OUT/'all_plot_leaf_observations.csv',index=False)
    valid=observations.loc[observations.n_numeric.gt(0)].copy()
    valid['series_id']=valid.TrialId+'|F'+valid.leaf_rank.astype(str)
    valid=valid.loc[valid.groupby('series_id').Date.transform('nunique').ge(3)].copy()
    if valid.duplicated(['TrialId','leaf_rank','Date']).any():
        raise ValueError('Duplicate external plot/leaf/date mean.')
    valid=valid.sort_values(['series_id','Date']).reset_index(drop=True)
    valid['location_id']='era5loc_ed480fff4117'
    valid['coordinate_year']=valid.location_id+'|'+valid.year.astype(str)
    valid['Country']='FRANCE'
    valid['cultivar']=valid.var_origin
    valid['season_year']=valid.year
    episodes=valid.groupby('series_id',sort=True).first().reset_index()
    episodes=episodes.rename(columns={'Date':'start','Value':'initial_percent'})
    episodes['end']=valid.groupby('series_id').Date.max().reindex(episodes.series_id).to_numpy()
    episodes['target_leaf_index']=episodes.leaf_rank-1
    valid['episode_index']=valid.series_id.map(dict(zip(episodes.series_id,episodes.index))).astype(int)
    valid['start']=valid.series_id.map(episodes.set_index('series_id').start)
    valid['day']=(valid.Date-valid.start).dt.days
    valid['conditioning']=valid.day.eq(0)
    valid['target_leaf_index']=valid.leaf_rank-1
    valid['fraction']=valid.Value/100
    max_days=int(valid.day.max())
    n=len(episodes)
    visible=np.zeros((n,3));active=np.zeros((n,3),bool)
    ages=np.zeros((n,3))
    temperature=np.full((n,max_days),18.);humidity=np.zeros_like(temperature);rain=np.zeros_like(temperature)
    weather=pd.read_parquet(ROOT/'data/era5/daily_weather.parquet')
    weather['date']=pd.to_datetime(weather.date)
    all_past=observations.loc[observations.n_numeric.gt(0)]
    for i,e in episodes.iterrows():
        visible[i],active[i],ages[i]=canopy_snapshot(all_past,e.TrialId,e.start,np.arange(1,4),return_age=True)
        w=weather_window(weather,e.location_id,e.start,e.end)
        temperature[i,:len(w)]=w.tmean_c.to_numpy()
        humidity[i,:len(w)]=w.rh_hours_ge_90pct.to_numpy()
        rain[i,:len(w)]=w.rain_mm.to_numpy()
    episodes.to_csv(OUT/'episodes.csv',index=False)
    valid.to_csv(OUT/'targets.csv',index=False)
    inventory={'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'source_rows':len(raw),'adult_rank_plot_date_cells':len(observations),
        'eligible_series':len(episodes),'eligible_observations':len(valid),
        'scored_future_observations':int((~valid.conditioning).sum()),
        'independent_station_years':2,'blocks_per_season':3,
        'sowing_dates':{'2018':'2017-10-17','2019':'2018-11-15'},
        'source_measure':'pycnidia-covered area, whether actively sporulating or not',
        'senescence_policy':'S excluded from numeric percent; count retained separately; no zero imputation',
        'sampling_policy':'plot/cultivar means over numerically assessed plants; not repeated physical-leaf identity',
        'country_station':'Thiverval-Grignon representative station coordinate, not exact plot location',
        'juvenile_L_ranks':'excluded from adult-F-rank transport; differing leaf numbering convention',
        'external_observation_mapping':'infectious plus removed tissue; excludes latent and nonsporulating compartments'}
    (OUT/'inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
    return FieldBatch(episodes,valid,visible,active,temperature,humidity,rain,np.arange(1,4),ages)


if __name__=='__main__':
    prepare()
    print((OUT/'inventory.json').read_text())
