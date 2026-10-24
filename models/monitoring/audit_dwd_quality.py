"""Read-only audit of quality flags for hourly rain actually used in retained forcing."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import pandas as pd

from .innovation_common import ROOT,DATA
from .inputs import sha


def run():
    out=ROOT/'outputs/disease_improvement/dwd_quality_flag_audit_20261022'
    if out.exists():raise FileExistsError('Fresh rainfall quality-audit destination required')
    weather_path=DATA/'daily_weather.parquet'
    matching_path=ROOT/'outputs/weather_validation_20261022/dwd/station_matches.csv'
    weather=pd.read_parquet(weather_path)
    used=weather.loc[weather.DWD_total_precip_used.eq(True),['location_id','date','precipitation_mm']].copy()
    used['date']=pd.to_datetime(used.date)
    matches=pd.read_csv(matching_path,dtype={'station_id':str})
    matches=matches.loc[matches.variable_group.eq('RR')&matches.within_distance]
    assert not matches.site_id.duplicated().any()
    used=used.merge(matches[['site_id','station_id']],left_on='location_id',right_on='site_id',validate='many_to_one')
    assert len(used)==int(weather.DWD_total_precip_used.eq(True).sum())
    raw=ROOT/'data/external_weather_validation_20261022/dwd'
    files={station:sorted(raw.glob(f'stundenwerte_RR_{station}_*_hist.zip')) for station in used.station_id.unique()}
    assert all(len(paths)==1 for paths in files.values())
    sources=[weather_path,matching_path,*[paths[0] for paths in files.values()]]
    out.mkdir(parents=True)
    registration=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        sources_sha256={str(p):sha(p) for p in sources},code_sha256=sha(Path(__file__)),
        scope='OnlyhourlyreportsmakingretainedDWDtotalprecipdays; no diseaseoutcomesread, noforcingchanges',
        quality_definition='DWDQN_8describesQCprocedure, notbinarycorrect/incorrect. Physicalrange/missingrule unchanged. Noautomaticdeletionfromanyflag',
        source_document='https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/hourly/precipitation/DESCRIPTION_obsgermany_climate_hourly_precipitation_en.pdf')
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    shutil.copyfile(__file__,out/'audit_source_snapshot.py')
    counts=[];paired=[]
    for station,group in used.groupby('station_id'):
        with zipfile.ZipFile(files[station][0]) as archive:
            product=[name for name in archive.namelist() if name.startswith('produkt') and name.endswith('.txt')]
            assert len(product)==1
            hourly=pd.read_csv(archive.open(product[0]),sep=';',skipinitialspace=True,
                usecols=['MESS_DATUM','QN_8','R1'])
        times=pd.to_datetime(hourly.MESS_DATUM.astype(str),format='%Y%m%d%H')
        hourly['date']=(times-pd.Timedelta(hours=1)).dt.normalize()
        hours=hourly.loc[hourly.date.isin(group.date.unique())].copy()
        for flag,count in hours.QN_8.value_counts(dropna=False).items():
            counts.append(dict(station_id=station,quality_flag=int(flag),hourly_reports=int(count)))
        valid=hours.R1.between(0,500)&np.isfinite(hours.R1)
        assert valid.all()
        daily=hours.groupby('date').agg(rebuilt_precipitation_mm=('R1','sum'),hours=('R1','size'))
        assert daily.hours.eq(24).all()
        block=group.merge(daily,left_on='date',right_index=True,validate='many_to_one')
        block['absolute_difference_mm']=np.abs(block.precipitation_mm-block.rebuilt_precipitation_mm)
        np.testing.assert_allclose(block.precipitation_mm,block.rebuilt_precipitation_mm,atol=1e-10,rtol=0)
        paired.append(block)
    counts=pd.DataFrame(counts);paired=pd.concat(paired,ignore_index=True)
    counts.to_csv(out/'hourly_quality_flag_counts.csv',index=False)
    paired.to_csv(out/'retained_daily_replay.csv',index=False)
    verification=dict(passed=True,sites=used.location_id.nunique(),stations=used.station_id.nunique(),site_days=len(used),
        distinct_station_days=len(paired[['station_id','date']].drop_duplicates()),
        distinct_hourly_reports=int(counts.hourly_reports.sum()),
        quality_flag_counts={str(k):int(v) for k,v in counts.groupby('quality_flag').hourly_reports.sum().items()},
        maximum_daily_total_difference_mm=float(paired.absolute_difference_mm.max()),
        source_unchanged=all(sha(Path(p))==digest for p,digest in registration['sources_sha256'].items()),
        no_disease_labels_read=True,no_forcing_change=True,
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='artifacts_sha256'},indent=2),flush=True)


if __name__=='__main__':run()
