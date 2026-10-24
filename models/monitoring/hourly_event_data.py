"""Materialize qualified subdaily event predictors without using disease outcomes."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .hourly_station_inputs import load_station_hours
from .hourly_event_features import daily_station_events,build_hourly_features
from .innovation_common import ROOT,DATA,PARENT
from .inputs import sha


def run(output=None):
    out=Path(output or ROOT/'data/disease_improvement/hourly_station_events_20261022').resolve()
    if out.exists():raise FileExistsError('Fresh hourly feature destination required')
    out.mkdir(parents=True)
    codes=[Path(__file__),*[Path(__file__).with_name(n+'.py') for n in
        ['hourly_station_inputs','hourly_event_features','dwd_station','meteo_france_station']]]
    registration=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        recipe='Observedstationcontinuousspells/rollingprecipitation/Humidity-temperaturecoincidence; no disease outcomes',
        windows_days=[7,21,42],run_cap_hours=24,coverage_gate='>=80%complete21dayprecipitationorhumiditytimingdays;>=7daysaftersowing',
        time_policy='RRreportendtime-1h pairedwithTUinstantatintervalstart; currentdayexcluded; rollingfeaturecomplete24hprehistoryneeded',
        total_precipitation_not_measured_liquid_rain=True,air_humidity_not_canopy_leafwetness=True,
        source_selection='Reuse existing nearestqualifiedDWD/MFstation andretaineddailyproviderqualification; no outcome selection',
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
            [DATA/'daily_weather.parquet',DATA/'fields.csv',PARENT/'training_observation_ledger.csv']})
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for code in codes:
        dest=out/'code_snapshot'/code.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code,dest)
    weather=pd.read_parquet(DATA/'daily_weather.parquet');fields=pd.read_csv(DATA/'fields.csv')
    queries=pd.read_csv(PARENT/'training_observation_ledger.csv',float_precision='round_trip').drop(columns='observed_percent')
    hourly,sources,matches=load_station_hours(ROOT,weather)
    for path in sources:registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    hourly.to_parquet(out/'qualified_station_hours.parquet',index=False)
    matches.to_csv(out/'station_provenance.csv',index=False)
    daily=daily_station_events(hourly);daily.to_parquet(out/'daily_event_summary.parquet',index=False)
    values,names,ledger=build_hourly_features(daily,fields,queries)
    assert values.shape==(2304,81)
    np.savez_compressed(out/'predictors.npz',target_ids=queries.global_target_index.to_numpy(),hourly_events=values,
        usable=ledger.qualified_hourly_context.to_numpy(bool))
    ledger.to_csv(out/'query_provenance.csv',index=False)
    (out/'feature_schema.json').write_text(json.dumps(dict(names=names,columns=len(names),
        units='precipitationaccumulations:mm; runlength/duration/weightedhumiditydose:h; coveragefractions:1',
        unavailable_channels='Zero-valued numerical summaries with explicit observed-day coverage fractions; not imputed measured dryweather'),indent=2)+'\n')
    manifest=dict(passed=True,source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        code_unchanged=all(sha(ROOT/p)==h for p,h in registration['code_sha256'].items()),
        unique_location_hours=not hourly.duplicated(['location_id','hour_start']).any(),
        locations=int(hourly.location_id.nunique()),hourly_rows=len(hourly),daily_rows=len(daily),
        features=len(names),queries=len(queries),qualified_queries=int(ledger.qualified_hourly_context.sum()),
        output_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    if not manifest['source_unchanged'] or not manifest['code_unchanged']:raise ValueError('Hourly source/code changed during materialization')
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({k:v for k,v in manifest.items() if k!='output_sha256'},indent=2),flush=True)
    return values,ledger


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path);args=p.parse_args();run(args.output)
