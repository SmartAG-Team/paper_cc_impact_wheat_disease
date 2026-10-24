"""Qualify observed regional night-temperature inputs with a nuisance control."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .innovation_common import ROOT,DATA,PARENT
from .inputs import sha
from .night_lst_inputs import build_night_features,temperature_content_columns

RAW=ROOT/'data/disease_improvement/night_lst_regional_20261022'
OUT=ROOT/'data/disease_improvement/night_lst_prefix_20261022'


def build():
    receipt=json.loads((RAW/'verification.json').read_text())
    assert receipt['passed'] and receipt['source_unchanged']
    parent=json.loads((RAW/'registration.json').read_text())
    for path,digest in parent['source_sha256'].items():assert sha(ROOT/path)==digest,path
    for path,digest in receipt['artifacts_sha256'].items():assert sha(RAW/path)==digest,path
    if OUT.exists():raise FileExistsError('Fresh night-input qualification destination required')
    OUT.mkdir(parents=True)
    sources=[RAW/'registration.json',RAW/'verification.json',RAW/'regional_night_lst.csv',
        RAW/'quality_interpretation_amendment.json',DATA/'daily_weather.parquet',DATA/'fields.csv',
        PARENT/'training_observation_ledger.csv']
    codes=[Path(__file__),Path(__file__).with_name('night_lst_inputs.py')]
    reg=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        qualification='Atleast20 quality-qualified cropped sample cells and valid-pixel fraction>=.2; sharemaskall3retrievalchannels; originalcohort retained',
        timing='CompleteUTCdayperiodend+fixed2dayassumedavailability; queryinputusesavailable_at<=query00:00; 14/42/84daywindows datedbyavailability, notobservationdate; latestqualifiednightage<=14days',
        predictors='94features: two sensors×47; skinT/spatialvariation/localviewtime/absoluteviewangle/dailyairTminandmeandewpoint/contrasts/positivecontrast/retrievalandprovidercoverage/airmatchingflags/age',
        physical_scope='Regional clear-sky skin temperature. Daily-air contrasts are NOT simultaneous dewpoint/air measurements and do not establish canopy wetness/condensation. Spatial SD is not measurement uncertainty.',
        coverage_scope='Cloud-dependent retrieval selection explicit; validfraction is withinstaticcropmask, not totalbuffercropfraction. Either sensor can satisfyquerycoverage. No forcingsources or precipitation changed.',
        air_control='Zero40satellite-derivedtemperature/spatialSD/contrast columns while preserving54dailyair/view/coverage/age/airavailability columns exactly; explicit16air-value columns retain dailyairinformation inBOTH arms; samefeatureshapeandrows, no outcome-dependentselection',
        scaling='SkinT and dailyairtemperatures/20; spatialSD and temperaturecontrasts/10; viewhour/24; angle/65; age/84. Fractions/flags unscaled.',
        publication_scope='Historical reprocessed product; +2daylag is a working assumption, not verified historical operational availability',
        disease_values_used_in_transformation=False,independent_disease_validation=False)
    (OUT/'registration.json').write_text(json.dumps(reg,indent=2)+'\n')
    for code in codes:
        dest=OUT/'code_snapshot'/code.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code,dest)
    raw=pd.read_csv(RAW/'regional_night_lst.csv')
    weather=pd.read_parquet(DATA/'daily_weather.parquet')
    queries=pd.read_csv(PARENT/'training_observation_ledger.csv',usecols=[
        'global_target_index','field_id','site_id','date','source','leaf_index'])
    x,names,available,provenance=build_night_features(raw,weather,queries)
    assert len(names)==94 and x.shape==(2304,94)
    content=temperature_content_columns(names);control=x.copy();control[:,content]=0
    assert len(content)==40
    np.savez_compressed(OUT/'predictors.npz',night=x,air_control=control,
        available=available,target_ids=queries.global_target_index.to_numpy(int),temperature_content_indices=np.asarray(content,int))
    provenance.to_csv(OUT/'query_provenance.csv',index=False)
    (OUT/'feature_schema.json').write_text(json.dumps(dict(feature_names=names,
        temperature_content_indices=content,scaling=reg['scaling'],scope=reg['physical_scope']),indent=2)+'\n')
    dated=pd.to_datetime(provenance.latest_period_end,errors='coerce')
    assert (dated.isna()|dated.lt(pd.to_datetime(provenance.query_date))).all()
    qualified=raw.loc[raw.night_lst_c_count.ge(20)&raw.valid_mean.ge(.2)]
    result=dict(passed=True,cohort_rows=len(queries),features=len(names),available_queries=int(available.sum()),
        sites_with_available_queries=queries.loc[available,'site_id'].nunique(),
        qualified_regional_sensor_days=len(qualified),raw_regional_sensor_days=len(raw),
        temperature_content_columns=len(content),same_air_and_retrieval_columns_exact=True,
        disease_targets_unchanged=True,disease_values_used=False,
        source_unchanged=all(sha(ROOT/p)==h for p,h in reg['source_sha256'].items()),
        artifacts_sha256={p.name:sha(p) for p in OUT.iterdir() if p.is_file()})
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},indent=2))


if __name__=='__main__':build()
