"""Freeze prior and stage-conditioned development inputs without disease values."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .innovation_common import ROOT,DATA,PARENT
from .inputs import sha
from .phenology_alignment import coupled_profiles,align_field,NAMES
from .phenology_transfer import development_increments
from .phenology_transfer_data import TRAIT
from .sowing_distribution_data import PHENO,EXTENSION,prior_weather_coverage
from .stage_features import build_stage_features,PRIOR_NAMES

SOW=ROOT/'data/disease_improvement/sowing_distribution_v2_20261022'
CALENDAR=ROOT/'outputs/raw_metadata_recovery_20261022/stage_calendar.csv'
OUT=ROOT/'data/disease_improvement/phenology_alignment_20261022'


def build(output=OUT):
    out=Path(output)
    if out.exists():raise FileExistsError('Fresh state-alignment destination required')
    out.mkdir(parents=True)
    sources=[PHENO/'admitted_trait_cycles.json',PHENO/'development_prior.joblib',
             PHENO/'registration.json',PHENO/'verification.json',PHENO/'predictors.npz',
             TRAIT/'daily_inputs.npz',SOW/'sowing_prior_nodes.csv',SOW/'verification.json',
             CALENDAR,EXTENSION,DATA/'fields.csv',DATA/'daily_weather.parquet',
             PARENT/'training_observation_ledger.csv']
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'phenology_alignment','phenology_transfer','phenology_transfer_data',
        'sowing_distribution_data','stage_features']]
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        profiles='All complete ordered four-event cycles from already admitted pre2013 >50km DWD trait population; joint thresholds from raw weather with existing historical process gates; equal station mass divided among complete cycles',
        clocks='GDD/TP/TPV equally weighted; event and query clocks strictly before their date',
        sowing='Existing nine equal-mass RF conditional empirical quantile nodes, no refit; query-specific matched all-node coverage or original scenario fallback',
        evidence='Only earlier usable same-field BBCH intervals; ordered ranks at10/31/51/87; no metric interpolation of BBCH codes',
        likelihood='Logistic threshold scale60 clock units; .95 ordered-rank probability + .05 uniform5-category probability; fixed working assumptions, no calibrated observation-error claim',
        controls='Prior-only and sequentially conditioned summaries have identical dimensions; both share nine raw prior-stage predictors and three context flags',
        information='No disease values read by transformation; same-day/future stages and other-field context excluded. Earlier raw stages may derive from other untreated endpoint records in same field; severity values never read.',
        disease_targets_unchanged=True,independent_disease_validation=False)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for path in codes:
        dest=out/'code_snapshot'/path.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,dest)
    for parent in [PHENO,SOW]:
        receipt=json.loads((parent/'verification.json').read_text());assert receipt['passed']
        old=json.loads((parent/'registration.json').read_text())
        for path,digest in old['source_sha256'].items():assert sha(Path(path))==digest
    started=time.monotonic()
    records=json.loads((PHENO/'admitted_trait_cycles.json').read_text())
    process=joblib.load(PHENO/'development_prior.joblib')
    with np.load(TRAIT/'daily_inputs.npz',allow_pickle=False) as arrays:weather=arrays['weather']
    bank=coupled_profiles(records,weather,process['parameters']);del weather
    joblib.dump(bank,out/'coupled_development_profiles.joblib')
    print('profiles',len(bank['cycle_ids']),'stations',len(set(bank['station_ids'])),flush=True)
    fields=pd.read_csv(DATA/'fields.csv')
    # Explicit column selection: real disease values do not enter the transformation.
    queries=pd.read_csv(PARENT/'training_observation_ledger.csv',usecols=[
        'global_target_index','field_id','site_id','date','source','leaf_index'])
    nodes=pd.read_csv(SOW/'sowing_prior_nodes.csv')
    calendar=pd.read_csv(CALENDAR);calendar=calendar.loc[calendar.usable.eq(True)].copy()
    weather=pd.read_parquet(DATA/'daily_weather.parquet')
    extension=pd.read_parquet(EXTENSION)
    for frame in [weather,extension]:frame['date']=pd.to_datetime(frame.date)
    keys=pd.MultiIndex.from_frame(weather[['location_id','date']])
    extension=extension.loc[~pd.MultiIndex.from_frame(extension[['location_id','date']]).isin(keys)]
    weather=pd.concat([weather,extension],ignore_index=True)
    assert not weather.duplicated(['location_id','date']).any()
    by_site={k:g.set_index('date').sort_index() for k,g in weather.groupby('location_id')}
    metadata=fields.set_index('field_id');provenance=[]
    prior_stage,stage_names,_=build_stage_features(weather,fields,queries,calendar)
    raw_stage=prior_stage[:,:len(PRIOR_NAMES)]
    width=len(NAMES)+3+len(PRIOR_NAMES)
    prior=np.zeros((len(queries),width));filtered=prior.copy()
    prior_probability=np.zeros((len(queries),4));filtered_probability=prior_probability.copy()
    available=np.zeros(len(queries),bool)
    for field_id,group in queries.groupby('field_id',sort=False):
        meta=metadata.loc[field_id];site=by_site[meta.site_id]
        starts=list(pd.to_datetime(nodes.loc[nodes.field_id.eq(field_id),'estimated_sowing_date']))
        assert len(starts)==9
        fallback=pd.Timestamp(meta.sowing_date)
        coverage=prior_weather_coverage(site,starts,group.date)
        field_calendar=calendar.loc[calendar.field_id.eq(field_id)]
        for covered in [True,False]:
            here=np.flatnonzero(coverage==covered)
            if not len(here):continue
            use_starts=starts if covered else [fallback]
            dates=pd.to_datetime(group.date.iloc[here]);last=dates.max()-pd.Timedelta(days=1)
            clocks={}
            for start in set(use_starts):
                timeline=pd.date_range(start,last)
                daily=site.reindex(timeline)[['tmean_c','tmax_c','tmin_c','precipitation_mm']].to_numpy(float)
                if not np.isfinite(daily).all():raise ValueError('Complete qualified prefix weather required')
                increments=development_increments(daily,timeline,meta.latitude,bank['parameters'])
                clocks[str(start.date())]=np.vstack([np.zeros((1,3)),np.cumsum(increments,axis=0)])
            result=align_field(use_starts,clocks,dates,field_calendar,bank)
            ix=group.index.to_numpy()[here]
            common=np.column_stack([result['stage_available'].astype(float),result['stage_count']/10,
                                    result['stage_age_days']/30,raw_stage[ix]])
            prior[ix]=np.column_stack([result['prior'],common])
            filtered[ix]=np.column_stack([result['filtered'],common])
            prior_probability[ix]=result['prior_probability'];filtered_probability[ix]=result['filtered_probability']
            available[ix]=result['stage_available']
            for local,index in enumerate(ix):
                now=pd.Timestamp(queries.loc[index,'date'])
                context=field_calendar.loc[field_calendar.date.ge(str(min(use_starts).date()))&field_calendar.date.lt(str(now.date()))]
                provenance.append(dict(global_target_index=int(queries.loc[index,'global_target_index']),
                    field_id=field_id,query_date=str(now.date()),weather_end=str((now-pd.Timedelta(days=1)).date()),
                    rf_prior_weather_complete=bool(covered),stage_available=bool(available[index]),
                    stage_count=int(result['stage_count'][local]),
                    latest_stage_date=context.date.max() if len(context) else None,
                    source_stage_dates=';'.join(context.date.astype(str)),
                    source_stage_rows=';'.join(context.source_rows.astype(str)),
                    posterior_effective_profiles=float(result['posterior_ess'][local]),
                    current_day_stage_used=False,real_disease_values_used=False))
        print('field',field_id,'queries',len(group),'elapsed',round(time.monotonic()-started,1),flush=True)
    names=NAMES+['stage_available','stage_count_10','stage_age_days_30']+PRIOR_NAMES
    np.savez_compressed(out/'predictors.npz',prior=prior,filtered=filtered,
        prior_probability=prior_probability,filtered_probability=filtered_probability,
        stage_available=available,target_ids=queries.global_target_index.to_numpy(int))
    pd.DataFrame(provenance).sort_values('global_target_index').to_csv(out/'query_provenance.csv',index=False)
    (out/'feature_schema.json').write_text(json.dumps(dict(feature_names=names,
        units='Explicit in names; probability/ratio/relative mass otherwise',milestones=[10,31,51,87]),indent=2)+'\n')
    assert np.isfinite(prior).all() and np.isfinite(filtered).all()
    assert np.all(np.diff(filtered_probability,axis=1)<=1e-12)
    np.testing.assert_array_equal(prior[~available],filtered[~available])
    result=dict(passed=True,profiles=len(bank['cycle_ids']),trait_stations=len(set(bank['station_ids'])),
        queries=len(queries),features=width,stage_context_queries=int(available.sum()),
        original_disease_values_read=False,source_unchanged=all(sha(Path(p))==v for p,v in registration['source_sha256'].items()),
        elapsed_seconds=time.monotonic()-started,
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='artifacts_sha256'},indent=2),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):build()
