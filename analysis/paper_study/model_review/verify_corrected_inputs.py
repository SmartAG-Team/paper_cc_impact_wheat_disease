"""Retrospective correction check; model and frozen archives remain read-only."""
from pathlib import Path
import sys, os, json, hashlib, importlib.util
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT))
os.environ['NUMBA_CACHE_DIR']=str(HERE/'numba_cache')
import numpy as np
import pandas as pd
from calibration.seasonal_septoria.field_data import prepare_fields, _leaf_index

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    checks=[]
    def check(name,condition):
        if not bool(condition):raise AssertionError(name)
        checks.append(name)
    hashes=json.loads((HERE/'review_source_hashes.json').read_text())
    old_path=HERE/'source_snapshot/model/seasonal_septoria/field_data.py'
    check('original_review_code_preserved',sha(old_path)==hashes['source_hashes']['model/seasonal_septoria/field_data.py'])
    spec=importlib.util.spec_from_file_location('model.seasonal_septoria.original_review_field_data',old_path)
    old=importlib.util.module_from_spec(spec);sys.modules[spec.name]=old;spec.loader.exec_module(old);old.ROOT=ROOT
    for label in ['F10','F14','LEAF, 14','LEAF, 14TH']:
        check('reject_multidigit_'+label,_leaf_index(label) is None and old._leaf_index(label)==0)
    for i in range(1,8):check('retain_F'+str(i),_leaf_index('F'+str(i))==i-1)
    a=pd.DataFrame(dict(dataset_id=['review']*2,physical_unit=['a']*2,site_id=['loc']*2,
        season_year=[2019]*2,country=['Germany']*2,latitude=[50.]*2,longitude=[10.]*2,
        organ=['F1','F2'],date=['2019-01-09','2019-01-10'],value=[1.,2.],
        metric=['infection_percent_unspecified_basis']*2,unit=['percent']*2,endpoint_series=['a1','a2']))
    c=pd.DataFrame(dict(dataset_id=['review'],point_id=['loc'],crop_season=['winter_wheat'],
        water_system=['rainfed'],calendar_valid=[True],planting_doy=[1.],maturity_doy=[200.]))
    w=pd.DataFrame(dict(location_id=['loc']*10,date=pd.date_range('2019-01-01',periods=10),
        tmean_c=[18.]*10,tmax_c=[23.]*10,rh_mean_pct=[90.]*10,precipitation_mm=[2.]*10))
    for metric,unit in [('lesion_count','count'),('pycnidia_count','count'),
                        ('infection_percent_unspecified_basis','proportion'),('unknown','percent')]:
        changed=a.copy();changed.loc[1,['metric','unit']]=[metric,unit]
        current=prepare_fields(changed,c,w);previous=old.prepare_fields(changed,c,w)
        check('reject_incompatible_'+metric+'_'+unit,len(current.targets)==1 and
              current.excluded.reason.tolist()==['unsupported_metric_or_unit'] and len(previous.targets)==2)
    try:prepare_fields(a.drop(columns='unit'),c,w);rejected=False
    except ValueError as e:rejected='units' in str(e)
    check('require_explicit_measurement_unit',rejected)
    archive=ROOT/'analysis/paper_study/seasonal_calibration_v1'
    source=pd.read_csv(archive/'basf_source_assessments_snapshot.csv')
    calendars=pd.read_csv(ROOT/'data/paper_study/wheat_area/trial_point_calendar_scenarios.csv')
    weather=pd.read_parquet(ROOT/'data/paper_study/field_weather/daily_weather.parquet')
    before=old.prepare_fields(source,calendars,weather);after=prepare_fields(source,calendars,weather)
    for key in ['temperature','humidity','rain','host_active','host_renewal']:
        check('exact_BASF_input_parity_'+key,np.array_equal(getattr(before,key),getattr(after,key),equal_nan=True))
    for key in ['metadata','targets','excluded']:
        pd.testing.assert_frame_equal(getattr(before,key),getattr(after,key));check('exact_BASF_table_parity_'+key,True)
    archive_changes={name:dict(before=value,after=sha(ROOT/name)) for name,value in hashes['prior_archive_hashes'].items() if sha(ROOT/name)!=value}
    check('all_34_frozen_archive_files_unchanged',not archive_changes)
    changes={name:dict(before=value,after=sha(ROOT/name)) for name,value in hashes['source_hashes'].items() if sha(ROOT/name)!=value}
    for name in ['core.py','host.py','endpoints.py','calibrate.py']:
        check('unchanged_numerical_model_'+name,'model/seasonal_septoria/'+name not in changes)
    receipt=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),
        scope='Retrospective corrected-input verification; no refitting or external-outcome evaluation',
        checks_passed=len(checks),checks=checks,source_changes=changes,prior_archive_changes=archive_changes,
        basf_fields=len(after.metadata),basf_targets=len(after.targets),external_outcomes_read=False,
        original_review_receipt_preserved=True,script_sha256=sha(Path(__file__)))
    (HERE/'corrected_inputs_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['status','checks_passed','basf_fields','basf_targets','prior_archive_changes']},indent=2))

if __name__=='__main__':main()
