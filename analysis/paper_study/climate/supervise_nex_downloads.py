"""Finish authorized public weather downloads with bounded retries and fresh audits."""
from datetime import datetime,timezone
from pathlib import Path
import argparse,json,os,shutil,subprocess,sys,time
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/climate'
STATE=HERE/'nex_download_supervision.json'

def now():return datetime.now(timezone.utc).isoformat()
def state(value):
    value={'updated_utc':now(),'supervisor_pid':os.getpid(),**value}
    temp=STATE.with_suffix('.tmp.json');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(STATE)
def counts():
    return {'era5_months':len([p for p in (DATA/'era5_daily').glob('*.parquet') if p.with_suffix('.json').exists()]),
            'nex_annual_variables':len([p for p in (DATA/'nasa_v2/variables').rglob('*.parquet') if p.with_suffix('.json').exists()]),
            'nex_merged_years':len([p for p in (DATA/'nasa_v2/daily').rglob('*.parquet') if p.with_suffix('.json').exists()]),
            'free_disk_bytes':shutil.disk_usage(ROOT).free}
def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--workers',type=int,default=24);parser.add_argument('--rounds',type=int,default=6)
    args=parser.parse_args();log=HERE/'nasa_full_download.log'
    for iteration in range(1,args.rounds+1):
        if shutil.disk_usage(ROOT).free<15_000_000_000:
            state({'status':'requires_disk_space','round':iteration,'workers':args.workers,'counts':counts()});raise SystemExit(2)
        with log.open('a') as output:
            output.write(json.dumps({'source':'nex_supervisor','status':'round_start','round':iteration,'workers':args.workers,'utc':now()})+'\n');output.flush()
            process=subprocess.Popen([sys.executable,str(HERE/'retrieve_climate.py'),'nasa','--workers',str(args.workers),'--fields','tmean_c','tmax_c','precipitation_mm','rh_mean_pct'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            while process.poll() is None:
                state({'status':'downloading','round':iteration,'workers':args.workers,'download_pid':process.pid,'counts':counts(),'coverage_status':'Incomplete until fresh full audit passes'})
                time.sleep(30)
            code=process.returncode
        with (HERE/'supervisor_audit.log').open('a') as output:
            audit=subprocess.run([sys.executable,str(HERE/'verify_downloads.py')],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
        if audit.returncode==0:
            with (HERE/'nex_full_deep_validation.log').open('w') as output:
                deep=subprocess.run([sys.executable,str(HERE/'verify_downloads.py'),'--deep','--source','nasa'],cwd=ROOT,stdout=output,stderr=subprocess.STDOUT)
            if deep.returncode==0:
                state({'status':'complete','round':iteration,'workers':args.workers,'counts':counts(),'download_returncode':code,'full_audit_returncode':0,'deep_nex_audit_returncode':0,'authoritative_audit':'analysis/paper_study/climate/actual_coverage_validation.json','deep_nex_audit':'analysis/paper_study/climate/actual_coverage_validation_nasa.json','completed_utc':now()});return
        state({'status':'retry_pending','round':iteration,'workers':args.workers,'counts':counts(),'download_returncode':code,'audit_returncode':audit.returncode})
        time.sleep(min(15*iteration,60))
    state({'status':'incomplete_after_bounded_retries','workers':args.workers,'counts':counts(),'rounds':args.rounds})
    raise SystemExit(1)
if __name__=='__main__':main()
