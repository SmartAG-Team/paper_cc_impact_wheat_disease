"""Check hourly precipitation timing against official 06–06 daily totals."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .inputs import sha
from .meteo_france_station import ROOT,CACHE,OUTPUT


def run(root=ROOT,output=None):
    root = Path(root)
    cache,out = root/CACHE,Path(output or root/OUTPUT)
    if (out/'timing_receipt.json').exists():
        raise FileExistsError('Use a new output directory to preserve the archived timing check')
    # Thresholds concern source timing, never disease errors or model scores.
    policy = dict(recorded_utc=datetime.now(timezone.utc).isoformat(),
        expected_shift_hours=7,alternative_shift_hours=[6,8],minimum_days=100,
        minimum_wet_days=25,agreement_tolerance_mm=.2,minimum_agreement=.98,
        validated_quality_codes=[0,1],disease_outcomes_used=False,
        definition='Provider daily RR at date J covers06UTC J through06UTC J+1; ending-hour RR1 reports07J through06J+1 should reproduce it')
    policy_path = out/'timing_policy.json'
    if policy_path.exists():
        old = json.loads(policy_path.read_text())
        for key,value in policy.items():
            if key != 'recorded_utc':
                assert old[key] == value
    else:
        policy_path.write_text(json.dumps(policy,indent=2)+'\n')
    candidates = pd.read_csv(out/'station_candidates.csv',dtype={'station_id':str})
    station_ids = set(candidates.station_id)
    rows,receipts = [],[]
    resources = json.loads((cache/'selected_daily_resources.json').read_text())
    for resource in resources:
        path = cache/(resource['title']+'.csv.gz')
        receipt = json.loads(path.with_suffix(path.suffix+'.receipt.json').read_text())
        assert receipt['sha256'] == sha(path)
        receipts.append(receipt)
        for block in pd.read_csv(path,sep=';',usecols=['NUM_POSTE','AAAAMMJJ','RR','QRR'],
                                 dtype={'NUM_POSTE':str,'AAAAMMJJ':str},chunksize=200000):
            use = block.NUM_POSTE.isin(station_ids)&block.AAAAMMJJ.between('20130101','20191231')
            if use.any():
                rows.append(block.loc[use])
    daily = pd.concat(rows,ignore_index=True)
    assert not daily.duplicated(['NUM_POSTE','AAAAMMJJ']).any()
    daily['date'] = pd.to_datetime(daily.AAAAMMJJ,format='%Y%m%d')
    daily['RR'] = daily.RR.where(daily.RR.between(0,2000)&daily.QRR.isin([0,1]))
    daily.to_parquet(out/'provider_daily_totals.parquet',index=False)
    comparisons,metrics = [],[]
    for station in sorted(station_ids):
        hourly = pd.read_parquet(out/f'{station}_hourly.parquet',columns=['time','RR1'])
        observed = daily.loc[daily.NUM_POSTE.eq(station),['date','RR']]
        for shift in [7,6,8]:
            key = (hourly.time-pd.Timedelta(hours=shift)).dt.normalize()
            grouped = hourly.RR1.groupby(key)
            reconstructed = grouped.sum(min_count=24).rename('reconstructed_mm').reset_index().rename(columns={'time':'date'})
            count = grouped.count()
            reconstructed.loc[reconstructed.date.map(count).ne(24),'reconstructed_mm'] = np.nan
            pair = observed.merge(reconstructed,on='date',validate='one_to_one').dropna()
            if not len(pair):
                continue
            error = pair.reconstructed_mm-pair.RR
            wet = pair.RR.gt(0)|pair.reconstructed_mm.gt(0)
            row = dict(station_id=station,shift_hours=shift,n=len(pair),wet_days=int(wet.sum()),
                bias=float(error.mean()),mae=float(abs(error).mean()),rmse=float(np.sqrt(np.mean(error**2))),
                agreement=float(error.abs().le(.200000001).mean()),
                wet_agreement=float(error.loc[wet].abs().le(.200000001).mean()))
            metrics.append(row)
            pair['station_id'] = station;pair['shift_hours'] = shift
            comparisons.append(pair)
    results = pd.DataFrame(metrics)
    pd.concat(comparisons,ignore_index=True).to_parquet(out/'timing_pairs.parquet',index=False)
    approved = []
    for station,group in results.groupby('station_id'):
        expected = group.loc[group.shift_hours.eq(7)]
        if not len(expected):
            continue
        row = expected.iloc[0]
        alternatives = group.loc[group.shift_hours.ne(7)]
        accepted = bool(row.n >= 100 and row.wet_days >= 25 and row.agreement >= .98
            and row.wet_agreement >= .98 and (row.mae < alternatives.mae).all())
        approved.append(dict(station_id=station,timing_approved=accepted,
            agreement=row.agreement,wet_agreement=row.wet_agreement,n=int(row.n),wet_days=int(row.wet_days)))
    results.to_csv(out/'timing_metrics.csv',index=False)
    approval = pd.DataFrame(approved)
    approval.to_csv(out/'timing_approval.csv',index=False)
    result = dict(source_sha256={str(p.relative_to(root)):sha(p) for p in [cache/'daily_catalog.json',
        cache/'daily_documentation.pdf',cache/'selected_daily_resources.json',out/'timing_policy.json']},
        archives=receipts,stations_tested=len(approval),stations_approved=int(approval.timing_approved.sum()),
        code_sha256=sha(Path(__file__)),requires_approved_station_for_rainfall_substitution=True)
    (out/'timing_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path)
    args = parser.parse_args()
    print(json.dumps(run(output=args.output),indent=2))
