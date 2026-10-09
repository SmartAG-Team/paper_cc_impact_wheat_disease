"""Fail-closed structural screening of future inputs; never certifies physiology.

Usage: python -B validate_contract.py /path/to/input_directory
Only stdout is written. Exit 0 means structural eligibility, not validated science.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CONTRACT = json.loads((HERE/'input_contract.json').read_text())


def validate_tables(tables):
    errors = []

    def fail(condition,message):
        if condition:
            errors.append(message)

    def result():
        return dict(input_ready=not errors,readiness_scope='necessary structural eligibility only',
            mechanistic_validation_established=False,causal_grain_loss_established=False,
            adaptation_benefit_established=False,errors=errors,
            manual_review_required=CONTRACT['manual_review_required'])

    for name,spec in CONTRACT['tables'].items():
        if name not in tables:
            errors.append('Missing table: '+name)
            continue
        d = tables[name]
        missing = set(spec['required'])-set(d.columns)
        fail(bool(missing),name+': missing columns '+','.join(sorted(missing)))
        fail(d.empty,name+': empty table')
        if not missing:
            absent = d[spec['required']].isna() | d[spec['required']].astype(str).apply(lambda s:s.str.strip().eq(''))
            fail(absent.any().any(),name+': missing required values')
    if errors:
        return result()
    t = {name:d.copy() for name,d in tables.items()}
    date_fields = {'plots':['sowing_date','anthesis_date','maturity_date','harvest_date'],
                   'canopy':['date'],'weather':['date'],'management':['date']}
    for name,fields in date_fields.items():
        for field in fields:
            source = t[name][field].astype(str)
            parsed = pd.to_datetime(source,format='%Y-%m-%d',errors='coerce')
            fail(parsed.isna().any() or not source.str.fullmatch(r'\d{4}-\d{2}-\d{2}').all(),name+': invalid ISO date '+field)
            t[name][field] = parsed
    numeric = {'plots':{'harvest_year':(1900,2200),'yield_t_ha':(0,None),'grain_moisture_percent':(0,100)},
        'canopy':{'leaf_rank':(1,3),'reference_lai':(0,None),'green_lai':(0,None),
                  'functional_loss_fraction':(0,1),'stb_severity_percent':(0,100)},
        'weather':{'tmin_c':(None,None),'tmax_c':(None,None),'precip_mm':(0,None),
                   'solar_mj_m2':(0,None),'relative_humidity_percent':(0,100)},
        'management':{'amount':(0,None)}}
    for name,fields in numeric.items():
        for field,(low,high) in fields.items():
            s = pd.to_numeric(t[name][field],errors='coerce')
            fail(not np.isfinite(s.to_numpy(float)).all(),name+': nonfinite '+field)
            fail(low is not None and s.lt(low).any(),name+': below range '+field)
            fail(high is not None and s.gt(high).any(),name+': above range '+field)
            t[name][field] = s
    if errors:
        return result()
    p,c,w,m = (t[n] for n in ['plots','canopy','weather','management'])
    fail(p.plot_id.duplicated().any(),'Duplicate physical plot-season ID')
    fail(p.duplicated(['source_ref','source_plot_id','harvest_year']).any(),'Duplicate source harvest alias')
    fail(not p.harvest_year.eq(p.harvest_date.dt.year).all(),'Harvest year/date mismatch')
    fail(not ((p.sowing_date<=p.anthesis_date)&(p.anthesis_date<p.maturity_date)&(p.maturity_date<=p.harvest_date)).all(),
         'Invalid phenological chronology')
    fail(not p.split.isin(['calibration','validation']).all(),'Unknown split label')
    fail(p.groupby('environment_id').split.nunique().gt(1).any(),'Environment leakage between calibration and validation')
    fail(p.groupby(['site_id','harvest_year']).split.nunique().gt(1).any(),'Site-year leakage through environment aliases')
    fail(p.groupby('environment_id')[['site_id','harvest_year']].nunique().gt(1).any().any(),'Environment identity conflicts')
    fail(p[p.split.eq('calibration')].empty,'No calibration environment')
    fail(len(p[p.split.eq('validation')][['site_id','harvest_year']].drop_duplicates())<2,'Fewer than two validation environments')
    fail(not p.measurement_origin.eq('observed').all(),'Harvest targets must be observed, not model-derived')
    fail(not c.measurement_origin.eq('observed').all(),'Canopy/function/severity must be independently observed')
    fail(not c.leaf_rank.isin([1,2,3]).all(),'Invalid final-leaf rank')
    fail(not c.lai_unit.eq('m2/m2').all(),'LAI unit must be m2/m2')
    fail(not c.severity_unit.eq('percent_leaf_area').all(),'Ordinal scores cannot substitute for percent leaf area')
    fail(c.duplicated(['plot_id','date','leaf_rank']).any(),'Duplicate canopy observation key')
    fail(w.duplicated(['environment_id','date']).any(),'Duplicate daily weather key')
    fail((w.tmin_c>w.tmax_c).any(),'Minimum temperature exceeds maximum')
    for name,d in [('canopy',c),('management',m)]:
        fail(not d.plot_id.isin(p.plot_id).all(),name+': orphan plot ID')
    fail(not w.environment_id.isin(p.environment_id).all(),'Weather: orphan environment ID')
    # Ambiguous keys and malformed identifiers prevent downstream joins.
    if errors:
        return result()
    for plot in p.itertuples():
        observed = c[c.plot_id.eq(plot.plot_id)]
        for rank in [1,2,3]:
            dates = observed.loc[observed.leaf_rank.eq(rank),'date']
            fail(dates.nunique()<3,f'{plot.plot_id}: fewer than three dated canopy assessments on rank {rank}')
            fail(dates.empty or dates.min()>plot.anthesis_date or dates.max()<plot.maturity_date,
                 f'{plot.plot_id}: canopy dates do not bracket grain filling on rank {rank}')
        fail((observed.date>plot.harvest_date).any() or (observed.date<plot.sowing_date).any(),
             f'{plot.plot_id}: canopy outside observed season')
        events = m[m.plot_id.eq(plot.plot_id)]
        fail(events.empty,f'{plot.plot_id}: missing management records')
        fail((events.date>plot.harvest_date).any(),f'{plot.plot_id}: management after harvest')
    for env,plots in p.groupby('environment_id'):
        needed = pd.date_range(plots.sowing_date.min(),plots.maturity_date.max())
        dates = w.loc[w.environment_id.eq(env),'date']
        fail(len(needed.difference(dates))>0,str(env)+': incomplete daily weather coverage')
    return result()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory',type=Path)
    args = parser.parse_args()
    tables = {name:pd.read_csv(args.directory/spec['file']) for name,spec in CONTRACT['tables'].items()
              if (args.directory/spec['file']).exists()}
    report = validate_tables(tables)
    print(json.dumps(report,indent=2,allow_nan=False))
    raise SystemExit(0 if report['input_ready'] else 2)


if __name__ == '__main__':
    main()
