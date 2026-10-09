"""Structural intake audit, not physiological validation. Exit 2 = no observations.

Run: python3 audit_inputs.py [input_directory]
The audit output is always written within the owned artifact directory.
"""
import collections
import csv
import datetime as dt
import json
import math
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parent


def audit(directory):
    schema=json.loads((ROOT/'schema.json').read_text())
    definitions=list(csv.DictReader((ROOT/'data_dictionary.csv').open()))
    types={(r['table'],r['column']):r['data_type'] for r in definitions}
    tables={};errors=[];profiles={}
    for name,spec in schema.items():
        p=directory/(name+'.csv')
        if not p.exists():
            tables[name]=[]
            if not spec['optional_table']:errors.append(f'{name}: missing table')
            continue
        with p.open(newline='') as f:
            reader=csv.DictReader(f);rows=list(reader)
            if reader.fieldnames != spec['columns']:errors.append(f'{name}: header mismatch')
        tables[name]=rows
        keys=[tuple(r.get(c,'') for c in spec['primary_key']) for r in rows]
        duplicates=len(keys)-len(set(keys))
        missing=sum(any(not r.get(c,'').strip() for c in spec['required_columns']) for r in rows)
        profiles[name]={'rows':len(rows),'duplicate_key_rows':duplicates,'rows_missing_required_values':missing}
        if duplicates:errors.append(f'{name}: {duplicates} duplicate key rows')
        if missing:errors.append(f'{name}: {missing} rows missing required values')
        for rownum,r in enumerate(rows,2):
            for c in spec['columns']:
                value=r.get(c,'')
                if not value:continue
                try:
                    datatype=types[name,c]
                    if datatype=='date':dt.date.fromisoformat(value)
                    elif datatype in ['number','integer']:
                        number=float(value)
                        if not math.isfinite(number):raise ValueError()
                        if datatype=='integer' and number!=int(number):raise ValueError()
                        if c not in ['latitude_deg','longitude_deg','tmin_degC','tmax_degC'] and number<0:raise ValueError()
                        if c in ['severity_pct','grain_moisture_pct'] and not 0<=number<=100:raise ValueError()
                        if c=='leaf_rank_from_flag' and number<1:raise ValueError()
                except (ValueError,TypeError):errors.append(f'{name}:{rownum}:{c}: invalid {types[name,c]} or range')
    # All joins below are many-to-one to a unique parent key; duplicate parents
    # above invalidate the input rather than expanding rows silently.
    joins=[]
    def link(child,child_key,parent,parent_key):
        parent_ids={r.get(parent_key,'') for r in tables[parent]}
        unmatched=sum(r.get(child_key,'') not in parent_ids for r in tables[child])
        n=len(tables[child])
        joins.append(dict(child=child,child_key=child_key,parent=parent,parent_key=parent_key,
                          child_rows=n,unmatched_rows=unmatched,match_rate=(n-unmatched)/n if n else None))
        if unmatched:errors.append(f'{child}.{child_key}: {unmatched} unmatched rows in {parent}')
    link('plots','environment_id','environments','environment_id')
    link('plots','treatment_id','treatments','treatment_id')
    for name in ['canopy','disease','stages','harvest','management','biomass_reserves']:
        link(name,'plot_id','plots','plot_id')
    link('reference_pairs','affected_plot_id','plots','plot_id')
    link('reference_pairs','reference_plot_id','plots','plot_id')
    for name in tables:
        if name!='sources':link(name,'source_id','sources','source_id')
    for row in tables['sources']:
        if row.get('evidence_type')!='field_plot_observation':errors.append('sources: non-observational input excluded')
        digest=row.get('source_file_sha256','')
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest.lower()):
            errors.append('sources: invalid SHA-256')
    for row in tables['disease']:
        if not row.get('severity_pct') and not row.get('symptom_area_index_m2_m2'):
            errors.append('disease: neither severity nor symptom area measured')
    for row in tables['harvest']:
        if row.get('grain_yield_unit') not in ['kg_ha','t_ha','dt_ha','g_m2']:
            errors.append('harvest: unsupported grain unit')
    for row in tables['stages']:
        if row.get('stage_scale') not in ['BBCH','Zadoks']:errors.append('stages: unresolved stage scale')
        if row.get('date_status') not in ['observed','interpolated']:errors.append('stages: invalid date status')
    plots={r.get('plot_id'):r for r in tables['plots']}
    harvest={r.get('plot_id'):r for r in tables['harvest']}
    canopy_dates=collections.defaultdict(set)
    for row in tables['canopy']:
        canopy_dates[row.get('plot_id'),row.get('leaf_rank_from_flag')].add(row.get('observation_date'))
        if not row.get('green_lai_m2_m2') and not row.get('functional_lai_m2_m2'):
            errors.append('canopy: neither green nor functional area measured')
        try:
            if row.get('total_lai_m2_m2') and float(row['green_lai_m2_m2'])>float(row['total_lai_m2_m2']):
                errors.append('canopy: green area exceeds total area')
        except (ValueError,KeyError):pass
    for name in ['canopy','disease']:
        for row in tables[name]:
            h=harvest.get(row.get('plot_id'),{})
            if h.get('harvest_date') and row.get('observation_date','')>h['harvest_date']:
                errors.append(f'{name}: post-harvest observation')
    for row in tables['reference_pairs']:
        a=plots.get(row.get('affected_plot_id'),{});b=plots.get(row.get('reference_plot_id'),{})
        if row.get('affected_plot_id')==row.get('reference_plot_id'):errors.append('reference_pairs: self-reference')
        if a and b and (a.get('environment_id'),a.get('cultivar_id'))!=(b.get('environment_id'),b.get('cultivar_id')):
            errors.append('reference_pairs: environment/cultivar mismatch')
    canopy_plot_ids={k[0] for k in canopy_dates}
    eligible_time_series=sum(len(dates)>=2 for dates in canopy_dates.values())
    groups={r.get('independent_environment_group') for r in tables['environments'] if r.get('independent_environment_group')}
    no_data=not any(tables[n] for n in ['plots','canopy','disease','harvest'])
    if not no_data:
        for name,spec in schema.items():
            if not spec['optional_table'] and not tables[name]:errors.append(f'{name}: required observation/design table is empty')
        if len(groups)<2:errors.append('environments: fewer than two independent groups')
        if not eligible_time_series:errors.append('canopy: no repeated same-plot same-leaf dates')
    status='no_observations' if no_data else ('structural_errors' if errors else 'structurally_screened_pending_scientific_review')
    result=dict(status=status,input_directory=str(directory.resolve()),table_profiles=profiles,
                join_checks=joins,errors=errors,independent_environment_groups=len(groups),
                repeated_plot_leaf_series=eligible_time_series,
                unique_canopy_plots=len(canopy_plot_ids),canopy_plots_with_harvest=len(canopy_plot_ids & set(harvest)),
                physiological_comparison_ready=False,
                unautomated_requirements=['Observed stage-window coverage and temporal gaps',
                    'Measured disease in reference plots and source-supported attribution',
                    'Weather completeness/units and soil/initial-state inputs for the chosen crop model',
                    'Actual leaf-area normalization, layer completeness and green versus functional area',
                    'Reserve parameter identifiability and independent calibration/evaluation allocation',
                    'Parent coordination before expansive modeling'],
                inference='No observations is not a passed join audit. Structural checks cannot establish physiological validity.')
    (ROOT/'input_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'status':status,'tables':len(profiles),'observation_rows':sum(len(tables[n]) for n in ['canopy','disease','harvest']),
                      'physiological_comparison_ready':False,'errors':len(errors)}))
    return 2 if no_data else (1 if errors else 0)


if __name__=='__main__':
    sys.exit(audit(Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/'templates'))
