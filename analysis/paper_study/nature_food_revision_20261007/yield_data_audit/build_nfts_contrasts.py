"""Arithmetic treatment contrasts from audited public treatment means."""
from pathlib import Path
import json
import pandas as pd
from bs4 import BeautifulSoup
from extract_nfts_means import grid, number

ROOT=Path(__file__).resolve().parent

def main():
    y=pd.read_csv(ROOT/'nfts_yield_treatment_means.csv',dtype={'cultivar_entry':str,'treatment_code':str}).fillna({'treatment_factor':''})
    d=pd.read_csv(ROOT/'nfts_disease_observations_long.csv',dtype={'cultivar_entry':str,'treatment_code':str}).fillna({'treatment_factor':''})
    metadata=json.loads((ROOT/'nfts_trial_metadata.json').read_text())
    result=[]
    catalogue=[]
    checks=[]
    for m in metadata:
        bad=any(v[0].startswith('Not OK') for v in m['validation'])
        yy=y[(y.record==m['record']) & y.value.notna()]
        dd=d[(d.record==m['record']) & d.value.notna()]
        year=int(pd.to_datetime(yy.date).dt.year.mode().iloc[0]) if len(yy) else None
        if len(yy) and (ROOT/f"nfts_native_{m['registry_id']}_sv.html").exists():
            soup=BeautifulSoup((ROOT/f"nfts_native_{m['registry_id']}_sv.html").read_text(),'html.parser')
            native={}
            for t in soup.find_all('table'):
                if not (t.get('id') or '').startswith('tabResultatLedNieveau'):continue
                g=grid(t)
                if len(g)<4 or 'Skörd dt/ha kärna 15%' not in g[2]:continue
                ci=g[2].index('Skörd dt/ha kärna 15%')
                for r in g[3:]:
                    key=(r[0]+r[1]) if ci==2 else r[0]
                    if number(r[ci]) is not None:native[key]=number(r[ci])
            for _,row in yy.iterrows():
                nv=native.get(row.treatment_code)
                checks.append({'record':m['record'],'treatment_code':row.treatment_code,
                    'english_value':row.value,'native_swedish_value':nv,
                    'matches':nv is not None and abs(nv-row.value)<1e-9})
        local=[]
        if len(yy) and (yy.treatment_factor=='A').any():
            aa=yy[yy.treatment_factor=='A'];bb=yy[yy.treatment_factor=='B']
            pairs=aa.merge(bb,on=['record','cultivar_entry'],suffixes=('_untreated','_treated'),validate='one_to_one')
            for _,r in pairs.iterrows():
                local.append(dict(record=m['record'],registry_id=m['registry_id'],trial_name=m['trial_name'],
                    year=year,latitude=m['latitude'],longitude=m['longitude'],replicates=m['replicates'],
                    cultivar_entry=r.cultivar_entry,cultivar=r.cultivar_untreated,
                    cultivar_identity_known=pd.notna(r.cultivar_untreated),
                    untreated_treatment_code=r.treatment_code_untreated,
                    treated_treatment_code=r.treatment_code_treated,
                    untreated_yield_native=r.value_untreated,treated_yield_native=r.value_treated,
                    untreated_yield_t_ha=r.yield_t_ha_15percent_moisture_untreated,
                    treated_yield_t_ha=r.yield_t_ha_15percent_moisture_treated,
                    untreated_lower_confidence_native=r.lower_confidence_limit_untreated,
                    untreated_upper_confidence_native=r.upper_confidence_limit_untreated,
                    treated_lower_confidence_native=r.lower_confidence_limit_treated,
                    treated_upper_confidence_native=r.upper_confidence_limit_treated,
                    treated_definition='Registry factor B: disease control (or stated fungicide schedule); not assumed disease-free.',
                    comparison_design='Within-trial cultivar entry under A untreated and B disease control, two-factor design.',
                    registry_quality_eligible=not bad,source_html=m['source_html'],source_url=m['source_url']))
        elif len(yy):
            untreated=yy[yy.treatment_code=='1']
            if len(untreated)!=1:
                raise ValueError(f"Unresolved untreated label in {m['record']}")
            u=untreated.iloc[0]
            for _,r in yy[yy.treatment_code!='1'].iterrows():
                local.append(dict(record=m['record'],registry_id=m['registry_id'],trial_name=m['trial_name'],
                    year=year,latitude=m['latitude'],longitude=m['longitude'],replicates=m['replicates'],
                    cultivar_entry=r.cultivar_entry,cultivar=r.cultivar,cultivar_identity_known=pd.notna(r.cultivar),
                    untreated_treatment_code='1',treated_treatment_code=r.treatment_code,
                    untreated_yield_native=u.value,treated_yield_native=r.value,
                    untreated_yield_t_ha=u.yield_t_ha_15percent_moisture,
                    treated_yield_t_ha=r.yield_t_ha_15percent_moisture,
                    untreated_lower_confidence_native=u.lower_confidence_limit,
                    untreated_upper_confidence_native=u.upper_confidence_limit,
                    treated_lower_confidence_native=r.lower_confidence_limit,
                    treated_upper_confidence_native=r.upper_confidence_limit,
                    treated_definition='Specified trial fungicide schedule; every treated comparator retained; complete protection not assumed.',
                    comparison_design='Randomized within-trial fungicide treatment vs untreated code1; comparators share one control.',
                    registry_quality_eligible=not bad,source_html=m['source_html'],source_url=m['source_url']))
        for r in local:
            r['fungicide_response_t_ha']=r['treated_yield_t_ha']-r['untreated_yield_t_ha']
            r['untreated_shortfall_relative_to_treated']=1-r['untreated_yield_native']/r['treated_yield_native']
            r['loss_interpretation']='Observed fungicide-associated yield difference; STB-only or disease-free counterfactual not identified.'
        result.extend(local)
        catalogue.append(dict(record=m['record'],registry_id=m['registry_id'],trial_name=m['trial_name'],year=year,
            registry_quality_eligible=not bad,validation=json.dumps(m['validation'],ensure_ascii=False),
            numeric_yield_means=len(yy),complete_yield_contrasts=len(local),
            disease_numeric_cells=len(dd),negative_contrasts=sum(x['fungicide_response_t_ha']<0 for x in local),
            source_html=m['source_html'],source_url=m['source_url'],latitude=m['latitude'],longitude=m['longitude'],
            replicates=m['replicates'],disease_measurements=json.dumps(sorted(dd.measurement.unique())),
            disease_date_scope=json.dumps(sorted(dd.date.dropna().unique())),
            source_statistical_notes=json.dumps(m['statistical_notes'],ensure_ascii=False)))
    contrasts=pd.DataFrame(result)
    contrasts.to_csv(ROOT/'nfts_fungicide_yield_contrasts_all.csv',index=False)
    eligible=contrasts[contrasts.registry_quality_eligible]
    eligible.to_csv(ROOT/'nfts_fungicide_yield_contrasts_eligible.csv',index=False)
    pd.DataFrame(catalogue).to_csv(ROOT/'nfts_trial_candidate_catalogue.csv',index=False)
    matched=[]
    for _,r in contrasts.iterrows():
        dx=d[(d.record==r.record)&d.treatment_code.isin([r.untreated_treatment_code,r.treated_treatment_code])&d.value.notna()]
        # Only same cultivar in factor trials; for one-factor trials code denotes the fungicide schedule.
        for _,obs in dx.iterrows():
            native=obs.get('native_swedish_measurement_label','')
            native=native if isinstance(native,str) else ''
            if 'Svartpricksjuka' in native:
                disease='STB (native Swedish Svartpricksjuka)'
            elif 'Septoria' in obs.measurement:
                disease='Septoria (registry label; species not expanded)'
            else:
                disease=obs.measurement
            leaf='unspecified'
            for n in range(1,8):
                if f'{n}nd leaf' in obs.measurement or f'{n}rd leaf' in obs.measurement or f'bladnivå {n}' in obs.measurement:
                    leaf=f'leaf{n} counted from top'
            bad_date=(pd.to_datetime(obs.date).year not in [r.year-1,r.year]) if pd.notna(obs.date) else None
            matched.append(dict(record=r.record,registry_id=r.registry_id,cultivar_entry=r.cultivar_entry,
                untreated_treatment_code=r.untreated_treatment_code,treated_treatment_code=r.treated_treatment_code,
                observed_treatment_code=obs.treatment_code,observed_on='untreated' if obs.treatment_code==r.untreated_treatment_code else 'treated',
                source_disease_label=obs.measurement,native_swedish_label=native,disease=disease,
                reported_leaf_scope=leaf,date=obs.date,growth_stage=obs.growth_stage,disease_value=obs.value,
                native_yield_t_ha_untreated=r.untreated_yield_t_ha,native_yield_t_ha_treated=r.treated_yield_t_ha,
                fungicide_response_t_ha=r.fungicide_response_t_ha,relative_shortfall=r.untreated_shortfall_relative_to_treated,
                registry_quality_eligible=r.registry_quality_eligible,source_date_outside_trial_crop_year=bad_date,
                source_html=obs.source_html,source_table=obs.source_table,source_row=obs.source_row,source_column=obs.source_column))
    pd.DataFrame(matched).to_csv(ROOT/'nfts_contrasts_with_observed_disease_long.csv',index=False)
    check=pd.DataFrame(checks)
    check.to_csv(ROOT/'nfts_native_english_yield_value_checks.csv',index=False)
    if not check.matches.all():
        raise ValueError('English vs native-language yield values do not agree')
    summary=dict(candidate_trials=len(metadata),trials_with_numeric_yield=contrasts.record.nunique(),
        quality_eligible_trials=eligible.record.nunique(),candidate_complete_contrasts=len(contrasts),
        quality_eligible_complete_contrasts=len(eligible),
        quality_eligible_cultivar_two_factor_contrasts=int(eligible.comparison_design.str.contains('two-factor').sum()),
        quality_eligible_single_factor_fungicide_contrasts=int((~eligible.comparison_design.str.contains('two-factor')).sum()),
        quality_eligible_negative_contrasts=int((eligible.fungicide_response_t_ha<0).sum()),
        quality_eligible_named_cultivar_contrasts=int(eligible.cultivar_identity_known.sum()),
        native_english_nonmissing_yield_comparisons=len(check),all_native_english_values_match=bool(check.matches.all()),
        yield_unit='t/ha at15% moisture; source dt/ha×0.1, verified in each native Swedish record.',
        replication_unit='Trial site-year for holdout; treatment/cultivar contrasts within one trial are not independent environments.',
        healthy_area_duration_available=False)
    (ROOT/'nfts_extraction_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
