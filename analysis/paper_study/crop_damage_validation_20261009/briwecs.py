"""Source-harmonized German protection-response associations, without HAD inference."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .run import HERE,COLLECTION,ROOT,sha,grouped_predictions,fit_predict,score

SOURCE=COLLECTION/'sources/public_septoria/briwecs-2025/extracted/Briwecs_data'
DISEASES=['Stripe_rust','Powdery_mildew','Leaf_rust','DTR','Fusarium']


def prepare():
    frames=[];invalid=0
    for path in sorted((SOURCE/'data/locations').glob('*.csv')):
        try:content=path.read_text(encoding='utf-8-sig');encoding='utf-8-sig'
        except UnicodeDecodeError:content=path.read_text(encoding='cp1252');encoding='cp1252'
        sep=';' if ';' in content.splitlines()[0] else ','
        frame=pd.read_csv(path,sep=sep,encoding=encoding)
        if not {'Septoria','Seedyield','BRISONr','Treatment','Location','Year'}<=set(frame):continue
        numeric=pd.to_numeric(frame.Septoria,errors='coerce')
        invalid+=int((frame.Septoria.notna()&numeric.isna()).sum())
        frame['Septoria']=numeric
        for col in ['Seedyield','BBCH59','BBCH87',*DISEASES]:
            frame[col]=pd.to_numeric(frame[col],errors='coerce') if col in frame else np.nan
        assert frame.Septoria.dropna().between(0,100).all()
        frame['source_file']=path.name
        frame['source_row']=np.arange(len(frame))+2
        # Source-author treatment correction is reproduced literally. GGE
        # unsuffixed plots in these years are irrigated; D/D D plots are rainfed.
        d=frame.Treatment.str.endswith(('_D','_DD'))
        irrigated=frame.Location.eq('GGE')&frame.Year.isin([2015,2018,2019])&~d
        frame['water_regime']=np.where(irrigated,'IR',np.where(frame.Location.eq('DKI'),'RO','RF'))
        frame['nitrogen']=frame.Treatment.str.split('_').str[0]
        frame['protection']=frame.Treatment.str.split('_').str[1]
        frames.append(frame)
    observations=pd.concat(frames,ignore_index=True)
    eligible=observations[observations.Seedyield.gt(0)&observations.Treatment.isin(['HN_NF','HN_WF','LN_NF','LN_WF'])]
    keys=['Location','Year','water_regime','nitrogen','BRISONr']
    aggregations={col:'mean' for col in ['Seedyield','Septoria','BBCH59','BBCH87',*DISEASES]}
    means=eligible.groupby(keys+['protection'],dropna=False).agg(**{
        **{col:(col,function) for col,function in aggregations.items()},
        'yield_replicates':('Seedyield','count'),'severity_replicates':('Septoria','count'),
        'source_rows':('source_row',lambda x:'|'.join(map(str,x))),
        'source_file':('source_file',lambda x:'|'.join(sorted(set(x))))}).reset_index()
    unprotected=means[means.protection.eq('NF')&means.Septoria.notna()]
    protected=means[means.protection.eq('WF')]
    pairs=unprotected.merge(protected,on=keys,suffixes=('_unprotected','_protected'),validate='one_to_one')
    pairs['severity_unprotected']=pairs.Septoria_unprotected/100
    pairs['severity_protected']=pairs.Septoria_protected/100
    pairs['severity_reduction']=pairs.severity_unprotected-pairs.severity_protected
    pairs['response_fraction']=1-pairs.Seedyield_unprotected/pairs.Seedyield_protected
    pairs['site_year']=pairs.Location+'_'+pairs.Year.astype(str)
    pairs['contrast_id']=pairs[keys].astype(str).agg('|'.join,axis=1)
    pairs['low_nitrogen']=pairs.nitrogen.eq('LN').astype(float)
    pairs['irrigated']=pairs.water_regime.eq('IR').astype(float)
    pairs['heading_doy']=pairs.BBCH59_unprotected
    pairs['heading_to_ripeness_days']=pairs.BBCH87_unprotected-pairs.BBCH59_unprotected
    pairs.loc[~pairs.heading_to_ripeness_days.between(1,100),'heading_to_ripeness_days']=np.nan
    for disease in DISEASES:pairs[disease+'_unprotected']/=100
    assert not pairs.contrast_id.duplicated().any()
    quality=dict(source_files=len(frames),source_rows=len(observations),
        non_numeric_severity_excluded=invalid,paired_cultivar_management_contrasts=len(pairs),
        source_site_years=int(pairs.site_year.nunique()),locations=int(pairs.Location.nunique()),
        negative_responses=int(pairs.response_fraction.lt(0).sum()),
        pairs_with_protected_severity=int(pairs.severity_protected.notna().sum()),
        reference_yield_basis='dry mass',grain_unit='Source dt ha-1 at 100% dry mass; relative response cancels common units',
        severity_unit='Source metadata: percent leaf area, whole-plot assessment; source decimals preserved',
        moisture_conversion='None; no absolute yields pooled with Tunisia or Nordic 15% moisture data',
        management='Same source location, year, cultivar, nitrogen and water regime; treatment means, not paired physical plots',
        limitations=['Disease dates and final-leaf ranks are undocumented in the location tables.',
            'Some severity means and harvest means have different replicate coverage.',
            'Protection targets multiple diseases; treatment responses do not identify STB-specific losses.',
            'Whole-site validation uses five or fewer geographic sites and does not validate stage-dependent canopy physiology.',
            'Site-level rating comparability and source-coded water assignment remain transfer limitations; no source rating is rescaled or rounded.'])
    return pairs,quality


def main():
    contrasts,quality=prepare()
    models={'training_mean':[], 'management_ridge':['low_nitrogen','irrigated'],
        'severity_ridge':['severity_unprotected'],
        'severity_management_ridge':['severity_unprotected','low_nitrogen','irrigated'],
        'nonnegative_damage':['severity_unprotected']}
    domains={'unprotected_endpoint':(contrasts,models)}
    common=contrasts.dropna(subset=['severity_reduction'])
    if common.site_year.nunique()>2:
        domains['observed_reduction']=(common,{'training_mean':[], 'severity_ridge':['severity_unprotected'],
            'reduction_ridge':['severity_reduction'],'nonnegative_damage':['severity_reduction']})
    common=contrasts.dropna(subset=['heading_doy','heading_to_ripeness_days'])
    if common.Location.nunique()>2:
        domains['phenology_available']=(common,{'training_mean':[],
            'severity_ridge':['severity_unprotected'],
            'phenology_ridge':['heading_doy','heading_to_ripeness_days'],
            'severity_phenology_ridge':['severity_unprotected','heading_doy','heading_to_ripeness_days']})
    common=contrasts.dropna(subset=[col+'_unprotected' for col in DISEASES])
    if common.Location.nunique()>2:
        domains['other_disease_available']=(common,{'training_mean':[],
            'severity_ridge':['severity_unprotected'],
            'multiple_disease_ridge':['severity_unprotected',*[col+'_unprotected' for col in DISEASES]]})
    protocol=dict(source='BRIWECS original site files; publication-rounded outputs excluded',
        source_hashes={str(p.relative_to(ROOT)):sha(p) for p in sorted((SOURCE/'data/locations').glob('*.csv'))},
        unit_metadata_sha256=sha(SOURCE/'metadata/Unit.xlsx'),
        treatment_correction_sha256=sha(SOURCE/'scripts/pre-processing/functions.R'),
        response='1 - unprotected cultivar treatment-mean yield / protected cultivar treatment-mean yield',
        pairing_keys=['Location','Year','water_regime','nitrogen','BRISONr'],
        source_treatments=['HN_NF','HN_WF','LN_NF','LN_WF'],
        drought_suffix='Excluded: drought-suffixed WF references are not pooled with unsuffixed WF plots.',
        severity='Unprotected severity alone unless protected severity is actually observed; missing stays unknown.',
        validation=['Whole site-year exclusion','Whole geographic-site exclusion','2015-2017 training, 2018-2019 testing'],
        weights='Equal site-years for primary and temporal comparisons; equal locations for location-held-out comparison; equal contrasts within each group.',
        models={domain:values for domain,(_,values) in domains.items()},ridge_penalty=.1,
        inference='Associational treatment-response validation; no leaf-stage damage function or climate model coefficient promoted.')
    path=HERE/'briwecs_protocol_before_fit.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    records=[];outputs=[]
    for domain,(frame,choices) in domains.items():
        for validation,group in [('leave_site_year_out','site_year'),('leave_location_out','Location'),('2015_2017_to_2018_2019','site_year')]:
            pieces={}
            for model,features in choices.items():
                if validation.startswith('leave'):
                    p=grouped_predictions(frame,group,model,features,'response_fraction')
                else:
                    train=frame[frame.Year.le(2017)];test=frame[frame.Year.ge(2018)].copy()
                    if len(train)==0 or len(test)==0:continue
                    prediction,parameters=fit_predict(train,test,model,features,'response_fraction',group)
                    p=test.copy();p['prediction']=prediction;p['observed']=p.response_fraction
                    p['model']=model;p['training_groups']='|'.join(sorted(train.site_year.unique()))
                    p['parameters']=json.dumps(parameters,sort_keys=True);p['validation_group']=group
                p['domain']=domain;p['validation']=validation;outputs.append(p);pieces[model]=p
            if not pieces:continue
            baseline=score(pieces['training_mean'],group,100)['RMSE']
            for model,p in pieces.items():
                s=score(p,group,100);records.append(dict(domain=domain,validation=validation,model=model,
                    **s,RMSE_pp=s['RMSE'],baseline_RMSE_pp=baseline,skill=1-(s['RMSE']/baseline)**2))
    contrasts.to_csv(HERE/'briwecs_treatment_response_features.csv',index=False)
    pd.concat(outputs,ignore_index=True).to_csv(HERE/'briwecs_held_out_predictions.csv',index=False)
    metrics=pd.DataFrame(records);metrics.to_csv(HERE/'briwecs_model_comparison_metrics.csv',index=False)
    (HERE/'briwecs_receipt.json').write_text(json.dumps(dict(quality=quality,protocol_sha256=sha(path),
        climate_yield_conversion_changed=False),indent=2)+'\n')
    print(json.dumps(quality,indent=2));print(metrics[['domain','validation','model','n','groups','RMSE_pp','baseline_RMSE_pp','skill']].to_string(index=False))


if __name__=='__main__':main()
