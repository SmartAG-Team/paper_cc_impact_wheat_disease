"""Reproduce canopy-impact sensitivity from compact exact annual inputs.

python -m analysis.paper_study.climate_robustness_20261009.run
Use --prepare-inputs --source-root PATH once to extract the public input bundle.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .statistics import pairing_statistics, separate_period_change, annual_means, distribution_statistics, production_components

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
MODELS=['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']
SCENARIOS=['ssp126','ssp245','ssp585']
YEARS=np.r_[np.arange(1991,2021),np.arange(2031,2061),np.arange(2071,2101)]
WEIGHTS={'harvested_area':'harvested_total_ha','baseline_production':'production_total_tonnes',
         'rainfed_production':'production_rainfed_tonnes'}
METRIC='GS65_85_lost_had3'
REGISTRY=ROOT/'analysis/paper_study/food_security_exposure_20261009/production_grid_input.csv'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare_inputs(source_root):
    source_root=Path(source_root)
    index=pd.read_csv(REGISTRY)
    if index.cell_id.duplicated().any():
        raise ValueError('Duplicate input cell identity')
    upstream=source_root/'analysis/paper_study/full_grid_climate_20261008/results/completion_receipt.json'
    metadata=json.loads(upstream.read_text())
    expected={r['path']:r['sha256'] for r in metadata['source_outputs']}
    values=np.full((3,3,90,len(index)),np.nan,dtype=np.float64)
    sources={}
    for mi,model in enumerate(MODELS):
        for si,scenario in enumerate(SCENARIOS):
            for yi,year in enumerate(YEARS):
                name=f'analysis/paper_study/nature_food_revision_20261007/continental_replay/annual_outputs/nasa/{model}/{scenario}/{year}.parquet'
                p=source_root/name
                digest=sha(p)
                if expected.get(name)!=digest:
                    raise ValueError('Annual output hash changed: '+name)
                f=pd.read_parquet(p,columns=['cell_id','valid_complete_season',METRIC])
                if not np.array_equal(index.cell_id.to_numpy(),f.cell_id.to_numpy()):
                    raise ValueError('Cell identities/order differ: '+name)
                x=f[METRIC].to_numpy(float)
                if np.isfinite(x[~f.valid_complete_season]).any() or np.any(x[np.isfinite(x)]<0):
                    raise ValueError('Invalid season or negative canopy value: '+name)
                values[mi,si,yi]=x
                sources[name]=digest
            print('Extracted',model,scenario,flush=True)
    np.savez_compressed(HERE/'annual_canopy_inputs.npz',values=values,
        models=np.array(MODELS),scenarios=np.array(SCENARIOS),years=YEARS,
        cell_id=index.cell_id.to_numpy(dtype=str))
    receipt={'status':'passed','metric':METRIC,'unit':'normalized HAD days',
        'annual_files':len(sources),'cells':len(index),'finite_annual_values':int(np.isfinite(values).sum()),
        'invalid_values_preserved_as_nan':True,'float_precision':'float64, no rounding',
        'source_completion_sha256':sha(upstream),'source_sha256':sources,
        'bundle_sha256':sha(HERE/'annual_canopy_inputs.npz'),'registry_sha256':sha(REGISTRY)}
    (HERE/'input_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')


def inputs():
    receipt=json.loads((HERE/'input_receipt.json').read_text())
    if sha(HERE/'annual_canopy_inputs.npz')!=receipt['bundle_sha256'] or sha(REGISTRY)!=receipt['registry_sha256']:
        raise ValueError('Bundled input or registry hash changed')
    index=pd.read_csv(REGISTRY)
    with np.load(HERE/'annual_canopy_inputs.npz',allow_pickle=False) as bundle:
        for key,expected in [('models',MODELS),('scenarios',SCENARIOS),('years',YEARS),('cell_id',index.cell_id)]:
            if not np.array_equal(bundle[key],expected):
                raise ValueError('Bundled input identities differ: '+key)
        values=bundle['values'].copy()
    if values.shape!=(3,3,90,len(index)):
        raise ValueError('Annual array shape differs')
    return index,values


def _ensemble(frame,groups,metrics):
    rows=[]
    for key,part in frame.groupby(groups,dropna=False,sort=True):
        row=dict(zip(groups,key if isinstance(key,tuple) else (key,)))
        if part.model.duplicated().any():
            raise ValueError('Repeated climate models within ensemble')
        row['models']=part.model.nunique()
        for metric in metrics:
            complete=len(part)==3 and part[metric].notna().all()
            row[metric]=part[metric].mean() if complete else np.nan
            row[metric+'_gcm_min']=part[metric].min() if complete else np.nan
            row[metric+'_gcm_max']=part[metric].max() if complete else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def run(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    index,values=inputs()
    index['environment_region']=index.environment_region.fillna('Unassigned')
    domains={'Europe':np.ones(len(index),bool)}
    domains.update({name:index.environment_region.eq(name).to_numpy() for name in sorted(index.environment_region.unique())})
    pairing,support,annual,distributions,coverage=[],[],[],[],[]
    for si,scenario in enumerate(SCENARIOS):
        for start,period in [(30,'2031-2060'),(60,'2071-2100')]:
            # A single spatial population across all years and all climate models.
            complete=np.isfinite(values[:,si,:30]).all(axis=(0,1)) & np.isfinite(values[:,si,start:start+30]).all(axis=(0,1))
            for weight_name,column in WEIGHTS.items():
                for domain,mask in domains.items():
                    key={'scenario':scenario,'period':period,'weighting':weight_name,'environment_region':domain}
                    selected=np.flatnonzero(mask)
                    w=index[column].to_numpy(float)[selected]
                    fixed_complete=complete[selected]
                    wc=w*fixed_complete
                    fraction=float(wc.sum()/w.sum()) if w.sum() else np.nan
                    for mi,model in enumerate(MODELS):
                        a=values[mi,si,:30,selected]
                        b=values[mi,si,start:start+30,selected]
                        mk=dict(key,model=model)
                        for shift in range(30):
                            pairing.append(dict(mk,shift=shift,**pairing_statistics(a,b,w,shift)))
                        original=pairing[-30]
                        support.append(dict(mk,method='positional_pairs',reference=original['reference'],future=original['future'],change=original['change'],coverage_fraction=original['coverage_fraction']))
                        separate=separate_period_change(a,b,w)
                        support.append(dict(mk,method='separate_available_periods',**separate,
                            coverage_fraction=min(separate['reference_coverage_fraction'],separate['future_coverage_fraction'])))
                        fixed=pairing_statistics(a,b,wc)
                        support.append(dict(mk,method='complete_common_cells',reference=fixed['reference'],future=fixed['future'],change=fixed['change'],coverage_fraction=fraction))
                        if wc.sum():
                            shifted=pairing_statistics(a,b,wc,17)
                            np.testing.assert_allclose(shifted['change'],fixed['change'],atol=1e-12,rtol=0)
                            previous,next_values=annual_means(a,wc),annual_means(b,wc)
                            stats=distribution_statistics(previous['mean'],next_values['mean'])
                            distributions.append(dict(mk,complete_support_fraction=fraction,**stats))
                            for period_name,yrs,result in [('1991-2020',YEARS[:30],previous),(period,YEARS[start:start+30],next_values)]:
                                for yi,year in enumerate(yrs):
                                    annual.append(dict(mk,observation_period=period_name,year=int(year),
                                        annual_mean=float(result['mean'][yi]),complete_support_fraction=fraction,
                                        valid_cells=int(result['valid_cells'][yi])))
                    coverage.append(dict(key,baseline_weight=float(w.sum()),complete_support_weight=float(wc.sum()),
                        complete_support_fraction=fraction,landuse_cells=int(mask.sum()),complete_support_cells=int(fixed_complete.sum())))
            print('Analyzed',scenario,period,flush=True)
    tables={'year_pairing_by_gcm':pd.DataFrame(pairing),'support_by_gcm':pd.DataFrame(support),
        'annual_domain_values':pd.DataFrame(annual),'annual_distribution_by_gcm':pd.DataFrame(distributions),
        'complete_population_coverage':pd.DataFrame(coverage)}
    keys=['scenario','period','weighting','environment_region']
    by_shift=_ensemble(tables['year_pairing_by_gcm'],keys+['shift'],['change','reference','future','coverage_fraction'])
    rows=[]
    for key,part in by_shift.groupby(keys,sort=True):
        original=part[part['shift'].eq(0)].iloc[0]
        if len(part)!=30: raise ValueError('Thirty cyclic pairings required')
        valid=part.change.notna().all()
        rows.append(dict(zip(keys,key),original_change=original.change,
            pairing_min=part.change.min() if valid else np.nan,
            pairing_max=part.change.max() if valid else np.nan,
            max_absolute_difference=(part.change-original.change).abs().max() if valid else np.nan,
            direction_consistent=bool(valid and (np.sign(part.change)==np.sign(original.change)).all()),
            original_gcm_min=original.change_gcm_min,original_gcm_max=original.change_gcm_max,
            minimum_coverage_fraction=part.coverage_fraction_gcm_min.min(),pairings=30))
    tables['year_pairing_ensemble']=by_shift
    tables['pairing_sensitivity']=pd.DataFrame(rows)
    tables['support_ensemble']=_ensemble(tables['support_by_gcm'],keys+['method'],['reference','future','change','coverage_fraction'])
    dmetrics=['historical_mean','future_mean','mean_change','historical_sd','future_sd','historical_q10','future_q10',
              'historical_q90','future_q90','q90_change','historical_top3_mean','future_top3_mean',
              'future_exceedance_fraction','complete_support_fraction']
    tables['annual_distribution_ensemble']=_ensemble(tables['annual_distribution_by_gcm'],keys,dmetrics)
    population=[]
    for domain,mask in domains.items():
        f=index.loc[mask]
        total=float(f.production_total_tonnes.sum())
        rainfed=float(f.production_rainfed_tonnes.sum())
        irrigated=float(f.production_irrigated_tonnes.sum())
        population.append({'environment_region':domain,'baseline_production_tonnes':total,
            'rainfed_production_tonnes':rainfed,'irrigated_production_tonnes':irrigated,
            **production_components(total,rainfed,irrigated),
            'spring_winter_partition_available':False})
    tables['crop_population_coverage']=pd.DataFrame(population)
    prior=pd.read_csv(ROOT/'analysis/paper_study/full_grid_climate_20261008/results/environmental_region_changes_by_gcm.csv')
    prior=prior[prior.metric.eq(METRIC)]
    current=tables['year_pairing_by_gcm'].query("shift == 0 and weighting == 'harvested_area'")
    checked=current.merge(prior,on=['model','scenario','period','environment_region'],suffixes=('_new','_prior'),validate='one_to_one')
    if len(checked)!=len(prior) or len(checked)!=len(current):
        raise ValueError('Incomplete independent comparison to published regional results')
    np.testing.assert_allclose(checked.change_new,checked.change_prior,atol=1e-10,rtol=0,equal_nan=True)
    for name,table in tables.items(): table.to_csv(output/(name+'.csv'),index=False)
    receipt={'status':'passed','protocol_sha256':sha(HERE/'protocol.json'),'input_receipt_sha256':sha(HERE/'input_receipt.json'),
        'input_bundle_sha256':sha(HERE/'annual_canopy_inputs.npz'),'registry_sha256':sha(REGISTRY),
        'prior_aggregation_comparisons':len(checked),'complete_support_pairing_invariance_checked':True,
        'models':MODELS,'scenarios':SCENARIOS,'cyclic_pairings':30,'model_refitted':False,
        'raw_simulations_rerun':False,'grain_loss_validated':False,'adaptation_validated':False,
        'output_sha256':{name+'.csv':sha(output/(name+'.csv')) for name in tables},
        'code_sha256':{name:sha(HERE/name) for name in ['run.py','statistics.py']}}
    (output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['output_sha256','code_sha256']},indent=2))


def read(name):
    receipt=json.loads((HERE/'receipt.json').read_text())
    filename=name+'.csv'
    if filename not in receipt['output_sha256'] or sha(HERE/filename)!=receipt['output_sha256'][filename]:
        raise ValueError('Climate sensitivity output hash changed: '+filename)
    return pd.read_csv(HERE/filename)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-inputs',action='store_true')
    parser.add_argument('--source-root',type=Path)
    parser.add_argument('--output',type=Path,default=HERE)
    args=parser.parse_args()
    if args.prepare_inputs:
        if args.source_root is None: parser.error('--prepare-inputs requires --source-root')
        prepare_inputs(args.source_root)
    run(args.output)
