"""Regional canopy observations augment first-visit and phase-growth forecasts."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .cold_ordinal import fit_cold_ordinal, predict_cold_ordinal
from .phenology_phase_growth import fit_phase_growth, predict_phase_growth, phase_weights
from .phenology_transfer_experiment import TRAIT_DATA, _anchor, _category
from .cold_ordinal_experiment import PHENO_MODELS
from .discrepancy_experiment import forecast_baseline
from .weather_sequence_model import design
from .innovation_common import ROOT, REFERENCE, load_cohort, register, assess, goal_decision
from .evaluation import score_predictions
from .inputs import sha

CANOPY = ROOT/'data/disease_improvement/canopy_prefix_20261022'
PRIMARY = 'canopy_joint_ordinal'
NAMES = [PRIMARY,'canopy_phase_growth','canopy_cold_ordinal','canopy_joint_mean']
CONTROL, PHENO_CONTROL = 'canopy_reference', 'canopy_phenology_control'


def run(output=None):
    out = Path(output or ROOT/'outputs/disease_improvement/canopy_augmentation_20261022').resolve()
    codes = [Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'canopy_inputs','canopy_data','gather_regional_canopy','cold_ordinal','cold_ordinal_experiment',
        'phenology_phase_growth','phenology_transfer_experiment','phenology_transfer','phenology_transfer_data',
        'discrepancy_experiment','discrepancy_filter','crossfit_router','field_history_models',
        'models','robust_models','history_interval_hybrid','weather_sequence_model','dynamic_model']]
    for directory in [CANOPY,TRAIT_DATA]:
        receipt=json.loads((directory/'verification.json').read_text())
        assert receipt['passed'] and receipt['source_unchanged']
        for path,digest in json.loads((directory/'registration.json').read_text())['source_sha256'].items():
            assert sha(ROOT/path)==digest,path
    registration = register(out,PRIMARY,dict(
        hypothesis='Observed regional crop condition provides useful information beyond weather and inferred development',
        input='512MODISfourdaycomposites at128sitebuffers10km/500m, NASAQAqualified; fixed2009GlobCoverclasses11/14 released2010, not field/wheat/cultivar identification',
        timing='Allcompositeperiods ended at least7daysbeforequery; fixedassumedavailabilitylag, nofuturepeaks/smoothing; real-timeproductlatencyunverified',
        canopy='52prefixfeatures latest/regionalvariation/retrievalSD/14,42,84daywindows datedbyassumedavailability(end+7days), soenddatewindows shift7daysearlier; latestagealsosinceassumedavailability; total256predictorspluscoldnominalkeys',
        coverage='2222of2304targets have>=20clearqualifiedcroppixels andQAvalidfraction>=.2 WITHINstaticcropmask,notwholebuffer cropcoverage;21sitesstaticcropfraction<.2stillretained; zeroencoding+missingflags, no targetdeletion',
        warm='Original21phase-growthrecipe:20members80%coordinateyear,HGB200/depth3/leaves15/min10/lr.05/L2=3/phasefloor.01; originalscientificweights andoutertrainingcoldpriors',
        cold='Fixed24mean-preserving5ppgrid jointCDF/meanrecipe,20members80%coldcoordinateyear,200iterations; encodedproperlossconservesoriginalmass, thresholddecreasing',
        variants='Jointnewwarm+coldCDF;newwarm+referencecold;fixed21warm+newcoldCDF;newwarm+newcoldmean',
        fallback='Withoutqualifiedcanopy, warmqueryretainsfrozen21phaseforecastandcoldqueryretainsoriginalreference. Unsupportednewsource-nativecategoriespreserveexistingcomponent.',
        same_day='Strictlyearlierfieldobservations only; currentallleaflabelswithheld; meteorologicalinputsunchangeddayminusone/station-preferredrainfall',
        interpretation='Retrospective regional canopy proxy; no identified leafdisease/cultivar/actualsowing, no calibratedcontinuousseverityCDF or autonomousfutureclimatevalidation',
        selection='Spatial iff anyregisteredformulation improvesoriginaltemporalreference; original10%andseveregates/cohort/splits unchanged',
        adaptive_extension=True),codes)
    for directory in [CANOPY,TRAIT_DATA]:
        registration['source_sha256'].update(json.loads((directory/'registration.json').read_text())['source_sha256'])
        for path in directory.iterdir():
            if path.is_file():registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    for path in [PHENO_MODELS/'registration.json',*sorted(PHENO_MODELS.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort()
    raw,measured,available=design(arrays,history,obs.drop(columns='observed_percent'))
    with np.load(TRAIT_DATA/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        original=np.column_stack([raw,source['phenology']]);phase=phase_weights(source['stage_probability'][:,2])
    with np.load(CANOPY/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        x=np.column_stack([original,source['canopy']]);covered=source['available']
    parts,records,started={},[],time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=(*NAMES[1:],CONTROL,PHENO_CONTROL))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES)>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Allregisteredcanopyformulations fail temporalreference improvement'),indent=2)+'\n');break
        ti,vi=position.loc[split['train']].to_numpy(),position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        warm=joblib.load(PHENO_MODELS/f'{split["fold"]}_model.joblib')
        assert warm['train_ids']==split['train'] and warm['test_ids']==split['test']
        weights=hierarchical_weights(train);mass=weights/weights.sum();y=train.observed_percent.to_numpy(float)/100
        priors=dict(pooled=float(np.average(y,weights=mass)),means={});categories=_category(train)
        for key in categories.unique():
            mask=categories.eq(key).to_numpy()
            priors['means'][key]=float((np.sum(mass[mask]*y[mask])+.04*priors['pooled'])/(mass[mask].sum()+.04))
        anchor=_anchor(priors,train,measured[ti],available[ti])
        growth=fit_phase_growth(x[ti],train,anchor,weights,phase[ti])
        cold=fit_cold_ordinal(x[ti],train,weights,available[ti])
        fitted=dict(growth=growth,cold=cold,priors=priors,supported_categories=categories.unique().tolist(),
            warm=warm,train_ids=split['train'],test_ids=split['test'])
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)

        def forecast(bundle):
            query=obs.iloc[vi].drop(columns='observed_percent');old=bundle['warm']
            parent=forecast_baseline(old['base'],arrays,history,obs,vi)
            old_use=available[vi]&_category(query).isin(old['supported_categories']).to_numpy()
            old_anchor=_anchor(old['priors'],query,measured[vi],available[vi])
            prior=predict_phase_growth(old['growth'],original[vi],old_anchor,phase[vi],parent,old_use)['stage']
            active=available[vi]&covered[vi]&_category(query).isin(bundle['supported_categories']).to_numpy()
            anchored=_anchor(bundle['priors'],query,measured[vi],available[vi])
            grown=predict_phase_growth(bundle['growth'],x[vi],anchored,phase[vi],prior,active)['stage']
            first=predict_cold_ordinal(bundle['cold'],x[vi],query,parent,available[vi])
            ordinal=np.where(covered[vi],first['ordinal'],parent)
            mean=np.where(covered[vi],first['mean'],parent)
            return dict(canopy_joint_ordinal=np.where(available[vi],grown,ordinal),
                canopy_phase_growth=grown,canopy_cold_ordinal=np.where(available[vi],prior,ordinal),
                canopy_joint_mean=np.where(available[vi],grown,mean),canopy_reference=parent,
                canopy_phenology_control=prior,warm_uses_canopy=active,
                cold_uses_canopy=covered[vi]&(first['member_count']>0))

        values,replay=forecast(fitted),forecast(joblib.load(path))
        for key in values:np.testing.assert_array_equal(values[key],replay[key])
        part=frames[label].loc[split['test']].reset_index().copy()
        for name in [*NAMES,CONTROL,PHENO_CONTROL]:part[name]=values[name]
        part['canopy_available']=covered[vi]
        part['canopy_warm_used'],part['canopy_cold_used']=values['warm_uses_canopy'],values['cold_uses_canopy']
        np.testing.assert_allclose(part[CONTROL],part[REFERENCE],atol=1e-11,rtol=0)
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            warm_canopy_rows=int(values['warm_uses_canopy'].sum()),cold_canopy_rows=int(values['cold_uses_canopy'].sum())))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,PRIMARY,out,extra_models=(*NAMES[1:],CONTROL,PHENO_CONTROL))
    groups=pd.read_csv(out/'subgroup_metrics.csv')
    decisions={n:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],n) for n in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,
        fitted_splits=len(records),source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path);args=parser.parse_args()
    with threadpool_limits(limits=1):run(args.output)
