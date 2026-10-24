"""Test thermal surface information against matched retrieval-only context."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .cold_ordinal import fit_cold_ordinal,predict_cold_ordinal
from .phenology_phase_growth import fit_phase_growth,predict_phase_growth,phase_weights
from .phenology_transfer_experiment import TRAIT_DATA,_anchor,_category
from .discrepancy_experiment import forecast_baseline
from .weather_sequence_model import design
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha
from .night_lst_data import OUT as NIGHT

CANOPY=ROOT/'data/disease_improvement/canopy_prefix_20261022'
MODELS=ROOT/'outputs/disease_improvement/canopy_augmentation_20261022'
PREVIOUS=ROOT/'outputs/disease_improvement/phenology_alignment_20261022'
PRIMARY='night_surface_joint'
NAMES=[PRIMARY,'night_surface_warm','night_surface_cold','night_air_control']
CONTROLS=['night_reference','night_canopy_control']


def run():
    previous=json.loads((PREVIOUS/'verification.json').read_text())
    assert previous['passed'] and not previous['goal_achieved']
    for directory in [NIGHT,CANOPY,TRAIT_DATA]:
        receipt=json.loads((directory/'verification.json').read_text())
        assert receipt['passed'] and receipt['source_unchanged']
    out=ROOT/'outputs/disease_improvement/night_surface_20261022'
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'night_lst_data','night_lst_inputs','gather_night_lst','cold_ordinal',
        'phenology_phase_growth','phenology_transfer_experiment','phenology_transfer',
        'discrepancy_experiment','discrepancy_filter','crossfit_router','field_history_models',
        'models','robust_models','history_interval_hybrid','weather_sequence_model','dynamic_model']]
    registration=register(out,PRIMARY,dict(
        hypothesis='Observed regional nighttime skin temperature adds useful information on surface cooling beyond retained rain, humidity, developmental clocks and regional canopy signals',
        data='NASA Terra/Aqua MOD11A1/MYD11A1.061 clear-sky1km; 10km rounded-site buffers/static2009cropland. MandatoryQA0/1 plus detaileddata0/emissivityerrorclass<=1/LSTerrorclass<=1/providerrawbounds; estimatedaverage-error categories, not individualerrorbounds',
        timing='Explicit daily periodend+2day assumedavailability, completeperiods strictlypastquery; latestage<=14d; windows14/42/84datedavailability. Historical actualproductlatencyunverified/reprocessedhindcast',
        physical='Dailyair/dewpoint contrasts notinstantaneouslycollocated; regionalspatialSD notmeasurementerror. Surfacecooling is neither measuredleafwetness norproofofcondensation/identifiedwheatfield',
        predictors='350:145original+59frozen21trait+52frozen25canopy+94nightfeatures(two47blocks)',
        matched_control='Same350predictors/schedules/memberships;40satellite-derivedtemperature/spatialSD/contrastcolumns zeroed,54dailyair/retrieval-and-providercoverage/view/age/airavailabilitycolumns retained exactly, including16explicitdailyairvaluecolumns. Tests surfacecontent beyondsameairandretrievalcontext; no causalinterpretation',
        warm='Original21fixedTPVfourphasegates;20members80%coordinateyear,HGB200/depth3/leaves15/minleaf10/L2=3/lr.05/floor.01, originalhierarchyweights; restoremeasured-or-outertrainprioranchoronce',
        cold='Original24meanpreserving5ppjointCDF,20members80%coldcoordinateyear,200iterations,40encodedcases/originaltarget withconservedscientificmass; source-nativecategoricalsupport',
        variants='Newwarm+cold;newwarm+referencecold;frozen25warm+newcold;matchedair-and-retrieval-controlnewwarm+cold',
        fallback='Withoutnightcoverage allvariants keepfrozen25canopyphase(warm) andoriginalreference(cold); absentoutertrain source-nativekey preservesreference; noquerydeletion',
        information='Same currentdateduntreatedcohort; strictlyearlier SAMEFIELDdiseasehistory only. Stationrain/forcingunchanged; noother-fieldcurrentseason diseasecontext/upload',
        screen='ONEsamecandidate passes ALLunchangedtemporalgoalgates before15originalspatialfits; otherwise spatialUNASSESSED',
        adaptive_extension=True),codes)
    for directory in [NIGHT,CANOPY,TRAIT_DATA]:
        reg=json.loads((directory/'registration.json').read_text())
        registration['source_sha256'].update(reg['source_sha256'])
        for path in directory.iterdir():
            if path.is_file():registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    for path in [PREVIOUS/'verification.json',MODELS/'registration.json',*sorted(MODELS.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    for path,digest in registration['source_sha256'].items():assert sha(ROOT/path)==digest,path
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort()
    raw,measured,available=design(arrays,history,obs.drop(columns='observed_percent'))
    with np.load(TRAIT_DATA/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        original=np.column_stack([raw,source['phenology']]);phase=phase_weights(source['stage_probability'][:,2])
    with np.load(CANOPY/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        basis=np.column_stack([original,source['canopy']]);canopy_available=source['available']
    with np.load(NIGHT/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        x={k:np.column_stack([basis,source[v]]) for k,v in [('thermal','night'),('nuisance','air_control')]}
        night_available=source['available']
    assert all(matrix.shape==(2304,350) for matrix in x.values())
    parts,records,started={},[],time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            metrics,_=assess({'temporal':frame},PRIMARY,out,extra_models=(*NAMES[1:],*CONTROLS))
            groups=pd.read_csv(out/'subgroup_metrics.csv');severe=groups.loc[groups.population.eq('severe_above75')]
            gates={n:goal_decision(metrics,severe,n)['gates'] for n in NAMES}
            eligible=[n for n,g in gates.items() if all(v for k,v in g.items() if k.startswith('temporal_'))]
            (out/'temporal_objective_screen.json').write_text(json.dumps(dict(candidate_gates=gates,eligible_candidates=eligible),indent=2)+'\n')
            if not eligible:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='No same nighttime-surface candidate passes all unchanged temporal objective gates; spatial unassessed',
                    screening_rule_registered_before_fit=True),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        old_path=MODELS/f'{split["fold"]}_model.joblib';old=joblib.load(old_path)
        assert old['train_ids']==split['train'] and old['test_ids']==split['test']
        weights=hierarchical_weights(train);mass=weights/weights.sum();y=train.observed_percent.to_numpy(float)/100
        categories=_category(train);priors=dict(pooled=float(np.average(y,weights=mass)),means={})
        for key in categories.unique():
            mask=categories.eq(key).to_numpy()
            priors['means'][key]=float((np.sum(mass[mask]*y[mask])+.04*priors['pooled'])/(mass[mask].sum()+.04))
        anchor=_anchor(priors,train,measured[ti],available[ti])
        print('start',split['fold'],flush=True)
        growth={k:fit_phase_growth(v[ti],train,anchor,weights,phase[ti]) for k,v in x.items()}
        cold={k:fit_cold_ordinal(v[ti],train,weights,available[ti]) for k,v in x.items()}
        fitted=dict(growth=growth,cold=cold,priors=priors,supported_categories=categories.unique().tolist(),
            old_canopy_model_path=str(old_path.relative_to(ROOT)),train_ids=split['train'],test_ids=split['test'])
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)

        def forecast(bundle):
            old=joblib.load(ROOT/bundle['old_canopy_model_path']);trait=old['warm']
            query=obs.iloc[vi].drop(columns='observed_percent')
            parent=forecast_baseline(trait['base'],arrays,history,obs,vi)
            use=available[vi]&_category(query).isin(trait['supported_categories']).to_numpy()
            anchor=_anchor(trait['priors'],query,measured[vi],available[vi])
            trait_value=predict_phase_growth(trait['growth'],original[vi],anchor,phase[vi],parent,use)['stage']
            use=available[vi]&canopy_available[vi]&_category(query).isin(old['supported_categories']).to_numpy()
            anchor=_anchor(old['priors'],query,measured[vi],available[vi])
            prior=predict_phase_growth(old['growth'],basis[vi],anchor,phase[vi],trait_value,use)['stage']
            supported=_category(query).isin(bundle['supported_categories']).to_numpy()
            active=available[vi]&night_available[vi]&supported
            anchor=_anchor(bundle['priors'],query,measured[vi],available[vi])
            warm={k:predict_phase_growth(bundle['growth'][k],v[vi],anchor,phase[vi],prior,active)['stage'] for k,v in x.items()}
            cold={k:predict_cold_ordinal(bundle['cold'][k],v[vi],query,parent,available[vi])['ordinal'] for k,v in x.items()}
            first={k:np.where(night_available[vi],v,parent) for k,v in cold.items()}
            return {PRIMARY:np.where(available[vi],warm['thermal'],first['thermal']),
                'night_surface_warm':warm['thermal'],'night_surface_cold':np.where(available[vi],prior,first['thermal']),
                'night_air_control':np.where(available[vi],warm['nuisance'],first['nuisance']),
                'night_reference':parent,'night_canopy_control':prior,'night_available':night_available[vi],
                'night_warm_used':active}

        values,replay=forecast(fitted),forecast(joblib.load(path))
        for name in values:np.testing.assert_array_equal(values[name],replay[name])
        part=frames[label].loc[split['test']].reset_index().copy()
        for name,value in values.items():part[name]=value
        np.testing.assert_allclose(part['night_reference'],part[REFERENCE],atol=1e-11,rtol=0)
        prior_frame=pd.read_csv(MODELS/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),float_precision='round_trip').set_index('global_target_index')
        np.testing.assert_array_equal(part.night_canopy_control,prior_frame.loc[split['test'],'canopy_phase_growth'])
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            warm_heads=200,cold_members_each_arm=len(fitted['cold']['thermal']['members']),
            qualified_queries=int(night_available[vi].sum())))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
              'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,PRIMARY,out,extra_models=(*NAMES[1:],*CONTROLS))
    groups=pd.read_csv(out/'subgroup_metrics.csv');severe=groups.loc[groups.population.eq('severe_above75')]
    decisions={n:goal_decision(metrics,severe,n) for n in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,
        fitted_splits=len(records),source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
