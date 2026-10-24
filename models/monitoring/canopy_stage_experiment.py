"""Transfer historically supervised canopy-stage probabilities into disease growth."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .canopy_stage_transfer import TRANSFER,CONTEXT,PREVIOUS
from .canopy_experiment import CANOPY
from .cold_ordinal import predict_cold_ordinal
from .phenology_phase_growth import fit_phase_growth,predict_phase_growth,phase_weights
from .phenology_transfer_experiment import TRAIT_DATA,_anchor,_category
from .discrepancy_experiment import forecast_baseline
from .weather_sequence_model import design
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PRIMARY='canopy_stage_growth'
NAMES=[PRIMARY,'canopy_stage_shared','canopy_stage_weather_gates','canopy_stage_joint_ordinal']
CONTROLS=['stage_reference','stage_phenology_control','stage_canopy_control']


def run():
    out=ROOT/'outputs/disease_improvement/canopy_stage_20261022'
    receipt=json.loads((TRANSFER/'verification.json').read_text())
    assert receipt['passed'] and receipt['auxiliary_screen_passed'] and receipt['source_unchanged']
    for path,digest in receipt['artifacts_sha256'].items():assert sha(TRANSFER/path)==digest,path
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'canopy_stage_transfer','canopy_stage_context','canopy_stage_model','canopy_stage_data',
        'canopy_experiment','canopy_inputs','cold_ordinal','phenology_phase_growth','phenology_transfer_experiment',
        'phenology_transfer','discrepancy_experiment','discrepancy_filter','crossfit_router',
        'field_history_models','models','robust_models','history_interval_hybrid','weather_sequence_model','dynamic_model']]
    registration=register(out,PRIMARY,dict(
        hypothesis='Canopycondition interpretedwithhistoricalobservedcropdevelopment improves within-fielddiseasegrowth beyondmanualcanopysummaries',
        auxiliary='Fixedpre2013DWDstationholdoutstageclassifier passedbothall/qualifiedcontextBriersvs199trainingstationcalendarCDF, thenrefit249stations; nocurrentdiseaselabels',
        features='260=145retained+59weatherdevelopment+52canopy+3learnedstagepassageprobabilities+4phaseentropy/log4',
        gates='LearnedP31>=P51>=P87 gives[1-P31,P31-P51,P51-P87,P87], .01floorandrenormalization; emergenceprobabilityisnotlearned',
        warm='20original80%coordinateyearmembers HGB200/depth3/leaves15/min10/lr.05/L2=3/noearlystop; originalscientificweights andoutertrainingcoldpriors',
        variants='Learnedphasegrowth/referencecold;sharedgrowth/referencecold;matchedsamefeaturesoriginalweathergategrowth/referencecold;learnedgrowth/frozen25coldCDF',
        control='Frozen21phase and25canopyphase plusoriginalreference; noqualifiedcanopywarmfallback21, coldoriginalreference; no source-nativeunsupportednewhead',
        timing='Weatherdayminusone;compositeend+7dayassumedlag; same-fieldstrictlyearlierdiseaselabelsonly, same-dayallleafwithheld; no regionaldisease surveillance',
        interpretation='Predictedregionalcropdevelopment, notobservedfieldphenology/actualsowing/cultivar/leafanatomy; historicalauxholdoutretrospectiveandoutcome-windowtruncated; meteorological/geographicaltransferunvalidated',
        selection='Original20splits/2304cohort/2017assessments/weights/10%andseveregates; spatialiffanyregisteredformulationbeatsoriginaltemporalreference',
        adaptive_extension=True),codes)
    for directory in [TRANSFER,CONTEXT,CANOPY,TRAIT_DATA]:
        for path in directory.iterdir():
            if path.is_file():registration['source_sha256'][str(path)]=sha(path)
    for path in [PREVIOUS/'registration.json',PREVIOUS/'verification.json',*sorted(PREVIOUS.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path)]=sha(path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort()
    raw,measured,available=design(arrays,history,obs.drop(columns='observed_percent'))
    with np.load(TRAIT_DATA/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        original=np.column_stack([raw,source['phenology']]);weather_phase=phase_weights(source['stage_probability'][:,2])
    with np.load(CANOPY/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        canopy_x=np.column_stack([original,source['canopy']]);covered=source['available']
    with np.load(TRANSFER/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        np.testing.assert_array_equal(source['available'],covered)
        probability=source['stage_probability']
    learned_phase=phase_weights(np.column_stack([np.ones(len(obs)),probability]))
    # Unavailable canopy does not create an apparent new stage measurement.
    probability=np.where(covered[:,None],probability,np.column_stack([
        1-weather_phase[:,0],weather_phase[:,2]+weather_phase[:,3],weather_phase[:,3]]))
    learned_phase=np.where(covered[:,None],learned_phase,weather_phase)
    entropy=-np.sum(learned_phase*np.log(np.maximum(learned_phase,1e-12)),axis=1)/np.log(4)
    x=np.column_stack([canopy_x,probability,entropy]);assert x.shape==(2304,260)
    parts,records,started={},[],time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            assess({'temporal':frame},PRIMARY,out,extra_models=(*NAMES[1:],*CONTROLS))
            if min(score_predictions(frame,n)['rmse'] for n in NAMES)>=score_predictions(frame,REFERENCE)['rmse']:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='Allregisteredcanopy-stageformulationsfailtemporalreferenceimprovement'),indent=2)+'\n');break
        ti,vi=position.loc[split['train']].to_numpy(),position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        previous=joblib.load(PREVIOUS/f'{split["fold"]}_model.joblib')
        assert previous['train_ids']==split['train'] and previous['test_ids']==split['test']
        weights=hierarchical_weights(train)
        anchor=_anchor(previous['priors'],train,measured[ti],available[ti])
        growth=fit_phase_growth(x[ti],train,anchor,weights,learned_phase[ti])
        control=fit_phase_growth(x[ti],train,anchor,weights,weather_phase[ti])
        fitted=dict(growth=growth,weather_gate_control=control,previous=previous,
            train_ids=split['train'],test_ids=split['test'])
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)

        def forecast(bundle):
            query=obs.iloc[vi].drop(columns='observed_percent');old=bundle['previous'];old21=old['warm']
            parent=forecast_baseline(old21['base'],arrays,history,obs,vi)
            old_use=available[vi]&_category(query).isin(old21['supported_categories']).to_numpy()
            old_anchor=_anchor(old21['priors'],query,measured[vi],available[vi])
            prior=predict_phase_growth(old21['growth'],original[vi],old_anchor,weather_phase[vi],parent,old_use)['stage']
            use=available[vi]&covered[vi]&_category(query).isin(old['supported_categories']).to_numpy()
            anchored=_anchor(old['priors'],query,measured[vi],available[vi])
            prior_canopy=predict_phase_growth(old['growth'],canopy_x[vi],anchored,weather_phase[vi],prior,use)['stage']
            learned=predict_phase_growth(bundle['growth'],x[vi],anchored,learned_phase[vi],prior,use)
            weather=predict_phase_growth(bundle['weather_gate_control'],x[vi],anchored,weather_phase[vi],prior,use)['stage']
            first=predict_cold_ordinal(old['cold'],canopy_x[vi],query,parent,available[vi])
            ordinal=np.where(covered[vi],first['ordinal'],parent)
            grown=np.where(use,np.clip(learned['stage'],0,100),prior)
            shared=np.where(use,np.clip(learned['shared'],0,100),prior)
            weather=np.where(use,np.clip(weather,0,100),prior)
            return dict(canopy_stage_growth=grown,canopy_stage_shared=shared,canopy_stage_weather_gates=weather,
                canopy_stage_joint_ordinal=np.where(available[vi],grown,ordinal),
                stage_reference=parent,stage_phenology_control=prior,stage_canopy_control=prior_canopy,
                stage_warm_used=use,stage_cold_reused=covered[vi]&(first['member_count']>0))

        values,replay=forecast(fitted),forecast(joblib.load(path))
        for key in values:np.testing.assert_array_equal(values[key],replay[key])
        part=frames[label].loc[split['test']].reset_index().copy()
        for name in [*NAMES,*CONTROLS]:part[name]=values[name]
        part['stage_warm_used']=values['stage_warm_used'];part['stage_cold_reused']=values['stage_cold_reused']
        np.testing.assert_allclose(part.stage_reference,part[REFERENCE],atol=1e-11,rtol=0)
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            learned_warm_rows=int(values['stage_warm_used'].sum())))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,PRIMARY,out,extra_models=(*NAMES[1:],*CONTROLS))
    groups=pd.read_csv(out/'subgroup_metrics.csv')
    decisions={n:goal_decision(metrics,groups.loc[groups.population.eq('severe_above75')],n) for n in NAMES}
    (out/'all_goal_decisions.json').write_text(json.dumps(decisions,indent=2)+'\n')
    (out/'verification.json').write_text(json.dumps(dict(passed=True,saved_model_replay=True,
        fitted_splits=len(records),source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        goal_achieved=any(d['achieved'] for d in decisions.values()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()}),indent=2)+'\n')
    return metrics,decisions


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
