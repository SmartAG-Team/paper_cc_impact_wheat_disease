"""Screen historical canopy-informed development on held stations before transfer."""
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from .canopy_stage_context import fit_calendar_cdf,predict_calendar_cdf,STAGES
from .canopy_stage_model import fit_canopy_stage,predict_canopy_stage
from .canopy_stage_data import REQUEST
from .innovation_common import ROOT
from .inputs import sha

CONTEXT=ROOT/'data/disease_improvement/canopy_stage_context_20261022'
PREVIOUS=ROOT/'outputs/disease_improvement/canopy_augmentation_20261022'
REUSE_HELD=ROOT/'data/disease_improvement/canopy_stage_transfer_20261022'
TRANSFER=ROOT/'data/disease_improvement/canopy_stage_transfer_scoring_repair_20261022'


def _score(y,prediction,queries,mask):
    part=queries.loc[mask].copy()
    mass=part.station_cycle_date_weight.to_numpy(float)
    # Each represented held station contributes equal mass, including in the covered subset.
    mass=mass/part.groupby('station_id').station_cycle_date_weight.transform('sum').to_numpy(float)
    error=(prediction[mask]-y[mask])**2
    return dict(brier=float(np.average(error.mean(axis=1),weights=mass)),
        per_stage={str(s):float(np.average(error[:,i],weights=mass)) for i,s in enumerate(STAGES)},
        rows=int(mask.sum()),stations=part.station_id.nunique())


def run():
    previous=json.loads((PREVIOUS/'verification.json').read_text())
    assert previous['passed'] and not previous['goal_achieved'] and previous['fitted_splits']==20
    receipt=json.loads((CONTEXT/'verification.json').read_text())
    assert receipt['passed'] and receipt['source_unchanged'] and receipt['no_model_fit']
    for name,digest in receipt['artifacts_sha256'].items():assert sha(CONTEXT/name)==digest,name
    if TRANSFER.exists():raise FileExistsError('Fresh auxiliary transfer destination required')
    sources=[CONTEXT/name for name in ['registration.json','verification.json','auxiliary_predictors.npz',
        'auxiliary_queries.csv','predictors.npz','feature_schema.json']]
    sources += [REQUEST/'admitted_cycles.json',REQUEST/'stations.csv',PREVIOUS/'verification.json',PREVIOUS/'all_goal_decisions.json']
    sources += [REUSE_HELD/name for name in ['registration.json','held_station_model.joblib','calendar_stage_control.joblib','execution_failure_amendment.json']]
    codes=[Path(__file__),Path(__file__).with_name('canopy_stage_model.py'),Path(__file__).with_name('canopy_stage_context.py'),
        Path(__file__).with_name('canopy_stage_data.py')]
    TRANSFER.mkdir(parents=True)
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p):sha(p) for p in sources},code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        validation='Fixed199train/50heldstations chosenbeforefit; heldstationweightedmeanBrier, averagedover31/51/87, bothallcontextsandqualifiedcanopycontexts',
        weights='Originalequalstationmasssplitamongcycles/datecontexts; each3stagecasehas1/3contextmass, mean-normalizedbeforeHGB. Coveredscore renormalizesretainedcontextmasswithinheldstation',
        comparator='PerstagecalendarCDF fitsONLY199auxtrainingstationeventdates, equalstation/cyclemass; no fulltraitprior orheldstationlabels reused',
        algorithm='OnejointHGBbinaryclassifier400iterations/lr.05/depth3/leaves15/min100encodedcases/L2=5/noearlystop/seed20261022; laststageindexmonotone-1',
        selection='Full249stationrefitONLYifbothheldstationBriers strictlyimprovecalendarcontrol. No tuning/retryingafterresult; no diseasetrainingifscreenfails',
        scope='Retrospectiveobserved87+14daytruncatedhistoricalwheatpopulation; 2009croplandmapreleased2010 predatescurrentdiseasequeriesbutnotearlyauxdates; notoperationalhistoricalvalidation',
        transfer='Allauxlabels/weather/canopybefore2013, allhistoricalbuffers>50kmcurrentdiseaselocations; DWDGermanweather→currentstation/reanalysisweather,regionalcroplandnotfield/wheatidentity',
        disease_information='Currentdateduntreatedcohortandown-fieldstrictlyearlierdiseasemeasurementsunchanged. No currentseasonother-fielddisease labels',
        disease_extension_if_passes='Fixed20member80%coordinateyear/HGB200phasegrowth, raw145+trait59+canopy52+learned31/51/87probabilitiesandphaseentropy=260features; primarylearnedgateswarm+referencecold; matchedweathergatecontrol;sharedheadcontrol; registeredjointvariantmayreuse25coldCDF. Noqualifiedcanopyfallbackfrozen21; sameoriginalsplits/weights/goal',
        current_stage_probabilities_are_predictions=True,independent_disease_validation=False)
    registration['execution_repair']='Initialauxiliarymodelcompletedunchanged199stationfit,butscreenstoppedbeforeanyscoreatread-onlypandasweightdivision. Resumeexacthash-boundheldmodelinnewdestination, allocatednormalizedweights;nofit/hyperparameter/data/screenchange.'
    (TRANSFER/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for code in codes:
        dest=TRANSFER/'code_snapshot'/code.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code,dest)
    queries=pd.read_csv(CONTEXT/'auxiliary_queries.csv');records=json.loads((REQUEST/'admitted_cycles.json').read_text())
    with np.load(CONTEXT/'auxiliary_predictors.npz') as source:
        x=source['context'];available=source['canopy_available']
        np.testing.assert_array_equal(source['query_ids'],np.arange(len(queries)))
    y=queries[[f'stage{s}_passed' for s in STAGES]].to_numpy(int)
    held=queries.auxiliary_validation_station.to_numpy(bool);train=~held
    training_stations=set(queries.loc[train,'station_id'].astype(int));held_stations=set(queries.loc[held,'station_id'].astype(int))
    assert len(training_stations)==199 and len(held_stations)==50 and not training_stations&held_stations
    assert np.isfinite(x).all() and np.all(np.diff(y,axis=1)<=0)
    weights=queries.station_cycle_date_weight.to_numpy(float)
    np.testing.assert_allclose(queries.groupby('station_id').station_cycle_date_weight.sum(),1,atol=1e-15)
    previous_registration=json.loads((REUSE_HELD/'registration.json').read_text())
    for name,digest in previous_registration['source_sha256'].items():assert sha(Path(name))==digest,name
    for name in ['canopy_stage_model.py','canopy_stage_context.py','canopy_stage_data.py']:
        relative=str(Path(__file__).with_name(name).relative_to(ROOT))
        assert sha(ROOT/relative)==previous_registration['code_sha256'][relative]
    print('reuse completed auxiliary fit',len(training_stations),'stations',int(train.sum()),'contexts',flush=True)
    fitted=joblib.load(REUSE_HELD/'held_station_model.joblib')
    assert fitted['training_stations']==sorted(training_stations) and fitted['held_stations']==sorted(held_stations)
    assert fitted['training_query_ids']==np.flatnonzero(train).tolist()
    np.testing.assert_array_equal(fitted['original_context_weight'],weights[train])
    assert fitted['features']==85 and fitted['iterations']==400
    path=TRANSFER/'held_station_model.joblib';joblib.dump(fitted,path)
    probability=predict_canopy_stage(fitted,x)
    np.testing.assert_array_equal(probability,predict_canopy_stage(joblib.load(path),x))
    prior=fit_calendar_cdf(records,training_stations);calendar=predict_calendar_cdf(prior,queries.date)
    previous_prior=joblib.load(REUSE_HELD/'calendar_stage_control.joblib')
    for stage in STAGES:
        for name in ['values','cumulative_weight']:
            np.testing.assert_array_equal(prior[stage][name],previous_prior[stage][name])
    joblib.dump(prior,TRANSFER/'calendar_stage_control.joblib')
    metrics={}
    for name,mask in [('all',held),('qualified_canopy',held&available)]:
        metrics[name]=dict(learned=_score(y,probability,queries,mask),calendar=_score(y,calendar,queries,mask))
    passed=all(result['learned']['brier']<result['calendar']['brier'] for result in metrics.values())
    decision=dict(passed=passed,criterion='StrictlylowerheldstationmeanBrierthantrain-onlycalendarCDFonbothallandqualifiedcontexts',metrics=metrics)
    (TRANSFER/'screen_decision.json').write_text(json.dumps(decision,indent=2)+'\n')
    predictions=queries.loc[held].copy();predictions['canopy_available']=available[held]
    for i,stage in enumerate(STAGES):
        predictions[f'learned_probability_{stage}']=probability[held,i]
        predictions[f'calendar_probability_{stage}']=calendar[held,i]
    predictions.to_csv(TRANSFER/'held_station_predictions.csv',index=False)
    if passed:
        print('screen passes; refit all historical stations',flush=True)
        full=fit_canopy_stage(x,y,weights);full['training_stations']=sorted(training_stations|held_stations)
        joblib.dump(full,TRANSFER/'full_stage_model.joblib')
        with np.load(CONTEXT/'predictors.npz') as source:
            current=source['context'];target_ids=source['target_ids'];current_available=source['canopy_available']
        current_probability=predict_canopy_stage(full,current)
        np.testing.assert_array_equal(current_probability,predict_canopy_stage(joblib.load(TRANSFER/'full_stage_model.joblib'),current))
        np.savez_compressed(TRANSFER/'predictors.npz',stage_probability=current_probability,
            available=current_available,target_ids=target_ids)
        support=pd.DataFrame(dict(global_target_index=target_ids,qualified_canopy_available=current_available,
            latitude_outside_auxiliary_support=(current[:,-1]<x[:,-1].min())|(current[:,-1]>x[:,-1].max()),
            feature_fraction_outside_auxiliary_minmax=np.mean((current<x.min(axis=0))|(current>x.max(axis=0)),axis=1)))
        support.to_csv(TRANSFER/'current_support.csv',index=False)
    verification=dict(passed=True,auxiliary_screen_passed=passed,saved_model_replay=True,
        source_unchanged=all(sha(Path(p))==h for p,h in registration['source_sha256'].items()),
        auxiliary_training_stations=199,auxiliary_validation_stations=50,full_refit_performed=passed,
        disease_model_fit=False,artifacts_sha256={p.name:sha(p) for p in TRANSFER.iterdir() if p.is_file()})
    (TRANSFER/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(decision,indent=2),flush=True)
    return decision


if __name__=='__main__':
    with threadpool_limits(limits=1):run()
