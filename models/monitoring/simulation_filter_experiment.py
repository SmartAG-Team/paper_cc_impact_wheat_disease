"""Test simulation-assisted recurrent filtering with a matched scratch prior."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .dynamic_data import build_sequences,subset_sequences
from .epidemic_world import world_inputs
from .simulation_filter import pretrain_filter,fit_filter,predict_filter
from .discrepancy_experiment import forecast_baseline,CACHE
from .weather_sequence_model import design
from .phenology_transfer_experiment import _category
from .innovation_common import ROOT,DATA,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PREVIOUS=ROOT/'outputs/disease_improvement/field_meta_kernel_20261022'
PRIMARY='simulation_pretrained_filter'
NAMES=[PRIMARY,'simulation_pretrained_direct','simulation_pretrained_half','scratch_state_filter']
CONTROL='simulation_filter_reference'


def run():
    previous=json.loads((PREVIOUS/'verification.json').read_text())
    assert previous['passed'] and not previous['goal_achieved']
    out=ROOT/'outputs/disease_improvement/simulation_filter_20261022'
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'simulation_filter','epidemic_world','dynamic_data','discrepancy_experiment','discrepancy_filter',
        'crossfit_router','field_history_models','models','robust_models','history_interval_hybrid',
        'weather_sequence_model','dynamic_model','phenology_transfer_experiment']]
    registration=register(out,PRIMARY,dict(
        hypothesis='Assumedepidemic-responsepretraining teacheslatentonset/pressure/progression soearlierhealthyobservationsdonotrequirepersistentlynegativeforecastcorrection',
        architecture='Weather6→12tanh;maskedassessments7+flags7→16tanh;staticlatitude/longitude/source3; GRU31→48;sourcecategoricalinitial3→48tanh;LayerNorm/Dropout.1/Linear7/sigmoid',
        timing='QueryqreadsstatebeforeqweatherandALLqassessments;onlyown-fieldhistory. Sourcefieldseasonyearfixedinmetadata, notfuturequerymaximum. No batch/time pooling orother-fieldcurrentdiseaselabels',
        simulation='2048virtualworlds bootstrapONLYouterhistoricaltrainweather/static/assessmentmasks, sourceprobabilityoriginalfieldscientificmass. Label-freewhitelist; ALLrealvaluescanbechangedwithoutchangingworldsorpretrainedstate',
        dynamics='S/E1/E2/E3/Vnominal7exchangeablechannels; positivewetness/T/rain-drivenprimary/secondaryhazards and3stageprogression; no code-rank/verticalleaf/anatomyassumption',
        randomization='Arrivaldayofyear15..185infixedseason;primary10^U(-4,-.9),secondary10^U(-2,.3);15%healthy;progressiontimescaleL=8..35at18C,threedaily-stageexpectedtransit3/(1-exp(-3/L))about9.6..36.5days;IIDchannelsusceptibilitylogSD.6normalizedfieldmean1;channeloffset-21..42days;measurementSDU(0,.07)*sqrt(V(1-V))',
        assumptions='Randomizedranges,irreversibilityandnoisemodelareunverifiedsyntheticresponseassumptions, not empirical diseaseparameterdistributions/additionalmeasurements/EPPOproof/cultivarobservations',
        pretrain='CPU,one48hiddenfilter/fold,512AdamWsteps/.003/wd.001/gradientclip1;32of2048worldsperbatch;syntheticqueryweightspreserveoriginalrelativeendpoint/recordmasswithinvirtualfield;seed20261022',
        fine_tune='Threefixedseeds20261022..24,150FULLBATCHeach;originalgloballynormalizedscientificqueryMSEinpercentagefraction;AdamW.002/wd.001/gradientclip1;lr*.3at60%and85%;MAP1e-4timesSUMparameterdeviationfromanchor',
        scratch='Threesame-seed/schedule/architecture/dropout/weight-decay/MAPconventionfits; anchorisownrandominitialization. Contrastincludespretrainingandchoiceofprioranchor; notinitialization-onlycausaleffect',
        variants='Warm-onlypretrained(firstreference);directpretrainedallsupported;halfneural/referencewarmblend(firstreference);matchedscratchwarm(firstreference). Unsupportedsource-nativequeriespreserveoriginalreference',
        information='Currentdateduntreatedcohort/weather/same-field-onlystrictlyearlierlabelsettingunchanged. Simulatedtargetsnotcountedasnewobserveddata',
        screen='AtleastONEsamecandidatepassesALLunchangedtemporalgates before15originalspatialfits; otherwise spatialunassessed. Full2017support/hierarchyweights/splits/goalthresholdsunchanged',
        adaptive_extension=True),codes)
    for path in [PREVIOUS/'verification.json',CACHE/'registration.json',*sorted(CACHE.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path)]=sha(path)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort()
    fields=pd.read_csv(DATA/'fields.csv');weather=pd.read_parquet(DATA/'daily_weather.parquet')
    full=build_sequences(weather,fields,obs,obs.drop(columns='observed_percent'))
    np.testing.assert_array_equal(full['query_ids'],obs.global_target_index)
    season=fields.set_index('field_id').season_year
    _,_,available=design(arrays,history,obs.drop(columns='observed_percent'))
    indexed=obs.set_index('global_target_index',drop=False)
    parts,records,started={},[],time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            metrics,_=assess({'temporal':frame},PRIMARY,out,extra_models=(*NAMES[1:],CONTROL))
            groups=pd.read_csv(out/'subgroup_metrics.csv');severe=groups.loc[groups.population.eq('severe_above75')]
            temporal={n:goal_decision(metrics,severe,n)['gates'] for n in NAMES}
            eligible=[n for n,g in temporal.items() if all(value for key,value in g.items() if key.startswith('temporal_'))]
            (out/'temporal_objective_screen.json').write_text(json.dumps(dict(candidate_gates=temporal,eligible_candidates=eligible),indent=2)+'\n')
            if not eligible:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='No single registered recurrent-filter candidate passes all unchanged temporal objective gates; spatial unassessed',
                    screening_rule_registered_before_fit=True),indent=2)+'\n');break
        train_ids=np.asarray(split['train'],int);vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[position.loc[train_ids].to_numpy()]
        assert train.season_year.max()<obs.iloc[vi].season_year.min()
        train_batch=subset_sequences(full,sorted(train.field_id.unique()))
        assert set(train_batch['query_ids'])==set(train_ids)
        ordered=indexed.loc[train_batch['query_ids']]
        weights=hierarchical_weights(ordered)
        prior=joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids']==split['train'] and prior['test_ids']==split['test']
        base={k:prior['experts'][k] for k in ['process','first','event','training_ids']}
        print('start',split['fold'],'trainfields',len(train_batch['field_ids']),flush=True)
        inputs=world_inputs(train_batch,season.loc[train_batch['field_ids']].to_numpy(int),weights)
        pretrained=pretrain_filter(inputs)
        joblib.dump(pretrained,out/f'{split["fold"]}_pretraining.joblib')
        np.savez_compressed(out/f'{split["fold"]}_world_parameters.npz',**pretrained['world_parameters'])
        trained=[];scratch=[]
        for member in range(3):
            config=dict(seed=20261022+member)
            trained.append(fit_filter(train_batch,ordered.observed_percent.to_numpy(float),weights,
                pretrained['state_dict'],config))
            scratch.append(fit_filter(train_batch,ordered.observed_percent.to_numpy(float),weights,config=config))
        fitted=dict(base=base,pretrained=pretrained,trained=trained,scratch=scratch,
            supported_categories=_category(ordered).unique().tolist(),train_ids=split['train'],test_ids=split['test'])
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)
        query_batch=subset_sequences(full,sorted(obs.iloc[vi].field_id.unique()))
        ci=position.loc[query_batch['query_ids']].to_numpy()
        qi=pd.Series(np.arange(len(ci)),index=query_batch['query_ids']).loc[split['test']].to_numpy()

        def forecast(bundle):
            baseline=forecast_baseline(bundle['base'],arrays,history,obs,ci)[qi]
            neural=np.mean([predict_filter(m,query_batch)[qi] for m in bundle['trained']],axis=0)
            untrained=np.mean([predict_filter(m,query_batch)[qi] for m in bundle['scratch']],axis=0)
            supported=_category(obs.iloc[vi]).isin(bundle['supported_categories']).to_numpy()
            warm=supported&available[vi]
            return {PRIMARY:np.where(warm,neural,baseline),
                'simulation_pretrained_direct':np.where(supported,neural,baseline),
                'simulation_pretrained_half':np.where(warm,.5*(neural+baseline),baseline),
                'scratch_state_filter':np.where(warm,untrained,baseline),CONTROL:baseline}

        values,replay=forecast(fitted),forecast(joblib.load(path))
        for key in values:np.testing.assert_array_equal(values[key],replay[key])
        part=frames[label].loc[split['test']].reset_index().copy()
        for key,value in values.items():part[key]=value
        np.testing.assert_allclose(part[CONTROL],part[REFERENCE],atol=1e-11,rtol=0)
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            simulation=pretrained['world_audit'],pretraining_trace=pretrained['trace'],
            real_traces=[m['trace'] for m in trained],scratch_traces=[m['trace'] for m in scratch]))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),
            'rmse',{n:round(score_predictions(part,n)['rmse'],3) for n in NAMES},flush=True)
    results={label:pd.concat(p,ignore_index=True) for label,p in parts.items()}
    for label,frame in results.items():frame.to_csv(out/('assessment_predictions.csv' if label=='temporal' else 'spatial_predictions.csv'),index=False)
    metrics,_=assess(results,PRIMARY,out,extra_models=(*NAMES[1:],CONTROL))
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
