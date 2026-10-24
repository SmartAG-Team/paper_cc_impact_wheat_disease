"""Sequential test of an end-to-end learned kernel for within-field adaptation."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .field_meta_kernel import fit_meta_kernel,predict_meta_kernel
from .canopy_stage_transfer import TRANSFER
from .canopy_experiment import CANOPY,TRAIT_DATA
from .phenology_phase_growth import phase_weights
from .cold_ordinal import predict_cold_ordinal
from .canopy_stage_transfer import PREVIOUS as CANOPY_MODELS
from .discrepancy_experiment import forecast_baseline,CACHE
from .weather_sequence_model import design
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .inputs import sha

PRIOR_STAGE=ROOT/'outputs/disease_improvement/canopy_stage_20261022'
PRIMARY='field_meta_kernel'
NAMES=[PRIMARY,'field_meta_kernel_half','field_meta_kernel_joint_ordinal']
CONTROL='field_meta_unfiltered'


def run():
    previous=json.loads((PRIOR_STAGE/'verification.json').read_text())
    assert previous['passed'] and not previous['goal_achieved']
    out=ROOT/'outputs/disease_improvement/field_meta_kernel_20261022'
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'field_meta_kernel','canopy_stage_transfer','canopy_stage_context','canopy_stage_model','canopy_inputs',
        'phenology_phase_growth','discrepancy_experiment','discrepancy_filter','crossfit_router',
        'field_history_models','models','robust_models','history_interval_hybrid','weather_sequence_model','dynamic_model','cold_ordinal']]
    registration=register(out,PRIMARY,dict(
        hypothesis='Learningafieldadaptationkernelbyforecastloss improves conditionalerrorupdates beyondfixedcovarianceorattentionweights',
        inputs='260causalnumericalfeatures matching26:145retained+59trait+52canopy+3historicalcanopy-supervisedstageprobabilities+phaseentropy; baselinefractionappendedand9nominalsource/nativeinputs',
        basis='12Gaussianfieldcoefficients:commonconstant,7nominalnativeintercepts,4contextualtanhfeatures;270→16tanh→4tanh; positivelearnedrelativeamplitudes',
        scale='Normalizeeachbasisvectortonorm.25severityfraction, soK(x,x)=.0625fixed; meanMSEdoesnotidentifyanabsolutecovariance/noisescale. Individualbasisfactorsnotidentified/physical',
        noise='Forecast-timecontextualnuggetSD.02+.48sigmoid(shared16hidden→1), initialized.15; no currenttargetorposteriorresidual input to noise',
        conditioning='Exact12x12GaussianridgeposteriorusingALLstrictlyearlierown-fieldinnovations; separatefloat64fieldprefixstatistics; allsame-daynativequeriesbeforeupdates; no memorytokenlimit',
        fitting='Threefixedseeds20261022..24,300epochsAdamW.003/wd.01/gradientclip1/lrtimes.3at60%and85%; originalhierarchyweightedUNCAPPEDprequentialMSE; first-visitlossconstant',
        training='Honestcached5innercoordinate-excludedreferenceforecasts/errors, no full-outerfittedforecast fallback; outermodelparametershistoricaltrainingonly',
        variants='Full/halfkernelcorrection withoriginalfirstvisitreference; fullwarmcorrection+frozen25coldCDF ONLYinouter-heldcomposition. No25coldpredictionsintraining',
        inference='Clippredictedseverity0..100onlyafterensemble; firstvisitsbitexactreference; earlierown-fieldlabelsallowed, no other-fieldcurrentseasondiseasesurveillance',
        screen='ProspectivelyrequireatleastONEcandidatepassesALLthreeTEMPORALgoalconditionsbefore15spatialfits. Ifnone, spatialunassessed andfullobjectivealreadyfalse; thresholds/cohort/weights/membershipsunchanged',
        interpretation='WorkingconditionalGaussianerrorkernel fittedbyempiricalrisk, notobservedcultivar/anatomy/causalmodel/calibratedpredictiveuncertainty',
        adaptive_extension=True),codes)
    for directory in [CANOPY,TRAIT_DATA,TRANSFER]:
        receipt=json.loads((directory/'verification.json').read_text());assert receipt['passed'] and receipt['source_unchanged']
        for path in directory.iterdir():
            if path.is_file():registration['source_sha256'][str(path)]=sha(path)
    sources=[PRIOR_STAGE/'verification.json',CACHE/'registration.json',CANOPY_MODELS/'verification.json']
    sources += [*sorted(CACHE.glob('*_model.joblib')),*sorted(CANOPY_MODELS.glob('*_model.joblib'))]
    registration['source_sha256'].update({str(p):sha(p) for p in sources})
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort()
    raw,_,available=design(arrays,history,obs.drop(columns='observed_percent'))
    with np.load(TRAIT_DATA/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        original=np.column_stack([raw,source['phenology']]);weather_phase=phase_weights(source['stage_probability'][:,2])
    with np.load(CANOPY/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        canopy_x=np.column_stack([original,source['canopy']]);covered=source['available']
    with np.load(TRANSFER/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        np.testing.assert_array_equal(source['available'],covered);probability=source['stage_probability']
    learned_phase=phase_weights(np.column_stack([np.ones(len(obs)),probability]))
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
            metrics,_=assess({'temporal':frame},PRIMARY,out,extra_models=(*NAMES[1:],CONTROL))
            groups=pd.read_csv(out/'subgroup_metrics.csv');severe=groups.loc[groups.population.eq('severe_above75')]
            temporal={n:goal_decision(metrics,severe,n)['gates'] for n in NAMES}
            eligible=[n for n,gates in temporal.items() if all(value for key,value in gates.items() if key.startswith('temporal_'))]
            (out/'temporal_objective_screen.json').write_text(json.dumps(dict(candidate_gates=temporal,eligible_candidates=eligible),indent=2)+'\n')
            if not eligible:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='No single registered candidate meets all unchanged temporal objective gates; spatial performance unassessed',
                    screening_rule_registered_before_fit=True),indent=2)+'\n');break
        ti,vi=position.loc[split['train']].to_numpy(),position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        print('start',split['fold'],flush=True)
        prior=joblib.load(CACHE/f'{split["fold"]}_model.joblib')
        assert prior['train_ids']==split['train'] and prior['test_ids']==split['test']
        oof=.5*(prior['inner_predictions'][:,0]+prior['inner_predictions'][:,1])
        base={k:prior['experts'][k] for k in ['process','first','event','training_ids']}
        meta=fit_meta_kernel(x[ti],train,oof,hierarchical_weights(train))
        previous=joblib.load(CANOPY_MODELS/f'{split["fold"]}_model.joblib')
        assert previous['train_ids']==split['train'] and previous['test_ids']==split['test']
        fitted=dict(base=base,meta=meta,cold=previous['cold'],train_ids=split['train'],test_ids=split['test'],
            inner_memberships=prior['inner_memberships'],training_baseline=oof)
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)
        ci=np.flatnonzero(obs.field_id.isin(obs.iloc[vi].field_id.unique()))
        qi=pd.Series(np.arange(len(ci)),index=ci).loc[vi].to_numpy()

        def forecast(bundle):
            baseline=forecast_baseline(bundle['base'],arrays,history,obs,ci)
            full=predict_meta_kernel(bundle['meta'],x[ci],obs.iloc[ci],baseline)[qi]
            half=predict_meta_kernel(bundle['meta'],x[ci],obs.iloc[ci],baseline,.5)[qi]
            first=predict_cold_ordinal(bundle['cold'],canopy_x[vi],obs.iloc[vi].drop(columns='observed_percent'),baseline[qi],available[vi])
            cold=np.where(covered[vi],first['ordinal'],baseline[qi])
            return {PRIMARY:full,'field_meta_kernel_half':half,
                'field_meta_kernel_joint_ordinal':np.where(available[vi],full,cold),CONTROL:baseline[qi]}

        values,replay=forecast(fitted),forecast(joblib.load(path))
        for key in values:np.testing.assert_array_equal(values[key],replay[key])
        part=frames[label].loc[split['test']].reset_index().copy()
        for key,value in values.items():part[key]=value
        np.testing.assert_allclose(part[CONTROL],part[REFERENCE],atol=1e-11,rtol=0)
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
            trace=[m['trace'] for m in meta['members']]))
        (out/'fit_progress.json').write_text(json.dumps(records,indent=2)+'\n')
        print('finished',split['fold'],'seconds',round(time.monotonic()-started,1),flush=True)
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
