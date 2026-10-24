"""Compare measured-stage alignment and matched weather-prior phase models."""
import json
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from analysis.final_study.metrics import hierarchical_weights
from .phenology_phase_growth import fit_phase_growth,predict_phase_growth,phase_weights
from .phenology_transfer_experiment import TRAIT_DATA,_anchor,_category
from .cold_ordinal_experiment import PHENO_MODELS
from .phenology_alignment_data import OUT as ALIGN_DATA
from .discrepancy_experiment import forecast_baseline
from .weather_sequence_model import design
from .innovation_common import ROOT,REFERENCE,load_cohort,register,assess,goal_decision
from .evaluation import score_predictions
from .inputs import sha

PREVIOUS=ROOT/'outputs/disease_improvement/simulation_filter_20261022'
PRIMARY='stage_aligned_growth'
NAMES=[PRIMARY,'stage_prior_control','stage_aligned_shared']
CONTROLS=['alignment_reference','alignment_phenology_control']


def run():
    previous=json.loads((PREVIOUS/'verification.json').read_text())
    assert previous['passed'] and not previous['goal_achieved']
    data_receipt=json.loads((ALIGN_DATA/'verification.json').read_text())
    assert data_receipt['passed'] and data_receipt['source_unchanged']
    out=ROOT/'outputs/disease_improvement/phenology_alignment_20261022'
    codes=[Path(__file__)]+[Path(__file__).with_name(n+'.py') for n in [
        'phenology_alignment','phenology_alignment_data','phenology_phase_growth',
        'phenology_transfer_experiment','phenology_transfer','phenology_transfer_data',
        'sowing_distribution_data','stage_features','discrepancy_experiment','discrepancy_filter',
        'crossfit_router','field_history_models','models','robust_models','history_interval_hybrid',
        'weather_sequence_model','dynamic_model','cold_ordinal_experiment','cold_ordinal']]
    registration=register(out,PRIMARY,dict(
        hypothesis='Earlier measured crop-stage intervals condition latent sowing/development profiles more informatively than weather-only phase priors or raw stage features alone',
        representation='Joint four-milestone thermal threshold profiles from complete pre2013 >50km historical cycles, equalstationmass; three clocks × nine existing RF sowingnodes; originalscenario querywisefallbackifnodeweatherincomplete',
        timing='Every query and historical trait milestone uses weather before its date. Only strictlyearlier SAMEFIELD qualified BBCH intervals update profile weights. All currentday stage/disease and other-field currentseason context withheld',
        likelihood='Fixed logistic transition60clockunits,5%uniform-rank contamination; ordinal milestone-rank interval likelihood. Structural measurement model, not calibrated biological/posterior uncertainty',
        predictors='145original+59frozen21trait+31newrepresentation/rawpriorstagefeatures=235. Matched prior representation has same nine raw stage predictors and three stage-context flags, but no mixture reweighting; inherited numeric BBCH/rate/native-code interaction descriptors remain unvalidated metric proxies in BOTH arms',
        architecture='Original21 shared/fourphasegrowth HGB20members80%coordinateyear/200iterations/depth3/leaves15/min10/lr.05/L2=3/phasefloor.01. Sameoriginalscientificmass and percentageanchor restoredonce',
        variants='Posterior phase growth; matched weather-prior phase control; posterior shared growth. No fitted current field disease parameter or anatomy mapping',
        fallback='Stage-unavailable or first visits preserve frozen21phase exactly; absent source-native endpoint preserves originalreference. Whole2017 assessment retained in every design',
        screen='ONEsamecandidate must pass ALL unchanged temporal goal gates before original15spatialfits; otherwise spatial UNASSESSED',
        information='Same current dated untreated cohort and field-only earlier-disease setting; no new real disease observations, global held-field fit, or currentseason other-field surveillance',
        adaptive_extension=True),codes)
    reg=json.loads((ALIGN_DATA/'registration.json').read_text())
    registration['source_sha256'].update(reg['source_sha256'])
    for directory in [ALIGN_DATA,TRAIT_DATA]:
        for path in directory.iterdir():
            if path.is_file():registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    for path in [PREVIOUS/'verification.json',PHENO_MODELS/'registration.json',
                 *sorted(PHENO_MODELS.glob('*_model.joblib'))]:
        registration['source_sha256'][str(path.relative_to(ROOT))]=sha(path)
    for path,digest in registration['source_sha256'].items():assert sha(ROOT/path)==digest,path
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    obs,arrays,history,frames,splits,position=load_cohort()
    raw,measured,available=design(arrays,history,obs.drop(columns='observed_percent'))
    with np.load(TRAIT_DATA/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        original=np.column_stack([raw,source['phenology']]);old_phase=phase_weights(source['stage_probability'][:,2])
    with np.load(ALIGN_DATA/'predictors.npz') as source:
        np.testing.assert_array_equal(source['target_ids'],obs.global_target_index)
        x={key:np.column_stack([original,source[key]]) for key in ['filtered','prior']}
        phase={key:phase_weights(source[key+'_probability']) for key in ['filtered','prior']}
        covered=source['stage_available']
    assert x['filtered'].shape[1]==235
    parts,records,started={},[],time.monotonic()
    for split in splits:
        label='temporal' if split['fold'].startswith('temporal') else 'spatial_50km'
        if label=='spatial_50km' and label not in parts:
            frame=pd.concat(parts['temporal'],ignore_index=True);frame.to_csv(out/'assessment_predictions.csv',index=False)
            metrics,_=assess({'temporal':frame},PRIMARY,out,extra_models=(*NAMES[1:],*CONTROLS))
            severe=pd.read_csv(out/'subgroup_metrics.csv');severe=severe.loc[severe.population.eq('severe_above75')]
            gates={n:goal_decision(metrics,severe,n)['gates'] for n in NAMES}
            eligible=[n for n,g in gates.items() if all(v for k,v in g.items() if k.startswith('temporal_'))]
            (out/'temporal_objective_screen.json').write_text(json.dumps(dict(candidate_gates=gates,eligible_candidates=eligible),indent=2)+'\n')
            if not eligible:
                (out/'screen_decision.json').write_text(json.dumps(dict(continue_spatial=False,
                    reason='No same registered stage-alignment candidate passes all unchanged temporal objective gates; spatial unassessed',
                    screening_rule_registered_before_fit=True),indent=2)+'\n');break
        ti=position.loc[split['train']].to_numpy();vi=position.loc[split['test']].to_numpy()
        train=obs.iloc[ti];assert train.season_year.max()<obs.iloc[vi].season_year.min()
        old=joblib.load(PHENO_MODELS/f'{split["fold"]}_model.joblib')
        assert old['train_ids']==split['train'] and old['test_ids']==split['test']
        weights=hierarchical_weights(train);mass=weights/weights.sum();y=train.observed_percent.to_numpy(float)/100
        priors=dict(pooled=float(np.average(y,weights=mass)),means={});categories=_category(train)
        for key in categories.unique():
            mask=categories.eq(key).to_numpy()
            priors['means'][key]=float((np.sum(mass[mask]*y[mask])+.04*priors['pooled'])/(mass[mask].sum()+.04))
        anchor=_anchor(priors,train,measured[ti],available[ti])
        print('start',split['fold'],flush=True)
        models={key:fit_phase_growth(x[key][ti],train,anchor,weights,phase[key][ti]) for key in ['filtered','prior']}
        fitted=dict(models=models,priors=priors,supported_categories=categories.unique().tolist(),old=old,
                    train_ids=split['train'],test_ids=split['test'])
        path=out/f'{split["fold"]}_model.joblib';joblib.dump(fitted,path)

        def forecast(bundle):
            query=obs.iloc[vi].drop(columns='observed_percent');old=bundle['old']
            parent=forecast_baseline(old['base'],arrays,history,obs,vi)
            old_use=available[vi]&_category(query).isin(old['supported_categories']).to_numpy()
            old_anchor=_anchor(old['priors'],query,measured[vi],available[vi])
            prior=predict_phase_growth(old['growth'],original[vi],old_anchor,old_phase[vi],parent,old_use)['stage']
            active=available[vi]&covered[vi]&_category(query).isin(bundle['supported_categories']).to_numpy()
            anchor=_anchor(bundle['priors'],query,measured[vi],available[vi])
            values={key:predict_phase_growth(bundle['models'][key],x[key][vi],anchor,phase[key][vi],prior,active)
                    for key in ['filtered','prior']}
            return {PRIMARY:values['filtered']['stage'],'stage_prior_control':values['prior']['stage'],
                'stage_aligned_shared':values['filtered']['shared'],'alignment_reference':parent,
                'alignment_phenology_control':prior,'alignment_used':active}

        values,replay=forecast(fitted),forecast(joblib.load(path))
        for name in values:np.testing.assert_array_equal(values[name],replay[name])
        part=frames[label].loc[split['test']].reset_index().copy()
        for name,value in values.items():part[name]=value
        np.testing.assert_allclose(part[CONTROLS[0]],part[REFERENCE],atol=1e-11,rtol=0)
        parts.setdefault(label,[]).append(part)
        records.append(dict(fold=split['fold'],elapsed_seconds=time.monotonic()-started,
                            new_heads=200,used_rows=int(values['alignment_used'].sum())))
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
