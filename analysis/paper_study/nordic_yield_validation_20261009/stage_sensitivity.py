"""Secondary sign-constrained sensitivity after the initial stage regression.

This is explicitly a post-diagnostic sensitivity, rather than a new external test.
"""
import json
import numpy as np
import pandas as pd
from .run import HERE,sha,prepare,leave_trial_out,score,bootstrap_comparison


def main():
    _,_,contrasts,_=prepare()
    frame=contrasts[contrasts.latest_stage.notna()].copy()
    frame['earlier_stage_basis']=frame.delta_severity*(85-frame.latest_stage)/10
    protocol=dict(analysis='Secondary sensitivity following inspection of initial model coefficient directions',
        positive_stage_model='Nonnegative coefficients on severity reduction and severity reduction times (85 - recorded assessment stage)/10',
        stage_scale='Empirical recorded-stage basis, not elapsed time or measured crop carbon allocation',
        zero_reduction='Zero predicted response when both severity-reduction predictors are zero',
        validation='Leave one entire trial out on the same four-trial common population',
        selection='No post-diagnostic model promoted to the climate simulation')
    path=HERE/'stage_sign_sensitivity_protocol.json'
    if path.exists():assert json.loads(path.read_text())==protocol
    else:path.write_text(json.dumps(protocol,indent=2)+'\n')
    predictions=leave_trial_out(frame,'positive_stage_linear',['delta_severity','earlier_stage_basis'])
    predictions['domain']='stage_known';predictions['validation']='leave_trial_out'
    prior=pd.read_csv(HERE/'held_out_predictions.csv')
    prior=prior[prior.domain.eq('stage_known')&prior.validation.eq('leave_trial_out')]
    comparison=pd.concat([prior,predictions],ignore_index=True)
    intervals=pd.DataFrame(bootstrap_comparison(comparison,'stage_known'))
    rows=[]
    for model,part in comparison.groupby('model'):
        rows.append(dict(domain='stage_known',validation='leave_trial_out',model=model,**score(part)))
    metrics=pd.DataFrame(rows).merge(intervals,on=['domain','model'],validate='one_to_one')
    baseline=float(metrics[metrics.model.eq('training_mean')].RMSE_pp.iloc[0])
    metrics['training_mean_RMSE_pp']=baseline
    metrics['skill_against_training_mean']=1-(metrics.RMSE_pp/baseline)**2
    predictions.to_csv(HERE/'positive_stage_predictions.csv',index=False)
    metrics.to_csv(HERE/'positive_stage_comparison.csv',index=False)
    derivatives=[]
    for trial,part in prior[prior.model.eq('stage_ridge')].groupby('registry_id'):
        params=json.loads(part.fit_parameters.iloc[0]);stage=float(part.latest_stage.iloc[0])
        slope=params['coefficients'][0]/params['training_scale'][0]+params['coefficients'][1]/params['training_scale'][1]*(stage-75)/10
        derivatives.append(dict(registry_id=trial,held_out_assessment_stage=stage,
            conditional_response_derivative=slope,expected_positive_damage_direction=slope>=0))
    pd.DataFrame(derivatives).to_csv(HERE/'stage_coefficient_diagnostics.csv',index=False)
    receipt=dict(status='complete',protocol_sha256=sha(path),
        negative_conditional_stage_model_slopes=sum(not row['expected_positive_damage_direction'] for row in derivatives),
        stage_model_folds=len(derivatives),additional_test_type='Post-diagnostic sensitivity',
        source_code_sha256={str(path.name):sha(path) for path in [HERE/'run.py',HERE/'stage_sensitivity.py']})
    (HERE/'stage_sensitivity_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(metrics[['model','RMSE_pp','training_mean_RMSE_pp','skill_against_training_mean']].to_string(index=False))
    print('Unconstrained stage regression has negative conditional severity-response slopes in',receipt['negative_conditional_stage_model_slopes'],'of',receipt['stage_model_folds'],'folds.')


if __name__=='__main__':main()
