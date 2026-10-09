"""Independent stored-date arithmetic and small public checkpoint replay."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from .train import sha
from .evaluate import predict_family

HERE=Path(__file__).resolve().parent


def verify():
    out=HERE/'results';receipt=json.loads((out/'receipt.json').read_text())
    for name,digest in receipt['output_sha256'].items():assert sha(out/name)==digest,name
    events=pd.read_csv(out/'event_predictions.csv.gz')
    assert not events.duplicated(['cohort','PEP_ID','SOWING_DATE','BBCH']).any()
    checks=0
    for table,shared in [('stage_metrics.csv',False),('common_stage_metrics.csv',True)]:
        summary=pd.read_csv(out/table)
        for row in summary.itertuples():
            f=events.loc[events.cohort.eq(row.cohort)&events.BBCH.eq(row.stage)&events.eligible]
            if shared:f=f.loc[f.all_model_common]
            column='T_P_V_position' if row.model=='T-P-V' else row.model+'_median'
            valid=f[column].notna();error=f.loc[valid,column]-f.loc[valid,'source_observed_position']
            assert len(f)==row.observed and int(valid.sum())==row.matched
            actual=[error.abs().mean(),np.sqrt(np.square(error).mean()),error.mean()]
            np.testing.assert_allclose(actual,[row.mae_days,row.rmse_days,row.bias_days],rtol=1e-7,atol=1e-6)
            penalty=(error.abs().sum()+366*(len(f)-len(error)))/len(f)
            np.testing.assert_allclose(penalty,row.penalized_mae_days,rtol=1e-7,atol=1e-6)
            checks+=1
    # Restore dates against the original public event tables, independent of the array packer.
    source_checks=0
    for cohort in ['validation','testing']:
        source=HERE.parents[1]/'phenology/heldout_validation'/f'{cohort}_tpv_event_dates.csv.gz'
        reference=pd.read_csv(source)
        reference=reference.loc[reference.true_date.notna()]
        own=events.loc[events.cohort.eq(cohort)]
        joined=own.merge(reference,on=['PEP_ID','SOWING_DATE','BBCH'],validate='one_to_one',suffixes=('','_source'))
        assert len(joined)==len(own)==len(reference)
        positions=(pd.to_datetime(joined.true_date)-pd.to_datetime(joined.SOWING_DATE)).dt.days+1
        np.testing.assert_array_equal(positions,joined.source_observed_position)
        keep=joined.T_P_V_position.notna()
        baseline_positions=(pd.to_datetime(joined.loc[keep,'predicted_date'])-pd.to_datetime(joined.loc[keep,'SOWING_DATE'])).dt.days+1
        np.testing.assert_array_equal(baseline_positions,joined.loc[keep,'T_P_V_position'])
        source_checks+=len(joined)
    selection=json.loads((out/'selection_before_test.json').read_text())
    assert not selection['test_scores_used']
    best=min(selection['validation_scores'],key=lambda k:selection['validation_scores'][k]['macro_penalized_mae'])
    assert selection['selected']==best
    replay=json.loads((HERE/'replay_manifest.json').read_text())
    for name,key in [('replay_inputs.npz','inputs_sha256'),('replay_expected.npz','expected_sha256'),('replay_environments.csv','environments_sha256'),('input_metadata.json','input_metadata_sha256')]:
        assert sha(HERE/name)==replay[key],name
    m=json.loads((HERE/'input_metadata.json').read_text())
    with np.load(HERE/'replay_inputs.npz') as z:a={k:z[k] for k in z.files}
    a['X']=((a['X']-np.asarray(m['mean'],np.float32))/np.asarray(m['std'],np.float32)).astype(np.float32)
    checkpoint_count=0
    with np.load(HERE/'replay_expected.npz') as expected:
        for family in ['lstm','tcn']:
            ps=[HERE/'fits'/f'{family}_s{s}.pt' for s in [17,29,43]]
            for p in ps:
                r=json.loads(p.with_suffix('.json').read_text());assert r['checkpoint_sha256']==sha(p)
                assert not r['test_labels_used'] and r['input_arrays_sha256']==m['input_arrays_sha256']
                checkpoint_count+=1
            actual=predict_family(a,m,np.arange(len(a['X'])),ps)
            for key,value in actual.items():
                np.testing.assert_allclose(value,expected[family+'_'+key],rtol=1e-6,atol=1e-6,equal_nan=True)
    result=dict(status='passed',independent_stage_metric_rows=checks,original_source_event_dates=source_checks,
                checkpoint_hashes_verified=checkpoint_count,public_replay_environments=len(a['X']),
                external_training_arrays_required=False,device='cpu',verifier_sha256=sha(__file__),
                scope='Stored event/date calculations and checkpoint inference; independent biological or climate validation is not implied')
    (HERE/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':
    torch.set_num_threads(2);verify()
