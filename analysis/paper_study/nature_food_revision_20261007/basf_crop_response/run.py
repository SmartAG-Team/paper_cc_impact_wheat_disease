"""Measured upper-leaf severity/yield-response comparisons, with a frozen year split."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
PARENT=HERE.parent
SOURCE=PARENT/'basf_yield_linkage'
DOMAINS={f'leaf{rank}_integral':[rank] for rank in [1,2,3]}
DOMAINS['all_three_separate_windows']=[1,2,3]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def weights(frame):
    # Equal coordinate-year, then source field, then treatment contrasts.
    groups=frame.groupby('coordinate_year').TrialId.transform('nunique').to_numpy()
    arms=frame.groupby(['coordinate_year','TrialId']).contrast_id.transform('size').to_numpy()
    w=1/(groups*arms);return w/w.sum()


def coefficient(x,y,w,link):
    z=y if link=='linear' else -np.log1p(-y)
    denominator=np.sum(w*x*x)
    return 0. if denominator<=0 else max(0.,float(np.sum(w*x*z)/denominator))


def predict(x,b,link):return b*x if link=='linear' else -np.expm1(-b*x)


def metrics(y,p,w):
    error=p-y;variance=np.sum(w*(y-np.sum(w*y))**2)
    return dict(MAE_pp=100*float(np.sum(w*abs(error))),RMSE_pp=100*float(np.sqrt(np.sum(w*error**2))),
        bias_pp=100*float(np.sum(w*error)),weighted_R2=float(1-np.sum(w*error**2)/variance) if variance else None)


def main():
    registry=dict(target='1 minus untreated/treated measured relative harvest yield',
        source='BASF documented MCALCR ratios; no mass units inferred',
        training_years=[2017,2018],evaluation_years=[2019],
        domains=DOMAINS,functions=['linear origin-constrained','exponential log-yield-ratio'],
        candidate_selection='No evaluation-driven model or leaf selection; report all predeclared domains',
        missing_leaves='Require every leaf in the declared domain; do not fill missing with zero',
        observation_windows='Common measured dates within each source leaf; all-three windows can differ',
        predictor_units='fraction-days of recorded severity contrast; not HAD or absolute healthy area',
        target_is_disease_free_yield_loss=False,existing_epidemic_data_reuse=True)
    protocol=HERE/'protocol_before_fit.json'
    if protocol.exists():assert json.loads(protocol.read_text())==registry
    else:protocol.write_text(json.dumps(registry,indent=2)+'\n')
    data=pd.read_csv(SOURCE/'paired_STB_leaf_window_integrals.csv')
    yields=pd.read_csv(SOURCE/'source_yield_contrasts.csv')
    rows=[];predictions=[];memberships=[]
    for domain,ranks in DOMAINS.items():
        grouped=data.loc[data.leaf_rank.isin(ranks)].groupby('contrast_id')
        records=[]
        for cid,group in grouped:
            if set(group.leaf_rank)!=set(ranks):continue
            records.append(dict(contrast_id=cid,x=float(group.measured_severity_integral_reduction_pp_days.sum()/100/len(ranks)),
                minimum_window_days=float(group.window_days.min()),maximum_window_days=float(group.window_days.max())))
        frame=pd.DataFrame(records).merge(yields,on='contrast_id',validate='one_to_one')
        frame['y']=1-1/frame.measured_relative_yield_ratio
        frame['domain']=domain;frame['role']=np.where(frame.season_year.eq(2019),'reused_year_transfer_2019','training_2017_2018')
        memberships.append(frame)
        train=frame.loc[frame.role.eq('training_2017_2018')];evaluation=frame.loc[frame.role.eq('reused_year_transfer_2019')]
        for link in ['linear','exponential']:
            x=train.x.to_numpy();y=train.y.to_numpy();w=weights(train)
            b=coefficient(x,y,w,link);baseline=float(np.sum(w*y))
            # Reconstruct without invoking the coefficient function.
            z=y if link=='linear' else np.log(train.measured_relative_yield_ratio.to_numpy())
            independent=max(0.,float(np.sum(w*x*z)/np.sum(w*x*x)))
            np.testing.assert_allclose(b,independent,rtol=0,atol=1e-12)
            part=evaluation.copy();part['response_function']=link
            part['fitted_training_coefficient']=b;part['prediction']=predict(part.x.to_numpy(),b,link)
            part['training_mean_prediction']=baseline;part['observed_response']=part.y
            predictions.append(part)
            wt=weights(part);score=metrics(part.y.to_numpy(),part.prediction.to_numpy(),wt)
            ref=metrics(part.y.to_numpy(),np.full(len(part),baseline),wt)
            rows.append(dict(domain=domain,function=link,training_contrasts=len(train),training_fields=train.TrialId.nunique(),
                evaluation_contrasts=len(part),evaluation_fields=part.TrialId.nunique(),training_coefficient=b,
                **score,baseline_RMSE_pp=ref['RMSE_pp'],skill_against_training_mean=1-(score['RMSE_pp']/ref['RMSE_pp'])**2,
                absolute_yield_prediction=False,causal_STB_loss_identified=False))
    pd.concat(memberships,ignore_index=True).to_csv(HERE/'source_memberships.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(HERE/'year_transfer_predictions.csv',index=False)
    result=pd.DataFrame(rows);result.to_csv(HERE/'year_transfer_metrics.csv',index=False)
    receipt=dict(status='complete',protocol_sha256=sha(protocol),
        source_sha256={str(p.relative_to(PARENT)):sha(p) for p in [SOURCE/'source_yield_contrasts.csv',SOURCE/'paired_STB_leaf_window_integrals.csv',Path(__file__)]},
        independently_reproduced_coefficients=True,all_predeclared_candidates_reported=True,
        assumptions=registry)
    (HERE/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(result.to_string(index=False))


if __name__=='__main__':main()
