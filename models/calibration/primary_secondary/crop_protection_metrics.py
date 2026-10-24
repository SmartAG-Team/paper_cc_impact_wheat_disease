"""Severity and threshold diagnostics for conditional wheat disease prediction."""
import numpy as np
import pandas as pd
from .field_data import hierarchical_weights


def attach_forecast_context(predictions,episodes):
    """Attach stage and disease known at issue time; reject conditioning scores."""
    if episodes.series_id.duplicated().any():
        raise ValueError('Issue-time episode keys are not unique.')
    f=predictions.copy()
    days=pd.to_numeric(f.day,errors='coerce')
    elapsed=(pd.to_datetime(f.Date,errors='raise')-pd.to_datetime(f.start,errors='raise')).dt.days
    if (not np.isfinite(days).all() or not days.gt(0).all()
            or not np.array_equal(days.to_numpy(),elapsed.to_numpy())
            or not f.conditioning.eq(False).all()):
        raise ValueError('Only future targets with exact issue/assessment chronology can be scored.')
    e=episodes[['series_id','initial_percent']].copy()
    e['issue_stage']=episodes.GsFrom.to_numpy() if 'GsFrom' in episodes else np.nan
    f=f.merge(e,on='series_id',how='left',validate='many_to_one')
    if f.initial_percent.isna().any() or not f.initial_percent.between(0,100).all():
        raise ValueError('Every scored episode needs a valid issue-time disease assessment.')
    f['horizon_group']=pd.cut(f.day,[0,7,14,28,np.inf],labels=['01-07','08-14','15-28','29+']).astype(str)
    f['stage_group']=np.select([f.issue_stage.lt(31),f.issue_stage.between(31,33),
        f.issue_stage.between(34,39),f.issue_stage.between(40,59),f.issue_stage.ge(60)],
        ['GS<31','GS31-33','GS34-39','GS40-59','GS60+'],default='unknown')
    return f


def threshold_diagnostics(frame,cutoff):
    """Diagnostic severity cutoffs are not probabilities or treatment thresholds.

    Each coordinate-year has equal weight, with equal series and equal future
    assessments nested within it. AUC ranks continuous predicted percentages.
    """
    if not np.isfinite(cutoff) or not 0<cutoff<=100 or frame.empty:
        raise ValueError('A cutoff in(0,100] and nonempty target records are required.')
    y=frame.observed_percent.to_numpy(float)
    p=frame.predicted_percent.to_numpy(float)
    if (not np.isfinite(y).all() or not np.isfinite(p).all()
            or np.any((y<0)|(y>100)) or np.any((p<0)|(p>100))):
        raise ValueError('Observed and predicted percentages must be finite and within[0,100].')
    if frame[['coordinate_year','series_id']].isna().any().any():
        raise ValueError('Field-year and series identities are required.')
    w=hierarchical_weights(frame)
    truth=y>=cutoff;warning=p>=cutoff
    tp=float(w[truth&warning].sum());fn=float(w[truth&~warning].sum())
    fp=float(w[~truth&warning].sum());tn=float(w[~truth&~warning].sum())
    def ratio(a,b):return float(a/b) if b>0 else None
    # Sorted score groups give half credit to ties in weighted concordance.
    ranks=pd.DataFrame({'score':p,'positive':w*truth,'negative':w*~truth})
    ranks=ranks.groupby('score',sort=True).sum()
    below=ranks.negative.cumsum()-ranks.negative
    concordance=float((ranks.positive*(below+.5*ranks.negative)).sum())
    return {'cutoff_percent':float(cutoff),'n_targets':len(frame),
        'n_series':frame.series_id.nunique(),'n_coordinate_years':frame.coordinate_year.nunique(),
        'n_positive_targets':int(truth.sum()),'n_negative_targets':int((~truth).sum()),
        'weighted_prevalence':tp+fn,'weighted_true_positive':tp,
        'weighted_false_positive':fp,'weighted_true_negative':tn,'weighted_false_negative':fn,
        'sensitivity':ratio(tp,tp+fn),'specificity':ratio(tn,tn+fp),
        'precision':ratio(tp,tp+fp),'false_alarm_fraction':ratio(fp,tp+fp),
        'missed_case_fraction':ratio(fn,tp+fn),'auc':ratio(concordance,(tp+fn)*(tn+fp))}
