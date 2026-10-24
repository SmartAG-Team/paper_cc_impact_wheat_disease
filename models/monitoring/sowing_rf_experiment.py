"""Train the fixed two-predictor sowing RF and estimate disease-field dates."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .inputs import sha
from .sowing_rf import FEATURES,calendar_day,fit_sowing_rf,predict_sowing_rf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/disease_improvement/sowing_rf_20261022'


def score(frame,prediction):
    weight=1/frame.groupby('station_id').station_id.transform('size').to_numpy(float)
    error=np.asarray(prediction)-calendar_day(frame.sowing_date)
    return dict(n=len(frame),stations=frame.station_id.nunique(),
        mae_days=float(np.average(abs(error),weights=weight)),
        rmse_days=float(np.sqrt(np.average(error**2,weights=weight))),
        bias_days=float(np.average(error,weights=weight)))


def run():
    source=ROOT/'data/disease_improvement/sowing_rf_source_20261022'
    climate=ROOT/'data/disease_improvement/sowing_climate_20261022'
    if OUT.exists():
        raise FileExistsError('Fresh RF destination required')
    table=pd.read_csv(source/'sowing_training_table.csv').rename(columns={'lat':'latitude','lon':'longitude'})
    table=table.loc[table.annual_temperature_complete].copy()
    table['sowing_date']=pd.to_datetime(table.sowing_date)
    assert not table.ambiguous_sowing.any() and table.annualvaliddays.eq(table.expected_days).all()
    assert table.annual_temperature_year.eq(table.sowing_date.dt.year-1).all()
    observed=table.loc[table.sowing_date.lt('2013-01-01')].copy()
    assert len(observed)==32232
    target=pd.read_csv(source/'target_field_climate.csv')
    annual=pd.read_csv(climate/'annual_temperature.csv')
    target=target.merge(annual,on=['site_id','annual_temperature_year'],how='left',validate='m:1')
    fallback=~target.annual_temperature_complete
    if not target.loc[fallback,'valid_months'].eq(12).all():
        raise ValueError('Missing complete annual temperature; no partial-year imputation allowed')
    target['temperature_product_transfer']=fallback
    target.loc[fallback,'annual_mean_temperature_c']=target.loc[fallback,'era5_annual_mean_temperature_c']
    target.loc[fallback,'annual_temperature_source']='ERA5_LAND_MONTHLY_AGGR_prior_calendar_year'
    target['outside_german_source_country']=target.country.ne('GERMANY')
    assert len(target)==218 and np.isfinite(target[FEATURES]).all().all()
    OUT.mkdir(parents=True)
    sources=[source/n for n in ['sowing_training_table.csv','target_field_climate.csv','registration.json']]
    sources += [climate/n for n in ['annual_temperature.csv','registration.json']]
    sources += [ROOT/'data/disease_improvement/station_preferred_ie_20261022/fields.csv']
    codes=[Path(__file__),Path(__file__).with_name('sowing_rf.py')]
    registration=dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        user_instruction='RF with annualmeanT andlatitude from EJA-PGN-Wheat; current disease cohort evaluation accepted',
        application_fit_cutoff='2012-12-31',features=FEATURES,
        model=dict(n_estimators=400,max_depth=12,min_samples_leaf=20,max_features=1.,random_state=20261022,station_equal_weights=True),
        source_selection='Unambiguous source-QC accepted observed sowing with complete365/366day precedingcalendar weather; preserve QC-valid extreme dates',
        annual_temperature='Previous complete calendar year, before autumn sowing; DWD<=50km where complete, explicitERA5-Land fallback at target sites',
        disease_labels_used=False,model_selection_on_disease_outcomes=False,
        spatial_auxiliary_test='Deterministic hashstation20percent holdout; no stationidentity asRFpredictor',
        disease_spatial_interpretation='Sowing RF is a globally shared auxiliary model;50km exclusions apply to disease-fitting observations. Auxiliary geographical transfer assessed separately.',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes})
    (OUT/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for path in codes:
        p=OUT/'code_snapshot'/path.relative_to(ROOT);p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(path.read_bytes())
    fitted=fit_sowing_rf(observed)
    joblib.dump(fitted,OUT/'fitted_sowing_rf.joblib')
    held={int(s) for s in observed.station_id.unique() if int(hashlib.sha256(str(int(s)).encode()).hexdigest(),16)%5==0}
    spatial_fit=fit_sowing_rf(observed.loc[~observed.station_id.isin(held)])
    joblib.dump(spatial_fit,OUT/'spatial_validation_rf.joblib')
    validation=[];metrics=[]
    masks=dict(temporal=table.sowing_date.between('2013-01-01','2019-12-31'),
        spatial=table.sowing_date.lt('2013-01-01')&table.station_id.isin(held),
        temporal_spatial=table.sowing_date.between('2013-01-01','2019-12-31')&table.station_id.isin(held))
    for name,mask in masks.items():
        part=table.loc[mask].copy();bundle=fitted if name=='temporal' else spatial_fit
        p=bundle['estimator'].predict(part[FEATURES].to_numpy(float))
        part['validation']=name;part['predicted_day_nonleap']=p;part['baseline_day_nonleap']=bundle['median_baseline_day']
        validation.append(part)
        for label,values in [('rf',p),('training_median',np.full(len(part),bundle['median_baseline_day']))]:
            metrics.append(dict(validation=name,model=label,**score(part,values)))
    pd.concat(validation,ignore_index=True).to_csv(OUT/'sowing_validation_predictions.csv',index=False)
    pd.DataFrame(metrics).to_csv(OUT/'sowing_validation_metrics.csv',index=False)
    queries=target[['field_id','site_id','season_year','latitude','longitude','country',
        'annual_mean_temperature_c','annual_temperature_year','annual_temperature_source',
        'temperature_product_transfer','outside_german_source_country']].copy()
    prediction=predict_sowing_rf(fitted,queries)
    old=pd.read_csv(sources[-1]).set_index('field_id')
    prediction['previous_scenario_sowing_date']=prediction.field_id.map(old.sowing_date)
    prediction['change_from_scenario_days']=(pd.to_datetime(prediction.estimated_sowing_date)-pd.to_datetime(prediction.previous_scenario_sowing_date)).dt.days
    prediction.to_csv(OUT/'field_estimated_sowing.csv',index=False)
    summary=dict(training_rows=fitted['training_rows'],training_stations=len(fitted['training_station_ids']),
        maximum_training_date=fitted['maximum_training_date'],target_field_seasons=len(prediction),
        dwd_annual_temperature_fields=int((~prediction.temperature_product_transfer).sum()),
        reanalysis_annual_temperature_fields=int(prediction.temperature_product_transfer.sum()),
        latitude_outside_support=int(prediction.outside_latitude_support.sum()),
        temperature_outside_support=int(prediction.outside_temperature_support.sum()),
        non_german_application_fields=int(prediction.outside_german_source_country.sum()),
        date_change_days=prediction.change_from_scenario_days.describe().to_dict(),
        feature_importance=dict(zip(FEATURES,fitted['estimator'].feature_importances_.tolist())),
        sowing_observed_at_application_fields=False,independent_disease_validation=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(pd.DataFrame(metrics).to_string(index=False),flush=True)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':
    run()
