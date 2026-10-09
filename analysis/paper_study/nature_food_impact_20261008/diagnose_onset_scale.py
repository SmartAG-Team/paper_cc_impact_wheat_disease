"""Inspect onset-map display bins and independently check recorded date arithmetic."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .data import ROOT,GRID,HERE,MODELS


def main():
    metric='F1_symptom_day_after_sowing'
    source=pd.read_parquet(GRID/'full_grid_ensemble_paired_changes.parquet')
    frame=source[source.metric.eq(metric)&source.scenario.eq('ssp585')&source.period.eq('2071-2100')&source.mean_change.notna()].copy()
    area=frame.harvested_total_ha.sum();median=float(frame.mean_change.median())
    summary=dict(valid_cells=len(frame),median_change_days=median,
        minimum_change_days=float(frame.mean_change.min()),maximum_change_days=float(frame.mean_change.max()),
        central90_percent_range_days=frame.mean_change.quantile([.05,.95]).tolist(),
        previous_display_bin=[-96,-20],
        previous_bin_area_percent=100*frame.loc[frame.mean_change.lt(-20),'harvested_total_ha'].sum()/area,
        below_minus90_cells=int(frame.mean_change.lt(-90).sum()),
        below_minus90_area_percent=100*frame.loc[frame.mean_change.lt(-90),'harvested_total_ha'].sum()/area,
        regional_aggregation='Regional means aggregate valid paired seasons within each climate model, before three-model averaging.')
    regions=pd.read_csv(GRID/'environmental_region_changes.csv')
    regional=regions[regions.environment_region.eq('Europe')&regions.metric.eq(metric)&regions.scenario.eq('ssp585')&regions.period.eq('2071-2100')].iloc[0]
    summary['europe_area_time_weighted_change_days']=float(regional.mean_change)
    picked=frame.nsmallest(5,'mean_change').copy()
    typical=frame.iloc[(frame.mean_change-median).abs().argmin()]
    identifiers=list(dict.fromkeys([*picked.cell_id,typical.cell_id]))
    outputs=[]
    columns=['cell_id','calendar_sowing_date','F1_symptom_date',metric,
        'F1_symptom_relative_anthesis_days','BBCH31_date','BBCH65_date','BBCH85_date']
    directory=ROOT/'analysis/paper_study/nature_food_revision_20261007/continental_replay/annual_outputs/nasa'
    for model in MODELS:
        for period,years in [('1991-2020',range(1991,2021)),('2071-2100',range(2071,2101))]:
            for year in years:
                path=directory/model/'ssp585'/f'{year}.parquet'
                records=pd.read_parquet(path,columns=columns,filters=[('cell_id','in',identifiers)])
                for column in ['calendar_sowing_date','F1_symptom_date','BBCH31_date','BBCH65_date','BBCH85_date']:
                    records[column]=pd.to_datetime(records[column])
                available=records.F1_symptom_date.notna()
                elapsed=(records.F1_symptom_date-records.calendar_sowing_date).dt.days
                np.testing.assert_allclose(elapsed[available],records.loc[available,metric],atol=0)
                flowering_relative=(records.F1_symptom_date-records.BBCH65_date).dt.days
                np.testing.assert_allclose(flowering_relative[available],records.loc[available,'F1_symptom_relative_anthesis_days'],atol=0)
                assert elapsed[available].ge(0).all()
                assert records.loc[available,'F1_symptom_date'].le(records.loc[available,'BBCH85_date']).all()
                records['model']=model;records['period']=period;records['pair_index']=year-(1991 if period=='1991-2020' else 2071)
                outputs.append(records)
    audited=pd.concat(outputs,ignore_index=True)
    past=audited[audited.period.eq('1991-2020')]
    future=audited[audited.period.eq('2071-2100')]
    pairs=past.merge(future,on=['cell_id','model','pair_index'],suffixes=('_past','_future'),validate='one_to_one')
    pairs['change']=pairs[metric+'_future']-pairs[metric+'_past']
    model_changes=pairs.groupby(['cell_id','model']).change.mean()
    ensemble=model_changes.groupby('cell_id').mean()
    target=frame.set_index('cell_id').mean_change.loc[ensemble.index]
    np.testing.assert_allclose(ensemble,target,atol=1e-11)
    summary.update(date_arithmetic='passed',audited_grid_seasons=len(audited),audited_cells=len(identifiers),
        independent_ensemble_means='passed',date_reference='Actual elapsed calendar days from each recorded sowing date; no day-of-year subtraction',
        model_parameters_changed=False,raw_climate_outputs_changed=False)
    output=HERE/'derived';output.mkdir(exist_ok=True)
    audited.to_csv(output/'onset_map_date_audit.csv',index=False)
    frame[['cell_id','latitude','longitude','mean_change','harvested_total_ha','dominant_source_country']].sort_values('mean_change').head(10).to_csv(output/'onset_map_extreme_cells.csv',index=False)
    (output/'onset_map_scale_diagnosis.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
