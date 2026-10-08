#!/usr/bin/env python3
"""Extract the official regional survey, without stacking the duplicate ADAS source."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/paper_study/observations'
HERE=Path(__file__).resolve().parent
SOURCE=OUT/'new_sources/Defra_regional-mean-time-series-wheat-october-2025.xlsx'


def main():
    repo=pd.read_csv(OUT/'new_sources/ADAS_data_pest_data.csv');rows=[];diffs=[]
    for leaf,sheet in [('L1','Leaf 1 (Flag leaf)'),('L2','Leaf 2'),('L3','Leaf 3')]:
        f=pd.read_excel(SOURCE,sheet_name=sheet,header=None)
        columns=[i for i,v in f.iloc[1].items() if str(v).strip()=='Zymoseptoria_tritici']
        assert len(columns)==3, (sheet,columns)
        for idx,row in f.iloc[2:].iterrows():
            year=pd.to_numeric(row[1],errors='coerce')
            if not np.isfinite(year):continue
            for column,metric in zip(columns,['disease_severity_percent','plant_incidence_percent','crop_incidence_percent']):
                value=pd.to_numeric(row[column],errors='coerce')
                if np.isfinite(value):assert 0<=value<=100
                rows.append(dict(dataset_id='defra-official-regional-survey',source_sheet=sheet,source_excel_row=idx+1,
                    source_excel_column=column+1,season_year=int(year),region=row[2],organ=leaf,metric=metric,unit='percent',
                    value=value,raw_value=row[column],timing_basis='annual summer survey; generally GS73-75; no exact assessment date',
                    management_status='commercial crop; fungicide status not separated',
                    independent_unit='region-year aggregate; constituent field identities unavailable',
                    duplicate_source_family='ADAS/Defra Survey; not independent of SPHERE-PPL annual dataset',source_file=str(SOURCE.relative_to(ROOT))))
                if leaf in ('L1','L2') and metric!='plant_incidence_percent':
                    col=f'{leaf}_Zymoseptoria_tritici_'+('Disease_Severity' if metric.startswith('disease') else 'Crop_Incidence')
                    m=repo[(repo.Year==int(year))&(repo.Region==row[2])]
                    if len(m) and np.isfinite(value) and not np.isclose(value,m.iloc[0][col]):
                        diffs.append(dict(year=int(year),region=row[2],organ=leaf,metric=metric,official=float(value),github=float(m.iloc[0][col])))
    result=pd.DataFrame(rows);result.to_csv(OUT/'defra_official_regional_target_inventory.csv',index=False)
    receipt={'source_SHA256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),'rows':len(result),'numeric_cells':int(result.value.notna().sum()),
        'years':sorted(result.season_year.unique().tolist()),'same_survey_as_github':True,'not_additional_independent_cohort':True,
        'overlapping_cell_differences':diffs,'source_chosen_for_current_29_column_inventory':'SPHERE-PPL subset; official Excel is separate corroborating/extended inventory; no duplicate stacking',
        'model_imported_or_fitted':False}
    (HERE/'defra_regional_inventory.json').write_text(json.dumps(receipt,indent=2,default=int)+'\n')
    print('Official regional cells',len(result),'numeric',result.value.notna().sum(),'source differences',len(diffs))


if __name__=='__main__':main()
