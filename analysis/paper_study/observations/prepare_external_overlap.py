#!/usr/bin/env python3
"""Geography-only external subset registration and source organ dictionary."""
from pathlib import Path
import json,hashlib
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'data/paper_study/observations'
HERE=Path(__file__).resolve().parent


def coordinate(lat,lon):return f'{float(lat):.6f},{float(lon):.6f}'


def main():
    p=OUT/'new_sources/corteva-2014-2018/corteva-wheat-diseases.txt'
    source=pd.read_csv(p,sep='\t');candidate=source.loc[source.Treatment.eq('Untreated')&source.PestCode.eq('SEPTTR')&source.EvaluationType.eq('INFECT')]
    labels=sorted(candidate.SamplingUnit.dropna().unique())
    dictionary=[]
    for label in labels:
        rank=int(label.split(' ')[1]) if label.startswith('LEAF ') and label.split(' ')[1].isdigit() else None
        supported=rank is not None and 1<=rank<=7
        dictionary.append(dict(source_label=label,ordinal_leaf_rank=rank,supported_ordinal_rank_1_to_7=supported,
            mapping_basis='exact source LEAF n label' if rank is not None else 'unranked label; ordinal mapping unavailable',
            leaf_numbering='adult ordinal source label; not equated to juvenile L numbering',
            source_dataset='corteva-2014-2018-external'))
    pd.DataFrame(dictionary).to_csv(OUT/'corteva_organ_dictionary.csv',index=False)
    raw=pd.read_csv(ROOT/'data/basf-wheat-diseases.txt',sep='\t')
    b=raw.loc[raw.Treatment.eq('Untreated')&raw.Organism.eq('SEPTTR')&raw.Clarifier.ne('CROP INJURY')].copy()
    b['year']=b.Date.str[:4].astype(int);b['coordinate']=[coordinate(x,y) for x,y in zip(b.Latitude,b.Longitude)]
    ordinal={'LEAF, 1ST / FLAG LEAF','LEAF, 2ND','LEAF, 3RD','LEAF, 4TH','LEAF, 5TH','LEAF, 6ST','LEAF, 7ST'}
    b['ordinal']=b.PlantPart.isin(ordinal)
    all_locations=set(b.coordinate);ordinal_locations=set(b.loc[b.ordinal,'coordinate'])
    all_location_years=set(zip(b.coordinate,b.year));ordinal_location_years=set(zip(b.loc[b.ordinal,'coordinate'],b.loc[b.ordinal,'year']))
    requests=pd.read_csv(OUT/'corteva_weather_requests.csv');requests['coordinate']=[coordinate(x,y) for x,y in zip(requests.latitude,requests.longitude)]
    requests['potential_same_coordinate_as_any_BASF']=requests.coordinate.isin(all_locations)
    requests['potential_same_coordinate_as_ordinal_BASF']=requests.coordinate.isin(ordinal_locations)
    requests['potential_same_coordinate_year_as_any_BASF']=[(c,int(y)) in all_location_years for c,y in zip(requests.coordinate,requests.season_year)]
    requests['potential_same_coordinate_year_as_ordinal_BASF']=[(c,int(y)) in ordinal_location_years for c,y in zip(requests.coordinate,requests.season_year)]
    requests['matched_BASF_trials_any_year']=[';'.join(str(v) for v in sorted(b.loc[b.coordinate.eq(c),'TrialId'].unique())) for c in requests.coordinate]
    requests['matched_BASF_trials_same_year']=[';'.join(str(v) for v in sorted(b.loc[b.coordinate.eq(c)&b.year.eq(int(y)),'TrialId'].unique())) for c,y in zip(requests.coordinate,requests.season_year)]
    requests['confirmed_duplicate_field']=False
    requests['overlap_basis']='exact published coordinate after formatting to six decimals; both sources rounded to 0.1 degree; true field identity unknown'
    requests['subset_selection_uses_outcome_values']=False
    requests.to_csv(OUT/'corteva_BASF_geographic_overlap.csv',index=False)
    requests.to_csv(OUT/'corteva_full_source_transfer_units.csv',index=False)
    requests.loc[~requests.potential_same_coordinate_year_as_any_BASF].to_csv(OUT/'corteva_coordinate_year_disjoint_units.csv',index=False)
    requests.loc[~requests.potential_same_coordinate_as_any_BASF].to_csv(OUT/'corteva_location_disjoint_external_units.csv',index=False)
    receipt={'eligible_source_units':len(requests),'potential_coordinate_year_overlap_any_BASF':int(requests.potential_same_coordinate_year_as_any_BASF.sum()),
        'potential_coordinate_overlap_any_year_any_BASF':int(requests.potential_same_coordinate_as_any_BASF.sum()),
        'potential_coordinate_year_overlap_ordinal_BASF':int(requests.potential_same_coordinate_year_as_ordinal_BASF.sum()),
        'coordinate_year_disjoint_source_units':int((~requests.potential_same_coordinate_year_as_any_BASF).sum()),
        'location_disjoint_source_units':int((~requests.potential_same_coordinate_as_any_BASF).sum()),
        'subset_defined_by_geography_only':True,'model_imported_or_fitted':False,'confirmed_duplicate_field_claimed':False,
        'Corteva_source_SHA256':hashlib.sha256(p.read_bytes()).hexdigest(),
        'BASF_source_SHA256':hashlib.sha256((ROOT/'data/basf-wheat-diseases.txt').read_bytes()).hexdigest(),
        'rounding_limitation':'Disjoint published coordinates do not establish exact biological site independence; same rounded coordinates flag possible overlap.'}
    (HERE/'external_geographic_registration.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
