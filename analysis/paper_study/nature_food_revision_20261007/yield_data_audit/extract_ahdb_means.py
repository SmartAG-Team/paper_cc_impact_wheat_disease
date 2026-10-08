"""Extract public AHDB annual summaries and preserve pairing limitations."""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent

def main():
    yields, disease = [], []
    for file in sorted(ROOT.glob('ahdb*.xlsx')):
        year = int(file.stem[-4:])
        excel = pd.ExcelFile(file)
        for sheet in excel.sheet_names:
            isyield = sheet in ['ww1yt-yld','ww1yu-yld']
            isdisease = sheet in ['Mildew','Yellow rust','Brown rust','Septoria','ww1yu-mild','ww1yu-yr','ww1yu-br','ww1yu-sept']
            if not (isyield or isdisease):
                continue
            cells = pd.read_excel(excel, sheet_name=sheet, header=None)
            for ci in range(cells.shape[1]):
                trial = cells.iat[3,ci]
                if not isinstance(trial,str) or not trial[0].isdigit() or len(trial)<5:
                    continue
                for ri in range(14,cells.shape[0]):
                    name, code = cells.iat[ri,0], cells.iat[ri,1]
                    if not isinstance(code,str) or not code.startswith('WW'):
                        continue
                    value = cells.iat[ri,ci]
                    if not isinstance(value,(float,int)) or not pd.notna(value):
                        continue
                    item = dict(year=year,trial_id=trial,site_pair_id=trial[:-1],trial_type=trial[-1],
                        site_name=cells.iat[4,ci],county=cells.iat[5,ci],cultivar=name,cultivar_code=code,
                        measurement=cells.iat[2,1],value=float(value),source_file=file.name,
                        source_sheet=sheet,source_cell=f'R{ri+1}C{ci+1}',
                        source_trial_mean=cells.iat[7,ci],source_lsd_5percent=cells.iat[8,ci],
                        source_cv_percent=cells.iat[9,ci],sowing_date=cells.iat[10,ci],
                        previous_crop=cells.iat[11,ci],soil_type=cells.iat[12,ci])
                    (yields if isyield else disease).append(item)
    y, d = pd.DataFrame(yields), pd.DataFrame(disease)
    y.to_csv(ROOT/'ahdb_yield_site_cultivar_means.csv', index=False)
    d.to_csv(ROOT/'ahdb_disease_site_cultivar_means.csv', index=False)
    pairs = y[y.trial_type=='T'].merge(y[y.trial_type=='U'],
        on=['year','site_pair_id','cultivar_code'],suffixes=('_treated','_untreated'),validate='one_to_one')
    pairs['fungicide_response_t_ha'] = pairs.value_treated-pairs.value_untreated
    pairs['untreated_shortfall_relative_to_treated'] = 1-pairs.value_untreated/pairs.value_treated
    for col in ['site_name','county','sowing_date','previous_crop','soil_type']:
        pairs[col+'_matches'] = pairs[col+'_treated'].fillna('')==pairs[col+'_untreated'].fillna('')
    matches = [col+'_matches' for col in ['site_name','county','sowing_date','previous_crop','soil_type']]
    pairs['all_pair_metadata_match'] = pairs[matches].all(axis=1)
    pairs['design_status'] = 'Distinct T/U trial IDs at same site; public summary does not establish randomized paired blocks.'
    pairs.to_csv(ROOT/'ahdb_matched_fungicide_yield_pairs.csv', index=False)
    pairs[pairs.all_pair_metadata_match].to_csv(ROOT/'ahdb_strict_metadata_matched_fungicide_yield_pairs.csv',index=False)
    joined = []
    for _, pair in pairs.iterrows():
        dx = d[(d.year==pair.year)&(d.trial_id==pair.trial_id_untreated)&(d.cultivar_code==pair.cultivar_code)]
        for _, obs in dx.iterrows():
            joined.append(dict(year=pair.year,site_pair_id=pair.site_pair_id,cultivar_code=pair.cultivar_code,
                disease=obs.measurement,disease_value_percent=obs.value,disease_trial_id=obs.trial_id,
                disease_source_file=obs.source_file,disease_source_sheet=obs.source_sheet,disease_source_cell=obs.source_cell,
                untreated_yield_t_ha=pair.value_untreated,treated_yield_t_ha=pair.value_treated,
                fungicide_response_t_ha=pair.fungicide_response_t_ha,
                relative_untreated_shortfall=pair.untreated_shortfall_relative_to_treated,
                all_pair_metadata_match=pair.all_pair_metadata_match,
                leaf_scope='Protocol assesses upper four leaves; individual leaf ranks not available in annual summaries.',
                assessment_date=None,healthy_area_duration=None))
    j = pd.DataFrame(joined)
    j.to_csv(ROOT/'ahdb_yield_pairs_with_same_untreated_trial_disease.csv',index=False)
    j[j.all_pair_metadata_match].to_csv(ROOT/'ahdb_strict_pairs_with_same_untreated_trial_disease.csv',index=False)
    info = dict(yield_cells=len(y),disease_cells=len(d),candidate_pairs=len(pairs),
        candidate_site_years=len(pairs[['year','site_pair_id']].drop_duplicates()),
        strict_pairs=int(pairs.all_pair_metadata_match.sum()),
        strict_site_years=len(pairs[pairs.all_pair_metadata_match][['year','site_pair_id']].drop_duplicates()),
        negative_responses=int((pairs.fungicide_response_t_ha<0).sum()),
        strict_negative_responses=int(((pairs.fungicide_response_t_ha<0)&pairs.all_pair_metadata_match).sum()),
        disease_scope='Top four leaves under published RL assessment protocol; actual assessment dates and leaf-specific observations absent from these summaries.',
        protected_definition='Fungicide-treated yield trial using full RL fungicide and PGR programme; untreated trial receives PGR but no fungicides.',
        yield_role='Matched site/cultivar fungicide response, not an identified STB-only disease-free yield counterfactual.')
    (ROOT/'ahdb_extraction_summary.json').write_text(json.dumps(info,indent=2))
    print(json.dumps(info,indent=2))

if __name__=='__main__':
    main()
