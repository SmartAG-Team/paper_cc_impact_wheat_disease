"""Independent source reconciliation and grouped-score calculations."""
from pathlib import Path
import hashlib
import json
import zipfile
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
COLLECTION=ROOT/'data/stb_collection_20261009'
NORDIC=ROOT/'analysis/paper_study/nordic_yield_validation_20261009'


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def expanded_cells(table):
    occupied={}
    rows=table.find_all('tr')
    for i,row in enumerate(rows):
        j=0
        for cell in row.find_all(['td','th'],recursive=False):
            while (i,j) in occupied:j+=1
            value=cell.get_text(' ',strip=True)
            for di in range(int(cell.get('rowspan',1))):
                for dj in range(int(cell.get('colspan',1))):occupied[(i+di,j+dj)]=value
            j+=int(cell.get('colspan',1))
    width=max(j for i,j in occupied)+1
    return [[occupied.get((i,j),'') for j in range(width)] for i in range(len(rows))]


def native_nordic_cells():
    source=COLLECTION/'sources/public_septoria/nfts-extension-20261009'
    records=pd.read_csv(source/'nfts_stb_severity_yield_matched.csv')
    checked=0
    for registry,part in records.groupby('registry_id'):
        soup=BeautifulSoup((source/f'nfts_native_{registry}_sv.html').read_text(),'html.parser')
        grids={}
        for table in soup.find_all('table'):
            identifier=table.get('id','')
            if not identifier.startswith('tabResultatLedNieveau'):continue
            grid=expanded_cells(table)
            label='yield' if 'Skörd dt/ha kärna 15%' in grid[2] else 'disease'
            if label=='disease' and 'Nedre konf.' in grid[2]:continue
            grids[(identifier,label)]=grid
        for row in part.itertuples():
            for kind in ['disease','yield']:
                grid=grids[(getattr(row,f'source_table_id_{kind}'),kind)]
                cell=grid[getattr(row,f'source_row_{kind}')][getattr(row,f'source_column_{kind}')]
                number=float(cell.replace(',','.'))
                assert number==getattr(row,f'value_{kind}'),(registry,kind,cell)
                checked+=1
            assert row.native_swedish_measurement_label_yield=='Skörd dt/ha kärna 15%'
            np.testing.assert_allclose(row.value_yield*.1,row.yield_t_ha_15percent_moisture,atol=1e-12)
    return checked


def native_tunisia():
    paper=COLLECTION/'sources/public_septoria/paper-model-data-20261009/derived'
    archive=COLLECTION/'sources/public_septoria/durum-mixtures-2020/Durum_mixtures_Data_out.zip'
    with zipfile.ZipFile(archive) as z:
        raw_severity=pd.read_excel(z.open('out/Cultivar_Mix_severity_means_data_all.xls'))
        ys=[]
        for year in [2018,2019]:
            y=pd.read_excel(z.open(f'out/{year}_Tunisia_Cultivar_Mix_Yield_Data.xls'));y['year']=year
            if year==2018:y['inoculate_no_fungicide']=1
            ys.append(y)
        raw_yield=pd.concat(ys,ignore_index=True)
        for year in [2018,2019]:
            for name,layer in [('flag',1),('flag-1',2)]:
                assert set(pd.read_excel(z.open(f'out/{year}_Output_final_{name}.xls')).leafLayer)=={layer}
    extracted=pd.read_csv(paper/'tunisia2020_damage_yield_leaf_records.csv')
    joined=extracted.merge(raw_severity,on=['year','treatment','replicate','leafLayer'],validate='one_to_one',suffixes=('_extract','_raw'))
    for key in ['PLACL','incidence','PLACL_weighted']:
        np.testing.assert_allclose(joined[key+'_extract'],joined[key+'_raw'],atol=1e-12)
    np.testing.assert_allclose(extracted.PLACL*extracted.incidence,extracted.PLACL_weighted,atol=1e-10)
    paired=extracted.merge(raw_yield[raw_yield.inoculate_no_fungicide.eq(1)],on=['year','treatment','replicate'],validate='many_to_one',suffixes=('_extract','_raw'))
    np.testing.assert_allclose(paired.kg_per_ha_extract,paired.kg_per_ha_raw,atol=1e-10)
    # All 40 protected-reference contrasts are independently reconstructed from
    # raw yields; neither disease severity nor a protected zero is added.
    key=['year','treatment','replicate']
    a=raw_yield[raw_yield.year.eq(2019)&raw_yield.inoculate_no_fungicide.eq(1)]
    b=raw_yield[raw_yield.year.eq(2019)&raw_yield.inoculate_no_fungicide.eq(0)]
    raw_pairs=a.merge(b,on=key,validate='one_to_one',suffixes=('_u','_p'))
    published=pd.read_csv(paper/'tunisia2019_protected_reference_yield_contrasts.csv')
    q=published.merge(raw_pairs,on=key,validate='one_to_one')
    np.testing.assert_allclose(q.relative_yield_gap_pct,100*(1-q.kg_per_ha_u/q.kg_per_ha_p),atol=1e-12)
    return dict(leaf_records=len(joined),yield_records=len(raw_yield),reference_contrasts=len(q))


def grouped_scores():
    comparisons=0
    for directory,normal in [(HERE,False),(NORDIC,True)]:
        predictions=pd.read_csv(directory/'held_out_predictions.csv')
        metrics=pd.read_csv(directory/'model_comparison_metrics.csv')
        for row in metrics.itertuples():
            frame=predictions[predictions.domain.eq(row.domain)&predictions.model.eq(row.model)]
            if normal:
                frame=frame[frame.validation.eq(row.validation)]
                observed=frame.response_fraction
                group='registry_id';scale=100;reported=row.RMSE_pp
                membership='training_trials'
            else:
                if row.domain in ['2018_to_2019','2019_to_2018']:
                    frame=predictions[predictions.domain.eq('absolute_year_transfer')&predictions.model.eq(row.model)&predictions.year.eq(int(row.domain[-4:]))]
                observed=frame.observed
                group=frame.validation_group.iloc[0]
                scale=1 if row.units=='t ha-1' else 100
                reported=row.RMSE;membership='training_groups'
            error=frame.prediction-observed
            per_group=pd.DataFrame({'group':frame[group],'square':error**2,'absolute':abs(error),'bias':error}).groupby('group').mean()
            np.testing.assert_allclose(scale*np.sqrt(per_group.square.mean()),reported,atol=1e-12)
            np.testing.assert_allclose(scale*per_group.absolute.mean(),row.MAE_pp if normal else row.MAE,atol=1e-12)
            for record in frame.itertuples():
                assert str(getattr(record,group)) not in str(getattr(record,membership)).split('|')
            comparisons+=1
    positive=pd.read_csv(NORDIC/'positive_stage_predictions.csv')
    metrics=pd.read_csv(NORDIC/'positive_stage_comparison.csv')
    for row in metrics.itertuples():
        if row.model!='positive_stage_linear':continue
        frame=positive[positive.model.eq(row.model)]
        error=frame.prediction-frame.response_fraction
        mse=pd.DataFrame({'trial':frame.registry_id,'squared':error**2}).groupby('trial').squared.mean().mean()
        np.testing.assert_allclose(100*np.sqrt(mse),row.RMSE_pp,atol=1e-12)
    predictions=pd.read_csv(HERE/'briwecs_held_out_predictions.csv')
    metrics=pd.read_csv(HERE/'briwecs_model_comparison_metrics.csv')
    for row in metrics.itertuples():
        frame=predictions[predictions.domain.eq(row.domain)&predictions.validation.eq(row.validation)&predictions.model.eq(row.model)]
        group=frame.validation_group.iloc[0]
        means=frame.assign(error2=(frame.prediction-frame.response_fraction)**2).groupby(group).error2.mean()
        np.testing.assert_allclose(100*np.sqrt(means.mean()),row.RMSE_pp,atol=1e-12)
        for record in frame.itertuples():
            assert str(getattr(record,group)) not in str(record.training_groups).split('|')
        comparisons+=1
    return comparisons


def collection_and_readiness():
    manifest=pd.read_csv(COLLECTION/'source_copy_manifest.csv')
    inventory=pd.read_csv(COLLECTION/'file_inventory.csv').set_index('relative_path')
    original_comparisons=0
    for row in manifest.itertuples():
        copied=sha(COLLECTION/row.packaged_path)
        if row.packaged_path in inventory.index:
            expected=inventory.loc[row.packaged_path,'sha256']
        else:
            # Collection inventories omit nested files also called
            # file_inventory.csv; their original packaged bytes remain in ZIP.
            archive_path=ROOT/'data/STB_research_sources_20261009.zip'
            if not archive_path.is_file():
                archive_path=ROOT/'data/STB_consolidated_collection_20261009.zip'
            with zipfile.ZipFile(archive_path) as archive:
                expected=hashlib.sha256(archive.read('stb_collection_20261009/'+row.packaged_path)).hexdigest()
        assert copied==expected
        if 'packaged_sha256' in manifest:
            assert copied==row.packaged_sha256
        original=ROOT/row.original_project_path
        if original.is_file():
            assert sha(original)==(row.original_sha256 if 'original_sha256' in manifest else copied)
            original_comparisons+=1
    catalogue=pd.read_csv(COLLECTION/'dataset_catalogue.csv')
    catalogue['modelling_scope']=catalogue.evidence_type
    catalogue.to_csv(HERE/'source_readiness.csv',index=False)
    briwecs=COLLECTION/'sources/public_septoria/briwecs-2025/extracted/Briwecs_data'
    rows=[]
    for path in sorted((briwecs/'data/locations').glob('*.csv')):
        try:content=path.read_text(encoding='utf-8-sig');encoding='utf-8-sig'
        except UnicodeDecodeError:content=path.read_text(encoding='cp1252');encoding='cp1252'
        sep=';' if ';' in content.splitlines()[0] else ','
        frame=pd.read_csv(path,sep=sep,encoding=encoding)
        if 'Septoria' not in frame or 'Seedyield' not in frame:
            rows.append(dict(file=path.name,source_rows=len(frame),paired_numeric_records=0,invalid_severity_tokens=0,scope='Disease/yield column absent'))
            continue
        severity=pd.to_numeric(frame.Septoria,errors='coerce');grain=pd.to_numeric(frame.Seedyield,errors='coerce')
        rows.append(dict(file=path.name,source_rows=len(frame),paired_numeric_records=int((severity.notna()&grain.notna()).sum()),
            invalid_severity_tokens=int((frame.Septoria.notna()&severity.isna()).sum()),
            severity_min=severity.min(),severity_max=severity.max(),
            source_file_sha256=sha(path),scope='Source numeric inventory; date and leaf scope not harmonized'))
    profile=pd.DataFrame(rows);profile.to_csv(HERE/'briwecs_source_profile.csv',index=False)
    # Older Nordic archive comprises yield, weather-derived recommendations and
    # regional average calendars; none supplies paired observed disease severity.
    older=COLLECTION/'sources/public_septoria/nordic-baltic-2012-2016'
    old=[]
    for path in sorted(older.glob('*sasdataset*.xls')):
        frame=pd.read_excel(path,header=5)
        old.append(dict(file=path.name,records=len(frame),columns=list(frame.columns)))
    return dict(copied_sources_verified=len(manifest),source_groups=len(catalogue),
        original_path_hash_comparisons=original_comparisons,
        briwecs_paired_numeric_inventory=int(profile.paired_numeric_records.sum()),
        briwecs_paired_site_year_files=int(profile.paired_numeric_records.gt(0).sum()),
        briwecs_invalid_tokens=int(profile.invalid_severity_tokens.sum()),
        briwecs_status='Original-site protection responses tested separately; dry-mass yield and undocumented disease timing limit physiological interpretation.',
        older_nordic_tables=old)


def briwecs_source_means():
    # Recompute treatment means directly from the raw site files. The reader
    # does not call the preparation/model code used to create the features.
    pairs=pd.read_csv(HERE/'briwecs_treatment_response_features.csv')
    directory=COLLECTION/'sources/public_septoria/briwecs-2025/extracted/Briwecs_data/data/locations'
    checked=0
    for (location,year),group in pairs.groupby(['Location','Year']):
        path=directory/f'{location}_{year}.csv'
        text=path.read_text(encoding='utf-8-sig')
        raw=pd.read_csv(path,sep=';' if ';' in text.splitlines()[0] else ',')
        raw['Seedyield']=pd.to_numeric(raw.Seedyield,errors='coerce')
        raw['Septoria']=pd.to_numeric(raw.Septoria,errors='coerce')
        for row in group.itertuples():
            for protected in [False,True]:
                suffix='protected' if protected else 'unprotected'
                treatment=row.nitrogen+('_WF' if protected else '_NF')
                values=raw[raw.BRISONr.eq(row.BRISONr)&raw.Treatment.eq(treatment)&raw.Seedyield.gt(0)]
                np.testing.assert_allclose(values.Seedyield.mean(),getattr(row,'Seedyield_'+suffix),atol=1e-12)
                np.testing.assert_allclose(values.Septoria.mean(),getattr(row,'Septoria_'+suffix),atol=1e-12,equal_nan=True)
                checked+=1
            np.testing.assert_allclose(row.response_fraction,1-row.Seedyield_unprotected/row.Seedyield_protected,atol=1e-12)
    return checked


def main():
    result=dict(status='passed',native_nordic_numeric_cells=native_nordic_cells(),
        native_tunisia=native_tunisia(),briwecs_source_treatment_means=briwecs_source_means(),grouped_metric_comparisons=grouped_scores(),
        collection=collection_and_readiness())
    (HERE/'verification_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='collection'},indent=2))
    print(json.dumps({k:v for k,v in result['collection'].items() if k!='older_nordic_tables'},indent=2))


if __name__=='__main__':main()
