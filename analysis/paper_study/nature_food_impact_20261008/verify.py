"""Independent arithmetic, artifact and source-consistency checks."""
from pathlib import Path
import argparse
import hashlib
import json
import re
from zipfile import ZipFile
import numpy as np
import pandas as pd
from docx import Document
from openpyxl import load_workbook
import pymupdf
from .data import ROOT,HERE,GRID,DERIVED,CANOPY,SEVERITY,ONSET,FREQUENCY,YIELD,pooled,sha


def pooled_arithmetic():
    onset=pd.read_csv(DERIVED/'pooled_symptom_onset_membership.csv')
    means=onset.groupby(['coordinate_year','field_id','leaf_index'])[['model','benchmark']].mean()
    means=means.groupby(['coordinate_year','field_id']).mean().groupby('coordinate_year').mean().mean()
    for key in ['model','benchmark']:
        np.testing.assert_allclose(means[key],pooled('symptom_onset',key).value,rtol=0,atol=1e-12)
    severity=pd.read_csv(DERIVED/'pooled_severity_membership.csv')
    scores={}
    for key in ['model','benchmark']:
        error=severity[key]-severity.observed
        d=severity[['coordinate_year','field_id','leaf_index']].assign(squared=error**2,bias=error)
        x=d.groupby(['coordinate_year','field_id','leaf_index'])[['squared','bias']].mean()
        x=x.groupby(['coordinate_year','field_id']).mean().groupby('coordinate_year').mean().mean()
        scores[key+'_rmse']=float(np.sqrt(x.squared));scores[key+'_bias']=float(x.bias)
        for metric in ['rmse','bias']:
            np.testing.assert_allclose(scores[key+'_'+metric],pooled('severity',key+'_'+metric).value,rtol=0,atol=1e-12)
    assert len(severity)==2033 and severity.field_id.nunique()==188
    assert len(onset)==136 and onset.field_id.nunique()==78
    return dict(symptom_onset_days=means.to_dict(),severity=scores,assessment_records=2033,onset_histories=136)


def domain_arithmetic(directory=GRID,period='2071-2100'):
    directory=Path(directory)
    registry=pd.read_csv(directory/'full_landuse_cell_registry.csv')
    rows=pd.read_csv(directory/'environmental_region_changes_by_gcm.csv')
    country=pd.read_csv(directory/'country_changes_by_gcm.csv')
    metrics=[CANOPY,SEVERITY,ONSET,FREQUENCY,YIELD]
    observed=[];direct=True
    start=int(period[:4])
    paths=[ROOT/f'analysis/paper_study/nature_food_revision_20261007/continental_replay/annual_outputs/nasa/{model}/ssp585/{year}.parquet'
        for model in ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0'] for year in list(range(1991,2021))+list(range(start,start+30))]
    if not all(p.exists() for p in paths):direct=False
    for model in ['ACCESS-CM2','MPI-ESM1-2-HR','MRI-ESM2-0']:
        if direct:
            numerator=np.zeros((len(registry),len(metrics)));denominator=np.zeros_like(numerator)
            for i in range(30):
                base=ROOT/f'analysis/paper_study/nature_food_revision_20261007/continental_replay/annual_outputs/nasa/{model}/ssp585'
                a=pd.read_parquet(base/f'{1991+i}.parquet',columns=['cell_id',*metrics])
                b=pd.read_parquet(base/f'{start+i}.parquet',columns=['cell_id',*metrics])
                assert np.array_equal(a.cell_id,registry.cell_id) and np.array_equal(b.cell_id,registry.cell_id)
                av=a[metrics].to_numpy();bv=b[metrics].to_numpy();valid=np.isfinite(av)&np.isfinite(bv)
                numerator+=np.where(valid,bv-av,0);denominator+=valid
        else:
            pairs=pd.read_parquet(directory/'full_grid_paired_changes_by_gcm.parquet')
            pairs=pairs[pairs.model.eq(model)&pairs.scenario.eq('ssp585')&pairs.period.eq(period)]
            numerator=np.empty((len(registry),len(metrics)));denominator=np.empty_like(numerator)
            for j,metric in enumerate(metrics):
                q=pairs[pairs.metric.eq(metric)].set_index('cell_id').loc[registry.cell_id]
                denominator[:,j]=q.valid_year_pairs
                numerator[:,j]=np.where(q.valid_year_pairs>0,q.change*q.valid_year_pairs,0)
        for name,mask,reference in [
            ('Europe',np.ones(len(registry),bool),rows[rows.environment_region.eq('Europe')]),
            ('Boreal',registry.environment_region.eq('Boreal').to_numpy(),rows[rows.environment_region.eq('Boreal')]),
            ('Mediterranean',registry.environment_region.eq('Mediterranean').to_numpy(),rows[rows.environment_region.eq('Mediterranean')]),
            ('France',registry.dominant_source_country.eq('France').to_numpy(),country[country.country.eq('France')])]:
            w=registry.harvested_total_ha.to_numpy()*mask
            for j,metric in enumerate(metrics):
                denominator_sum=float(w@denominator[:,j])
                value=float(w@numerator[:,j]/denominator_sum) if denominator_sum else np.nan
                q=reference[reference.model.eq(model)&reference.scenario.eq('ssp585')&reference.period.eq(period)&reference.metric.eq(metric)].iloc[0]
                np.testing.assert_allclose(value,q.change,rtol=0,atol=1e-11,equal_nan=True)
                observed.append(dict(domain=name,model=model,metric=metric,change=value))
    return dict(status='passed',comparisons=len(observed),source='Annual simulation outputs' if direct else 'Bundled paired grid values and valid-year counts',period=period)


def verify(output):
    output=Path(output).resolve();main=(output/'Manuscript.txt').read_text();si=(output/'Supplementary_Information.txt').read_text()
    abstract=(output/'abstract.txt').read_text().strip()
    assert len(abstract.split())<=150 and not re.search(r'\[[1-9]\d*',abstract)
    assert main.count(abstract)==1
    # Archive names occur in source references; scientific Results do not divide
    # their observations into corporate datasets.
    result=main.split('\n\nResults\n\n',1)[1].split('\n\nDiscussion\n\n',1)[0]
    assert not re.search(r'\b(?:nan|inf)\b',result,flags=re.I)
    assert not re.search(r'BASF|Corteva|sampling weights|draw identities|area-proportional draws',result)
    assert not result.startswith('European wheat area')
    captions=json.loads((output/'figures/captions.json').read_text())
    assert len(captions)==4 and len(list((output/'figures').glob('*.png')))==4
    for stem,caption in captions.items():
        assert len(caption.split())<=220 and all((output/'figures'/f'{stem}.{ext}').exists() for ext in ['png','pdf','svg'])
    assert 'harvested area and field' not in list(captions.values())[0].lower()
    order=[int(n) for n in re.findall(r'(?<!Supplementary )Fig\. (\d+)',main.split('Figure legends')[0])]
    assert list(dict.fromkeys(order))==[1,2,3,4]
    declared=set(re.findall(r'^Table (S\d+[a-z]?) \|',si,flags=re.M))
    for identifier in re.findall(r'(?:Table|Tables) (S\d+[a-z]?)',main):
        assert identifier in declared or any(x.startswith(identifier) for x in declared),identifier
    doc=Document(output/'Manuscript.docx');assert len(doc.inline_shapes)==4 and len(doc.tables)==1
    word='\n'.join(p.text for p in doc.paragraphs);assert abstract in word
    i=next(i for i,p in enumerate(doc.paragraphs) if p.text==abstract)
    assert doc.paragraphs[i+1].paragraph_format.page_break_before
    pages={}
    for name in ['Manuscript.pdf','Supplementary_Information.pdf','Supplementary_Model_Specification.pdf']:
        pdf=pymupdf.open(output/name);pages[name]=len(pdf)
        assert all(len(p.get_text().strip())>5 for p in pdf)
        if name=='Manuscript.pdf':assert abstract in ' '.join(pdf[0].get_text().split())
    workbook=load_workbook(output/'Source_Data.xlsx',read_only=True)
    for sheet,name,count,expected in list(workbook['Source_manifest'].values)[1:]:
        path=output/Path(name).relative_to('publication/european_wheat_stb') if name.startswith('publication/european_wheat_stb/') else ROOT/name
        assert sha(path)==expected,name
    archive_count=None
    if (output/'Software_and_Evidence.zip').exists():
        with ZipFile(output/'Software_and_Evidence.zip') as z:
            assert z.testzip() is None
            manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
            for name,record in manifest['members'].items():
                content=z.read(name);assert hashlib.sha256(content).hexdigest()==record['sha256'] and len(content)==record['bytes'],name
            archive_count=len(manifest['members'])
    arithmetic=pooled_arithmetic();domains=domain_arithmetic()
    report=dict(status='passed',scope='Numerical reporting and provenance; biological and absolute-yield predictions remain unvalidated',
        abstract_words=len(abstract.split()),main_figures=4,main_tables=1,Results_focus='Full-grid climate impacts',
        pooled_evaluation=arithmetic,full_grid_domain_check=domains,PDF_pages=pages,
        source_workbook_sheets=len(workbook.sheetnames),archived_members_checked=archive_count)
    (output/'verification_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    print(json.dumps(verify(parser.parse_args().output),indent=2))
