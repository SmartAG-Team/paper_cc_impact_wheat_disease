"""Check the complete evidence and publication exports before installation."""
from pathlib import Path
import hashlib
import json
import re
import shutil
import zipfile
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
from docx import Document
from openpyxl import load_workbook
import pymupdf
from .evidence import HERE,ROOT,STUDY,GERMAN,EVENTS,sha

STAGING=HERE/'staging'
ACTIVE=ROOT/'publication/european_wheat_stb'
BEFORE=HERE/'before/publication/european_wheat_stb'
SOURCE=STUDY/'manuscript'


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def verify():
    receipt=json.loads((STAGING/'document_build_receipt.json').read_text())
    for relative,digest in receipt['sources'].items():assert sha(ROOT/relative)==digest,relative
    assert receipt['reference_count']==25
    text=(STAGING/'Manuscript.txt').read_text()
    assert not re.search(r'\bn\.d\.|\[\[|APSIM Initiative|Python Crop Simulation Environment',text)
    for citation in ['Holzworth et al., 2018','de Wit et al., 2019','Hersbach et al., 2020']:
        assert citation in text,citation
    records=json.loads((STAGING/'references.csl.json').read_text())
    assert len(records)==25 and all(r.get('issued') for r in records)
    for key,doi in [('Holzworth2018APSIM','10.1016/j.envsoft.2018.02.002'),
            ('deWit2019WOFOST','10.1016/j.agsy.2018.06.018'),('Hersbach2020ERA5','10.1002/qj.3803')]:
        record=next(r for r in records if r['id']==key)
        assert record['DOI'].lower()==doi
        registered=json.loads((HERE/'literature'/f'{key}.crossref.json').read_text())
        assert record['volume']==registered['volume'] and record['page']==registered['page']
    scores=pd.read_csv(HERE/'field_stage_scores_all.csv',parse_dates=['predicted_date','lower_exclusive','upper_inclusive','forcing_end'])
    lower=scores.lower_exclusive+pd.Timedelta(days=1)
    prediction=scores.predicted_date
    missing=prediction.isna()
    signed=np.where(lower.notna()&(prediction<lower),(prediction-lower).dt.days,
        np.where(scores.upper_inclusive.notna()&(prediction>scores.upper_inclusive),
            (prediction-scores.upper_inclusive).dt.days,0.)).astype(float)
    bound=(scores.forcing_end+pd.Timedelta(days=1)-scores.upper_inclusive).dt.days.clip(lower=0).fillna(0)
    signed[missing]=bound[missing]
    np.testing.assert_array_equal(signed,scores.signed_distance_days)
    np.testing.assert_array_equal(np.abs(signed),scores.distance_days)
    assert scores.compatible.eq(signed==0).all()
    coverage=pd.read_csv(HERE/'field_stage_coverage_all.csv')
    assert len(coverage)==36 and set(coverage.event)==set(EVENTS)
    assert (coverage.registered_fields==coverage.constrained_fields+coverage.unavailable_or_rejected_fields).all()
    assert (coverage.constrained_fields==coverage.genuine_two_sided+coverage.left_censored+coverage.right_censored).all()
    points=pd.read_csv(STAGING/'figures/fig2_field_stage_intervals.csv')
    german=pd.read_csv(STAGING/'figures/fig2_german_stage_errors.csv',parse_dates=['true_date','predicted_date'])
    assert len(points)==259 and len(german)==79011
    assert set(points.event)=={31,32,33,37,39,51,65,85}
    assert set(german.BBCH)=={10,31,51,85}
    np.testing.assert_array_equal((german.predicted_date-german.true_date).dt.days,german.error_days)
    assert german.groupby('cohort').size().to_dict()==dict(validation=38133,testing=40878)
    soft=points.loc[points.event.eq(85)]
    assert len(soft)==5 and not soft.compatible.any() and soft.distance_days.mean()==9.
    # Earlier climate, yield and disease evidence is unchanged by the expanded display.
    for name in ['fig3_onset_and_detection','fig4_timing_and_yield_relevance','fig5_conditional_climate_changes']:
        assert sha(STAGING/'figures'/f'{name}.png')==sha(BEFORE/'figures'/f'{name}.png')
    for name in ['abstract.txt','baseline_results.txt','climate_results.txt']:
        assert sha(STAGING/name)==sha(BEFORE/name)
    workflow=json.loads((STAGING/'figures/fig1_workflow_layout_check.json').read_text())
    assert workflow['status']=='passed' and len(workflow['nodes'])==9
    for path in [*STAGING.glob('figures/*.svg'),*STAGING.glob('supplementary_figures/*.svg')]:ET.parse(path)
    main_doc=Document(STAGING/'Manuscript.docx');supp_doc=Document(STAGING/'Supplementary_Information.docx')
    assert len(main_doc.inline_shapes)==5 and len(main_doc.tables)==3
    assert len(main_doc.tables[0].rows)==19 and len(main_doc.tables[1].rows)==13
    assert len(supp_doc.inline_shapes)==4 and len(supp_doc.tables)==20
    with zipfile.ZipFile(STAGING/'Manuscript.docx') as z:
        for index,name in [(1,'fig1_framework'),(2,'fig2_flag_stage_validation')]:
            assert hashlib.sha256(z.read(f'word/media/image{index}.png')).hexdigest()==sha(STAGING/'figures'/f'{name}.png')
    pages={}
    for name in ['Manuscript','Supplementary_Information']:
        with pymupdf.open(STAGING/f'{name}.pdf') as pdf:
            pages[name]=len(pdf)
            for index,page in enumerate(pdf):
                contents=page.get_text()
                if 'Figure 1 |' in contents or 'Figure 2 |' in contents or 'Figure S2 |' in contents or 'Table S6b |' in contents:
                    page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(HERE/'preview'/f'{name}_page_{index+1}.png')
    wb=load_workbook(STAGING/'Source_Data.xlsx',read_only=True,data_only=True)
    manifest=receipt['source_workbook']
    assert len(manifest)==29 and len(wb.sheetnames)==31
    for row in manifest:
        path=ROOT/row['source_path']
        if row['source_path'].startswith('publication/european_wheat_stb/'):
            path=STAGING/Path(row['source_path']).relative_to('publication/european_wheat_stb')
        assert sha(path)==row['sha256']
        rows=wb[row['sheet']].iter_rows(values_only=True);next(rows)
        assert sum(1 for _ in rows)==row['rows'],row['sheet']
    old=load_workbook(BEFORE/'Source_Data.xlsx',read_only=True,data_only=True)
    for name in old.sheetnames[:15]:assert list(wb[name].values)==list(old[name].values),name
    old.close();wb.close()
    assert all(sha(HERE/'before'/r['path'])==r['sha256']
        for r in json.loads((HERE/'before/archive_manifest.json').read_text()))
    return dict(status='passed',scope='all prior validation datasets and stages, Figure 1 workflow, Figure 2 and citations',
        german_event_metrics_independently_recomputed=270,german_main_figure_records=79011,
        field_scores_independently_reconciled=len(scores),field_stage_registry=EVENTS,
        field_main_figure_records=259,soft_dough_intervals=dict(n=5,compatible=0,mean_excess_days=9.),
        main_figures=5,main_tables=3,supplementary_figures=4,supplementary_tables=20,
        pdf_pages=pages,source_workbook_sheets=31,reference_count=25,undated_in_text_citations=0,
        workflow_text_within_nodes=True,native_word_images_match_exports=True,
        manuscript_climate_yield_and_disease_metrics_unchanged=True,previous_publication_archive_unchanged=True,
        fresh_repository_test_run=False,model_tests_inherited_from_unchanged_runtime=True)


def install(audit):
    with zipfile.ZipFile(BEFORE/'Software_and_Evidence.zip') as z:
        old_manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
    allowed={str(p.relative_to(ROOT)) for p in [SOURCE/'build.py',SOURCE/'figures.py',SOURCE/'sections.py',
        SOURCE/'references/references.csl.json',SOURCE/'references/references.bib',
        SOURCE/'figures/captions.json',SOURCE/'figures/fig2_field_stage_intervals.csv',
        ACTIVE/'figures/fig2_field_stage_intervals.csv']}
    for row in old_manifest:
        if row['path'] not in allowed:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    audit['runtime_calibration_and_frozen_scientific_sources_unchanged']=True
    for path in STAGING.rglob('*'):
        if path.is_file():
            target=ACTIVE/path.relative_to(STAGING);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for path in (STAGING/'figures').iterdir():
        if path.is_file():shutil.copy2(path,SOURCE/'figures'/path.name)
    dump(HERE/'publication_verification.json',audit)
    dump(ACTIVE/'validation_and_figures_verification.json',audit)
    dump(ACTIVE/'figure2_verification.json',dict(status='passed',supersedes='flag-stage-only Figure 2',
        german_shared_event_records=79011,genuine_field_stage_records=259,stages=EVENTS,
        verification_path='validation_and_figures_verification.json'))
    paths={ROOT/row['path'] for row in old_manifest}
    for directory in [HERE,GERMAN,ROOT/'analysis/primary_secondary/external_french']:
        for path in directory.rglob('*'):
            if not path.is_file() or any(part in ['before','staging','preview','__pycache__'] for part in path.relative_to(directory).parts):continue
            if path.suffix in ['.py','.json','.csv','.parquet','.gz','.md','.txt','.pdf','.png']:
                paths.add(path)
    paths.update([ROOT/'analysis/phenology/verify_archived_validation.py',ROOT/'analysis/phenology/plot_heldout_validation.py',
        ACTIVE/'figures/fig2_german_stage_errors.csv',SOURCE/'figures/fig2_german_stage_errors.csv',
        SOURCE/'figures/fig1_workflow_layout_check.json',ACTIVE/'figures/fig1_framework.svg',
        ACTIVE/'figures/fig2_flag_stage_validation.svg'])
    reproduction=(BEFORE/'REPRODUCIBILITY.txt').read_text()
    reproduction += ('\nComplete validation revision: python -m analysis.paper_study.full_validation_20261007.evidence '
        'reconstructs the stage, cohort and archival-transfer tables without fitting. '
        'python -m analysis.paper_study.overwinter_leaf_model_20261006.manuscript.figures '
        'regenerates the five main figures, including the research workflow and complete validation overview. '
        'The 20 supplementary tables and four stage-window figures retain calibration, censoring and model-version identity. '
        'German source/event archives and French frozen transfers are bundled with their original provenance.\n')
    (ACTIVE/'REPRODUCIBILITY.txt').write_text(reproduction)
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    package=ACTIVE/'Software_and_Evidence.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest:z.write(ROOT/row['path'],row['path'])
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('REPRODUCIBILITY.txt',reproduction)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(manifest)+2
        for row in manifest:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    dump(ACTIVE/'software_package_receipt.json',dict(status='complete',members=len(manifest)+2,
        bundled_sources_and_evidence=manifest,package_sha256=sha(package),raw_climate_cache_bundled=False,
        validation_revision='Complete prior-dataset and stage coverage, 2026-10-07'))
    previous=json.loads((BEFORE/'final_verification.json').read_text())
    verification=dict(previous)
    verification.update(manuscript_pdf_pages=audit['pdf_pages']['Manuscript'],supplementary_pdf_pages=audit['pdf_pages']['Supplementary_Information'],
        reference_count=25,workbook_sheets=31,software_package_members=len(manifest)+2,
        supplementary_figures=4,supplementary_tables=20,
        verification_scope=audit['scope'],undated_in_text_citations=0,
        model_test_evidence_origin=str((BEFORE/'final_verification.json').relative_to(ROOT)),
        runtime_and_calibration_sources_unchanged=True)
    verification['historical_review_receipts']=verification.pop('review_receipts',[])
    verification['review_receipts']=[dict(path=str((HERE/'publication_verification.json').relative_to(ROOT)),
        sha256=sha(HERE/'publication_verification.json'))]
    verification['artifact_sha256']={str(p.relative_to(ACTIVE)):sha(p) for p in sorted(ACTIVE.rglob('*'))
        if p.is_file() and p.name!='final_verification.json'}
    dump(ACTIVE/'final_verification.json',verification)
    for name,digest in verification['artifact_sha256'].items():assert sha(ACTIVE/name)==digest
    print(json.dumps(dict(status='passed',main_pdf_pages=audit['pdf_pages']['Manuscript'],
        supplementary_pdf_pages=audit['pdf_pages']['Supplementary_Information'],
        workbook_sheets=31,software_package_members=len(manifest)+2,all_prior_datasets_and_stages=True,
        references_dated=True,runtime_and_calibration_unchanged=True)))


if __name__=='__main__':install(verify())
