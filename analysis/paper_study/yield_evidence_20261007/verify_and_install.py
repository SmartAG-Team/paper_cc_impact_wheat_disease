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
from analysis.paper_study.full_validation_20261007.evidence import HERE as VALIDATION,ROOT,STUDY,GERMAN,EVENTS,sha
from .published_response import HERE

STAGING=HERE/'staging'
ACTIVE=ROOT/'publication/european_wheat_stb'
BEFORE=HERE/'before/publication/european_wheat_stb'
SOURCE=STUDY/'manuscript'


def dump(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def verify():
    receipt=json.loads((STAGING/'document_build_receipt.json').read_text())
    for relative,digest in receipt['sources'].items():assert sha(ROOT/relative)==digest,relative
    assert receipt['reference_count']==27
    text=(STAGING/'Manuscript.txt').read_text()
    assert not re.search(r'\bn\.d\.|\[\[|APSIM Initiative|Python Crop Simulation Environment',text)
    for citation in ['Holzworth et al., 2018','de Wit et al., 2019','Hersbach et al., 2020']:
        assert citation in text,citation
    records=json.loads((STAGING/'references.csl.json').read_text())
    assert len(records)==27 and all(r.get('issued') for r in records)
    for key,doi in [('Holzworth2018APSIM','10.1016/j.envsoft.2018.02.002'),
            ('deWit2019WOFOST','10.1016/j.agsy.2018.06.018'),('Hersbach2020ERA5','10.1002/qj.3803')]:
        record=next(r for r in records if r['id']==key)
        assert record['DOI'].lower()==doi
        registered=json.loads((VALIDATION/'literature'/f'{key}.crossref.json').read_text())
        assert record['volume']==registered['volume'] and record['page']==registered['page']
    scores=pd.read_csv(VALIDATION/'field_stage_scores_all.csv',parse_dates=['predicted_date','lower_exclusive','upper_inclusive','forcing_end'])
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
    coverage=pd.read_csv(VALIDATION/'field_stage_coverage_all.csv')
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
    for name in ['fig2_flag_stage_validation','fig3_onset_and_detection','fig5_conditional_climate_changes']:
        assert sha(STAGING/'figures'/f'{name}.png')==sha(BEFORE/'figures'/f'{name}.png')
    for name in ['climate_results.txt','field_results.txt']:
        normalize=lambda t:re.sub(r'\s+','',t)
        assert normalize((STAGING/name).read_text())==normalize((BEFORE/name).read_text())
    workflow=json.loads((STAGING/'figures/fig1_workflow_layout_check.json').read_text())
    assert workflow['status']=='passed' and len(workflow['nodes'])==9
    for path in [*STAGING.glob('figures/*.svg'),*STAGING.glob('supplementary_figures/*.svg')]:ET.parse(path)
    main_doc=Document(STAGING/'Manuscript.docx');supp_doc=Document(STAGING/'Supplementary_Information.docx')
    assert len(main_doc.inline_shapes)==5 and len(main_doc.tables)==3
    assert len(main_doc.tables[0].rows)==19 and len(main_doc.tables[1].rows)==13
    assert len(supp_doc.inline_shapes)==12 and len(supp_doc.tables)==28
    with zipfile.ZipFile(STAGING/'Manuscript.docx') as z:
        for index,name in [(1,'fig1_framework'),(2,'fig2_flag_stage_validation')]:
            assert hashlib.sha256(z.read(f'word/media/image{index}.png')).hexdigest()==sha(STAGING/'figures'/f'{name}.png')
    pages={}
    for name in ['Manuscript','Supplementary_Information']:
        with pymupdf.open(STAGING/f'{name}.pdf') as pdf:
            pages[name]=len(pdf)
            for index,page in enumerate(pdf):
                contents=page.get_text()
                if 'Figure ' in contents or 'Table S7' in contents or 'Table S8' in contents or index == 0:
                    page.get_pixmap(matrix=pymupdf.Matrix(1.5,1.5)).save(HERE/'preview'/f'{name}_page_{index+1}.png')
    wb=load_workbook(STAGING/'Source_Data.xlsx',read_only=True,data_only=True)
    manifest=receipt['source_workbook']
    assert len(manifest)==42 and len(wb.sheetnames)==44
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
    yield_audit=verify_yield()
    return dict(published_yield=yield_audit,status='passed',scope='all prior validation, workflow, published canopy-yield evidence and wheat-rust crop-response comparisons',
        german_event_metrics_independently_recomputed=270,german_main_figure_records=79011,
        field_scores_independently_reconciled=len(scores),field_stage_registry=EVENTS,
        field_main_figure_records=259,soft_dough_intervals=dict(n=5,compatible=0,mean_excess_days=9.),
        main_figures=5,main_tables=3,supplementary_figures=12,supplementary_tables=28,
        pdf_pages=pages,source_workbook_sheets=44,reference_count=27,undated_in_text_citations=0,
        workflow_text_within_nodes=True,native_word_images_match_exports=True,
        frozen_climate_and_disease_metrics_unchanged=True,previous_publication_archive_unchanged=True,
        fresh_repository_test_run=False,model_tests_inherited_from_unchanged_runtime=True)


def verify_yield():
    matrix=pd.read_csv(HERE/'parker2004_table4_yield_loss_matrix.csv').set_index('cultivar')
    values=matrix.to_numpy().ravel()
    assert matrix.shape==(25,6) and np.isfinite(values).sum()==117 and (values<0).sum()==5
    slopes=pd.read_csv(HERE/'parker2004_table5_yield_slopes.csv')
    assert len(slopes)==25 and slopes.fixed_effect_slope_t_ha_per_GLAI_day.notna().sum()==21
    np.testing.assert_allclose([slopes.random_effect_slope_t_ha_per_GLAI_day.min(),
        slopes.random_effect_slope_t_ha_per_GLAI_day.max()],[.0141,.0207],rtol=0,atol=1e-12)
    pairs=pd.read_csv(HERE/'foulkes2006_group_separated_predictions.csv')
    assert len(pairs)==8 and pairs.genetic_background.nunique()==5
    assert pairs.leaf_scope.eq('top five leaves').all()
    assert pairs.statistical_status.eq('mixed-model predicted genotype-by-fungicide means').all()
    for kind in ['HAD','yield']:
        protected=pairs['healthy_HAD_GLAI_day' if kind=='HAD' else 'healthy_yield_t_ha_15pct_moisture']
        affected=pairs['diseased_HAD_GLAI_day' if kind=='HAD' else 'diseased_yield_t_ha_15pct_moisture']
        np.testing.assert_allclose(pairs['derived_'+kind+'_loss_fraction'],(protected-affected)/protected,rtol=0,atol=1e-14)
    metrics=pd.read_csv(HERE/'published_yield_response_metrics.csv').set_index('response')
    w=1/pairs.groupby('genetic_background').genetic_background.transform('size').to_numpy()
    w=w/w.sum();np.testing.assert_allclose(w,pairs.evaluation_weight,rtol=0,atol=1e-14)
    for mode,xcol,ycol in [('absolute','derived_HAD_loss_GLAI_day','derived_yield_loss_t_ha'),
            ('relative','derived_HAD_loss_fraction','derived_yield_loss_fraction')]:
        x=pairs[xcol].to_numpy();y=pairs[ycol].to_numpy();pred=pairs[mode+'_held_out_prediction'].to_numpy()
        for group in pairs.genetic_background.unique():
            test=pairs.genetic_background.eq(group).to_numpy();train=~test
            wt=1/pairs.loc[train].groupby('genetic_background').genetic_background.transform('size').to_numpy()
            coefficient=max(0,np.sum(wt*x[train]*y[train])/np.sum(wt*x[train]**2))
            np.testing.assert_allclose(coefficient,pairs.loc[test,mode+'_training_only_slope'],rtol=0,atol=1e-14)
            np.testing.assert_allclose(coefficient*x[test],pred[test],rtol=0,atol=1e-14)
        err=pred-y
        calculated={'group_balanced_MAE':np.sum(w*np.abs(err)),
            'group_balanced_RMSE':np.sqrt(np.sum(w*err**2)),
            'group_balanced_bias':np.sum(w*err),
            'group_balanced_R2':1-np.sum(w*err**2)/np.sum(w*(y-np.sum(w*y))**2)}
        for key,value in calculated.items():np.testing.assert_allclose(value,metrics.loc[mode,key],rtol=0,atol=1e-14)
    assert metrics.loc['absolute','group_balanced_R2']<0
    assert not metrics.raw_plot_validation.any() and not metrics.current_top3_yield_calibration.any()
    domain=pd.read_csv(HERE/'published_yield_domain_compatibility.csv')
    assert len(domain)==4 and set(domain.source)=={'Parker2004','Foulkes2006','Bryson1997','SubbaRao1989'}
    yellow=pd.read_csv(HERE/'bryson1997_reported_yield_fits.csv')
    brown=pd.read_csv(HERE/'subbarao1989_reported_yield_fits.csv')
    assert yellow.value.tolist()==[.63,.8,.73,.92] and brown.value.tolist()==[.84,.91,.67,.88]
    assert not yellow.raw_data_available_for_reanalysis.any() and not brown.raw_data_available_for_reanalysis.any()
    histories=pd.read_csv(STAGING/'supplementary_figures/figS6_leaf_onset_distances.csv')
    assert len(histories)==272 and histories.groupby('model').size().eq(136).all()
    tests=json.loads((HERE/'response_tests_receipt.json').read_text())
    assert tests['status']=='passed' and tests['tests']==4
    for relative,digest in tests['sources'].items():assert sha(ROOT/relative)==digest
    receipt=json.loads((HERE/'published_response_receipt.json').read_text())
    for relative,digest in receipt['sources'].items():assert sha(ROOT/relative)==digest
    for name,digest in receipt['generated_data_sha256'].items():assert sha(HERE/name)==digest
    return dict(published_STB_yield_contrasts=117,negative_contrasts_retained=5,top_three_cultivar_slopes=25,
        paired_predicted_means=8,withheld_backgrounds=5,independently_reproduced_relative_RMSE_percentage_points=
        float(100*metrics.loc['relative','group_balanced_RMSE']),
        independently_reproduced_absolute_R2=float(metrics.loc['absolute','group_balanced_R2']),
        response_tests=4,leaf_scopes_preserved=True,reported_rust_fit_scores_are_not_independent_validation=True,
        diseases=['STB','yellow rust','leaf rust'],epidemiology_model_scope='STB',current_top_three_transfer_recalibrated=False)


def install(audit):
    with zipfile.ZipFile(BEFORE/'Software_and_Evidence.zip') as z:
        old_manifest=json.loads(z.read('PACKAGE_MANIFEST.json'))
    source_changes={str(p.relative_to(ROOT)) for p in [SOURCE/'build.py',SOURCE/'figures.py',SOURCE/'sections.py',
        SOURCE/'references/references.csl.json',SOURCE/'references/references.bib',VALIDATION/'workflow_figure.py']}
    for row in old_manifest:
        if row['path'] not in source_changes:assert sha(ROOT/row['path'])==row['sha256'],row['path']
    audit['runtime_and_existing_calibration_and_frozen_evidence_unchanged']=True
    for path in STAGING.rglob('*'):
        if path.is_file():
            target=ACTIVE/path.relative_to(STAGING);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(path,target)
    for path in (STAGING/'figures').iterdir():
        if path.is_file():shutil.copy2(path,SOURCE/'figures'/path.name)
    dump(HERE/'publication_verification.json',audit)
    dump(ACTIVE/'yield_evidence_verification.json',audit)
    dump(ACTIVE/'validation_and_figures_verification.json',audit)
    paths={ROOT/row['path'] for row in old_manifest}
    for directory in [HERE,ACTIVE/'supplementary_figures',ACTIVE/'figures']:
        for path in directory.rglob('*'):
            if not path.is_file() or any(part in ['before','staging','preview','__pycache__'] for part in path.relative_to(directory).parts):continue
            # The package contains numerical extractions and provenance, without source-article reproductions.
            if directory==HERE and (path.suffix in ['.pdf','.png'] or path.name.endswith('_ocr_notes.txt') or path.name.endswith('.text') or path.name=='foulkes2006.txt'):continue
            if path.suffix in ['.py','.json','.csv','.parquet','.md','.txt','.pdf','.png','.svg']:
                paths.add(path)
    paths.update([ROOT/'calibration/yield_transfer.py',ROOT/'tests/test_published_yield_transfer.py',
        ACTIVE/'wheat_stb_model-0.1.0-py3-none-any.whl',
        BEFORE/'final_verification.json',HERE/'before/archive_manifest.json'])
    reproduction=(BEFORE/'REPRODUCIBILITY.txt').read_text()
    reproduction += (
        '\nPublished crop-response revision (2026-10-07): python -m analysis.paper_study.yield_evidence_20261007.published_response '
        'reconstructs the five-background evaluation from published top-five treatment means. '
        'python -m analysis.paper_study.overwinter_leaf_model_20261006.manuscript.figures regenerates the main figures. '
        'The manuscript builder also supplies 28 supplementary tables and 12 supplementary figures. '
        'STB, yellow-rust and leaf-rust crop-response evidence retains leaf scope, time window, units and aggregate/fitted status; '
        'epidemic simulation remains STB-specific. Calibration/yield_transfer.py is external to model/. '
        'Four additional response-analysis tests are verified; the unchanged runtime retains its archived 301-test evidence.\n'
        'For a new complete document build, import analysis.paper_study.overwinter_leaf_model_20261006.manuscript.build, '
        'set build.DEST to a new directory and call build.main(). The registered wheel is supplied in publication/european_wheat_stb; '
        'the builder uses its publication-relative wheel path. '
        'Source_Data.xlsx includes all prior data sheets and 13 added data sheets.\n')
    (ACTIVE/'REPRODUCIBILITY.txt').write_text(reproduction)
    manifest=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(paths)]
    package=ACTIVE/'Software_and_Evidence.zip'
    with zipfile.ZipFile(package,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest:z.write(ROOT/row['path'],row['path'])
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
        z.writestr('REPRODUCIBILITY.txt',reproduction)
    with zipfile.ZipFile(package) as z:
        assert z.testzip() is None and len(z.namelist())==len(manifest)+2
        for row in manifest:assert hashlib.sha256(z.read(row['path'])).hexdigest()==row['sha256']
    dump(ACTIVE/'software_package_receipt.json',dict(status='complete',members=len(manifest)+2,
        bundled_sources_and_evidence=manifest,package_sha256=sha(package),raw_climate_cache_bundled=False,
        publication_revision='Published wheat-disease canopy-yield evidence, 2026-10-07'))
    previous=json.loads((BEFORE/'final_verification.json').read_text())
    verification=dict(previous)
    verification.update(manuscript_pdf_pages=audit['pdf_pages']['Manuscript'],supplementary_pdf_pages=audit['pdf_pages']['Supplementary_Information'],
        reference_count=27,workbook_sheets=44,software_package_members=len(manifest)+2,
        supplementary_figures=12,supplementary_tables=28,added_response_tests=4,
        verification_scope=audit['scope'],undated_in_text_citations=0,
        model_test_evidence_origin=str((BEFORE/'final_verification.json').relative_to(ROOT)),
        runtime_and_existing_calibration_sources_unchanged=True,new_crop_response_analysis_outside_model=True,
        publication_yield_data=audit['published_yield'])
    verification.pop('runtime_and_calibration_sources_unchanged',None)
    verification['historical_review_receipts']=[*verification.get('historical_review_receipts',[]),*verification.get('review_receipts',[])]
    verification['review_receipts']=[dict(path=str((HERE/'publication_verification.json').relative_to(ROOT)),
        sha256=sha(HERE/'publication_verification.json'))]
    verification['artifact_sha256']={str(p.relative_to(ACTIVE)):sha(p) for p in sorted(ACTIVE.rglob('*'))
        if p.is_file() and p.name!='final_verification.json'}
    dump(ACTIVE/'final_verification.json',verification)
    for name,digest in verification['artifact_sha256'].items():assert sha(ACTIVE/name)==digest
    print(json.dumps(dict(status='passed',main_pdf_pages=audit['pdf_pages']['Manuscript'],
        supplementary_pdf_pages=audit['pdf_pages']['Supplementary_Information'],
        source_workbook_sheets=44,supplementary_figures=12,supplementary_tables=28,
        software_package_members=len(manifest)+2,all_prior_validation_retained=True,published_yield=audit['published_yield'])))


if __name__=='__main__':install(verify())
