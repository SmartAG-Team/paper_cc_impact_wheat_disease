"""Validate the compact GCB article, plotting data and file provenance."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from zipfile import ZipFile

from docx import Document
from docx.shared import Cm
from lxml import etree
from PIL import Image
import numpy as np
import pandas as pd
import pymupdf
from .figure_registry import MAIN_ROLES,SUPPLEMENT_ROLES,ROLE_NAMES

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / 'paper'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def publication_files(directory):
    """Include linked data and model references once in the delivery inventory."""
    for parent,dirs,names in os.walk(directory,followlinks=True):
        dirs[:]=[name for name in dirs if name not in {'__pycache__','exports'}]
        for name in names:
            path=Path(parent)/name
            if name!='publication_manifest.json' and path.suffix not in {'.pyc','.pyo'}:yield path


def validate(write_manifest=False):
    p = PACKAGE
    source = (p/'Manuscript.md').read_text()
    abstract = source.split('## Abstract\n\n')[1].split('\n\nKeywords:')[0]
    body = source.split('## Introduction\n\n')[1].split('## Data availability')[0]
    acknowledgements = source.split('## Acknowledgements\n\n')[1].split('\n\n## ')[0]
    words = lambda s: len(re.findall(r"\b[\w]+(?:[’'–−-][\w]+)*\b",
        re.sub(r'^!\[[^\]]*\]\([^\n]+\)\s*$', '', s, flags=re.M)))
    assert words(abstract) <= 300 and words(body+acknowledgements) < 8000
    running_title=source.split('Running title: ')[1].split('\n')[0]
    assert len(running_title) < 45
    keywords=source.split('Keywords: ')[1].split('\n')[0].split(';')
    assert 6<=len(keywords)<=10
    assert not re.search(r'\([^()]*,\s*\d{4}[^()]*\)\s*\([^()]*,\s*\d{4}',source.split('## References')[0])
    assert not re.search(r'\b(?:VERIFY|TODO|TBD)\b', source)
    affiliation = ('State Key Laboratory of Soil and Water Conservation and Desertification Control, '
        'College of Soil and Water Conservation Science and Engineering, Northwest A&F University, '
        'Yangling, Shaanxi 712100, China.')
    assert affiliation in source
    assert (p/'Manuscript.txt').read_text() == source
    references=source.split('## References\n\n')[1]
    assert '&amp;' not in references and ', ,' not in references
    assert len(references.strip().split('\n\n'))>=38
    gcb_reference_count=references.count('Global Change Biology,')
    assert gcb_reference_count>=9
    # Current journal rules, distinct from author information awaiting confirmation.
    journal=json.loads((p/'data/journal_submission_requirements.json').read_text())
    assert journal['checked_on']=='2026-10-24' and journal['initial_checklist_updated']=='2025-12-01'
    cover=(p/'Cover_Letter.md').read_text()
    answers=[]
    for section in cover.split('\n## ')[1:]:
        heading,text=section.split('\n',1)
        answer=text.split('\n\nSincerely,')[0].strip()
        answers.append((heading,words(answer)))
    assert len(answers)==7 and all(n<=50 for _,n in answers),answers
    candidates=json.loads((p/'verification/reviewer_candidates.json').read_text())['candidates']
    assert len(candidates)==5 and len({row['email'].lower() for row in candidates})==5
    assert all(row['email'] in cover and row['email_evidence_url'].startswith('https://') for row in candidates)
    boundary='Map lines delineate study areas and do not necessarily depict accepted national boundaries.'
    for role in ['data_overview','climate_change']:
        assert boundary in (p/'figures'/f'{ROLE_NAMES[role]}_caption.txt').read_text(),role
    assert acknowledgements.strip() and 'Conceptualization' in source
    assert all(path.stat().st_size<10*1024*1024 for path in (p/'figures').glob('*.png'))
    status=json.loads((p/'submission_status.json').read_text())
    assert status['author_declarations_confirmed'] and not status['phone_and_orcid_pending']
    assert 'https://orcid.org/'+status['submitting_author_orcid'] in source
    assert status['corresponding_author_telephone'] in source
    for name in ['Manuscript','Supplementary_Information','Cover_Letter','Mathematical_Model_Specification']:
        path = p/f'{name}.docx'
        with ZipFile(path) as archive:
            for item in archive.namelist():
                if item.endswith('.xml'): etree.fromstring(archive.read(item))
        Document(path)
    figure_manifest = json.loads((p/'figure_manifest.json').read_text())
    assert len(figure_manifest['figures']) == len(ROLE_NAMES)==14
    assert figure_manifest['role_to_name']==ROLE_NAMES
    assert figure_manifest['main_figures']==[ROLE_NAMES[r] for r in MAIN_ROLES]
    def figure_data(role):return p/'data'/('monitoring_diagnostics' if role=='site_errors' else ROLE_NAMES[role])
    for path, value in figure_manifest['artifacts_sha256'].items():
        assert digest(p/path) == value, path
    assert digest(ROOT/figure_manifest['builder_path']) == figure_manifest['code_sha256']
    assert digest(ROOT/'paper/scripts/figures.py') == figure_manifest['entrypoint_sha256']
    assert digest(ROOT/'paper/scripts/figure_registry.py') == figure_manifest['registry_sha256']
    scientific_audit=json.loads((p/'verification/figure_scientific_audit.json').read_text())
    assert scientific_audit['status']=='passed' and set(scientific_audit['figures'])==set(ROLE_NAMES.values())
    assert scientific_audit['raw_Corteva_assessments_reconstructed']==1720
    assert scientific_audit['raw_BASF_assessments_reconstructed']==584
    assert scientific_audit['treatment_eligibility_verified_for_all_retained_records']
    assert digest(ROOT/'paper/scripts/figure_scientific_audit.py')==scientific_audit['auditor_sha256']
    for path,value in scientific_audit['source_sha256'].items():assert digest(ROOT/path)==value,path
    for figure in figure_manifest['figures']:
        assert figure['width_cm'] == 19 and figure['font_pt'] == 12
        assert figure['png_dpi']==300 and figure['formats']==['png']
        for path, value in figure['source_sha256'].items(): assert digest(ROOT/path) == value, path
        png=Image.open(p/'figures'/f"{figure['name']}.png")
        assert abs(png.width-19/2.54*300)<=1
        assert abs(png.height-figure['height_cm']/2.54*300)<=1
        assert all(abs(v-300)<.1 for v in png.info['dpi'])
    assert not list((p/'figures').glob('*.pdf')) and not list((p/'figures').glob('*.svg'))
    mapping=json.loads((p/'verification/figure_data_cleanup.json').read_text())['allowed_files_by_figure']
    for name,files in mapping.items():
        assert set(q.name for q in (p/'data'/name).iterdir() if q.is_file())==set(files),name
    from .figure_placement import place_figures,IMAGE
    for document_name,roles in [('Manuscript',MAIN_ROLES),('Supplementary_Information',SUPPLEMENT_ROLES)]:
        text=(p/f'{document_name}.md').read_text()
        assert text==place_figures(text,[ROLE_NAMES[r] for r in roles])
    rendered=json.loads((p/'document_profiles.json').read_text())
    assert rendered['source_renderer_sha256']==digest(ROOT/'paper/scripts/documents.py')
    for record in rendered['documents']:
        assert not record['appended_figures']
    assert rendered['documents'][0]['inline_figures']==[ROLE_NAMES[r] for r in MAIN_ROLES]
    assert rendered['documents'][1]['inline_figures']==[ROLE_NAMES[r] for r in SUPPLEMENT_ROLES]
    docs = {}
    for name, count in [('Manuscript',len(MAIN_ROLES)),('Supplementary_Information',len(SUPPLEMENT_ROLES)),('Cover_Letter',0)]:
        doc = Document(p/f'{name}.docx')
        assert len(doc.inline_shapes) == count
        assert all(abs(shape.width-Cm(19)) < 10 for shape in doc.inline_shapes)
        if name=='Supplementary_Information':
            assert len(doc.tables)==5
            assert [len(table.rows) for table in doc.tables]==[7,9,7,17,10]
            assert doc.styles['Normal'].paragraph_format.line_spacing==1.25
            assert len([q for q in doc.paragraphs if q.style.name=='Heading 1' and re.match(r'S[1-7]\.',q.text)])==7
            for table in doc.tables:
                assert table.rows[0]._tr.xpath('./w:trPr/w:tblHeader')
                assert all(row._tr.xpath('./w:trPr/w:cantSplit') for row in table.rows)
                assert all(run.font.size.pt==12 for row in table.rows for cell in row.cells for paragraph in cell.paragraphs for run in paragraph.runs)
            provenance=json.loads((p/'data/supplementary_tables/parameter_provenance.json').read_text())
            displayed=[row.cells[1].text for table in doc.tables[3:] for row in table.rows[1:]]
            assert displayed==[row['value_exact_from_current_table'] for row in provenance]
        if name=='Manuscript':
            assert len(doc._element.xpath('.//w:lnNumType'))==len(doc.sections)
            assert all(node.get('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}restart')=='continuous'
                for node in doc._element.xpath('.//w:lnNumType'))
            paragraphs=doc.paragraphs
            assert 'References' in [q.text for q in paragraphs]
            # Preserve the complete bibliography through figure-caption updates
            # and both document renderers, including references after captions.
            reference_paragraphs={q.text for q in paragraphs}
            for reference in references.strip().split('\n\n'):
                assert reference.replace('*','') in reference_paragraphs, reference
            method_index=next(i for i,q in enumerate(paragraphs) if q.text=='Materials and methods')
            result_index=next(i for i,q in enumerate(paragraphs) if q.text=='Results')
            images=[i for i,q in enumerate(paragraphs) if q._p.xpath('.//w:drawing')]
            assert sum(method_index<i<result_index for i in images)==3
        pdf = pymupdf.open(p/f'{name}.pdf')
        if name=='Manuscript':
            rendered_text=' '.join(' '.join(page.get_text().split()) for page in pdf)
            assert 'References' in rendered_text
            for title in ['Defoliation, disease, and growth.',
                    'Velocity of temperature and flowering time in wheat']:
                assert title in rendered_text,title
        outside = []
        for number, page in enumerate(pdf, 1):
            for block in page.get_text('dict')['blocks']:
                if block['type'] != 0: continue
                box = pymupdf.Rect(block['bbox'])
                if not (page.rect+(-1,-1,1,1)).contains(box): outside.append(number)
        assert not outside, (name, outside)
        docs[name] = dict(pages=len(pdf), embedded_word_figures=count, pdf_text_within_page=True)
    points = pd.read_csv(figure_data('monitoring')/'predictions.csv')
    original_points=pd.read_csv(ROOT/'data/final_monitoring/reference/data/fig1/predictions.csv')
    pd.testing.assert_frame_equal(original_points.reset_index(drop=True),
        points[original_points.columns].reset_index(drop=True),check_exact=True)
    monitoring = {}
    expected = [17.151664891542353,17.30622057488964]
    for i, (_, frame) in enumerate(points.groupby('panel', sort=True)):
        assert len(frame) == 2017 and frame.field_id.nunique() == 192 and frame.site_id.nunique() == 121
        rmse = float(np.sqrt(np.average((frame.prediction_percent-frame.observed_percent)**2,
            weights=frame.scientific_weight)))
        assert abs(rmse-expected[i]) < 1e-10
        monitoring[str(frame.panel.iloc[0])] = rmse
    densities=pd.read_csv(figure_data('monitoring')/'density.csv')
    assert densities.groupby('evaluation').assessments.sum().eq(2017).all()
    assert np.allclose(densities.groupby('evaluation').scientific_mass_percent.sum(),100)
    caption=(p/'figures'/f'{ROLE_NAMES["monitoring"]}_caption.txt').read_text()
    assert '2.5 percentage points on both axes' in caption and 'not prediction error or confidence' in caption
    assert 'RMSE 17.15' in caption and 'RMSE 17.31' in caption
    coverage=pd.read_csv(figure_data('monitoring')/'severity_year_coverage.csv')
    for design,part in coverage.groupby('evaluation'):
        assert part.sort_values('season_year').season_year.tolist()==[2015,2016,2017,2018,2019]
        assert part.sort_values('season_year').assessments.tolist()==[425,435,426,401,330]
    assert '2015–2019' in caption
    onset_panel=pd.read_csv(figure_data('monitoring')/'onset_records.csv')
    onset_original=pd.read_csv(p/'data/onset_evaluation/onset_records.csv')
    pd.testing.assert_frame_equal(onset_panel[onset_original.columns],onset_original,check_exact=True)
    layout=json.loads((figure_data('monitoring')/'figure_layout.json').read_text())
    assert set(layout['panels'])=={'a','b','c','d'} and layout['onset_evaluation_histories']==4
    assert layout['onset_calibration_histories']==3 and not layout['calibration_is_validation']
    assert layout['F1_field_seasons']==55 and layout['positive_field_seasons']==52
    assert not layout['exact_onset_MAE_identified'] and layout['small_sample']
    assert not layout['onset_legend_overlaps_density_label']
    counts=layout['cohort_counts']
    assert counts['all_2019_fields']==45 and counts['all_2019_sites']==36 and counts['qualified_F1_fields']==33
    assert counts['eligibility_status_counts']=={'F1_positive_at_first_assessment':27,'no_qualified_F1_observation':12,'two_sided_F1_onset':4,'no_positive_F1_assessment':2}
    assert '(b) Forecast calibration' in caption and '(c) Cumulative evaluation weight' in caption
    assert '(d) Signed onset interval-excess errors' in caption
    assert layout['density_evaluation']=='temporal' and layout['error_coverage_full_range_pp']==[0,100]
    assert layout['calibration_resamples']==1000 and layout['error_coverage_resamples']==2000
    assert (figure_data('monitoring')/'calibration.csv').is_file()
    assert not (figure_data('diagnostics')/'calibration.csv').exists()
    audit_r2=pd.read_csv(figure_data('monitoring')/'r2_audit.csv')
    assert np.allclose(audit_r2.predictive_r2,1-audit_r2.mse/audit_r2.var_y)
    np.testing.assert_allclose(audit_r2.set_index('evaluation').loc[['temporal','spatial_50km'],'predictive_r2'],
        [.6246080320805238,.6178121472072579],rtol=0,atol=1e-12)
    scores=pd.read_csv(figure_data('monitoring')/'scores.csv')
    resamples=pd.read_csv(figure_data('monitoring')/'metric_resamples.csv')
    assert len(resamples)==4000 and np.allclose(resamples.r2,1-resamples.sse/resamples.sst)
    for design,row in scores.set_index('evaluation').iterrows():
        part=resamples[resamples.evaluation.eq(design)]
        for metric in ['r2','rmse']:
            np.testing.assert_allclose([row[metric+'_lower95'],row[metric+'_upper95']],
                part[metric].quantile([.025,.975]),atol=1e-12,rtol=0)
    contexts=pd.read_csv(figure_data('diagnostics')/'visit_context.csv')
    for group,count in [('First visit',494),('Last visit',403),('Only visit',18)]:
        assert contexts.loc[contexts.visit_context.eq(group),'n'].eq(count).all()
    for role,file,count in [('diagnostics','field_metrics.csv',192),('site_errors','site_metrics.csv',121)]:
        grouped=pd.read_csv(figure_data(role)/file)
        assert grouped.groupby('evaluation').size().eq(count).all()
        for design,part in grouped.groupby('evaluation'):
            expected_rmse=scores.set_index('evaluation').loc[design,'rmse']
            assert abs(np.sqrt(np.average(part.rmse**2,weights=part.weight_sum))-expected_rmse)<1e-10
    geometry=json.loads((figure_data('climate_change')/'map_geometry.json').read_text())
    assert geometry['cells_per_scenario']==14941 and geometry['shared_geographical_scales']==0
    assert geometry['map_axes']==4 and geometry['cell_footprints_inside_extent']==14941
    agreement=pd.read_csv(figure_data('climate_change')/'direction_agreement.csv')
    expected_category=np.select([agreement.available_gcms.ne(3),agreement.positive_gcm_count.eq(3),
        agreement.negative_gcm_count.eq(3)],['unavailable','all_3_increase','all_3_decrease'],default='mixed_or_zero')
    assert np.array_equal(agreement.agreement_category,expected_category)
    monitoring_parameters=json.loads((p/'parameters/monitoring_discrepancy_parameters.json').read_text())
    assert len(monitoring_parameters['folds'])==20
    assert all(row['converged'] for row in monitoring_parameters['folds'])
    assert digest(ROOT/monitoring_parameters['source'])==monitoring_parameters['source_sha256']
    attribution = pd.read_csv(figure_data('attribution_results')/'attribution.csv')
    ens = attribution[attribution.model.eq('ensemble')].set_index('setting')
    assert len(ens) == 8
    assert abs(ens.loc['baseline','host_had_days']+5.299410108700939) < 1e-9
    assert abs(ens.loc['fixed_common_overlap','host_had_days']-.569024) < 1e-6
    assert np.max(abs(attribution.weather_had_days+attribution.host_had_days-attribution.net_had_days)) < 1e-10
    regions = pd.read_csv(figure_data('production_exposure')/'display_regions.csv')
    assert len(regions) == 6 and not regions.domain.duplicated().any()
    assert abs(regions.baseline_production_tonnes.sum()-241844854.561714) < 1e-5
    components = ['unanimous_increase_tonnes','unanimous_decrease_tonnes','mixed_or_zero_tonnes','unavailable_tonnes']
    assert np.allclose(regions[components].sum(axis=1),regions.baseline_production_tonnes)
    cohort_replay=json.loads((p/'data/onset_cohorts/checks.json').read_text())
    assert cohort_replay['status']=='passed' and cohort_replay['field_seasons']==55
    assert cohort_replay['positive_field_seasons']==52 and cohort_replay['archived_onset_days_reproduced']==55
    assert cohort_replay['bracketed_causal_prefix_maximum_state_error']==0 and not cohort_replay['fitting_performed']
    assert digest(p/'code/replay_onset_cohorts.py')==digest(ROOT/'paper/scripts/replay_onset_cohorts.py')==cohort_replay['replay_code_sha256']
    for path,expected in cohort_replay['source_sha256'].items():assert digest(p/path)==expected,path
    onset=json.loads((p/'data/onset_evaluation/checks.json').read_text())
    assert onset['status']=='passed' and onset['records']==4 and onset['all_eligible_two_sided_histories_retained']
    assert onset['onset_day_prefix_state_maximum_error']==0 and not onset['models_fitted']
    assert digest(p/'code/replay_onset.py')==digest(ROOT/'paper/scripts/replay_onset.py')==onset['replay_code_sha256']
    for path,expected in onset['source_sha256'].items():assert digest(p/path)==expected,path
    onset_metrics=pd.read_csv(p/'data/onset_evaluation/onset_metrics.csv').set_index('model')
    np.testing.assert_allclose(onset_metrics.loc[['frozen_process','archived_calendar'],'mean_interval_excess_days'],[3.75,6.25],rtol=0,atol=1e-12)
    assert onset_metrics.loc['frozen_process','inside_observed_intervals']==2
    onset_results=source.split('## Results')[1].split('## Discussion')[0]
    assert all(value in onset_results for value in ['1.00 days','3.75 days','6.25 days','70.18%','70.40%','80.68%','80.81%'])
    methods=source.split('## Materials and methods')[1].split('## Results')[0]
    assert '21 of 22' in methods and '31 of 33' in methods
    f1 = json.loads((p/'verification/portable_f1_replay.json').read_text())
    assert f1['status'] == 'passed' and f1['maximum_state_error'] == 0 and not f1['fitted']
    code = json.loads((p/'code/source_manifest.json').read_text())
    for record in code:
        assert digest(p/record['snapshot']) == record['sha256'] == digest(ROOT/record['source'])
    assert digest(p/'code/replay_f1.py') == digest(ROOT/'paper/scripts/replay_f1.py')
    endpoint=json.loads((p/'verification/portable_endpoint_scores.json').read_text())
    assert endpoint['status']=='passed' and endpoint['candidate_comparison_reproduced'] and endpoint['current_supplementary_table']==3 and not endpoint['fitted']
    assert digest(p/'code/recompute_endpoint_scores.py')==digest(ROOT/'paper/scripts/recompute_endpoint_scores.py')
    archival = json.loads((p/'data/process_f1/manifest.json').read_text())
    for path, record in archival['output_hashes'].items():
        assert digest(p/'data/process_f1'/path) == record['sha256'], path
    climate = json.loads((p/'verification/climate_forcing_integrity.json').read_text())
    assert climate['forcing_content_hashes_verified'] and climate['forcing_files'] == 687
    inventory=json.loads((ROOT/'data/repository_layout.json').read_text())
    assert inventory['status']=='passed' and inventory['scientific_values_unchanged']
    assert not (ROOT/'outputs').exists()
    for row in inventory['retained_result_sources']:
        assert digest(ROOT/row['path'])==row['sha256'],row['path']
    profile = dict(main_words=words(body+acknowledgements), abstract_words=words(abstract),
        counted_sections='Introduction, Materials and Methods, Results, Discussion, Acknowledgements',
        running_title_characters=len(running_title),keywords=len(keywords),
        references=len(source.split('## References\n\n')[1].strip().split('\n\n')),
        gcb_journal_references=gcb_reference_count,main_figures=len(MAIN_ROLES),supplement_figures=len(SUPPLEMENT_ROLES),
        final_affiliation_verified=True, retrospective_F1_not_independent=True,
        monitoring_not_projection_validation=True)
    receipt = dict(status='passed', text=profile, documents=docs, monitoring_rmse=monitoring,
        figures_verified=len(ROLE_NAMES), unchanged_code_snapshots=len(code),
        figure_role_numbering_verified=True,all_figures_embedded_after_first_citation=True,continuous_manuscript_line_numbers=True,methods_figures_embedded_in_Methods=3,
        figure_scientific_audits_verified=len(scientific_audit['figures']),
        journal_format_checks_passed=True,cover_answers_words=dict(answers),
        submission_ready=status['submission_ready'],author_items_pending=status['author_submission_items'],
        raw_source_disease_scores_and_untreated_eligibility_verified=2304,
        weighted_density_mass_verified=True,field_and_site_MSE_aggregation_verified=True,
        original_prediction_observation_weight_columns_unchanged_exactly=True,
        all_references_retained_in_editable_document=True,
        metric_resampling_rows_verified=4000,
        monitoring_fold_parameter_records_verified=20,
        portable_f1_replay=f1, qualified_onset_replay=onset, climate_forcing_hashes_verified=687,
        portable_endpoint_score_recomputation=endpoint,
        final_result_sources_verified=len(inventory['retained_result_sources']),
        repository_layout_verified=True)
    if write_manifest:
        (p/'text_profile.json').write_text(json.dumps(profile,indent=2)+'\n')
        (p/'verification/package_checks.json').write_text(json.dumps(receipt,indent=2)+'\n')
        files = {str(path.relative_to(p)):dict(bytes=path.stat().st_size,sha256=digest(path))
            for path in sorted(publication_files(p))}
        manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(),
            files=files, local_review_revision=True, public_deposition_pending=True)
        (p/'publication_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    else:
        manifest = json.loads((p/'publication_manifest.json').read_text())
        for path, record in manifest['files'].items():
            assert digest(p/path) == record['sha256'], path
        assert not [path for path in publication_files(p) if str(path.relative_to(p)) not in manifest['files']]
    print(json.dumps(receipt,indent=2))
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--write-manifest',action='store_true')
    args = parser.parse_args()
    validate(args.write_manifest)
