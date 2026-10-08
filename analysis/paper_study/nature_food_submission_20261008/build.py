"""Build the revised crop-production manuscript and its evidence files."""
from __future__ import annotations
import argparse, hashlib, html, json, re, shutil, subprocess, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document
from docx.shared import Inches, Pt
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from openpyxl import Workbook, load_workbook
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, PageBreak, Image, LongTable, TableStyle, Spacer
from PIL import Image as PILImage

from analysis.paper_study.manuscript import build_documents as layout
from analysis.paper_study.overwinter_leaf_model_20261006.manuscript import build as legacy
from analysis.paper_study.full_validation_20261007.figures import supplementary as validation_figures
from analysis.paper_study.yield_evidence_20261007.figures import supplementary as yield_figures
from analysis.paper_study.nature_food_submission_20261008.figures import current_map_figures as map_figures
from analysis.paper_study.full_validation_20261007.tables import supplementary_tables
from analysis.paper_study.yield_evidence_20261007.tables import supplementary_tables as yield_tables
from analysis.paper_study.map_restoration_20261007.tables import supplemental_tables as map_tables
from analysis.paper_study.nature_food_submission_20261008.sections import sections, TITLE, HERE, AUTHOR, AFFILIATION, EMAIL
from analysis.paper_study.nature_food_submission_20261008.figures import main as draw_figures, supplementary_framework, supplementary_phenology
from analysis.paper_study.nature_food_submission_20261008.spatial import supplementary_tables as spatial_tables
from analysis.paper_study.nature_food_submission_20261008.spatial_figures import environmental_comparison
from analysis.paper_study.nature_food_submission_20261008.tables import main_tables, offset_table
from analysis.paper_study.nature_food_submission_20261008.prepare import prepare

ROOT = HERE.parents[2]
EVIDENCE = ROOT / 'analysis/paper_study/nature_food_fix_20261007'
PUBLICATION = ROOT / 'publication/european_wheat_stb'
TOKEN = re.compile(r'\[\[([A-Za-z0-9_|]+)\]\]')
CITATION = re.compile(r'\[([1-9]\d*(?:\s*,\s*[1-9]\d*)*)\]')
MAIN_ORDER = ['abstract.txt','introduction.txt','climate_results.txt','field_results.txt','baseline_results.txt',
              'discussion_field_and_limits.txt','methods.txt','availability.txt']


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def polish(text: str) -> str:
    text = re.sub(r'(?<=[a-z])(?=\d)', ' ', text)
    text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)
    text = re.sub(r'(?<=[A-Za-z])(?=[+−]\d)', ' ', text)
    text = re.sub(r'\bA(?=\d)', 'A ', text)
    text = re.sub(r'\b(MAE|RMSE|SED|HAD|R²)(?=\d)', r'\1 ', text)
    text = re.sub(r'\b(BBCH|BASF|SSP|GS|LAI)(?=\d)', r'\1 ', text)
    text = re.sub(r'(?<=\d)(?=[a-z]{2})', ' ', text)
    text = re.sub(r';(?=\S)', '; ', text)
    text = re.sub(r'(?<!\d),(?=\S)', ', ', text)
    text = re.sub(r',(?!\d{3}\b(?!\.\d))(?=\S)', ', ', text)
    text = re.sub(r'\b(SSP|GS) (?=\d)', r'\1', text)
    text = text.replace('NEX-GDDP-CMIP6v 2', 'NEX-GDDP-CMIP6v2')
    return text


def html_text(value: str) -> str:
    value = re.sub(r'</?div[^>]*>', '', value)
    value = re.sub(r'</?span[^>]*>', '', value)
    value = re.sub(r'</?i>', '', value)
    value = re.sub(r'<[^>]+>', '', value)
    return html.unescape(value).strip()


def number_references(raw: dict[str, list[str]]):
    records = json.loads((HERE/'references/references.csl.json').read_text())
    lookup = {row['id']: row for row in records}
    ordered_ids = []
    for name in MAIN_ORDER + ['supplementary_methods.txt']:
        for paragraph in raw[name]:
            for match in TOKEN.finditer(paragraph):
                for key in match.group(1).split('|'):
                    if key not in ordered_ids:
                        ordered_ids.append(key)
    missing = sorted(set(ordered_ids) - set(lookup))
    if missing:
        raise ValueError(f'Missing CSL records: {missing}')
    all_tokens = set(TOKEN.findall('\n'.join(p for group in raw.values() for p in group)))
    clusters = {token: token.split('|') for token in all_tokens}
    inp = {'items': [lookup[key] for key in ordered_ids], 'ordered_ids': ordered_ids, 'clusters': clusters}
    (HERE/'citation_input.json').write_text(json.dumps(inp, indent=2, ensure_ascii=False)+'\n')
    output = HERE/'citation_rendered.json'
    subprocess.run(['node', str(HERE/'render_nature_csl.js'), str(HERE/'citation_input.json'), str(output)], check=True)
    rendered = json.loads(output.read_text())
    number = {key: i+1 for i, key in enumerate(rendered['bibliography_ids'])}
    def replace(text: str) -> str:
        def sub(match):
            ids = match.group(1).split('|')
            nums = sorted(number[key] for key in ids)
            return '['+','.join(str(n) for n in nums)+']'
        return polish(TOKEN.sub(sub, text))
    converted = {name: [replace(p) for p in paragraphs] for name, paragraphs in raw.items()}
    references = [re.sub(r'^\d+\.\s*', '', html_text(entry)) for entry in rendered['bibliography_html']]
    return converted, rendered, references


def _bootstrap_tables():
    root = EVIDENCE/'validation'
    d = pd.read_csv(root/'paired_cluster_bootstrap_summary.csv')
    rows = [['Endpoint','Evaluation','Scope','Comparator','n','Groups','Estimate','95% grouped interval']]
    for partition, label in [('reused_BASF2019','BASF 2019'),('reused_strict_Corteva','Corteva')]:
        onset = d[(d.endpoint.eq('genuine_two_sided_onset_distance_days')) & d.partition.eq(partition) &
                  d.leaf_scope.eq('top3_source_numbered') & d.comparison_population.eq('original_eligible_records') &
                  d.comparator.eq('phenology_only')]
        for metric, name in [('difference','Model − benchmark'),('model','Seasonal model'),('benchmark','Phenology benchmark')]:
            r = onset[onset.metric.eq(metric)].iloc[0]
            ci = f'{r.lower95:.2f} to {r.upper95:.2f}' if metric == 'difference' else '—'
            rows.append(['Onset interval excess (days)',label,'F1–F3 source-numbered',name,
                         str(int(r.records)),str(int(r.coordinate_year_groups)),f'{r.point:.2f}',ci])
        severity = d[(d.endpoint.eq('all_assessments_severity_pp')) & d.partition.eq(partition) &
                     d.leaf_scope.eq('all_source_numbered_leaves') & d.comparison_population.eq('original_eligible_records') &
                     d.comparator.eq('calibration_leaf_mean')]
        for metric, name in [('model_rmse','Seasonal model RMSE'),('benchmark_rmse','Calibration-leaf mean RMSE'),('rmse_difference','Paired RMSE difference')]:
            r = severity[severity.metric.eq(metric)].iloc[0]
            ci = f'{r.lower95:.2f} to {r.upper95:.2f}' if metric == 'rmse_difference' else '—'
            rows.append(['Severity RMSE (percentage points)',label,'All numeric source leaves',name,
                         str(int(r.records)),str(int(r.coordinate_year_groups)),f'{r.point:.2f}',ci])
    caption = ('Table S10a | Frozen seasonal-model and comparator errors and paired differences. Onset error is distance outside a genuinely two-sided observed interval. '
               'Severity error uses all numeric assessment records and source-specific severity percentages. Groups are coordinate-years; bootstrap intervals use 20,000 intact paired resamples and condition on frozen predictions.')

    b = [['Scope or sensitivity','Evaluation','Histories or groups','Model − benchmark (days)','95% grouped interval']]
    selectors = [
        ('Confirmed BASF flag leaf','reused_BASF2019','source_leaf1','original_eligible_records'),
        ('Corteva source leaf 1; final rank unknown','reused_strict_Corteva','source_leaf1','original_eligible_records'),
        ('Corteva top three; same-day ordered stages','reused_strict_Corteva','top3_source_numbered','same_day_ordered_two_bound_stage'),
        ('Corteva top three; exclude pre-BBCH37 leaf-1 zeros','reused_strict_Corteva','top3_source_numbered','exclude_observed_pre37_leaf1_zeros'),
        ('Corteva top three; exclude pre-BBCH39 leaf-1 zeros','reused_strict_Corteva','top3_source_numbered','exclude_observed_pre39_leaf1_zeros')]
    for label, partition, scope, population in selectors:
        r = d[(d.endpoint.eq('genuine_two_sided_onset_distance_days')) & d.partition.eq(partition) &
              d.leaf_scope.eq(scope) & d.comparison_population.eq(population) & d.comparator.eq('phenology_only') &
              d.metric.eq('difference')].iloc[0]
        b.append([label,partition.replace('reused_',''),f'{int(r.records)} / {int(r.coordinate_year_groups)}',
                  f'{r.point:+.2f}',f'{r.lower95:.2f} to {r.upper95:.2f}'])
    capb = ('Table S10b | Leaf-identity and observation-population sensitivities for paired onset distance. BASF source leaf 1 is explicitly the flag leaf; the final-rank identity of Corteva source leaf 1 is unresolved. '
            'Intervals are 95% percentiles from 20,000 paired coordinate-year resamples. Zero-assessment exclusion is a sensitivity definition, not an inferred placeholder label.')

    nested = pd.read_csv(root/'nested_calibration/nested_selection_comparison.csv')
    fold = pd.read_csv(root/'nested_calibration/fold_stage_identification.csv')
    c = [['Model / stage','Selected threshold or setting','Nested loss / threshold range','Ties or exact minima','Original result','Original loss','Bracketed evidence']]
    for r in nested.itertuples():
        c.append([r.model.replace('_',' '),r.nested_selected_candidate,f'{r.nested_pooled_distance_days:.3f}',str(r.nested_primary_ties),
                  r.original_candidate,f'{r.original_pooled_distance_days:.3f}',f'{r.top3_two_sided_histories} / {r.top3_two_sided_fields}'])
    for r in fold.itertuples():
        c.append([f'{r.fold.replace("spatial_","fold ")} BBCH {r.event}',f'{r.selected_threshold:.2f}',
                  f'{r.joint_exact_minimum_threshold_low:.2f} to {r.joint_exact_minimum_threshold_high:.2f}',
                  str(r.joint_exact_minimum_grid_values),'Censored-loss fit','—',
                  f'{r.genuine_two_sided_fields} genuinely two-sided fields'])
    capc = ('Table S10c | Fully nested selection re-estimates added stage thresholds, flowering proxy and weather preprocessing within each original-calibration training fold. '
            'Disease and phenology settings reproduce the original choices; three disease candidates share the nested minimum. Exact-minimum values and bracket counts show remaining threshold identification.')
    return [(rows, caption) for rows, caption in [(rows, caption), (b, capb), (c, capc)]]


def _climate_tables():
    d = pd.read_csv(EVIDENCE/'climate/headline_robustness.csv')
    wanted = {'F1_symptom_relative_anthesis_days':'Flag-leaf symptoms (days relative to anthesis)',
              'GS65_85_lost_had3':'HAD deficit (days per nominal LAI)',
              'conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai':'Conditional yield effect (t ha⁻¹ per nominal LAI)'}
    rows = [['Setting','Outcome','Mean change','Three-GCM range','Spatial MCSE','Common-pair coverage']]
    for setting in d.setting.drop_duplicates():
        for metric,label in wanted.items():
            q = d[(d.setting.eq(setting)) & d.metric.eq(metric) & d.population.eq('all_settings_common_paired')]
            if q.empty:
                q = d[(d.setting.eq(setting)) & d.metric.eq(metric) & d.population.eq('within_setting_paired')]
            if q.empty: continue
            r = q.iloc[0]
            rows.append([setting.replace('_',' '),label,f'{r.gcm_mean_change:+.3f}',
                         f'{r.gcm_min_change:+.3f} to {r.gcm_max_change:+.3f}',f'{r.spatial_mcse_gcm_mean_change:.3f}',
                         f'{100*r.minimum_gcm_paired_coverage:.1f}–{100*r.maximum_gcm_paired_coverage:.1f}%'])
    caption = ('Table S11a | Registered late-century SSP5–8.5 structural, detection, canopy and functional-conversion settings. '
               'Where available, outcomes use the all-setting common paired population; the yield-effect metric uses its within-setting paired population. '
               'The three-GCM range is deterministic model spread and spatial MCSE describes 64 stratified area draws; neither is a probability interval. Yield effects are conditional model products.')

    p = pd.read_csv(EVIDENCE/'specification/Parameter_inventory.csv')
    keep = [c for c in ['symbol','parameter','value','unit','role','uncertainty'] if c in p.columns]
    p = p[keep]
    paramrows = [[c.replace('_',' ').title() for c in keep]]
    for r in p.itertuples(index=False, name=None):
        paramrows.append(['—' if pd.isna(x) else str(x) for x in r])
    capp = ('Table S12 | Parameter and scenario inventory. Values retain their model, calibration or registered-sensitivity provenance; uncertainty describes identification or scenario status, not a probability distribution unless explicitly stated. '
            'The complete nine-column machine-readable inventory is supplied as Parameter_inventory.csv.')

    de = pd.read_csv(EVIDENCE/'climate/decomposition/supported_forcing_ensemble_decomposition.csv')
    labels = {'weather_shapley':'Disease-weather contribution','host_shapley':'Host-development contribution',
              'total_change':'Net HAD change','interaction':'Weather × host interaction'}
    rowsd = [['Component','Mean (HAD days)','GCM range','Spatial MCSE','Coverage','Draw-year pairs']]
    for component,label in labels.items():
        r = de[de.component.eq(component)].iloc[0]
        rowsd.append([label,f'{r.gcm_mean:+.3f}',f'{r.gcm_min:+.3f} to {r.gcm_max:+.3f}',
                      f'{r.spatial_mcse_gcm_mean:.3f}',
                      f'{100*r.minimum_gcm_coverage:.2f}–{100*r.maximum_gcm_coverage:.2f}%',
                      f'{int(r.minimum_gcm_pairs)}–{int(r.maximum_gcm_pairs)}'])
    capd = ('Table S13 | Four-combination Shapley decomposition for GS65–85 upper-three-leaf HAD proxy loss under SSP5–8.5. '
            'Interaction is allocated equally between the two contributions; weather plus host equals the net. Twenty of 5,751 finite pairs are excluded because a hybrid active-day disease-weather mean exceeds 40 °C. '
            'The supported-forcing population and all decomposition terms share the remaining 5,731 pairs. Contributions are model-response components, not physical causal effects or measured yield.')
    return [(rows, caption), (paramrows, capp), (rowsd, capd)]


def _numbered_text(text: str) -> str:
    return CITATION.sub(lambda m: f'<super>{m.group(1)}</super>', html.escape(text))


def _add_text_paragraph(document, text):
    p = document.add_paragraph()
    for part in re.split(r'(\[[1-9]\d*(?:\s*,\s*[1-9]\d*)*\])', text):
        if not part: continue
        run = p.add_run(part)
        if CITATION.fullmatch(part): run.font.superscript = True
    return p


def _line_numbers(document):
    for section in document.sections:
        element = OxmlElement('w:lnNumType')
        element.set(qn('w:countBy'), '1')
        element.set(qn('w:restart'), 'continuous')
        section._sectPr.append(element)


def _pdf_body(flow, texts, styles):
    for text in texts:
        style = styles['PaperHeading' if layout.is_heading(text) else 'PaperBody']
        flow.append(Paragraph(_numbered_text(text), style))


def _word_doc(body, references_html, captions, dest: Path, tables, declarations):
    document = Document(); layout.configure_doc(document)
    document.core_properties.title=TITLE
    document.core_properties.author=AUTHOR
    document.styles['Normal'].paragraph_format.line_spacing = 2.0
    document.add_heading(TITLE, level=0)
    document.add_paragraph(AUTHOR+'*')
    document.add_paragraph(AFFILIATION)
    document.add_paragraph('*Correspondence: '+EMAIL)
    document.add_heading('Abstract', level=1)
    for index, text in enumerate(body):
        paragraph = _add_text_paragraph(document, text)
        if index == 1:
            paragraph.paragraph_format.page_break_before = True
    document.add_heading('References', level=1)
    for entry in references_html:
        layout.add_reference_doc(document, entry)
    for paragraph in declarations:
        _add_text_paragraph(document,paragraph)
    _table_docx(document,tables)
    for stem, caption in captions.items():
        document.add_page_break()
        document.add_picture(str(dest/'figures'/f'{stem}.png'),width=Inches(6.25))
        document.add_paragraph(caption)
    _line_numbers(document)
    document.save(dest/'Manuscript.docx')


def _pdf_table(rows,caption,styles,main=False):
    available=A4[0]-90
    widths=([75,112,125,available-312] if main and len(rows[0])==4
            else [available/len(rows[0])]*len(rows[0]))
    cells=[[Paragraph(html.escape(str(v)),styles['PaperTable']) for v in row] for row in rows]
    table=LongTable(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
    table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef1f3')),
        ('LINEBELOW',(0,0),(-1,0),.5,colors.grey),('LINEBELOW',(0,-1),(-1,-1),.5,colors.grey),
        ('LEFTPADDING',(0,0),(-1,-1),4),('RIGHTPADDING',(0,0),(-1,-1),4),
        ('TOPPADDING',(0,0),(-1,-1),4),('BOTTOMPADDING',(0,0),(-1,-1),4)]))
    return [Paragraph(html.escape(caption),styles['PaperCaption']),table]


def _pdf_doc(body, refs_html, captions, figdir, dest: Path, tables, declarations):
    styles = layout.font_styles()
    flow = [Paragraph(html.escape(TITLE), styles['PaperTitle']),
            Paragraph(html.escape(AUTHOR+'*'),styles['PaperBody']),
            Paragraph(html.escape(AFFILIATION),styles['PaperBody']),
            Paragraph(html.escape('*Correspondence: '+EMAIL),styles['PaperBody']),
            Paragraph('Abstract', styles['PaperHeading'])]
    _pdf_body(flow, body[:1], styles)
    flow.append(PageBreak())
    _pdf_body(flow, body[1:], styles)
    flow.append(PageBreak()); flow.append(Paragraph('References', styles['PaperHeading']))
    for ref in refs_html:
        flow.append(Paragraph(layout.reference_inline(ref), styles['PaperReference']))
    _pdf_body(flow,declarations,styles)
    for rows,caption in tables:
        flow.append(PageBreak());flow.extend(_pdf_table(rows,caption,styles,main=True))
    for stem, caption in captions.items():
        path = figdir/f'{stem}.png'
        with PILImage.open(path) as im: width, height = im.size
        scale = min(450/width, 530/height)
        flow.extend([PageBreak(), Image(str(path), width*scale, height*scale)])
        flow.append(Paragraph(html.escape(caption), styles['PaperCaption']))
    SimpleDocTemplate(str(dest/'Manuscript.pdf'), pagesize=A4, rightMargin=45, leftMargin=45,
                      topMargin=45, bottomMargin=45, title=TITLE, author='Gang Zhao').build(
                          flow, onFirstPage=layout.footer, onLaterPages=layout.footer)


def _table_docx(document, table_data):
    for rows, caption in table_data:
        document.add_page_break(); document.add_paragraph(caption)
        table = document.add_table(rows=1, cols=len(rows[0])); table.style = 'Table Grid'
        for j, value in enumerate(rows[0]): table.rows[0].cells[j].text = str(value)
        for row in rows[1:]:
            cells = table.add_row().cells
            for j, value in enumerate(row): cells[j].text = str(value)
        for i, row in enumerate(table.rows):
            for cell in row.cells:
                for p in cell.paragraphs:
                    p.paragraph_format.space_after = Pt(2); p.paragraph_format.line_spacing = 1
                    for run in p.runs: run.font.size = Pt(8); run.bold = i == 0


def _supplement_doc(paragraphs, tables, fig_captions, dest: Path, references_html):
    document = Document(); layout.configure_doc(document)
    document.add_heading('Supplementary Information', level=0)
    document.add_paragraph(TITLE)
    document.add_paragraph(AUTHOR)
    for p in paragraphs:
        if layout.is_heading(p): document.add_heading(p, level=1 if p in ['Supplementary Methods','Methods'] else 2)
        else: _add_text_paragraph(document, p)
    document.add_heading('References',level=1)
    for entry in references_html:layout.add_reference_doc(document,entry)
    _table_docx(document, tables)
    document.add_page_break(); document.add_heading('Supplementary figures', level=1)
    for stem, caption in fig_captions.items():
        path = dest/'supplementary_figures'/f'{stem}.png'
        document.add_page_break(); document.add_picture(str(path), width=Inches(6.35)); document.add_paragraph(caption)
    document.add_page_break(); document.add_heading('Supplementary model specification', level=1)
    document.add_paragraph('The complete editable mathematical specification and its rendered PDF are supplied as separate files. The 84-entry parameter inventory is reproduced in Table S12 and Parameter_inventory.csv.')
    document.save(dest/'Supplementary_Information.docx')


def _supplement_pdf(paragraphs, table_data, captions, dest: Path, refs_html):
    styles = layout.font_styles()
    flow = [Paragraph('Supplementary Information', styles['PaperTitle']),
            Paragraph(html.escape(TITLE),styles['PaperBody']),Paragraph(AUTHOR,styles['PaperBody'])]
    _pdf_body(flow, paragraphs, styles)
    flow.append(Paragraph('References',styles['PaperHeading']))
    for ref in refs_html:flow.append(Paragraph(layout.reference_inline(ref),styles['PaperReference']))
    available = A4[0]-90
    for rows, caption in table_data:
        flow.append(PageBreak()); flow.append(Paragraph(html.escape(caption), styles['PaperCaption']))
        widths = [available/len(rows[0])]*len(rows[0])
        cells = [[Paragraph(html.escape(str(v)), styles['PaperTable']) for v in row] for row in rows]
        table = LongTable(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
        table.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#eef1f3')),
                                   ('LINEBELOW',(0,0),(-1,0),.5,colors.grey),('LINEBELOW',(0,-1),(-1,-1),.5,colors.grey),
                                   ('LEFTPADDING',(0,0),(-1,-1),3),('RIGHTPADDING',(0,0),(-1,-1),3),
                                   ('TOPPADDING',(0,0),(-1,-1),3),('BOTTOMPADDING',(0,0),(-1,-1),3)]))
        flow.append(table)
    for stem, caption in captions.items():
        path = dest/'supplementary_figures'/f'{stem}.png'
        with PILImage.open(path) as im: width, height = im.size
        scale = min(450/width, 520/height)
        flow.extend([PageBreak(), Image(str(path), width*scale, height*scale),
                     Paragraph(html.escape(caption), styles['PaperCaption'])])
    SimpleDocTemplate(str(dest/'Supplementary_Information.pdf'), pagesize=A4, rightMargin=45,
                      leftMargin=45, topMargin=45, bottomMargin=45, title='Supplementary Information').build(
                          flow, onFirstPage=layout.footer, onLaterPages=layout.footer)


def _curated_figures(figures: dict[str,str], folder: Path):
    rename = {
        'figS13_spatial_symptom_changes':'figS12_spatial_symptom_changes',
        'figS14_spatial_yield_transfers':'figS13_spatial_yield_transfers',
        'figS19_full_grid_example_2001':'figS14_full_grid_example_2001',
        'figS20_climate_robustness':'figS15_climate_robustness',
        'figS21_weather_host_decomposition':'figS16_weather_host_decomposition'}
    sources = {}; curated={}
    for stem, caption in list(figures.items()):
        if stem in rename: target = rename[stem]
        elif stem.startswith(tuple(f'figS{i}_' for i in range(1,12))): target = stem
        else: continue
        sources[stem] = target
        newnum = int(re.search(r'figS(\d+)', target).group(1))
        caption = re.sub(r'^Figure S\d+', f'Figure S{newnum}', caption)
        if stem != target:
            for ext in ['png','pdf','svg','csv']:
                src = folder/f'{stem}.{ext}'; dst = folder/f'{target}.{ext}'
                if src.exists(): shutil.copy2(src,dst)
        curated[target] = polish(caption)
    for path in folder.iterdir():
        if path.suffix.lower() in {'.png','.pdf','.svg','.csv'} and path.stem.startswith('figS'):
            if path.stem not in set(curated): path.unlink()
    (folder/'supplementary_captions.json').write_text(json.dumps(curated,indent=2)+'\n')
    return curated


def _curated_tables():
    all_tables = supplementary_tables()+yield_tables()+map_tables()+_bootstrap_tables()+_climate_tables()
    moved = legacy.tables()
    for letter,(rows,caption,widths) in zip('abc',moved):
        caption = re.sub(r'^Table[123]',f'Table S9{letter}',caption)
        if letter == 'c':
            caption = ('Table S9c | Wheat-harvested-area-weighted paired changes in modeled disease and conditional yield-effect proxies across European wheat cells. '
                       'Each period is compared with its SSP-specific 1991–2020 reference using three climate models and 64 stratified area-proportional spatial draws. '
                       'HAD3 is GLAI-days per unit nominal upper-three-leaf LAI; conditional transfer is t ha⁻¹ per nominal LAI at b=0.018. '
                       'Climate-model ranges and spatial Monte Carlo standard errors are distinct summaries and exclude ecological-parameter, canopy-transfer and observation uncertainty. '
                       'Yield-effect values are conditional model products, not observed or total European production losses.')
        all_tables.append((rows,caption))
    results=[]
    for rows,caption,*_ in all_tables:
        m = re.match(r'Table\s*S?(\d+)([a-z]?)',caption)
        if not m: continue
        n,letter=int(m.group(1)),m.group(2)
        if n in {5,6} and caption.startswith(f'Table S{n}'): continue # archived v1/French evidence
        if n==10 and letter in {'a','b'} and caption.startswith('Table S10') and ('Published yellow-rust' in caption or 'Published leaf-rust' in caption): continue # rust yield studies
        remap={(7,'a'):(5,'a'),(7,'b'):(5,'b'),(8,'a'):(6,'a'),(8,'b'):(6,'b'),(8,'c'):(6,'c'),
               (9,''):(7,''),(12,'a'):(8,'a'),(12,'b'):(8,'b'),(13,'a'):(10,'a'),(13,'b'):(10,'b'),(13,'c'):(10,'c'),
               (14,''):(11,''),(15,''):(12,''),(16,''):(13,'')}
        if (n,letter) in remap:
            nn,ll=remap[(n,letter)]
            caption=re.sub(r'^Table\s*S?\d+[a-z]?',f'Table S{nn}{ll}',caption)
        if caption.startswith('Table S11') and n==11 and not letter: continue
        results.append((rows,caption))
    # The three moved main-summary tables become S9a–c.
    results=[(r,re.sub(r'^Table[123]',f'Table S9{letter}',c) if re.match(r'^Table[123]',c) else c)
             for letter,(r,c) in zip('abc',[(r,c) for r,c in results if re.match(r'^Table[123]',c)])] + \
            [(r,c) for r,c in results if not re.match(r'^Table[123]',c)]
    def key(item):
        m=re.match(r'Table S(\d+)([a-z]?)',item[1]); return (int(m.group(1)),m.group(2))
    results=[(r,polish(c.replace('Table S13 |','Table S13a |',1))) for r,c in results]
    results.append(offset_table())
    results.extend(spatial_tables())
    return sorted(results,key=key)


def _source_workbook(dest: Path):
    validation=EVIDENCE/'validation'; study=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006'
    yieldroot=ROOT/'analysis/paper_study/yield_evidence_20261007'
    maproot=ROOT/'analysis/paper_study/map_restoration_20261007'
    sources=[]
    for name in ['field_stage_metrics_all.csv','field_stage_constraints_all.csv','field_stage_coverage_all.csv',
                 'german_stage_metrics_all.csv','disease_assessment_detection_metrics_baseline.csv',
                 'disease_onset_bracket_metrics_baseline.csv','disease_observed_window_occurrence_metrics_baseline.csv']:
        sources.append((validation/name))
    sources += [validation/'paired_cluster_bootstrap_summary.csv',validation/'paired_onset_primary.csv',
                validation/'paired_severity_primary.csv',validation/'bootstrap_group_membership.csv',
                validation/'leaf_evidence_and_stage_coverage.csv',
                validation/'nested_calibration/nested_selection_comparison.csv',
                validation/'nested_calibration/fold_stage_identification.csv']
    sources += [study/'climate/ensemble_paired_period_changes.csv',study/'climate/coverage_by_period_and_status.csv',
                EVIDENCE/'climate/headline_robustness.csv',EVIDENCE/'climate/paired_changes_by_setting_gcm.csv',
                EVIDENCE/'climate/coverage_by_setting_gcm_period_status.csv',
                EVIDENCE/'climate/decomposition/supported_forcing_ensemble_decomposition.csv',
                EVIDENCE/'climate/decomposition/supported_forcing_decomposition_by_gcm.csv',
                EVIDENCE/'climate/decomposition/supported_forcing_coverage.csv',
                EVIDENCE/'climate/decomposition/host_stage_calendar_diagnostics.csv']
    sources += [yieldroot/'parker2004_table4_yield_loss_matrix.csv',yieldroot/'parker2004_table5_yield_slopes.csv',yieldroot/'foulkes2006_group_separated_predictions.csv',yieldroot/'published_yield_response_metrics.csv',yieldroot/'published_yield_domain_compatibility.csv',
                yieldroot/'foulkes_group_separated_predictions.csv',
                yieldroot/'stb_independent/foulkes2006_table2_paired_predicted_means.csv',
                yieldroot/'stb_independent/foulkes2006_table3_coefficients.csv',
                ROOT/'analysis/paper_study/nature_food_revision_20261007/disease_data_audit/BASF_observed_treatment_yield_contrasts.csv']
    sources += [maproot/name for name in ['map_wheat_domain.csv','map_field_locations.csv','map_draw_values.csv',
                                           'map_ensemble_values.csv','map_metric_coverage.csv']]
    sources += [EVIDENCE/'specification/Parameter_inventory.csv',HERE/'derived/weather_host_offset.csv',HERE/'derived/map_canopy_changes.csv',dest/'Main_Table1.csv']
    sources += [HERE/'spatial_sources/wheat_cells_eea_regions.csv']
    sources += sorted((HERE/'derived/spatial').glob('*.csv'))
    sources += sorted((dest/'figures').glob('*.csv'))
    sources += sorted((dest/'supplementary_figures').glob('*.csv'))
    sources=[p for p in sources if p.exists()]
    book=Workbook(write_only=True);manifest=[];used=set()
    for i,path in enumerate(sources,1):
        frame=pd.read_csv(path)
        stem=re.sub(r'[^A-Za-z0-9]+','_',path.stem).strip('_')
        sheetname=f'{i:02d}_{stem}'[:31]
        if sheetname in used: sheetname=f'{i:02d}_source_{i}'[:31]
        used.add(sheetname);sheet=book.create_sheet(sheetname);sheet.append(list(frame.columns))
        for row in frame.itertuples(index=False,name=None):
            sheet.append([None if pd.isna(v) else v.item() if isinstance(v,np.generic) else v for v in row])
        manifest_path=(Path('publication/european_wheat_stb')/path.relative_to(dest)).as_posix() if path.is_relative_to(dest) else str(path.relative_to(ROOT))
        manifest.append([sheetname,manifest_path,len(frame),sha(path)])
    sheet=book.create_sheet('Source_manifest');sheet.append(['Sheet','Source','Rows','SHA256'])
    for row in manifest:sheet.append(row)
    sheet=book.create_sheet('Definitions');sheet.append(['Metric','Definition'])
    for row in [
        ('Genuine two-sided onset','Last zero plus one day through first positive; no exact date imputation.'),
        ('Disease severity','Source-specific percentage points; not healthy-area duration.'),
        ('Severity integral','Disease percentage-days; not HAD or absolute green area.'),
        ('HAD proxy','Nominal reference upper-three leaf area multiplied by modeled functional-loss fractions.'),
        ('Conditional yield effect','Fixed coefficient multiplied by HAD proxy loss; not an observed yield loss.'),
        ('BASF treated–untreated yields','87 contrasts in 11 fields; plot pairing and fully protected reference status are unresolved.'),
        ('Weather–host Shapley terms','Four forcing combinations; interaction split equally; terms sum to net change.'),
        ('Host offset percentage','Minus 100 times host contribution divided by weather contribution on the same population; ensemble ratio differs from the mean of individual percentages.'),
        ('Supported forcing','20 of 5,751 finite pairs excluded above 40 °C on any active hybrid disease-weather day.'),
        ('Spatial Monte Carlo error','Sampling error for 64 registered area-proportional draws.'),
        ('GCM range','Range across three deterministic climate models; not a probability interval.')]:sheet.append(row)
    for row in [
        ('Environmental regions','Independent EEA 2016 polygons intersect wheat-cell centroids; unsampled regions have no response estimate.'),
        ('Regional paired change','Original design-weighted valid-area-time domain ratio; GCM means averaged after estimating each domain ratio.'),
        ('Spatial sign area share','Estimated reference wheat-area share from original 64 draw weights; no observed affected hectares inferred.'),
        ('Regional coverage','Main comparisons include named regions with at least four draws; sparse-domain MCSE is withheld.')]:sheet.append(row)
    book.save(dest/'Source_Data.xlsx')


def _package(dest: Path, evidence_archive: Path | None = None):
    base = Path(evidence_archive) if evidence_archive is not None else PUBLICATION/'Software_and_Evidence.zip'
    if not base.exists():
        raise FileNotFoundError('Supply --evidence-archive or use --documents-only for publication regeneration.')
    temp=dest/'Software_and_Evidence.zip.tmp'
    excluded={'before','__pycache__','figure_preview','render_check','numba_cache','.pytest_cache'}
    snapshots={'source_snapshot','pre_review','before_conversion','aborted_before_juvenile_handover'}
    record={}
    with zipfile.ZipFile(base,'r') as source, zipfile.ZipFile(temp,'w',zipfile.ZIP_DEFLATED,compresslevel=4) as out:
        def write_member(name,data):
            if name in record:return
            out.writestr(name,data)
            record[name]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data)}
        for info in source.infolist():
            name=info.filename
            if info.is_dir() or name in {'REPRODUCIBILITY.txt','PACKAGE_MANIFEST.json','Publication_requirements.txt'}:continue
            if name.startswith('publication/european_wheat_stb/'):continue
            if name.startswith('analysis/paper_study/nature_food_submission_20261008/'):continue
            arc=Path(name)
            if arc.is_absolute() or '..' in arc.parts:raise ValueError('Invalid evidence archive member path.')
            local=ROOT/arc
            if local.is_file() and not snapshots.intersection(arc.parts) and local.suffix in {'.py','.js','.xml','.csl','.bib','.toml'}:
                data=local.read_bytes()
            else:data=source.read(info)
            write_member(name,data)
        for code_root in ['model','process_model','analysis/paper_study','analysis/phenology']:
            for file in (ROOT/code_root).rglob('*.py'):
                relative=file.relative_to(ROOT)
                if excluded.intersection(relative.parts) or snapshots.intersection(relative.parts):continue
                write_member(relative.as_posix(),file.read_bytes())
        for asset in (ROOT/'analysis/paper_study/manuscript/csl_vendor').iterdir():
            if asset.is_file():write_member(asset.relative_to(ROOT).as_posix(),asset.read_bytes())
        geography=ROOT/'data/geography/ne_110m_admin_0_countries.zip'
        write_member(geography.relative_to(ROOT).as_posix(),geography.read_bytes())
        grid_replay=ROOT/'analysis/paper_study/nature_food_revision_20261007/continental_replay/annual_outputs/nasa/ACCESS-CM2/ssp585/2001.parquet'
        write_member(grid_replay.relative_to(ROOT).as_posix(),grid_replay.read_bytes())
        skip_large={'climate/draw_season_outputs.parquet','validation/bootstrap_metric_replicates.csv.gz',
                    'validation/bootstrap_draw_indices.npz','validation/nested_calibration/overwinter_source_model_inner_assessment_predictions.csv'}
        for folder in [EVIDENCE,HERE]:
            for file in folder.rglob('*'):
                if not file.is_file():continue
                rel=file.relative_to(folder)
                if excluded.intersection(rel.parts) or str(rel) in skip_large:continue
                if file.name=='clean_extraction.json':continue
                name=file.relative_to(ROOT).as_posix()
                if name in record:continue
                write_member(name,file.read_bytes())
        for file in dest.rglob('*'):
            if not file.is_file() or file.suffix in {'.zip','.tmp'}:continue
            if file.name in {'publication_receipt.json','verification_receipt.json','reproduction_receipt.json'}:continue
            name=(Path('publication/european_wheat_stb')/file.relative_to(dest)).as_posix()
            write_member(name,file.read_bytes())
        write_member('REPRODUCIBILITY.txt',(HERE/'Reproducibility.txt').read_bytes())
        write_member('Publication_requirements.txt',(HERE/'Publication_requirements.txt').read_bytes())
        manifest={'package':'Nature Food wheat climate–host production analysis, 8 October 2026',
                  'entry_point':'analysis/paper_study/nature_food_submission_20261008/build.py',
                  'publication_command':'python -m analysis.paper_study.nature_food_submission_20261008.build --output publication/european_wheat_stb_regenerated --documents-only',
                  'member_count_excluding_manifest':len(record),
                  'checksums_exclude_this_manifest':True,'members':record}
        out.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2)+'\n')
    temp.replace(dest/'Software_and_Evidence.zip')


def build(destination: Path, documents_only: bool = False, evidence_archive: Path | None = None):
    dest = Path(destination).expanduser().resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f'Output directory must be empty: {dest}')
    dest.mkdir(parents=True, exist_ok=True)
    prepare()
    raw = sections()
    converted, citeout, references = number_references(raw)
    for name, parts in converted.items():
        (dest/name).write_text('\n\n'.join(parts)+'\n')

    figdir = dest/'figures'; figdir.mkdir()
    captions = draw_figures(figdir)
    main_table_data=main_tables()
    pd.DataFrame(main_table_data[0][0][1:],columns=main_table_data[0][0][0]).to_csv(dest/'Main_Table1.csv',index=False)
    # Retain source CSVs created by the figure builders and copy a full evidence table.
    _word_doc([p for name in MAIN_ORDER for p in converted[name]], citeout['bibliography_html'], captions, dest,main_table_data,converted['declarations.txt'])
    _pdf_doc([p for name in MAIN_ORDER for p in converted[name]], citeout['bibliography_html'], captions, figdir, dest,main_table_data,converted['declarations.txt'])
    main_text = TITLE+'\n\n'+AUTHOR+'*\n\n'+AFFILIATION+'\n\n*Correspondence: '+EMAIL+'\n\nAbstract\n\n'+'\n\n'.join(converted['abstract.txt'])+'\n\n'
    main_text += '\n\n'.join(p for name in MAIN_ORDER[1:] for p in converted[name])
    main_text += '\n\nReferences\n\n'+'\n\n'.join(f'{i}. {r}' for i,r in enumerate(references,1))
    main_text += '\n\n'+'\n\n'.join(converted['declarations.txt'])
    main_text += '\n\nTables\n\n'+'\n\n'.join(cap+'\n\n'+'\n'.join('\t'.join(map(str,row)) for row in rows) for rows,cap in main_table_data)
    main_text += '\n\nFigure legends\n\n'+'\n\n'.join(captions.values())+'\n'
    (dest/'Manuscript.txt').write_text(main_text)
    (dest/'references.txt').write_text('References\n\n'+'\n\n'.join(f'{i}. {r}' for i,r in enumerate(references,1))+'\n')
    (dest/'references.csl.json').write_text(json.dumps(citeout['bibliography_ids'],indent=2)+'\n')
    (dest/'references.bib').write_text((HERE/'references/references.bib').read_text())

    # Retain current crop-stage, Septoria and crop-response evidence only.
    sfig = dest/'supplementary_figures'; sfig.mkdir()
    figcaps = {}
    figcaps.update(validation_figures(sfig, legacy.export))
    figcaps.update(yield_figures(sfig, legacy.export))
    figcaps.update(map_figures(sfig, legacy.export))
    # The archived supplementary series labels the three moved summaries S11a–c.
    for stem, source, caption in [
        ('figS20_climate_robustness', EVIDENCE/'climate/climate_robustness.png',
         'Figure S20 | Late-century SSP5–8.5 climate-model changes across fourteen registered structural, detection, canopy and functional-conversion settings. Panels retain their metric-specific paired populations and spatial coverage. Model ranges are deterministic and do not represent probabilities.'),
        ('figS21_weather_host_decomposition', EVIDENCE/'climate/decomposition/weather_host_decomposition.png',
         'Figure S21 | Four-combination decomposition of modeled weather and host-development contributions to upper-three-leaf HAD proxy loss and flag-leaf symptom timing. The interaction is allocated equally between contributions. Supported-forcing results exclude hybrid active days with mean temperature above 40 °C. Hybrid forcing terms describe model response and do not establish physical causality or observed yield effects.')]:
        target = sfig/f'{stem}.png'; shutil.copy2(source, target)
        for ext in ['pdf']:
            sibling = source.with_suffix('.pdf')
            if sibling.exists(): shutil.copy2(sibling, sfig/f'{stem}.{ext}')
        figcaps[stem] = caption
    figcaps = _curated_figures(figcaps,sfig)
    stem,caption=supplementary_framework(sfig);figcaps[stem]=polish(caption)
    stem,caption=supplementary_phenology(sfig);figcaps[stem]=polish(caption)
    figcaps['figS19_environmental_region_responses']=polish(environmental_comparison(sfig,legacy.export))
    (sfig/'supplementary_captions.json').write_text(json.dumps(figcaps,indent=2)+'\n')
    table_data = _curated_tables()
    for stem, caption in figcaps.items():
        (sfig/f'{stem}.caption.txt').write_text(caption+'\n')

    supplement_paras = converted['supplementary_methods.txt']
    _supplement_doc(supplement_paras, table_data, figcaps, dest,citeout['bibliography_html'])
    _supplement_pdf(supplement_paras, table_data, figcaps, dest,citeout['bibliography_html'])
    supplement_text = 'Supplementary Information\n\n'+TITLE+'\n\n'+AUTHOR+'\n\n'+'\n\n'.join(supplement_paras)
    supplement_text += '\n\nReferences\n\n'+'\n\n'.join(f'{i}. {r}' for i,r in enumerate(references,1))
    for i,(rows,caption) in enumerate(table_data,1):
        supplement_text += '\n\n'+caption+'\n\n'+'\n'.join('\t'.join(map(str,row)) for row in rows)
    supplement_text += '\n\nSupplementary figures\n\n'+'\n\n'.join(figcaps.values())+'\n'
    (dest/'Supplementary_Information.txt').write_text(supplement_text)

    _source_workbook(dest)
    for name in ['Supplementary_Model_Specification.docx','Supplementary_Model_Specification.pdf',
                 'Supplementary_Model_Specification.md','Parameter_inventory.csv']:
        shutil.copy2(EVIDENCE/'specification'/name, dest/name)
    shutil.copy2(HERE/'Reproducibility.txt',dest/'REPRODUCIBILITY.txt')
    shutil.copy2(HERE/'Publication_requirements.txt',dest/'Publication_requirements.txt')
    if not documents_only:_package(dest,evidence_archive)
    receipt = {'status':'generated','title':TITLE,'abstract_words':len(' '.join(converted['abstract.txt']).split()),
               'main_text_words_excluding_methods':sum(len(' '.join(converted[n]).split()) for n in MAIN_ORDER if n not in ['abstract.txt','methods.txt','availability.txt']),
               'main_figures':len(captions),'main_tables':len(main_table_data),'display_items':len(captions)+len(main_table_data),'references':len(references),'supplementary_figures':len(figcaps),
               'supplementary_tables':len(table_data),'source_workbook_sheets':len(load_workbook(dest/'Source_Data.xlsx',read_only=True).sheetnames),
               'files':{name:sha(dest/name) for name in ['Manuscript.docx','Manuscript.pdf','Manuscript.txt','Supplementary_Information.docx',
                                                          'Supplementary_Information.pdf','Source_Data.xlsx','Software_and_Evidence.zip'] if (dest/name).exists()}}
    (dest/'publication_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    return receipt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--documents-only',action='store_true')
    parser.add_argument('--evidence-archive',type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.output,args.documents_only,args.evidence_archive), indent=2))


if __name__ == '__main__': main()
