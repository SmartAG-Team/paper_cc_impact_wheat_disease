"""Formal mid-century results and figures from the completed 540 annual jobs."""
from pathlib import Path
import json
import html
import pandas as pd
import geopandas as gpd
from docx import Document
from docx.shared import Inches
from PIL import Image as PILImage
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate,Paragraph,Image,PageBreak
from . import data,figures
from analysis.paper_study.manuscript import build_documents as layout


def build():
    evidence=data.HERE/'midcentury_evidence'
    receipt=json.loads((evidence/'completion_receipt.json').read_text())
    assert receipt['annual_jobs']==540 and receipt['periods']==['1991-2020','2031-2060']
    for name,expected in receipt['outputs_sha256'].items():
        assert data.sha(evidence/name)==expected
    destination=data.ROOT/'publication/european_wheat_stb_climate_preview_20261008'
    destination.mkdir(exist_ok=True);folder=destination/'figures';folder.mkdir(exist_ok=True)
    grid=pd.read_parquet(evidence/'full_grid_ensemble_paired_changes.parquet')
    regions=pd.read_csv(evidence/'environmental_region_changes.csv');countries=pd.read_csv(evidence/'country_changes.csv')
    geography=gpd.read_file(data.ROOT/'data/geography/ne_110m_admin_0_countries.zip');cells=grid.drop_duplicates('cell_id')
    figures.style()
    figures.scenario_map(folder,grid,geography,cells,data.CANOPY,'fig1_grid_climate_canopy_impacts',
        'Change in HAD deficit (days per reference LAI)')
    figures.disease_map(folder,grid,geography,cells,period='2031-2060')
    figures.domain_figure(folder,regions,countries,period='2031-2060')
    def score(metric,region='Europe',scenario='ssp585'):
        return regions[regions.metric.eq(metric)&regions.environment_region.eq(region)&regions.scenario.eq(scenario)].iloc[0]
    had=score(data.CANOPY);timing=score(data.ONSET);severity=score(data.SEVERITY);frequency=score(data.FREQUENCY)
    summary=[
      'Climate impacts on European wheat in 2031–2060',
      'Spatially divergent changes in canopy damage',
      'The European healthy-area-duration (HAD) deficit changed by '+', '.join(f'{score(data.CANOPY,scenario=s).mean_change:+.2f}' for s in data.SCENARIOS)+
      ' days per reference leaf area under SSP1–2.6, SSP2–4.5 and SSP5–8.5, respectively, in 2031–2060 relative to 1991–2020. '
      f'Under SSP5–8.5, the three-climate-model mean deficit increased from {had.reference_on_common:.2f} to {had.future_on_common:.2f} days, with changes of {had.gcm_min:+.2f} to {had.gcm_max:+.2f} days across models (Fig. 1).',
      'Disease occurrence, timing and damage severity',
      f'Flag-leaf symptoms shifted {abs(timing.mean_change):.2f} days earlier relative to flowering under mid-century SSP5–8.5. '
      f'The reference-area-weighted mean functional-canopy loss during grain filling increased from {100*severity.reference_on_common:.2f}% to {100*severity.future_on_common:.2f}%, a change of {100*severity.mean_change:+.2f} percentage points. '
      f'The frequency of flag-leaf symptoms before soft dough changed from {100*frequency.reference_on_common:.3f}% to {100*frequency.future_on_common:.3f}%. '
      'Occurrence remained close to saturation, whereas symptom timing and canopy damage showed spatial contrasts (Fig. 2).',
      'Environmental-region and country differences',
      'Mid-century SSP5–8.5 HAD changes were '+', '.join(f'{score(data.CANOPY,region=r).mean_change:+.2f} days in the {r} region' for r in data.REGIONS)+
      '. Environmental and country averages retain exact fixed harvested-area weights and metric-specific valid paired support (Fig. 3).',
      'Conditional yield response',
      f'At the reference coefficient 0.018 t ha⁻¹ per GLAI-day, the European canopy response translated into a conditional yield change of {-1000*.018*had.mean_change:+.1f} kg ha⁻¹ per unit reference LAI. '
      f'The climate-model range was {-1000*.018*had.gcm_max:+.1f} to {-1000*.018*had.gcm_min:+.1f} kg ha⁻¹ per reference LAI. '
      'This disease-related yield index depends on a nominal upper-three-leaf canopy and fixed canopy–yield conversion; actual harvested yields and national production losses remain unvalidated.',
      'Pooled field evaluation',
      f'The pooled disease evaluation contains 2,033 assessments from 188 field seasons and 136 informative symptom-onset histories. '
      f'Model onset error is {data.pooled("symptom_onset","model").value:.2f} days versus {data.pooled("symptom_onset","benchmark").value:.2f} days for the phenology benchmark. '
      f'Severity RMSE is {data.pooled("severity","model_rmse").value:.2f} versus {data.pooled("severity","benchmark_rmse").value:.2f} percentage points. '
      'Calibration records remain separate from evaluation, and source measurement conventions remain in the underlying evidence.'
    ]
    captions={
      'fig1_grid_climate_canopy_impacts':'Figure 1 | Mid-century climate effects on simulated Septoria-related canopy damage. '
        'Panels show three-climate-model mean changes in 2031–2060 relative to 1991–2020 under three emissions pathways. '
        'Every 0.25° pixel is an independently simulated wheat land-use cell. The fixed mask retains 14,941 cells, including 14,932 eligible winter-wheat rainfed calendars. '
        'Positive values indicate greater grain-fill HAD deficit. All panels share symmetric discrete color intervals retaining the full range; gray wheat cells have unavailable estimates.',
      'fig2_disease_frequency_timing_severity':'Figure 2 | Mid-century effects on disease occurrence, timing and canopy damage under SSP5–8.5. '
        '(a) Percentage-point change in the frequency of flag-leaf symptoms before soft dough. '
        '(b) First symptoms relative to flowering; negative values indicate earlier symptoms. '
        '(c) Percentage-point change in reference-area-weighted functional-canopy loss during grain filling. '
        '(d) Change in days from the fixed sowing date to first symptoms. Each panel has its own symmetric discrete color scale. '
        'Timing requires detected symptoms in both paired years; the severity proxy differs from observed lesion percentage.',
      'fig3_regional_and_country_impacts':'Figure 3 | Environmental-region and country-level impacts in 2031–2060 relative to 1991–2020. '
        '(a) Harvested-area-weighted HAD-deficit changes across eight EEA regions. '
        '(b) Conditional yield change for the twelve country groups with the largest reference wheat area. '
        'Markers show three-model means and whiskers their range, rather than confidence intervals. '
        'Country assignment follows the dominant SPAM source-country label of each grid cell and is restricted to the study domain. '
        'Conditional yield equals −0.018 times the HAD change, converted to kg ha⁻¹ per unit reference LAI.'}
    rows=[['Environmental region','Δ HAD (days / LAI)','Δ timing (days)','Δ canopy damage (pp)','Conditional Δ yield (kg ha⁻¹ / LAI)']]
    for region in ['Europe']+data.REGIONS:
        rows.append([region,f'{score(data.CANOPY,region).mean_change:+.2f}',f'{score(data.ONSET,region).mean_change:+.2f}',
            f'{100*score(data.SEVERITY,region).mean_change:+.2f}',f'{-1000*score(data.YIELD,region).mean_change:+.1f}'])
    table_caption=('Table 1 | Environmental-region climate impacts under mid-century SSP5–8.5. '
        'Values are harvested-area-weighted three-model means for 2031–2060 relative to 1991–2020. '
        'Canopy damage is the reference-area-weighted functional-loss proxy; pp denotes percentage points. '
        'Conditional yield uses the fixed canopy–yield coefficient and nominal upper-three-leaf reference LAI.')
    pd.DataFrame(rows[1:],columns=rows[0]).to_csv(destination/'Main_Table1.csv',index=False)
    (destination/'Climate_Results_Preview.txt').write_text('\n\n'.join(summary)+'\n\n'+table_caption+'\n\n'+
        '\n'.join('\t'.join(row) for row in rows)+'\n\n'+'\n\n'.join(captions.values())+'\n')
    (folder/'captions.json').write_text(json.dumps(captions,indent=2)+'\n')
    doc=Document();layout.configure_doc(doc)
    doc.add_heading(summary[0],level=0)
    for paragraph in summary[1:]:
        if layout.is_heading(paragraph):doc.add_heading(paragraph,level=1)
        else:doc.add_paragraph(paragraph)
    from analysis.paper_study.nature_food_submission_20261008.build import _table_docx,_pdf_table
    _table_docx(doc,[(rows,table_caption)])
    for stem,caption in captions.items():
        doc.add_page_break();doc.add_picture(str(folder/f'{stem}.png'),width=Inches(6.3));doc.add_paragraph(caption)
    doc.save(destination/'Climate_Results_Preview.docx')
    styles=layout.font_styles();flow=[Paragraph(html.escape(summary[0]),styles['PaperTitle'])]
    for paragraph in summary[1:]:
        flow.append(Paragraph(html.escape(paragraph),styles['PaperHeading' if layout.is_heading(paragraph) else 'PaperBody']))
    flow.append(PageBreak());flow.extend(_pdf_table(rows,table_caption,styles,main=True))
    for stem,caption in captions.items():
        with PILImage.open(folder/f'{stem}.png') as im:w,h=im.size
        scale=min(450/w,520/h)
        flow += [PageBreak(),Image(str(folder/f'{stem}.png'),width=w*scale,height=h*scale),Paragraph(html.escape(caption),styles['PaperCaption'])]
    SimpleDocTemplate(str(destination/'Climate_Results_Preview.pdf'),pagesize=A4,rightMargin=45,leftMargin=45,topMargin=45,bottomMargin=45).build(flow)
    from .build import source_workbook
    original=data.GRID
    try:
        data.GRID=evidence
        source_workbook(destination,canonical=str(destination.relative_to(data.ROOT)))
    finally:data.GRID=original
    (destination/'preview_receipt.json').write_text(json.dumps({'status':'generated','scope':'Completed 1991–2020 and 2031–2060 full-grid comparisons only',
      'late_century_results_included':False,'annual_jobs':540,'pooled_evaluation':True,
      'source_completion_receipt_sha256':data.sha(evidence/'completion_receipt.json')},indent=2)+'\n')
    return destination


if __name__=='__main__':print(build())
