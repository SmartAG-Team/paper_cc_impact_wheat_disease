"""Shared widths and alignment for readable12pt supplementary tables."""
import re


def table_layout(headers):
    if headers[0]=='Component':return [3.1,6.4,6.5],set()
    if headers[0]=='Endpoint':return [2.6,3.3,3.7,2.1,2.1,2.2],{3,4,5}
    if headers[0]=='Candidate':return [4.4,2.3,2.6,3.35,3.35],{2,3,4}
    if headers==['Parameter','Value','Unit','Interpretation']:return [4.6,1.8,2.7,6.9],{1}
    if headers[0]=='Stage code':return [2.3,2.8,10.9],{0,1}
    return [16/len(headers)]*len(headers),set()


def numeric_cell(text):
    return bool(re.fullmatch(r'[+−-]?\d+(?:\.\d+)?',text.strip()))


def style_word_table(table,widths,numeric_columns):
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn
    from docx.shared import Cm,Pt,RGBColor
    table.autofit=False
    for column,width in zip(table.columns,widths):column.width=Cm(width)
    pr=table._tbl.tblPr
    borders=OxmlElement('w:tblBorders')
    for name in ['top','left','bottom','right','insideH','insideV']:
        node=OxmlElement('w:'+name);node.set(qn('w:val'),'single' if name in ['top','bottom'] else 'nil')
        node.set(qn('w:sz'),'6');node.set(qn('w:color'),'70818C');borders.append(node)
    pr.append(borders)
    for i,row in enumerate(table.rows):
        row._tr.get_or_add_trPr().append(OxmlElement('w:cantSplit'))
        for j,(cell,width) in enumerate(zip(row.cells,widths)):
            cell.width=Cm(width);tc=cell._tc.get_or_add_tcPr()
            margins=OxmlElement('w:tcMar')
            for side,value in [('top','60'),('bottom','60'),('left','75'),('right','75')]:
                node=OxmlElement('w:'+side);node.set(qn('w:w'),value);node.set(qn('w:type'),'dxa');margins.append(node)
            tc.append(margins)
            if i==0 or i%2==0:
                fill=OxmlElement('w:shd');fill.set(qn('w:fill'),'EAF1F5' if i==0 else 'F7F9FA');tc.append(fill)
            if i==0:
                edge=OxmlElement('w:tcBorders');line=OxmlElement('w:bottom');line.set(qn('w:val'),'single');line.set(qn('w:sz'),'6');line.set(qn('w:color'),'70818C');edge.append(line);tc.append(edge)
            for p in cell.paragraphs:
                p.paragraph_format.line_spacing=1.1;p.paragraph_format.space_after=Pt(0)
                p.alignment=2 if j in numeric_columns else 0
                for r in p.runs:
                    r.font.size=Pt(12)
                    if i==0:r.bold=True;r.font.color.rgb=RGBColor.from_string('254F66')
