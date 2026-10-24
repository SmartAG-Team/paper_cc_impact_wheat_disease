"""Place manuscript figures immediately after their first-citing paragraph."""
import re

IMAGE=re.compile(r'^!\[[^\]]*\]\(figures/(figs?\d+)\.png\)$')
CITATION=re.compile(r'\bFig(?:ure)?s?\.?\s+((?:S?\d+[a-z]?)(?:\s*(?:,|and|&|–|-|to)\s*S?\d+[a-z]?)*)')


def cited_figures(paragraph):
    """Ordered figure names, with main/SI identities and numeric ranges distinct."""
    names=[]
    for match in CITATION.finditer(paragraph):
        pieces=re.split(r'\s*(,|and|&|–|-|to)\s*',match[1])
        previous=None
        for i in range(0,len(pieces),2):
            token=re.fullmatch(r'(S?)(\d+)[a-z]?',pieces[i].strip())
            if not token:continue
            prefix,number=token[1],int(token[2]);operator=pieces[i-1] if i else None
            if previous and operator in ['–','-','to']:
                prefix=prefix or previous[0]
                if prefix!=previous[0] or number<previous[1]:raise ValueError('Invalid figure range')
                selected=range(previous[1]+1,number+1)
            else:selected=[number]
            for n in selected:
                name=('figs' if prefix else 'fig')+str(n)
                if name not in names:names.append(name)
            previous=(prefix,number)
    return names


def place_figures(source,figure_names):
    """Canonical Markdown preserves captions/references and never appends a figure."""
    figure_names=list(figure_names)
    if len(set(figure_names))!=len(figure_names):raise ValueError('Duplicate declared figure')
    if not figure_names:return source
    placed=set();output=[];excluded=False
    for paragraph in re.split(r'\n\s*\n',source.strip()):
        paragraph=paragraph.strip()
        if IMAGE.fullmatch(paragraph):continue
        if paragraph.startswith('## '):
            excluded=paragraph in ['## Figure captions','## References','## Contents']
        output.append(paragraph)
        if excluded or paragraph.startswith('#'):continue
        for name in cited_figures(paragraph):
            if name in figure_names and name not in placed:
                output.append(f'![Figure {name.removeprefix("fig").upper()}](figures/{name}.png)')
                placed.add(name)
    missing=set(figure_names)-placed
    if missing:raise ValueError(f'Figures lack a body-text citation: {sorted(missing)}')
    order=[IMAGE.fullmatch(p)[1] for p in output if IMAGE.fullmatch(p)]
    if order!=figure_names:raise ValueError(f'Figure first-citation order differs from numbering: {order}')
    return '\n\n'.join(output)+'\n'
