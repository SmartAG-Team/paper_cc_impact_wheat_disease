"""CSL reference records and Zotero-importable BibTeX from retained source metadata."""
from pathlib import Path
import hashlib, html, json, re, subprocess
from html.parser import HTMLParser

ROOT=Path(__file__).resolve().parents[3]
DEST=ROOT/'publication/european_wheat_stb'
HERE=Path(__file__).resolve().parent
PATTERN=r'\[([1-9]\d*(?:(?:,\s*|[–-])[1-9]\d*)*)\]'


def plain(value):
    return html.unescape(re.sub('<[^>]+>','',value)).strip()


def numbers(token):
    for part in token[1:-1].split(','):
        ends=re.split('[–-]',part.strip())
        yield from range(int(ends[0]),int(ends[-1])+1)


def records():
    retained={}
    for name in ['primary_reference_metadata.json','nature_reference_metadata.json','crop_disease_reference_metadata.json']:
        metadata=json.loads((HERE/name).read_text())
        if isinstance(metadata,dict):metadata=[dict(value,DOI=key) for key,value in metadata.items()]
        retained.update({m['DOI'].lower():m for m in metadata})
    dois={1:'10.1038/s41559-018-0793-y',2:'10.1038/s41558-021-01104-8',
        3:'10.1111/j.1365-3059.2011.02455.x',4:'10.1111/j.1365-3059.1990.tb02501.x',
        5:'10.1111/j.0032-0862.2004.00983.x',6:'10.1094/PHYTO.2003.93.10.1329',
        7:'10.1016/j.agrformet.2012.04.019',16:'10.1038/s41558-023-01902-2',
        17:'10.1038/s41467-022-34411-5',18:'10.1038/nplants.2017.102'}
    items={}
    for number,doi in dois.items():
        m=retained[doi.lower()];authors=[]
        for author in m['author']:
            family=author['family'];family=family.title() if family.isupper() else family
            if family.lower()=='van den bosch':family='van den Bosch'
            authors.append({'family':family,'given':author.get('given','')})
        title=m['title'][0] if isinstance(m['title'],list) else m['title']
        container=m.get('container-title','');container=container[0] if isinstance(container,list) else container
        if number==6:title='Modeling of relationships between weather and Septoria tritici epidemics on winter wheat: A critical approach'
        item=dict(id=f'ref-{number:02d}',type='article-journal',author=authors,
            title=html.unescape(re.sub(r'\s+',' ',title)),DOI=doi,URL='https://doi.org/'+doi,
            issued=m.get('published') or m['issued'],**{'container-title':html.unescape(container).replace('®','')})
        for field in ['volume','issue','page']:
            if m.get(field):item[field]=str(m[field])
        if number==17 and not item.get('page'):item['page']='7079'
        if number==18 and not item.get('page'):item['page']='17102'
        items[number]=item
    def dataset(number,title,author,year,publisher,doi=None,version=None,url=None,kind='dataset'):
        item=dict(id=f'ref-{number:02d}',type=kind,title=title,author=author,publisher=publisher)
        if year:item['issued']={'date-parts':[[year]]}
        if doi:item.update(DOI=doi,URL='https://doi.org/'+doi)
        elif url:item['URL']=url
        if version:item['version']=version
        items[number]=item
    dataset(8,'Data from 56 fungicide trials in wheat fields across Europe 2017–2019',
        [{'family':'Holst','given':'N.'}],2022,'Zenodo','10.5281/zenodo.6521175','1.0')
    dataset(9,'Data from 168 fungicide trials in wheat fields across Europe 2014–2018',
        [{'family':'Holst','given':'N.'},{'family':'Donner','given':'M.'}],2022,'Zenodo','10.5281/zenodo.6352615','1.0')
    dataset(10,'ERA5 hourly data on single levels',[{'literal':'Copernicus Climate Change Service'}],
        None,'Copernicus Climate Data Store','10.24381/cds.adbb2d47')
    dataset(11,'Historical Weather API documentation',[{'literal':'Open-Meteo'}],None,'Open-Meteo',
        url='https://open-meteo.com/en/docs/historical-weather-api',kind='webpage')
    items[11]['accessed']={'date-parts':[[2026,10,5]]}
    dataset(12,'GGCMI Phase 3 crop calendar',[{'family':f,'given':g} for f,g in
        [('Jägermeyr','J.'),('Müller','C.'),('Minoli','S.'),('Ray','D.'),('Siebert','S.')]],
        2021,'Zenodo','10.5281/zenodo.5062513','1.01')
    dataset(13,'Global Spatially-Disaggregated Crop Production Statistics Data for 2020, SPAM2020 v2r2',
        [{'literal':'International Food Policy Research Institute'}],2026,'Harvard Dataverse','10.7910/DVN/SWPENT','6')
    dataset(14,'NEX-GDDP-CMIP6 version 2 technical note',[{'literal':'NASA Center for Climate Simulation'}],
        2025,'NASA',url='https://www.nccs.nasa.gov/wp-content/uploads/2025/06/NEX-GDDP-CMIP6-v2-Tech_Note.pdf',kind='report')
    dataset(15,'Irrigated and rainfed crop areas and cropping calendars',[{'literal':'MIRCA-OS'}],
        2026,'HydroShare',version='2',url='https://www.hydroshare.org/resource/e4582ca0042148338bb5e0148b749ed6/')
    assert set(items)==set(range(1,19))
    return items


def bibtex(items):
    output=[]
    def safe(value):return plain(str(value)).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_')
    for number,item in sorted(items.items()):
        kind='article' if item['type']=='article-journal' else 'techreport' if item['type']=='report' else 'misc'
        author=' and '.join('{'+a['literal']+'}' if 'literal' in a else a['family']+', '+a.get('given','') for a in item['author'])
        fields={'author':author,'title':item['title']}
        if item.get('issued'):fields['year']=item['issued']['date-parts'][0][0]
        for field,bibfield in [('container-title','journal'),('volume','volume'),('issue','number'),('page','pages'),('DOI','doi'),('URL','url')]:
            if item.get(field):fields[bibfield]=item[field]
        if item.get('publisher'):fields['institution' if kind=='techreport' else 'publisher']=item['publisher']
        if item.get('version'):fields['note']='Version '+item['version']
        if item.get('accessed'):fields['urldate']='2026-10-05'
        output.append('@'+kind+'{'+item['id'].replace('-','_')+',\n'+''.join('  '+key+' = {'+safe(value)+'},\n' for key,value in fields.items())+'}\n')
    return '\n'.join(output)


def main():
    items=records();(DEST/'references.csl.json').write_text(json.dumps(list(items.values()),ensure_ascii=False,indent=2)+'\n')
    (DEST/'references.bib').write_text(bibtex(items))
    clusters={}
    sources=['abstract.txt','introduction.txt','field_results.txt','baseline_results.txt','climate_results.txt',
        'discussion_field_and_limits.txt','climate_discussion.txt','methods.txt','availability.txt','supplementary_methods.txt']
    for name in sources:
        for match in re.finditer(PATTERN,(DEST/name).read_text()):
            clusters[match[0]]=[items[n]['id'] for n in numbers(match[0])]
    source=DEST/'citation_input.json';source.write_text(json.dumps(dict(items=list(items.values()),clusters=clusters),ensure_ascii=False,indent=2)+'\n')
    result=DEST/'citation_rendered.json'
    subprocess.run(['node',str(HERE/'render_csl.js'),str(source),str(result)],check=True)
    rendered=json.loads(result.read_text())
    references=[plain(entry) for entry in rendered['bibliography_html']]
    (DEST/'references_author_date.txt').write_text('References\n\n'+'\n\n'.join(references)+'\n')
    assert len(references)==18 and len(set(rendered['bibliography_ids']))==18
    print(json.dumps(dict(style=rendered['style_title'],reference_count=18,citation_clusters=len(clusters),
        examples={key:plain(value) for key,value in list(rendered['citations'].items())[:4]},
        zotero_word_plugin_fields=False,zotero_library_modified=False),ensure_ascii=False))


if __name__=='__main__':main()
