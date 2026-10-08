"""Source-backed CSL/BibTeX records rendered by the existing APA 7 pipeline."""
from pathlib import Path
import json, copy, hashlib, re, html, subprocess, importlib.util

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
ORIGINAL=ROOT/'publication/european_wheat_stb/references.csl.json'
RENDER=ROOT/'analysis/paper_study/manuscript/render_csl.js'
LEGACY_SCRIPT=ROOT/'analysis/paper_study/manuscript/author_date_citations.py'
spec=importlib.util.spec_from_file_location('legacy_author_date',LEGACY_SCRIPT)
legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
original=json.loads(ORIGINAL.read_text())
index={record['id']:record for record in original}
keys={1:'Savary2019',2:'Chaloner2021',3:'SuffertSache2011',4:'Shaw1990',5:'Lovell2004',6:'Pietravalle2003',7:'Gouache2013',8:'Holst2022BASF',9:'HolstDonner2022Corteva',10:'ERA5',11:'OpenMeteo',12:'GGCMIPhase32021',13:'SPAM2020',14:'NEXGDDPCmip6V2',17:'Minoli2022',18:'Wang2017'}
items=[];provenance={}
for number,key in keys.items():
 item=copy.deepcopy(index[f'ref-{number:02d}']);item['id']=key
 items.append(item)
 provenance[key]={'metadata_source':str(ORIGINAL.relative_to(ROOT)),'source_original_id':f'ref-{number:02d}','original_metadata_reused':True,'source_url':item.get('URL'),'scope':'main'}

newkeys=['Parker2004','Bancal2015','Foulkes2006','vanDenBerg2017','Karisto2018']
titles={
 'Foulkes2006':'Major genetic changes in wheat with potential to affect disease tolerance',
 'vanDenBerg2017':'Physiological traits determining yield tolerance of wheat to foliar diseases',
 'Karisto2018':'Ranking quantitative resistance to <i>Septoria tritici</i> blotch in elite wheat cultivars using automated image analysis',
 'Bancal2015':'Identifying traits leading to tolerance of wheat to <i>Septoria tritici</i> blotch',
}
primary_urls={
 'Parker2004':'https://bsppjournals.onlinelibrary.wiley.com/doi/10.1111/j.1365-3059.2004.00951.x',
 'Bancal2015':'https://www.sciencedirect.com/science/article/abs/pii/S0378429015001628',
 'Foulkes2006':'https://apsjournals.apsnet.org/doi/10.1094/PHYTO-96-0680',
 'vanDenBerg2017':'https://apsjournals.apsnet.org/doi/10.1094/PHYTO-07-16-0283-R',
 'Karisto2018':'https://apsjournals.apsnet.org/doi/10.1094/PHYTO-04-17-0163-R',
}
for key in newkeys:
 raw=json.loads((HERE/f'{key}_crossref.json').read_text())
 authors=[{'family':a['family'],'given':a.get('given','')} for a in raw['author']]
 correction=None
 if key=='Bancal2015':
  hal=json.loads((HERE/'Bancal2015_hal.json').read_text())['response']['docs'][0]
  authors=[{'family':f,'given':g} for f,g in zip(hal['authLastName_s'],hal['authFirstName_s'])]
  correction='Crossref-deposited family/given fields are reversed for all four authors. Corrected using primary HAL API record hal-01535231.'
 item={'id':key,'type':'article-journal','author':authors,'title':titles.get(key,raw['title'][0]),'container-title':raw['container-title'][0].replace('®',''),'issued':raw.get('published-print') or raw['issued'],'DOI':raw['DOI'],'URL':'https://doi.org/'+raw['DOI']}
 for field in ['volume','issue','page']:
  if field in raw:item[field]=raw[field]
 items.append(item)
 provenance[key]={'metadata_source':f'{key}_crossref.json','crossref_api_url':'https://api.crossref.org/works/'+raw['DOI'],'primary_publisher_url':primary_urls[key],'retrieved_date':'2026-10-06','scope':'main','title_formatting':'APA sentence case; scientific binomial italics where present; factual wording unchanged.'}
 if correction:provenance[key].update(author_correction=correction,author_correction_source='Bancal2015_hal.json',author_correction_api_url='https://api.archives-ouvertes.fr/search/?q=halId_s:hal-01535231&fl=authLastName_s,authFirstName_s&wt=json')
optional=copy.deepcopy(index['ref-15']);optional['id']='MIRCAOS2026'
provenance['MIRCAOS2026']={'metadata_source':str(ORIGINAL.relative_to(ROOT)),'source_original_id':'ref-15','original_metadata_reused':True,'source_url':optional['URL'],'scope':'optional supplementary calendar allocation only'}
allitems=items+[optional]
for filename,records in [('references.csl.json',items),('references_optional_supplement.csl.json',[optional]),('references_all.csl.json',allitems)]:
 (HERE/filename).write_text(json.dumps(records,ensure_ascii=False,indent=2)+'\n')
for filename,records in [('references.bib',items),('references_all.bib',allitems)]:
 # Reuse original BibTeX exporter, retaining stable keys instead of numeric ids.
 (HERE/filename).write_text(legacy.bibtex({i:record for i,record in enumerate(records,1)}))

results={}
for stem,records in [('citation',items),('citation_all',allitems)]:
 clusters={item['id']:[item['id']] for item in records}
 clusters['Holst_dataset_pair']=['Holst2022BASF','HolstDonner2022Corteva']
 inputfile=HERE/f'{stem}_input.json';resultfile=HERE/f'{stem}_rendered.json'
 inputfile.write_text(json.dumps({'items':records,'clusters':clusters},ensure_ascii=False,indent=2)+'\n')
 subprocess.run(['node',str(RENDER),str(inputfile),str(resultfile)],check=True)
 rendered=json.loads(resultfile.read_text());results[stem]=rendered
 plainentries=[legacy.plain(s) for s in rendered['bibliography_html']]
 txtname='references_author_date.txt' if stem=='citation' else 'references_all_author_date.txt'
 (HERE/txtname).write_text('References\n\n'+'\n\n'.join(plainentries)+'\n')
 htmlname='bibliography.html' if stem=='citation' else 'bibliography_all.html'
 (HERE/htmlname).write_text('<!doctype html>\n<html lang="en"><head><meta charset="utf-8"><title>References</title><style>body{font-family:Georgia,serif;line-height:1.5;max-width:1000px;margin:48px auto;padding:0 24px}.csl-entry{padding-left:2em;text-indent:-2em;margin-bottom:1em}a{color:#164d70}</style></head><body><h1>References</h1>\n'+''.join(rendered['bibliography_html'])+'\n</body></html>\n')
 assert len(rendered['bibliography_ids'])==len(records)

labels={}
for item in allitems:
 key=item['id'];label=legacy.plain(results['citation_all']['citations'][key])
 labels[key]={'parenthetical':label,'author_date':label[1:-1] if label.startswith('(') and label.endswith(')') else label,'year':item.get('issued',{}).get('date-parts',[['n.d.']])[0][0],'source_url':item.get('URL'),'scope':provenance[key]['scope']}
(HERE/'citation_keys_labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n')
manifest={'style_title':results['citation']['style_title'],'style_id':results['citation']['style_id'],'processor_version':results['citation']['processor_version'],'main_reference_count':len(items),'all_reference_count':len(allitems),'citation_keys':list(labels),'metadata_provenance':provenance,'original_csl_sha256':hashlib.sha256(ORIGINAL.read_bytes()).hexdigest(),'unchanged_archived_reference_ids':['ref-15','ref-16'],'journal_eligibility':{'allowed_new_journals':['Plant Pathology','Phytopathology','Field Crops Research'],'Plant Pathology_basis':'Established specialist society flagship: owned and edited by BSPP; explicit crop-loss/epidemiology scope; https://www.bspp.org.uk/publications/plant-pathology/','concern':'Leading specialist eligibility is a disciplinary judgment. BSPP ownership alone is not a numerical ranking proof; no use of journal impact factor as eligibility evidence.','excluded_new_publishers':['MDPI','Frontiers']},'data_interpretation':{'SPAM2020':'Production-background allocation/context; not disease-free attainable yield Y0, not observed untreated/protected matched yields.','MIRCAOS2026':'Optional legacy supplementary irrigation/calendar allocation only.','Holst2022BASF':'BASF 56 trial source archive, 2017–2019.','HolstDonner2022Corteva':'Corteva 168 trial source archive, 2014–2018.'},'APA_Holst_disambiguation':{'combined_parenthetical':legacy.plain(results['citation_all']['citations']['Holst_dataset_pair']),'year_letter_suffix_required':False,'basis':'Author groups differ: Holst alone versus Holst and Donner. APA author-date strings already distinguish them.'}}
(HERE/'reference_provenance.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
# Numeric factual support from verified prior evidence accompanies metadata, without new literature claims.
evidence=ROOT/'analysis/paper_study/infection_priority_20261006/literature/yield_loss_primary_evidence.json'
if evidence.exists():(HERE/'yield_loss_numeric_support.json').write_bytes(evidence.read_bytes())
assert len({x['id'] for x in allitems})==len(allitems)
assert all('mdpi' not in str(x).lower() and 'frontiers' not in str(x).lower() for x in allitems)
assert labels['Bancal2015']['author_date']=='Bancal et al., 2015'
assert labels['Holst2022BASF']['author_date']=='Holst, 2022'
assert labels['HolstDonner2022Corteva']['author_date']=='Holst & Donner, 2022'
assert len(next(x for x in items if x['id']=='Wang2017')['author'])==56
print(json.dumps({'main_reference_count':len(items),'optional_reference_count':1,'Holst_pair':manifest['APA_Holst_disambiguation']['combined_parenthetical'],'Bancal':labels['Bancal2015']['parenthetical'],'Wang_full_author_count':56,'processor_version':results['citation']['processor_version']},ensure_ascii=False))
