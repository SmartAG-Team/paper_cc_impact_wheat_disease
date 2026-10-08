"""Primary source evidence and APA metadata for computational STB forecasting."""
from pathlib import Path
import json,re,html,subprocess,importlib.util,hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
source_names={'SuffertSache2011':'SuffertSache2011','LovellCanopy2004':'LovellCanopy2004','Lovell2004':'LovellLatent2004','Shaw1990':'Shaw1990','Jamieson1995':'Jamieson1995','Abichou2018':'Abichou2018','ShawRoyle1989':'ShawRoyle1989','ShawRoyle1993':'ShawRoyle1993','Baccar2011':'Baccar2011'}
items=[]
for key,stem in source_names.items():
 raw=json.loads(HERE.joinpath(stem+'_crossref.json').read_text())
 authors=[]
 for a in raw['author']:
  family=a['family'];family=family.title() if family.isupper() else family
  if family.lower()=='van den bosch':family='van den Bosch'
  authors.append({'family':family,'given':a.get('given','').replace('\xa0',' ')})
 title=re.sub(r'\s+',' ',raw['title'][0]).strip().replace('( <i>','(<i>').replace('</i> )','</i>)')
 record={'id':key,'type':'article-journal','author':authors,'title':title,'container-title':raw['container-title'][0].replace('®',''),'issued':raw.get('published-print') or raw['issued'],'DOI':raw['DOI'],'URL':'https://doi.org/'+raw['DOI']}
 for field in ['volume','issue','page']:
  if field in raw:record[field]=raw[field]
 items.append(record)
(HERE/'references.csl.json').write_text(json.dumps(items,ensure_ascii=False,indent=2)+'\n')
spec=importlib.util.spec_from_file_location('legacy',ROOT/'analysis/paper_study/manuscript/author_date_citations.py');legacy=importlib.util.module_from_spec(spec);spec.loader.exec_module(legacy)
(HERE/'references.bib').write_text(legacy.bibtex(dict(enumerate(items,1))))
source=HERE/'citation_input.json';result=HERE/'citation_rendered.json'
source.write_text(json.dumps({'items':items,'clusters':{x['id']:[x['id']] for x in items}},ensure_ascii=False,indent=2)+'\n')
subprocess.run(['node',str(ROOT/'analysis/paper_study/manuscript/render_csl.js'),str(source),str(result)],check=True)
r=json.loads(result.read_text());labels={k:legacy.plain(v) for k,v in r['citations'].items()}
(HERE/'citation_keys_labels.json').write_text(json.dumps(labels,ensure_ascii=False,indent=2)+'\n')
(HERE/'references_author_date.txt').write_text('References\n\n'+'\n\n'.join(legacy.plain(x) for x in r['bibliography_html'])+'\n')
(HERE/'bibliography.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><title>References</title></head><body>'+''.join(r['bibliography_html'])+'</body></html>\n')

claims={
 'SuffertSache2011':{
  'source_type':'Primary three-year field experiment; author-deposited original journal PDF.',
  'primary_pdf_url':'https://bioger.versailles-saclay.hub.inrae.fr/media/files/pages-perso/page-perso-suffert-frederic/pdf-publis/suffert-et-sache-2011-plant-pathol',
  'verified_locations':['Printed p.878 abstract','Printed pp.887–888 Discussion','Printed p.886 Fig.6 caption'],
  'direct_findings':['Infected wheat debris at the soil surface affected early disease; effect was transient and varied among years.','First lesions occurred in December after mid-October sowing.','Both sexual and asexual inoculum could initiate infection.','By late autumn the living canopy was the main local source of asexual secondary inoculum.','Winter epidemic progress slowed; remote airborne inoculum could sustain later disease progress.'],
  'supports':['Local residue and living-crop source states; weather-conditioned continuity through autumn/winter.','Local plus external inoculum, with uncertain season-specific relative contribution.'],
  'does_not_establish':['A free-living soil pathogen stock or calibrated soil winter-survival rate.','A universal hard spring activation temperature or a fixed local/external mixture.'],
  'scope_limit':'North-west European winter wheat; volunteer/grass/seed roles are discussed via reviewed prior literature, not all directly quantified by this experiment.'},
 'ShawRoyle1989':{
  'source_type':'Primary three-year field comparison; original publisher abstract.',
  'verified_locations':['Publisher abstract; Plant Pathology38:35–43'],
  'direct_findings':['Primary infection was broadly airborne and distributed across sites; crop/seed history and soil treatment did not explain lesions as strongly as excluding airborne inoculum.','Exposure in autumn and sometimes winter/spring produced infections even away from wheat residue sources.'],
  'supports':['An external airborne background source independent of assumed local residue availability.','Separation of residue on soil surface from an autonomous soil-borne infection mechanism.'],
  'does_not_establish':['Zero local residue contribution in all climates or modern fields.','A numerical external background forcing for the present European grid.']},
 'ShawRoyle1993':{
  'source_type':'Primary observational epidemics in11winter-wheat crops over4years; cv.Longbow.',
  'verified_locations':['Publisher abstract; Plant Pathology42:882–899'],
  'direct_findings':['Upper two leaves suffered infection from lower crop layers near their emergence, followed by within-layer secondary cycles.','Relative timing of inoculum arrival and leaf emergence mattered more than weather infection suitability alone.','Early upper-leaf infection left more time for subsequent cycles before senescence.'],
  'supports':['Leaf-stage-aware exposure gates and explicit linkage from earlier lower-leaf infections to later upper-leaf risk.','Retained infection cohorts and phenology-specific epidemic opportunity.'],
  'does_not_establish':['A stage below which any infection is impossible.','Universal cycle counts or modern-cultivar coefficients.']},
 'LovellCanopy2004':{
  'source_type':'Primary canopy/inoculum-proximity experiment; official publisher-deposited Crossref abstract.',
  'verified_locations':['Crossref abstract deposited by Wiley; Plant Pathology53:11–21'],
  'direct_findings':['Final-three-leaf infection could occur without splashy rainfall.','Proximity of new leaves to older diseased leaves and crop architecture changed upper-leaf risk.'],
  'supports':['Weather plus existing inoculum plus exposed leaf area; rain-only event gates can miss favorable wet-contact pathways.','Cultivar and stem-elongation effects on vertical source-target connectivity.'],
  'does_not_establish':['Any universal maximum transfer distance, rainfall cutoff or leaf-number-to-node mapping.'],
  'APA_note':'Same first author and year as Lovell2004; citeproc expands author groups.'},
 'Lovell2004':{
  'source_type':'Primary outdoor-temperature latency study over2years; official publisher-deposited Crossref abstract.',
  'verified_locations':['Crossref abstract deposited by Wiley; Plant Pathology53:170–181'],
  'direct_findings':['Symptom development followed temperature accumulation and varied among cohorts/cultivars.','Modelled development at cold temperatures supported pathogen progress below the corresponding crop-growth base.'],
  'supports':['Continuous weather-driven latent cohort progression through cold periods, with slowed rates rather than a spring calendar reset.','Different crop and pathogen developmental clocks.'],
  'does_not_establish':['Winter residue-survival kinetics.','Numerical equivalence of latency time, symptom onset, sporulation onset and initial infection suitability.']},
 'Shaw1990':{
  'source_type':'Primary temperature/cultivar latency study with field comparison.',
  'verified_locations':['Publisher-deposited Crossref abstract; Plant Pathology39:255–268'],
  'direct_findings':['Lesions from an infection cohort developed over a distributed time interval.','Latency varied with temperature and cultivar.'],
  'supports':['Distributed latent-stage completion rather than all infected tissue maturing on one calendar day.'],
  'does_not_establish':['One universal fixed-day infection-to-symptom delay.']},
 'Jamieson1995':{
  'source_type':'Primary field-crop leaf-appearance modelling article.',
  'verified_locations':['ScienceDirect publisher abstract; Field Crops Research41:35–44'],
  'direct_findings':['Leaf appearance was predicted using apical-meristem temperature and leaf number.','Near-surface temperature before stem extension and canopy temperature thereafter were more suitable than unqualified air-temperature thermal sums.'],
  'supports':['A leaf-appearance clock distinguished from stage-only anthesis phenology.','Temperature-based phyllochron units need a defined temperature source and local phenological observations.'],
  'does_not_establish':['A phyllochron in TPV units, whose photoperiod/vernalisation multipliers change the coordinate.','Universal total leaf count or node-stage-to-flag-rank equivalence.']},
 'Abichou2018':{
  'source_type':'Primary architecture dataset:55situations,11seasons,13commercial cultivars.',
  'verified_locations':['ScienceDirect abstract and Introduction; Field Crops Research218:213–230'],
  'direct_findings':['Leaf-stage progress was linear or bilinear even within a genotype.','Changes in phyllochron could coincide with flag-leaf initiation.','Final leaf number, tiller emergence and tiller cessation varied with cultivar and environment.'],
  'supports':['Variable or phased leaf-appearance descriptions and uncertainty in effective upper-leaf spacing.','Separate tip emergence, leaf unfolding and flag-relative ordinal identity.'],
  'does_not_establish':['A universal TPV-index spacing or default numerical range in that index.','A fixed whole-season leaf count from BBCH31/32.']},
 'Baccar2011':{
  'source_type':'Primary coupled virtual-plant/epidemic model evaluated against a winter-wheat density experiment.',
  'verified_locations':['Oxford Academic original abstract; Annals of Botany108:1179–1194'],
  'direct_findings':['Vertical disease progression and dynamic canopy architecture were coupled.','Representing plant developmental variability improved correspondence with field epidemics relative to an identical-average-plant canopy.'],
  'supports':['Dynamic source-target leaf topology and uncertainty from among-plant stage variation.'],
  'does_not_establish':['Validation of the present two-dimensional regional reservoir coefficients or exact TPV leaf-rank thresholds.']}
}
for record in items:
 claims[record['id']]['DOI']=record['DOI'];claims[record['id']]['URL']=record['URL'];claims[record['id']]['citation_label']=labels[record['id']]
standard={
 'type':'Ancillary primary coding standard, not journal article and excluded from article CSL/BibTeX.',
 'primary_url':'https://www.openagrar.de/servlets/MCRFileNodeServlet/openagrar_derivate_00010428/BBCH-Skala_en.pdf',
 'verified_location':'Cereals principal growth stage3 table, official OpenAgrar indexed document.',
 'codes':{'31':'First elongated stem node criterion; not31leaves or one total mainstem leaf.','32':'Second elongated stem node criterion; not a fixed upper-leaf count.','33':'Third elongated stem node criterion.','37':'Flag-leaf tip visible while rolled.','39':'Flag leaf fully unfolded, ligule visible.'},
 'interpretation':'F1/F2/F3 are ordinal ranks relative to the eventual flag leaf, not the BBCH node number or known full mainstem leaf number.',
 'retrieval_limit':'Direct download returned server security challenge; official source text accessible in web search index. Exact PDF page not verified.'
}
model={
 'supported_state_structure':['Local infected wheat residue availability','Living canopy infection cohorts and later infectious lower-leaf source','External airborne inoculum background','Leaf-specific exposure/healthy area from emergence through senescence','Weather-conditioned infection pressure','Temperature-driven latent-cohort progression','Later secondary linkage to earlier crop infection states'],
 'assumption_separation':[
  'A nonzero local reservoir is a conditional scenario unless previous wheat/residue/living-host information exists.',
  'Local reservoir persistence and external forcing amplitudes are unidentified by severity snapshots alone.',
  'No universal hard spring temperature switch is justified by these sources.',
  'Secondary contribution must be driven by earlier pathogen states rather than independently reinitialised at every stage.',
  'Leaf37and39can be calibrated as crop-stage events; rank-spacing p in TPV units remains an effective sensitivity if no leaf-emergence records exist.',
  'C39−C37describes within-flag unfolding duration, not directly an interleaf phyllochron.',
  'True leaf appearance requires independent leaf-number/Haun-stage observations or a validated leaf model; no total-leaf-count reconstruction is supported here.',
  'Phenology gates set each not-yet-exposed leaf area to zero; host absence is different from climate unsuitability.',
  'Visible-symptom development must not be silently equated to spore-producing infectiousness or total functional green-area loss.'
 ],
 'parameter_policy':'No calibrated winter survival rate, local/external weight, TPV phyllochron default, fixed full leaf count, or universal rainfall threshold is supplied by this audit.',
 'validation_requirements':['Earlier-season residue/crop-history/source observations for local reservoir','Autumn/winter disease observations for persistence','Leaf appearance and unfolding observations independent of lesion dates','Strict field holdout evaluation of secondary timing','Observation definitions distinguishing incidence, affected area, pycnidia and latent infection']
}
obj={'retrieved_date':'2026-10-06','scope':'Computational crop-protection forecasting; primary journal sources only. No pathogen cultivation, experimental dissemination or pathogen engineering procedures.','sources':claims,'ancillary_BBCH_standard':standard,'model_structure_and_uncertainty':model,'allowed_journals':['Plant Pathology','Field Crops Research','Annals of Botany'],'excluded_publishers':['MDPI','Frontiers'],'metadata_provenance':{key:source_names[key]+'_crossref.json' for key in source_names},'local_source_pdf_sha256':hashlib.sha256((HERE/'SuffertSache2011.pdf').read_bytes()).hexdigest()}
(HERE/'overwinter_leaf_evidence.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
memo='''小麦叶斑病越冬菌源与上部叶层发育

　　Zymoseptoria tritici的本地越冬菌源主要依附于感染的小麦残体和越冬活体植株。土表残体是菌源载体，不能等同于具有独立增殖和侵染参数的土壤病原库。西北欧洲三年田间试验显示，土表小麦残体增加早期病害，但影响随季节推进减弱；晚秋以后，活体冠层成为重要的本地次生菌源，冬季疫情减慢而并未完全中断。英国三年试验同时表明，外源空气传播菌源可在缺少邻近残体的地点形成初侵染。本地残体、越冬植株和区域外源输入共同构成菌源边界条件，贡献比例随地区、年份和耕作历史变化（Suffert & Sache,2011；Shaw & Royle,1989）。

　　上部叶层的病害风险由菌源可获得性、叶片暴露、湿润天气和感染后的发展共同决定。十一块冬小麦田的四年观测显示，上部叶层刚出现时的侵染时间对后续病害尤为关键，早期侵染为次生发展留下更长的叶片有效寿命。冠层研究还发现，末三叶侵染可发生于没有强飞溅降雨的条件，病叶与新叶距离及冠层结构改变感染风险。因此，关键生育期提供叶层暴露与源—靶连接条件，强降雨阈值不能单独代表所有侵染途径（Shaw & Royle,1993；Lovell, Parker, et al.,2004）。

　　越冬作物中的潜伏感染具有温度驱动的持续发展过程。不同感染批次的症状在一段时间内陆续出现，潜育过程受温度和品种影响；病原与作物的发育温度响应并不相同。秋季感染状态通过冬季保留并向后续症状发展，可以连接早期侵染与上部叶层风险。症状出现时间、具传染性的产孢病斑和功能叶面积损失属于不同状态，潜育时钟的参数不能直接替代菌源越冬存活率或侵染启动阈值（Shaw,1990；Lovell, Hunter, et al.,2004）。

　　叶片出现需要区别于茎节生育期。谷物BBCH31、32和33分别依据伸长茎节判定，BBCH37为旗叶叶尖可见，BBCH39为旗叶完全展开且叶舌可见。F1—F3表示相对最终旗叶的叶位，不表示已知的主茎总叶数。叶片出现的phyllochron通常定义在温度坐标中，并受温度来源、叶位及发育阶段影响；田间数据存在单阶段或双阶段叶片出现进程。温度—光周期—春化发育指数包含额外权重，文献中的积温叶间期不能直接当作该指数的固定叶位间距（Jamieson et al.,1995；Abichou et al.,2018）。

　　阶段阈值可约束旗叶出现和完全展开日期。缺少叶数或Haun叶龄观测时，从旗叶阈值向前推算F2、F3所用的间距属于有效发育间距情景；旗叶出现至展开的时间差不是直接观测的相邻叶片出现间隔。本地越冬存活系数、残体与外源输入的比例、真实总叶数和温度—光周期—春化指数中的叶间期均缺乏独立识别依据。动态叶层与病害状态的耦合具有田间模型支持，具体区域模型的参数仍需独立观测和留出样本约束（Baccar et al.,2011）。

'''
# Formal prose uses standard citation spacing.
memo=re.sub(r',(\d{4})',r', \1',memo)
(HERE/'overwinter_leaf_evidence_zh.txt').write_text(memo)
assert len(items)==9 and len(r['bibliography_ids'])==9
assert labels['Lovell2004']!='(Lovell et al., 2004)'
assert labels['LovellCanopy2004']!='(Lovell et al., 2004)'
print(json.dumps({'references':9,'Lovell_latent_label':labels['Lovell2004'],'Lovell_canopy_label':labels['LovellCanopy2004'],'evidence_file':'overwinter_leaf_evidence.json'},ensure_ascii=False))
