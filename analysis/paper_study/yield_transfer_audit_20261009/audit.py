"""Reproduce the bounded evidence audit from public_source_subset.zip only.

No network access, model fitting, manuscript changes, or writes outside this directory.
"""
from pathlib import Path
import hashlib
import io
import json
import platform
import zipfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def harvest_units(frame):
    """Repeated ratings retain one yield; numeric coincidence is not identity."""
    if frame.yield_unit_id.isna().any():
        raise ValueError('Missing harvest identifier')
    if frame.groupby('yield_unit_id').yield_t_ha.nunique(dropna=False).gt(1).any():
        raise ValueError('Conflicting yields for one harvest identifier')
    return frame.drop_duplicates('yield_unit_id').copy()


def raw_plot_yields(frame):
    keys = ['row','column']
    if frame.duplicated(keys+['sub_column']).any():
        raise ValueError('Duplicate French subrow key')
    result = frame.groupby(keys,as_index=False).agg(
        subrows=('GY','size'), nonmissing=('GY','count'), raw_mean_g_m2=('GY','mean'))
    complete = result.subrows.eq(4) & result.nonmissing.eq(4)
    result['yield_raw_t_ha'] = (result.raw_mean_g_m2 * .01).where(complete)
    return result


def prediction_gate(frame):
    """Structural sufficiency for supplementary grouped association testing only."""
    protocol = json.loads((HERE/'protocol.json').read_text())['gate']
    sites = frame.postcode.nunique() if 'postcode' in frame else 0
    counts = frame.groupby('site_year').size() if 'site_year' in frame else pd.Series(dtype=int)
    conditions = dict(enough_sites=sites >= protocol['minimum_sites'],
        enough_site_years=len(counts) >= protocol['minimum_site_years'],
        no_sparse_site_years=len(counts)>0 and counts.min() >= protocol['minimum_plots_per_site_year'],
        enough_training_sites=sites-1 >= protocol['minimum_train_sites'])
    return dict(eligible=bool(all(conditions.values())), sites=int(sites),
                site_years=int(len(counts)), conditions={k:bool(v) for k,v in conditions.items()})


class Bundle:
    def __init__(self):
        self.path = HERE/'public_source_subset.zip'
        with zipfile.ZipFile(self.path) as z:
            if z.testzip() is not None:
                raise ValueError('Source ZIP CRC failure')
            self.members = {n:z.read(n) for n in z.namelist()}
        expected = json.loads(self.members['CHECKSUMS.json'])
        if set(expected) != set(self.members)-{'CHECKSUMS.json'}:
            raise ValueError('Unindexed bundle members')
        for name, value in expected.items():
            if sha256(self.members[name]) != value:
                raise ValueError('Source checksum mismatch: '+name)

    def csv(self,name):
        return pd.read_csv(io.BytesIO(self.members[name]))

    def json(self,name):
        return json.loads(self.members[name])


def main():
    b = Bundle()
    checks, duplicates, profiles, numeric_profiles, outputs = [], [], [], [], []

    def check(name, passed, detail):
        checks.append(dict(check=name, passed=bool(passed), evidence=detail))

    def save_csv(name, rows):
        frame = rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
        frame.to_csv(HERE/name,index=False)
        outputs.append(name)

    inventory = b.csv('context/source_inventory.csv')
    save_csv('source_files.csv',inventory)
    member_index = [dict(member=n, bytes=len(data), sha256=sha256(data)) for n,data in sorted(b.members.items())]
    save_csv('bundle_member_hashes.csv',member_index)
    for _, row in inventory.iterrows():
        same = sha256(b.members[row.bundle_member]) == row.sha256
        check('source_bytes:'+row.original_path,same,'Packaged member SHA-256 equals retained source/copy SHA-256')
    aliases = inventory.groupby('sha256').filter(lambda d:len(d)>1)
    save_csv('identical_file_copies.csv',aliases)
    for name in ['zenodo_17432866.json','zenodo_10209176.json','zenodo_5393959.json']:
        meta = b.json('sources/'+name)
        check('licence:'+name,meta['metadata']['license']['id']=='cc-by-4.0','Retained original repository metadata')
    journal_records = json.loads((HERE/'journal_provenance.json').read_text())
    journal_map = {r['doi']:r for r in journal_records}
    citations = []
    for dataset,record,role,article in [
        ('swiss','17432866','measurement release','10.1002/csc2.21151'),
        ('swiss','10209176','identifier recovery; same observations','10.1002/csc2.21151'),
        ('france','5393959','phenotype source and original analysis code','10.1111/nph.17915')]:
        meta = b.json('sources/zenodo_'+record+'.json')
        journal = journal_map[article]
        citations.append(dict(dataset=dataset,role=role,data_doi=meta['doi'],data_url=meta['doi_url'],
            dataset_title=meta['metadata']['title'],creators='; '.join(c['name'] for c in meta['metadata']['creators']),
            metadata_publication_date=meta['metadata']['publication_date'],record_created=meta['created'],
            licence=meta['metadata']['license']['id'],associated_article_doi=article,
            associated_journal='; '.join(journal.get('journal',[])),
            journal_policy=('Associated Crop Science paper retained as provenance only; top-disciplinary eligibility not established; cite dataset records'
                            if dataset=='swiss' else 'New Phytologist: specialist leading plant-science journal; primary dataset citation preferred for these observations'),
            provenance='Retained Zenodo metadata; journal identity from DOI-registry metadata; no excluded-journal evidence added'))
    save_csv('dataset_citations.csv',citations)
    for meta_name, member in [('zenodo_17432866.json','swiss-mixtures/blendit_full_pub.csv'),
                              ('zenodo_17432866.json','swiss-mixtures/README.txt'),
                              ('zenodo_10209176.json','swiss-mixtures-v1/blendit_full.csv')]:
        files = b.json('sources/'+meta_name)['files']
        source = next(f for f in files if f['key']==Path(member).name)
        check('repository_md5:'+member,hashlib.md5(b.members['sources/'+member]).hexdigest()==source['checksum'][4:],
              'Local bytes checked against retained repository MD5; no remote refresh')
    for item in b.json('sources/acquisition.json'):
        member = 'sources/'+item['file']
        if member in b.members:
            check('acquisition_sha256:'+item['file'],sha256(b.members[member])==item['sha256'],item['url'])

    for name in sorted(b.members):
        if name.endswith('.csv') and name.startswith(('sources/','analysis_ready/')):
            d = b.csv(name)
            profiles.append(dict(member=name, rows=len(d),columns=len(d.columns),exact_duplicate_excess=int(d.duplicated().sum())))
            for col in d:
                values = pd.to_numeric(d[col],errors='coerce')
                numeric = int(values.notna().sum())
                numeric_profiles.append(dict(member=name,column=col,dtype=str(d[col].dtype),
                    nonmissing=int(d[col].notna().sum()),missing=int(d[col].isna().sum()),distinct=int(d[col].nunique()),
                    numeric_count=numeric,numeric_min=values.min() if numeric else None,
                    numeric_max=values.max() if numeric else None))
    save_csv('source_profile.csv',profiles)
    save_csv('column_profile.csv',numeric_profiles)

    swiss = b.csv('analysis_ready/swiss_stb_yield_724.csv')
    france = b.csv('analysis_ready/france_stb_yield_224.csv')
    all_france = b.csv('analysis_ready/france_all_226_disease_plots.csv')
    common = b.csv('analysis_ready/additional_ordinal_stb_yield_948.csv')
    v2 = b.csv('sources/swiss-mixtures/blendit_full_pub.csv')
    v1 = b.csv('sources/swiss-mixtures-v1/blendit_full.csv')
    shared = list(v2.columns)
    a = v2.dropna(subset=['septo_2','yield_dtha']).reset_index(names='v2_row')
    c = v1.dropna(subset=['septo_2','yield_dtha']).reset_index(names='v1_row')
    matched = a.merge(c,on=shared,how='left',validate='one_to_one',indicator=True)
    check('swiss_cross_version_identity',len(matched)==len(swiss) and matched._merge.eq('both').all()
          and matched.plot_id.nunique()==len(swiss),'All 19 common source fields; v1 supplies plot identifiers only')
    sm = swiss.merge(matched,on='plot_id',validate='one_to_one',suffixes=('','_original'))
    check('swiss_csv_line_provenance',sm.source_v2_csv_line.eq(sm.v2_row+2).all()
          and sm.source_v1_csv_line.eq(sm.v1_row+2).all(),'CSV line numbers include header')
    for field in shared:
        check('swiss_reconstructed:'+field,
              ((sm[field]==sm[field+'_original']) | (sm[field].isna() & sm[field+'_original'].isna())).all(),
              'Analysis-ready values equal matched original source values')
    check('swiss_units',np.allclose(swiss.yield_t_ha,swiss.yield_dtha*.1),'dt/ha x 0.1 = t/ha; no additional moisture correction')
    check('swiss_score_unchanged',swiss.stb_score_ordinal.eq(swiss.septo_2).all(),'Source ordinal values retained')
    check('swiss_identifiers_unique',not swiss.record_id.duplicated().any(),'One harvest per recovered plot ID')
    check('swiss_environment_consistency',swiss.site_year.eq(swiss.postcode.astype(str)+'_'+swiss.year.astype(str)).all(),
          'Location and harvest year are observed grouping variables')
    duplicates += [dict(dataset='swiss_v2_full',kind='exact_rows',rows=len(v2),units=len(v2.drop_duplicates()),
        excess=len(v2)-len(v2.drop_duplicates()),interpretation='Repeated public-source rows outside paired subset; no independent yields inferred'),
        dict(dataset='swiss_paired',kind='paired_plot_ids',rows=len(swiss),units=swiss.plot_id.nunique(),
             excess=int(swiss.plot_id.duplicated().sum()),interpretation='v1 and v2 are versions of the same observations')]

    raw = b.csv('sources/montazeaud2022/raw_yield_variables.csv')
    rawplots = raw_plot_yields(raw)
    fr = france.merge(rawplots,on=['row','column'],suffixes=('','_recomputed'),validate='one_to_one')
    check('france_raw_yield_reconstruction',np.allclose(fr.yield_raw_t_ha,fr.yield_raw_t_ha_recomputed),
          'Four complete equal-area row representations averaged; g/m2 x 0.01 = t/ha')
    check('france_four_subrows',fr.nonmissing.eq(4).all() and fr.subrows.eq(4).all(),'Subrows are not independent plot harvests')
    scores = b.csv('sources/montazeaud2022/STB_symptoms.csv')
    scoreplot = scores.groupby(['row','column'],as_index=False).severity.mean()
    sf = all_france.merge(scoreplot,on=['row','column'],validate='one_to_one')
    check('france_component_score_mean',np.allclose(sf.stb_score_ordinal,sf.severity),'One plot-level score from component scores; no time series')
    source_scores = b.csv('sources/montazeaud2022/STB_RAW_RYT.csv')
    source_scores['original_score_line'] = source_scores.index+2
    score_keys = all_france.merge(source_scores,on=['row','column'],suffixes=('','_source'),validate='one_to_one')
    check('france_score_source_line',score_keys.stb_source_csv_line.eq(score_keys.original_score_line).all(),
          'Score plot keys and original CSV line pointers')
    adjusted = b.csv('sources/montazeaud2022/GY_RAW_RYT.csv')
    adjusted['original_yield_line'] = adjusted.index+2
    adj = france.merge(adjusted,on=['row','column'],suffixes=('','_source'),validate='one_to_one')
    check('france_adjusted_units',np.allclose(adj.yield_author_spatial_blup_t_ha,adj.RAW_GY_source*.01)
          and adj.yield_source_csv_line.eq(adj.original_yield_line).all(),'Author spatial BLUP retained separately from observed raw harvest')
    code1 = b.members['source_code/Spatial_analyses_yield_variables.R'].decode()
    code2 = b.members['source_code/Allelic_richness_phenotypic_file_prep.R'].decode()
    check('france_blup_provenance','BLUPSPT' in code1 and 'GY_spatial_estimates.csv' in code2,
          'Original archived R scripts identify the adjusted yield route')
    complete_pure = raw[raw.assoc.eq('M')].groupby(['row','column']).filter(lambda d:d.GY.notna().all())
    mirrored = complete_pure.groupby(['row','column']).apply(
        lambda d: d.sort_values('sub_column').GY.iloc[0]==d.sort_values('sub_column').GY.iloc[1]
        and d.sort_values('sub_column').GY.iloc[2]==d.sort_values('sub_column').GY.iloc[3], include_groups=False)
    check('france_pure_stand_repeated_representations',mirrored.all(),
          'Both adjacent row representations agree in every complete pure stand; pooled samples are not four independent observations')
    missing = all_france[all_france.yield_raw_t_ha.isna()]
    check('france_missing_harvest_retained',len(missing)==2 and len(france)==224 and len(all_france)==226,
          'Two scored mixture plots without harvest remain excluded from paired subset')
    duplicates += [dict(dataset='france_raw_yield',kind='four_subrow_representations',rows=len(raw),units=len(rawplots),
        excess=len(raw)-len(rawplots),interpretation='Four row representations per physical plot; not four harvest replicates'),
        dict(dataset='france_disease_components',kind='same_plot_components',rows=len(scores),units=len(scoreplot),
        excess=len(scores)-len(scoreplot),interpretation='Components assessed at one time; not repeated disease dates'),
        dict(dataset='france_paired',kind='plot_keys',rows=len(france),units=france.record_id.nunique(),
        excess=int(france.record_id.duplicated().sum()),interpretation='Raw and BLUP yields are two measures of the same harvest')]

    expected = pd.concat([swiss[['record_id','yield_t_ha','stb_score_ordinal','site_year']],
        france[['record_id','yield_raw_t_ha','stb_score_ordinal','site_year']].rename(columns={'yield_raw_t_ha':'yield_t_ha'})])
    combined = common.merge(expected,on='record_id',validate='one_to_one',suffixes=('','_original'))
    check('common_union_identity',len(combined)==len(common)==len(expected)
          and np.allclose(combined.yield_t_ha,combined.yield_t_ha_original)
          and np.allclose(combined.stb_score_ordinal,combined.stb_score_ordinal_original)
          and combined.site_year.eq(combined.site_year_original).all(),
          '948-row common index is the union of 724 Swiss and 224 French plots, not additional evidence')
    check('no_invented_percent_or_loss',common.severity_percent.isna().all()
          and common.disease_attributable_yield_loss_percent.isna().all(),'No ordinal conversion or healthy-yield counterfactual imputation')
    duplicates.append(dict(dataset='additional_common_index',kind='cohort_union',rows=len(common),units=0,
        excess=len(common),interpretation='All 948 rows already present in the two detailed cohorts; adds zero independent plots'))

    traits = b.csv('sources/montazeaud2022/Traits_monocultures.csv')
    linked = france.merge(traits,on=['row','column'],suffixes=('','_trait'),validate='one_to_one')
    check('french_trait_identity',linked.focal.eq(linked.genotype).all() and linked.assoc.eq('M').all(),
          'Trait and harvest field-grid positions link to the same pure-stand genotype')
    keep = ['record_id','row','column','genotype','LAI','Heading','Maturity','yield_raw_t_ha']
    trait_out = linked[keep].copy()
    trait_out['trait_assessment_date'] = pd.NA
    trait_out['phenology_unit_status'] = 'not defined in retained trait table; no calendar-date conversion'
    trait_out['canopy_scope'] = 'one LAI trait value; functional/green and leaf-rank basis unverified'
    save_csv('french_trait_linkage.csv',trait_out)
    environment = swiss.groupby(['postcode','site_year','year'],as_index=False).agg(
        plots=('record_id','size'),severity_levels=('stb_score_ordinal','nunique'),
        severity_min=('stb_score_ordinal','min'),severity_max=('stb_score_ordinal','max'),
        heading_nonmissing=('day_epi','count'),density_levels=('density','nunique'),
        mixtures=('mono_mix',lambda x:int(x.eq('mix').sum())),pure_stands=('mono_mix',lambda x:int(x.eq('mono').sum())))
    environment['no_within_environment_severity_variation'] = environment.severity_levels.eq(1)
    environment['dated_disease_observations'] = 0
    environment['dated_canopy_observations'] = 0
    save_csv('swiss_environment_profile.csv',environment)

    prior = b.json('context/existing_evidence.json')
    for name, detail in prior['grain'].items():
        duplicates.append(dict(dataset=name,kind='previous_assessment_vs_harvest_grain',
            rows=detail['assessment_rows'],units=detail['harvest_units'],excess=detail['repeated_harvest_representations'],
            interpretation='Frozen prior-source diagnostic; no validation rerun; duplicate aliases/eligibility require prior receipts'))
    save_csv('duplicate_checks.csv',duplicates)

    # Every derived plot links to source bytes and physical source records.
    lineage = []
    for _, r in swiss.iterrows():
        lineage.append(dict(record_id=r.record_id,dataset_id=r.dataset_id,environment=r.site_year,
            disease_member='sources/swiss-mixtures/blendit_full_pub.csv',disease_lines=str(r.source_v2_csv_line),
            yield_member='sources/swiss-mixtures/blendit_full_pub.csv',yield_lines=str(r.source_v2_csv_line),
            identifier_member='sources/swiss-mixtures-v1/blendit_full.csv',identifier_line=r.source_v1_csv_line,
            yield_unit='dt/ha',yield_conversion_to_t_ha=.1))
    raw['csv_line'] = raw.index+2
    for _, r in france.iterrows():
        lines = raw.loc[raw.row.eq(r.row)&raw.column.eq(r.column),'csv_line'].astype(str)
        lineage.append(dict(record_id=r.record_id,dataset_id=r.dataset_id,environment=r.site_year,
            disease_member='sources/montazeaud2022/STB_RAW_RYT.csv',disease_lines=str(r.stb_source_csv_line),
            yield_member='sources/montazeaud2022/raw_yield_variables.csv',yield_lines=';'.join(lines),
            identifier_member='sources/montazeaud2022/STB_RAW_RYT.csv',identifier_line=r.stb_source_csv_line,
            yield_unit='g/m2',yield_conversion_to_t_ha=.01))
    save_csv('plot_provenance.csv',lineage)

    readiness = [
        dict(dataset='swiss',plots=724,site_years=10,sites=5,disease='one undated ordinal score per plot; observed 0,2,3,4,5,6',
             yield_measure='observed plot harvest, dt/ha; package reports 15% moisture',
             canopy='no LAI, green area, leaf-rank function, or radiation observations',
             phenology='724 heading day-of-year values; no anthesis/maturity dates',
             repeated_disease=False,compatible_functional_canopy=False,identified_stb_yield_cause=False,
             permissible_use='retrospective within-source grouped yield prediction; separate supplementary comparison',
             restriction='two seasons; repeated sites/cultivars; score/date metadata conflicts; multiple diseases; no protected control'),
        dict(dataset='france',plots=224,site_years=1,sites=1,disease='one 0-3 ordinal component-average score, 2018-03-23, approximately GS30',
             yield_measure='observed complete four-row mean g/m2; dried samples, moisture percentage unspecified; BLUP separate',
             canopy='166 paired pure stands link to one LAI trait value; functional basis, timing and leaf ranks unverified',
             phenology='166 paired Heading and Maturity trait values; units and calendar linkage unresolved',
             repeated_disease=False,compatible_functional_canopy=False,identified_stb_yield_cause=False,
             permissible_use='same-environment trait/disease/harvest association and data-quality checks',
             restriction='fungicides after early assessment; no independent environment test; no matched untreated/protected canopy sequence'),
        dict(dataset='common_index',plots=948,site_years=11,sites=6,disease='two incompatible ordinal scales retained separately',
             yield_measure='duplicate index of the two source cohorts; different moisture bases',canopy='no additional observations',
             phenology='no additional observations',repeated_disease=False,compatible_functional_canopy=False,
             identified_stb_yield_cause=False,permissible_use='discovery/provenance index',restriction='not a third dataset; no pooled severity coefficient')]
    save_csv('source_readiness.csv',readiness)
    gate_rows = [
        ('disease',True,True,'Observed source-specific disease ratings and existing external disease/response benchmarks',
         'Dated repeated quantitative leaf-area severity, leaf ranks and observation functions for canopy-linked validation',
         'Ordinal Swiss/French scales are incompatible with percent leaf area; no infection timing inferred'),
        ('HAD',False,False,'Conditional proxy diagnostics and archived physiological summaries only',
         'Repeated dated functional green/reference LAI by leaf rank across anthesis to maturity, measured senescence and uncertainty',
         'A single LAI trait, heading date, ordinal score or modeled HAD is not observed HAD'),
        ('grain_yield',True,True,'948 unique observed plot harvests: 724 Swiss and 224 French; existing benchmarks retained',
         'Matched canopy/phenology/weather/soil/management and independent yield predictions for end-to-end crop-model validation',
         'Observed harvest availability establishes a target; no physiological or climate-transfer validation is established'),
        ('grain_loss',False,False,'Existing treatment-associated yield responses; new sources lack an identified STB yield counterfactual',
         'Comparable randomized/matched reference and exposed plots with measured disease in both, functional canopy, harvest and other stresses',
         'Severity regressions and treatment packages cannot isolate STB-attributable grain loss'),
        ('tonnes',False,False,'Conditional scenarios only; no validated regional grain-loss conversion',
         'Validated absolute yield changes, crop area and production baselines, moisture consistency, spatial representativeness and uncertainty',
         'Multiplying a conditional proxy by regional production does not establish actual tonnes lost'),
        ('adaptation',False,False,'Recorded mixture composition and management; existing treatment-response comparisons',
         'Replicated feasible intervention contrasts with matched physiology, yield, multiple environments and implementation/cost constraints',
         'Density is site-confounded; cultivar composition and fungicide packages do not identify causal or future-climate benefits')]
    gate = [dict(quantity=q,observed_quantity_supported=obs,claim_ready=ready,
                 currently_supportable_scope=scope,missing_observations=missing,claim_boundary=boundary,
                 end_to_end_physiological_validation=False,
                 evidence_files='source_readiness.csv;duplicate_checks.csv;french_trait_linkage.csv;public_source_subset.zip::context/existing_evidence.json')
            for q,obs,ready,scope,missing,boundary in gate_rows]
    save_csv('evidence_gate.csv',gate)
    limitations = [
        ('swiss_score_scale','README specifies 0-9; retained package reports paper 1-9; zeros occur','Keep ordinal categories; obtain author clarification'),
        ('swiss_collection_dates','README date field 2021-2023 conflicts with title and paired data 2019-2020','Use observed harvest years; no invented assessment dates'),
        ('swiss_sowing_density','README gives 350 seeds/m2; paired source has 350 and 400, fixed by site','Retain observed density; no causal density effect'),
        ('swiss_mixture_composition','Paired data include 29 four-cultivar mixture plots; README describes two-cultivar mixtures','Retain var_numb and original cultivar names'),
        ('swiss_harvest_missingness','750 plots have disease scores; 724 have yield; missingness concentrated at Delley','No imputed harvest; observed subset may be selective'),
        ('french_traits','179 single-record monoculture trait profiles; 166 link to paired disease/harvest plots','Obtain trait methods, units and dates before canopy interpretation'),
        ('french_blup','RAW_GY is an author spatial BLUP using the experiment, not independent raw yield','Use reconstructed raw harvest for future association work; no random-split BLUP target'),
        ('future_end_to_end','No new cohort has compatible observed functional canopy sequences or an identified STB-specific yield cause','Matched physiology and harvest required for actual grain-loss/adaptation claims')]
    save_csv('evidence_limitations.csv',[dict(issue=k,evidence=v,consequence=r) for k,v,r in limitations])
    save_csv('audit_checks.csv',checks)
    failed = sum(not c['passed'] for c in checks)
    receipt = dict(checks=len(checks),checks_failed=failed,new_unique_plot_yields=len(common),
        swiss_plot_yields=len(swiss),swiss_sites=swiss.postcode.nunique(),swiss_site_years=swiss.site_year.nunique(),
        swiss_constant_score_site_years=int(environment.no_within_environment_severity_variation.sum()),
        swiss_missing_harvest_among_scored=int(v2.septo_2.notna().sum()-len(swiss)),
        swiss_full_source_exact_duplicate_excess=int(v2.duplicated().sum()),
        swiss_four_cultivar_paired_plots=int(swiss.var_numb.eq(4).sum()),
        french_plot_yields=len(france),french_disease_plots=len(all_france),french_site_years=france.site_year.nunique(),
        french_trait_profiles=len(traits),french_trait_linked_pairs=len(linked),
        french_raw_subrows=len(raw),french_raw_plot_keys=len(rawplots),
        french_complete_raw_harvests=int(rawplots.yield_raw_t_ha.notna().sum()),
        french_complete_pure_stands_with_mirrored_row_values=int(mirrored.sum()),
        additional_compatible_functional_canopy_series=0,additional_identified_stb_yield_counterfactuals=0,
        model_fits_in_readiness_audit=0,supplementary_prediction_receipt='prediction_receipt.json',
        swiss_supplementary_prediction_gate=prediction_gate(swiss),
        france_supplementary_prediction_gate=prediction_gate(france.assign(postcode='Mauguio')),
        existing_benchmarks='Frozen Tunisia, German BRIWECS, Nordic receipts; no rerun',
        existing_assessment_grain=prior['grain'],
        source_bundle_sha256=sha256(b.path.read_bytes()),source_bundle_bytes=b.path.stat().st_size,
        runtime=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        protocol_sha256=sha256((HERE/'protocol.json').read_bytes()),
        code_sha256=sha256(Path(__file__).read_bytes()),
        journal_provenance_sha256=sha256((HERE/'journal_provenance.json').read_bytes()),
        output_hashes={name:sha256((HERE/name).read_bytes()) for name in sorted(outputs)})
    (HERE/'audit_receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:receipt[k] for k in ['checks','checks_failed','new_unique_plot_yields',
         'french_trait_linked_pairs','additional_compatible_functional_canopy_series','model_fits_in_readiness_audit']},indent=2))
    if failed:
        raise SystemExit('Audit failed; see audit_checks.csv')


if __name__ == '__main__':
    main()
