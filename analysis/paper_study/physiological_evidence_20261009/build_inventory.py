"""Curated source decisions plus reproducible public-repository file listings."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ARCHIVE = '/Users/gangzhao/Documents/workspace/paper_cc_impact_wheat_disease/data/STB_consolidated_collection_20261009.zip'
PREFIX = 'stb_collection_20261009/sources/public_septoria/paper-model-data-20261009/'
rows = []


def add(source_id, title, source_url, **kwargs):
    row = dict(source_id=source_id, title=title, source_url=source_url,
               reviewed_date='2026-10-09', retrieval_state='', exact_retrieval_url='',
               evidence_path='', raw_data_license='Not established; article access is not a raw-data license',
               retrieved_material_license='', unit_of_observation='', reported_environment_coverage='',
               canopy_measurement_units='', disease_measurement_units='', stage_and_date_support='',
               grain_units='', protected_reference='', weather_and_management='',
               raw_plot_keys='', compatible_new_raw_rows=0, compatibility='not_eligible',
               reason='', smallest_missing_input='', cohort_overlap='Not established')
    row.update(kwargs)
    rows.append(row)


add('bancal2007', 'Bancal 2007 canopy layers and reserves', 'https://doi.org/10.1093/aob/mcm163',
    retrieval_state='Existing article and 13-row aggregate read in place; current HAL metadata acquired',
    exact_retrieval_url='https://pmc.ncbi.nlm.nih.gov/articles/PMC2749629/',
    evidence_path=ARCHIVE+'::'+PREFIX+'derived/bancal2007_treatment_aggregates.csv; metadata/hal_leads.json',
    retrieved_material_license='Archived article: all rights reserved; no open raw-data license',
    unit_of_observation='Published treatment-year aggregate; three field blocks reported',
    reported_environment_coverage='Five years at Grignon; 13 treatment combinations',
    canopy_measurement_units='GLAI integral: m2 leaf m-2 ground degC day, base 0; reserves: g DM m-2',
    disease_measurement_units='Mixed leaf rust and STB; published STB integral includes senescence',
    stage_and_date_support='Anthesis-to-maturity integrals; no dated plot trajectories in retained CSV',
    grain_units='No direct plot grain-yield observations in retained aggregate',
    protected_reference='Paired protected/diseased treatment summaries',
    weather_and_management='Documented at article level; raw forcing not deposited in inspected HAL record',
    raw_plot_keys='No plot/date/leaf key',
    reason='Aggregate HAD/reserves and harvest-index summaries cannot recover individual grain observations',
    smallest_missing_input='Original plot/date/leaf green-area and disease files, plot harvests, stage dates, daily weather and design key')

add('bancal2015', 'Bancal 2015 wheat tolerance experiments', 'https://doi.org/10.1016/j.fcr.2015.05.006',
    retrieval_state='Current HAL metadata only; publisher HTTP 403; no linked file in returned HAL record',
    exact_retrieval_url='https://hal.science/hal-01535231v1',
    evidence_path='metadata/hal_leads.json; metadata/rdg_bancal_search.json; metadata/datacite_bancal_related.json',
    unit_of_observation='Reported paired plots; 161 genotype-year-site-management combinations',
    reported_environment_coverage='18 genotypes; 5 years; 6 sites; combinations are not 161 independent environments',
    canopy_measurement_units='LAI m2 m-2; green lamina trajectories reported',
    disease_measurement_units='STB predominant; other diseases require plot-level attribution',
    stage_and_date_support='Reported trajectories; raw dates unavailable', grain_units='g grain DM m-2 in article',
    protected_reference='Reported treated/untreated pairs, 2-3 repetitions',
    weather_and_management='Documented experimental conditions; raw forcing not acquired',
    raw_plot_keys='Not public in inspected records',
    reason='Relevant experimental design, but no downloadable observations or verified joins',
    smallest_missing_input='Original per-leaf dated observations and matched harvest/design/weather tables',
    cohort_overlap='Bancal2022 explicitly reuses this experimental database; do not count twice')

add('bancal2022', 'Bancal 2022 HAD/HAA cultivar characterization', 'https://doi.org/10.1016/j.eja.2021.126421',
    retrieval_state='14-page author manuscript acquired; publisher retrieval blocked; supplements unresolved',
    exact_retrieval_url='https://agroparistech.hal.science/hal-03660399/file/Tol%C3%A9rance-05052022.pdf',
    evidence_path='documentation/bancal2022_author_manuscript.response; metadata/bancal2022_hal.json',
    retrieved_material_license='HAL authorisation v1; https://about.hal.science/hal-authorisation-v1/',
    unit_of_observation='Article analyses of paired trials; no machine-readable raw observations',
    reported_environment_coverage='Same database as Bancal2015',
    canopy_measurement_units='HAD: m2 m-2 degC day after heading; HAA based on intercepted radiation',
    disease_measurement_units='STB dominant mixed field context',
    stage_and_date_support='Heading-to-harvest fitted Gompertz integrals; no raw dated rows supplied',
    grain_units='g grain m-2 in article', protected_reference='Reported paired protected/unprotected sub-trials',
    raw_plot_keys='Absent from acquired manuscript',
    reason='Paper section 2.1 establishes cohort reuse; fitted trajectories are not newly acquired observations',
    smallest_missing_input='Bancal2015 original records; public publisher supplements still require verification',
    cohort_overlap='Confirmed reuse of Bancal2015')

add('bregaglio2021', 'Bregaglio 2021 process-based model comparison', 'https://doi.org/10.1016/j.fcr.2021.108108',
    retrieval_state='Current TUM author-institution record and HAL metadata acquired',
    exact_retrieval_url='https://portal.fis.tum.de/en/publications/comparing-process-based-wheat-growth-models-in-their-simulation-o/',
    evidence_path='documentation/bregaglio2021_TUM.html; metadata/hal_leads.json',
    unit_of_observation='Observed disease-free benchmark plus simulated disease scenarios',
    reported_environment_coverage='Dutch benchmark; scenario runs are not new environments',
    canopy_measurement_units='Observed green LAI for healthy-crop calibration; simulated diseased LAI',
    disease_measurement_units='Imposed disease profiles, not paired field disease measurements',
    stage_and_date_support='Crop development used for calibration', grain_units='Grain dry biomass/yield in model outputs',
    protected_reference='Attainable/diseased paired simulations',
    raw_plot_keys='No joined diseased field-plot harvest table supplied',
    reason='Supports architecture/healthy-growth calibration; cannot independently validate disease-related grain loss',
    smallest_missing_input='Independent observed diseased/protected canopy-harvest pairs; further scenario outputs do not remedy this')

add('foulkes2006', 'Foulkes 2006 canopy tolerance field experiments', 'https://doi.org/10.1094/PHYTO-96-0680',
    retrieval_state='Current Rothamsted landing page acquired; existing eight-row predicted means reused',
    exact_retrieval_url='https://repository.rothamsted.ac.uk/id/eprint/14844/',
    evidence_path='metadata/foulkes_repository.html; existing_rejection_audit.json',
    retrieved_material_license='Repository labels deposited article CC BY 4.0; separate raw-data license not established',
    unit_of_observation='Eight genotype-by-protection mixed-model predicted means in existing CSV',
    reported_environment_coverage='Four field experiments over 1999-2000; STB/stripe-rust strata differ',
    canopy_measurement_units='HAD: GLAI calendar days; upper five leaves',
    disease_measurement_units='Disease strata; raw leaf scores not acquired',
    stage_and_date_support='Post-anthesis to full canopy senescence; no raw trajectories in means',
    grain_units='t ha-1 at 15% moisture in archived table', protected_reference='Reported fungicide contrasts',
    raw_plot_keys='No plot identifiers in predicted means; repository lists article PDF',
    reason='Predicted means erase plot-level temporal joins and independent environmental outcomes',
    smallest_missing_input='Four-experiment raw plot records with separate disease identities, canopy dates and harvest keys')

add('taylor2023', 'Taylor-Cunniffe 2023 underlying yield dataset D', 'https://doi.org/10.1371/journal.pcbi.1010969',
    retrieval_state='Archived article/201-point grid reused; complete current author repository tree checked',
    exact_retrieval_url='https://api.github.com/repos/nt409/quantitative-resistance/git/trees/master?recursive=1',
    evidence_path='metadata/taylor_repository_tree_master.json; documentation/taylor_repository_README.md; existing_rejection_audit.json',
    raw_data_license='Underlying dataset D supplied by request; no public raw-data license established',
    retrieved_material_license='Article CC BY; repository code MIT (not a license for unpublished field data)',
    unit_of_observation='201 fitted severity-yield grid points; underlying field trial units unavailable',
    reported_environment_coverage='Yield fit from Soenderborg 2019, one location-year',
    canopy_measurement_units='No measured GLAI series', disease_measurement_units='Leaf-2 severity percent at GS75',
    stage_and_date_support='One endpoint GS75', grain_units='Fitted yield t ha-1; raw moisture basis not verified',
    protected_reference='Not verified for underlying rows',
    raw_plot_keys='No dated canopy/harvest table in inspected repository tree',
    reason='A response curve and one-environment endpoint data cannot meet the physiology/transfer gate',
    smallest_missing_input='Raw dataset D alone is insufficient; independent environments and repeated actual leaf area also required')

add('collin2018', 'Collin 2018 source limitation and tolerance', 'https://doi.org/10.1016/j.fcr.2017.11.022',
    retrieval_state='Current HAL record acquired; indexed Worktribe file listing; direct Worktribe HTTP 403',
    exact_retrieval_url='https://nottingham-repository.worktribe.com/output/918103',
    evidence_path='metadata/hal_leads.json; web_evidence.json; retrieval_log.jsonl',
    retrieved_material_license='Indexed manuscript license CC BY-NC-ND 4.0; raw-data terms unverified',
    unit_of_observation='Reported field plot source-sink contrasts',
    reported_environment_coverage='One detailed source-sink experiment plus three tolerance experiments reported',
    canopy_measurement_units='Green lamina area, leaf-3 contribution and assimilate-supply traits reported',
    disease_measurement_units='STB field context; raw leaf measurements unavailable',
    stage_and_date_support='Not verified from raw records', grain_units='Not verified from raw records',
    protected_reference='Reported protection, late N and source-sink treatments need separate coding',
    raw_plot_keys='Unavailable', reason='Repository offers a manuscript; raw observations and join structure not verified',
    smallest_missing_input='Plot/time/leaf/harvest tables and explicit treatment and cohort identifiers',
    cohort_overlap='Possible overlap with van_den_bosch2022; identifiers required before pooling')

add('bosch2022_tolerance', 'van den Bosch 2022 tolerance cohort', 'https://doi.org/10.1111/ppa.13509',
    retrieval_state='Existing PDF read in place; explicit reasonable-request data statement',
    exact_retrieval_url='https://repository.rothamsted.ac.uk/id/eprint/31726/1/ppa.13509.pdf',
    evidence_path=ARCHIVE+'::'+PREFIX+'van_den_bosch2022.pdf (PDF pp.4-5,10)',
    unit_of_observation='257 completed line-site-years; tolerance estimates for 179, not 257 independent experiments',
    reported_environment_coverage='Eight site-years', canopy_measurement_units='GLAI m2 m-2; HAD calendar days, upper four leaves',
    disease_measurement_units='STB-targeted contrasts with non-target disease control; exact raw definitions unavailable',
    stage_and_date_support='GS59 to canopy senescence, approximately GS83; does not cover GS31-85',
    grain_units='t ha-1; moisture basis requires original file', protected_reference='Intensive protection vs STB-affected plots',
    raw_plot_keys='No raw public keys in retained PDF', reason='Relevant multi-environment physiology, restricted to author-supplied raw records',
    smallest_missing_input='Dated leaf observations and harvests including all 257 combinations and exclusion flags',
    cohort_overlap='Possible Collin2018 overlap; the 120-site-year HAD-loss archive is a separate component')

add('uk_had120', 'Historical UK HAD-loss archive described in van den Bosch 2022 section 2.3',
    'https://doi.org/10.1111/ppa.13509', retrieval_state='Documented in existing PDF; no public numerical archive verified',
    exact_retrieval_url='https://repository.rothamsted.ac.uk/id/eprint/31726/1/ppa.13509.pdf',
    evidence_path=ARCHIVE+'::'+PREFIX+'van_den_bosch2022.pdf (PDF p.5)',
    unit_of_observation='Plot/assessment; three replicates reported', reported_environment_coverage='120 UK site-years, 1998-2002',
    canopy_measurement_units='GLAI and symptom area per ground area, m2 m-2',
    disease_measurement_units='STB symptom area index excluding natural senescence',
    stage_and_date_support='Approximately 10-day observations GS31 to senescence', grain_units='Harvest linkage not established from this description',
    protected_reference='Treated/untreated plots; observed treated symptoms retained',
    raw_plot_keys='Not acquired', reason='High-priority trajectory lead; same-unit harvest and weather joins remain unverified',
    smallest_missing_input='Historical plot keys and dated canopy/symptom/stage records joined to grain harvest and weather',
    cohort_overlap='Must be distinguished from eight-site-year tolerance cohort and 14-site-year 1993-1995 cohort')

add('uk_canopy14', 'Historical UK canopy data reused in Corkley 2025', 'https://doi.org/10.1111/ppa.14080',
    retrieval_state='Publisher-indexed methods and data statement inspected; stale Rothamsted PDF URL returned 404',
    exact_retrieval_url='https://bsppjournals.onlinelibrary.wiley.com/doi/full/10.1111/ppa.14080',
    evidence_path='web_evidence.json; metadata/datacite_corkley_wheat.json; retrieval_log.jsonl',
    raw_data_license='Dataset 1: no deposited raw license; dataset 2 AHDB license restrictions are separate',
    unit_of_observation='Four cultivars × four replicates per site-year reported',
    reported_environment_coverage='14 UK site-years, 1993-1995',
    canopy_measurement_units='GLAI m2 m-2; top three leaves', disease_measurement_units='Dated STB observations reported',
    stage_and_date_support='Dated observations and thermal time reported', grain_units='Same-unit grain harvest not verified',
    protected_reference='With/without fungicide field treatments', weather_and_management='Daily weather within 1 km reported',
    raw_plot_keys='No public file identified', reason='Relevant canopy archive, but no public raw rows or verified harvest joins',
    smallest_missing_input='Historical canopy dataset with original harvest/design/stage tables; model parameter files are insufficient',
    cohort_overlap='1993-1995 provenance distinguishes it from UK 1998-2002 HAD archive')

add('phenotol2019_2021', 'FSOV PHENOTOL field trials', 'https://www.fsov.org/uploads/2024/5/FSOV-2018-O-PHENOTOL-Article.pdf',
    retrieval_state='Official 10-page project report acquired; no numerical dataset found in scoped repository searches',
    exact_retrieval_url='https://www.fsov.org/uploads/2024/5/FSOV-2018-O-PHENOTOL-Article.pdf',
    evidence_path='documentation/phenotol2024.pdf (pp.2-3); metadata/datacite_phenotol.json; metadata/rdg_phenotol_search.json',
    retrieved_material_license='Public project PDF; reuse license not stated in inspected report',
    unit_of_observation='Field microplots; manual canopy reference sampled on one block in calibration trials',
    reported_environment_coverage='145482 Ouzouer 2019; 146672 Villers-Saint-Christophe 2020; 149157 Aubigny 2021; BH-Pheno-Or Orsonville 2021',
    canopy_measurement_units='Manual GLAI m2 m-2; repeated green fractions; thermal-time IFVI; sensor proxies',
    disease_measurement_units='Separate per-leaf disease/senescence attribution not verified',
    stage_and_date_support='Heading and repeated post-heading measurements; GS31 not established',
    grain_units='q ha-1 in report; raw moisture basis unknown', protected_reference='Fungicide and N/water factorial contrasts',
    weather_and_management='Design descriptions; daily forcing files not acquired',
    raw_plot_keys='Trial IDs known; plot keys unavailable', reason='Strong newer lead, but figures and one-block manual reference do not supply replicated raw validation data',
    smallest_missing_input='Named trial exports including manual GLAI, raw sensor dates, harvests, disease-by-leaf ratings and block keys')

add('stics2025', 'French wheat STICS varietal parameter deposit v3', 'https://doi.org/10.57745/FDPIXY',
    retrieval_state='Repository metadata with all 20 files and four-page README acquired',
    exact_retrieval_url='https://entrepot.recherche.data.gouv.fr/api/datasets/:persistentId/?persistentId=doi:10.57745/FDPIXY',
    evidence_path='metadata/rdg_stics_french_wheat.json; documentation/stics_readme.pdf',
    raw_data_license='License of underlying ARVALIS observations not established',
    retrieved_material_license='Etalab Open License 2.0 for deposited material',
    unit_of_observation='Parameters, RMSE tables, figures and README; no raw plot observation file',
    reported_environment_coverage='Multiple environments for 11 cultivars in underlying calibration data',
    canopy_measurement_units='Underlying LAI described from sowing to anthesis only; no raw observations deposited',
    disease_measurement_units='No disease observation table listed',
    stage_and_date_support='README labels BBCH50 as flag leaf: unresolved metadata inconsistency', grain_units='Underlying biomass and yield t ha-1',
    protected_reference='No paired disease-protection table listed', weather_and_management='Underlying station/SAFRAN and ARVALIS data described, not deposited',
    raw_plot_keys='Not present in listed files', reason='Parameter/validation-summary release cannot replace canopy-disease-harvest field data',
    smallest_missing_input='Underlying ARVALIS observations including post-anthesis canopy and disease/control measurements')

add('eth_segmentation2023', 'ETH WheatSegmentationModels training/validation data', 'https://doi.org/10.3929/ethz-b-000610997',
    retrieval_state='DataCite metadata and complete author code-tree/README acquired; 2.42 GB image payload not downloaded',
    exact_retrieval_url='https://api.datacite.org/dois/10.3929/ethz-b-000610997',
    evidence_path='metadata/eth_segmentation_data.json; metadata/eth_segmentation_tree.json; documentation/eth_segmentation_README.md',
    unit_of_observation='Image/segmentation annotation', reported_environment_coverage='Same-unit multi-environment harvest cohort not established',
    canopy_measurement_units='Image segmentation is not verified ground-area-normalized leaf-layer LAI',
    disease_measurement_units='No joint dated leaf-disease-harvest table identified',
    grain_units='No grain table in inspected author tree or metadata', raw_plot_keys='Harvest link not established',
    reason='Useful measurement-method asset; compatibility with grain-loss validation unproven',
    smallest_missing_input='Plot keys, ground-area calibration and contemporaneous disease/stage/harvest/reference tables')

add('existing_tunisia2020', 'Already archived Tunisian durum mixtures', 'https://doi.org/10.5061/dryad.r2280gbb5',
    retrieval_state='Prior inventory consulted; no re-download or re-analysis',
    evidence_path=ARCHIVE+'::'+PREFIX+'DATA_README.txt',
    raw_data_license='Existing source terms retained; not re-verified in this search',
    unit_of_observation='82 damaged plot-season harvests, 122 total yields, 40 protection contrasts',
    reported_environment_coverage='Two seasons', canopy_measurement_units='No observed GLAI trajectory',
    disease_measurement_units='Lesion percent and incidence by leaf rank', stage_and_date_support='Two ranks assessed on different dates; not same-leaf time series',
    grain_units='kg ha-1; moisture basis unresolved', protected_reference='40 management-package contrasts; protected severity absent',
    raw_plot_keys='Existing verified year/treatment/replicate joins', reason='Real yield records lack repeated actual canopy area and measured reference disease',
    smallest_missing_input='Actual repeated leaf-layer area and protected/unprotected dated disease/stage measurements',
    cohort_overlap='Already used; not new acquisition')

add('existing_swiss', 'Already archived Swiss Blend-it plots', 'https://doi.org/10.5281/zenodo.17432866',
    retrieval_state='Follow-up inventory and 948-row combined file read in place',
    evidence_path='existing_rejection_audit.json; original follow-up START_HERE.txt',
    raw_data_license='CC BY 4.0 as recorded in supplied follow-up inventory',
    unit_of_observation='724 plot harvests', reported_environment_coverage='Five sites; ten site-years, 2019-2020',
    canopy_measurement_units='No functional canopy trajectory', disease_measurement_units='Ordinal final disease scores, not percent leaf area',
    stage_and_date_support='No assessment dates', grain_units='dt ha-1; 15% moisture', protected_reference='No protected yield control',
    raw_plot_keys='Previously recovered plot IDs; versions overlap', reason='Multiple environments but missing physiological trajectory and reference outcome',
    smallest_missing_input='Contemporaneous dated canopy/disease records and protected comparators', cohort_overlap='All 724 already analyzed')

add('existing_france', 'Already archived French durum mixtures', 'https://doi.org/10.5281/zenodo.5393959',
    retrieval_state='Follow-up inventory read in place', evidence_path='existing_rejection_audit.json; original follow-up START_HERE.txt',
    raw_data_license='CC BY 4.0 as recorded in supplied follow-up inventory',
    unit_of_observation='224 plot harvests; 226 scored plots', reported_environment_coverage='Mauguio 2018 only',
    canopy_measurement_units='No functional canopy trajectory', disease_measurement_units='0-3 ordinal ratings near GS30',
    stage_and_date_support='Single 2018-03-23 rating followed by fungicide', grain_units='g m-2; dried grain, moisture percent unknown',
    protected_reference='No season-long protected/untreated contrast', raw_plot_keys='Existing verified plot joins',
    reason='One environment and early endpoint; no repeated functional canopy or disease counterfactual',
    smallest_missing_input='Additional independent environments with repeated physiological observations', cohort_overlap='All 224 already analyzed')


def main():
    with (ROOT / 'source_inventory.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    listings = []
    d = json.loads((ROOT/'metadata/rdg_stics_french_wheat.json').read_text())
    for f in d['data']['latestVersion']['files']:
        info = f['dataFile']
        listings.append(dict(source_id='stics2025', name=f['label'], bytes=info.get('filesize'),
                             exact_url=f"https://entrepot.recherche.data.gouv.fr/api/access/datafile/{info['id']}",
                             role='documentation' if f['label']=='readme.pdf' else 'parameter_figure_or_error_summary',
                             acquired=f['label']=='readme.pdf'))
    for sid, filename in [('taylor2023','taylor_repository_tree_master.json'),('eth_segmentation2023','eth_segmentation_tree.json')]:
        tree = json.loads((ROOT/'metadata'/filename).read_text())
        for item in tree['tree']:
            if item['type'] != 'blob':
                continue
            listings.append(dict(source_id=sid, name=item['path'], bytes=item.get('size'),
                                 exact_url=item['url'], role='author_repository_file_not_assumed_observation',
                                 acquired=item['path']=='README.md'))
    with (ROOT/'repository_file_inventory.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(listings[0]));writer.writeheader();writer.writerows(listings)
    print(json.dumps({'sources':len(rows),'repository_files_listed':len(listings),
                      'compatible_new_raw_rows':sum(r['compatible_new_raw_rows'] for r in rows)}))


if __name__ == '__main__':
    main()
