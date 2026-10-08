"""Produce source-grounded climate methods only after complete audits pass."""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import hashlib,json
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis/paper_study/climate'
DATA=ROOT/'data/paper_study/climate'

def sha(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):value.update(block)
    return value.hexdigest()

def main():
    audits={name:json.loads((OUT/file).read_text()) for name,file in {
        'all':'actual_coverage_validation.json','era5':'actual_coverage_validation_era5.json',
        'nasa':'actual_coverage_validation_nasa.json','provenance':'nex_native_provenance_validation.json',
        'native_values':'nex_independent_native_value_validation.json'}.items()}
    for name in ['all','era5','nasa']:assert audits[name]['coverage_status']=='Complete'
    assert audits['era5']['deep'] and audits['nasa']['deep']
    assert audits['all']['summary']['era5']['valid']==372 and audits['all']['summary']['nasa']['valid']==687
    assert audits['provenance']['status']=='Complete and passed' and audits['provenance']['complete_component_bytes_checked']
    assert audits['native_values']['status']=='passed' and audits['native_values']['passed_checks']==48
    registry=DATA.parent/'wheat_area/europe_wheat_cells_025.parquet';cells=pd.read_parquet(registry)
    registry_hash=sha(registry)
    assert all(audits[name]['registry_sha256']==registry_hash for name in ['all','era5','nasa','provenance'])
    source=pd.read_csv(OUT/'nex_native_file_provenance.csv');assert len(source)==2748
    license_count=Counter(source.native_license);transport=audits['provenance']['transport_components']
    land={p.stem:json.loads(p.read_text()) for p in (DATA/'nasa_v2/coastal_alignment').glob('*.json')}
    assert len(land)==3 and all(r['imputed_cells']==r['excluded_cells']==r['native_land_masked_cells']==0 for r in land.values())
    receipts=['actual_coverage_validation.json','actual_coverage_validation_era5.json','actual_coverage_validation_nasa.json',
              'nex_native_provenance_validation.json','nex_independent_native_value_validation.json','climate_registry_contract.json']
    main_schema=[('date','string','YYYY-MM-DD'),('cell_id','string','g025_r{row:03d}_c{col:04d}'),
                 ('tmean_c','float32','degC'),('tmax_c','float32','degC'),
                 ('precipitation_mm','float32','mm/day'),('rh_mean_pct','float32','percent')]
    pd.DataFrame(main_schema,columns=['field','dtype','units_or_format']).to_csv(OUT/'climate_wide_schema.csv',index=False)
    text=f"""Daily meteorological forcing covers a fixed 2020 wheat-area registry of {len(cells):,} positive-area cells on a 0.25° geographic grid. Cell centres follow longitude = −180 + (column + 0.5) × 0.25 and latitude = 90 − (row + 0.5) × 0.25, with EPSG:4326 transform [0.25, 0, −180, 0, −0.25, 90]. Registry boundaries span −9.5° to 59.5° longitude and 34.75° to 67.75° latitude. Harvested wheat area totals {cells.harvested_total_ha.sum():,.2f} ha. The registry checksum is {registry_hash}; the original source-cell identifiers and fixed area denominators are retained across periods.

The reanalysis baseline contains 372 complete calendar-month partitions from 1 January 1990 through 31 December 2020, acquired from ECMWF/ERA5/HOURLY through Google Earth Engine. Daily mean temperature and daily maximum temperature are the arithmetic mean and maximum of the 24 hourly 2 m temperatures at 00:00–23:00 UTC. Temperatures are converted from kelvin to degrees Celsius. Hourly relative humidity is calculated from paired 2 m temperature and dew-point temperature as 100 × exp[17.625 Td/(243.04 + Td) − 17.625 T/(243.04 + T)], with T and Td in degrees Celsius; the daily field is the arithmetic mean of the hourly values bounded to 0–100%. Daily precipitation sums the preceding-hour accumulations ending at 01:00 UTC on date D through 00:00 UTC on D + 1 and converts metres of water equivalent to millimetres. Daily continuous fields are bilinearly interpolated from the native ERA5 grid, whose longitude and latitude origins are −180.125° and 90.125°, to the common cell centres.

Climate projections use native NASA NEX-GDDP-CMIP6 version 2.0 files for ACCESS-CM2, MPI-ESM1-2-HR and MRI-ESM2-0, each with realization r1i1p1f1. Scenarios comprise historical, SSP1–2.6, SSP2–4.5 and SSP5–8.5. Historical 1990–2014 supplies 75 model-years; scenario-specific 2015–2020 continuations supply 54 model-scenario-years. Future calendar periods 2030–2060 and 2070–2100 each supply 279 model-scenario-years. The archive therefore contains 687 merged daily model-scenario-year partitions and 2,748 component variable-year partitions. The antecedent calendar years 1990, 2030 and 2070 support the harvest-year windows 1991–2020, 2031–2060 and 2071–2100. All selected files use the proleptic Gregorian calendar.

The native variables are tas, tasmax, pr and hurs. Archived fields are tmean_c, tmax_c, precipitation_mm and rh_mean_pct, respectively. Temperature is converted as K − 273.15 and precipitation as kg m−2 s−1 × 86,400. NEX daily tas follows the provider's mean of daily temperature extrema, whereas ERA5 tmean_c averages hourly temperatures. NEX bias correction uses the GMFD/Princeton daily forcing reference for 1960–2014. Native hurs values are preserved in rh_mean_pct_raw, with the physical humidity field bounded to 0–100%. Precipitation rounding artifacts between −0.01 and 0 mm/day are set to zero, with raw values and adjustment counts retained whenever they occur.

Native version 2.0 files are selected from archived official NASA catalogs and spatially subset through the NCCS THREDDS NetCDF Subset Service. Query boundaries equal native pixel centres, which prevents an observed longitude-axis displacement produced by off-centre bounds. The native 0.25° NEX grid already matches the common cell centres. Regional bounding-box responses retain positive-wheat cells in the Parquet archive. Acquisition used {transport.get('quarterly',0):,} variable-years assembled from four quarterly responses and {transport.get('annual',0):,} earlier variable-years acquired as annual responses. Original response digests, native filenames, source attributes and calendars are retained in the adjacent component JSON records. Temporary quarter responses are removed after the annual selected-cell archive and provenance records are committed.

One frozen, jointly valid four-field land lookup was derived for each model from historical 1 January 1990. The lookup permits the nearest valid native centre within 0.5° and records source coordinates and distance. All three model lookups cover every target cell directly: zero cells require coastal replacement and zero cells are excluded. Native missing-value indicators remain available for all four variables, together with source-grid row and column identifiers.

Complete-byte checksums, field schemas, registry hashes and expected partition dates pass for all 372 ERA5 monthly and 687 NEX annual partitions. Separate deep audits verify the complete daily date–cell combinations, absence of duplicate keys, finite required fields, physical humidity and precipitation bounds, and daily mean temperature not exceeding the maximum. An independent provenance audit checks all 2,748 native component identities and byte digests, archived catalogs, units, grid edges, calendars, licence attributes and {audits['provenance']['quarter_metadata_records_checked']:,} quarter metadata records. Independently retrieved native annual subsets verify {audits['native_values']['compared_cell_days']:,} field-specific cell-days across all three models, historical conditions and all three SSPs, including {audits['native_values']['quarter_boundary_and_leap_cell_days']:,} quarter-transition and leap-day comparisons. Differences remain within an absolute tolerance of 0.0001 and a relative tolerance of 10−6 in the archived field units.

The archive retains the daily wide schema keyed by date and cell_id within each model–scenario partition. Model, scenario, realization, source version and year are directory or adjacent metadata identifiers. Main physical fields use float32 values and lossless Zstandard Parquet compression. ERA5 hourly relative humidity, NEX raw humidity, source-grid coordinates and per-file licences remain traceable through the source contract and adjacent metadata.

ERA5 attribution is “Contains modified Copernicus Climate Change Service information 2026”, with data supplied through Google Earth Engine. The European Commission and ECMWF bear no responsibility for the use of the Copernicus information in this archive. The archived Copernicus licence and catalog snapshots accompany the dataset. NEX native licence attributes include {license_count.get('CC-BY 4.0',0):,} component files labelled CC-BY 4.0 and {license_count.get('CC-BY-SA 4.0',0):,} labelled CC-BY-SA 4.0. These per-file attributes and the original CMIP6 terms are preserved. The NASA version 2 technical note dated 31 May 2025 directs users to the source-file cmip6_license attribute and original CMIP6 terms. Earlier provider statements concerning blanket CC0 licensing remain archived as dated source provenance.

NASA Earth Exchange, the Climate Analytics Group, NASA Ames Research Center and NASA NCCS receive credit for producing and distributing NEX-GDDP-CMIP6. CMIP6 coordination and data provision are credited to the World Climate Research Programme, its Working Group on Coupled Modelling, participating climate modelling groups, the Earth System Grid Federation and their supporting funding agencies.

Source records: https://developers.google.com/earth-engine/datasets/catalog/ECMWF_ERA5_HOURLY ; https://apps.ecmwf.int/datasets/licences/copernicus/ ; https://doi.org/10.7917/OFSG3345 ; https://www.nccs.nasa.gov/wp-content/uploads/2025/06/NEX-GDDP-CMIP6-v2-Tech_Note.pdf ; https://pcmdi.llnl.gov/CMIP6/TermsOfUse/TermsOfUse6-1.html .
"""
    (OUT/'CLIMATE_METHODS.txt').write_text(text)
    receipt={'checked_utc':datetime.now(timezone.utc).isoformat(),'status':'complete','registry_sha256':registry_hash,
        'coverage':audits['all']['summary'],'source_licenses':dict(license_count),'source_roles':{'baseline':'ECMWF/ERA5/HOURLY','projections':'Native NEX-GDDP-CMIP6 2.0'},
        'source_receipts':{str((OUT/r).relative_to(ROOT)):sha(OUT/r) for r in receipts},
        'methods_path':'analysis/paper_study/climate/CLIMATE_METHODS.txt','methods_sha256':sha(OUT/'CLIMATE_METHODS.txt'),
        'native_cmip6_candidate_used':False,'earth_engine_gddp_collection_used':False}
    (OUT/'climate_full_acquisition_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
