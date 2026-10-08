"""Independent source-byte, monthly-area and denominator verification."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,zlib
import numpy as np
import pandas as pd
from netCDF4 import Dataset

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/wheat_area'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    results=[]
    def check(name,condition,details=None):
        if not bool(condition):raise AssertionError(name)
        results.append(dict(check=name,status='passed',details=details))
    manifest=json.loads((DATA/'mirca_monthly_wheat_download_manifest.json').read_text())
    check('no_full_archive_checksum_claim',not manifest['entire_archive_checksum_verified'] and not manifest['entire_archive_downloaded'])
    for item in manifest['members']:
        p=ROOT/item['local_path'];b=p.read_bytes()
        check('member_sha_'+p.name,hashlib.sha256(b).hexdigest()==item['sha256'])
        check('member_crc_'+p.name,f'{zlib.crc32(b):08x}'==item['crc32'] and len(b)==item['bytes'])
        container=p.parent.parent/(p.stem+'.member.rar');block=container.read_bytes()[16:-8]
        ranges=[x for x in manifest['range_receipts'] if x['start']==item['header_offset']]
        check('original_packed_range_'+p.name,len(ranges)==1 and hashlib.sha256(block).hexdigest()==ranges[0]['sha256'])
    receipt=json.loads((HERE/'seasonal_allocation_receipt.json').read_text())
    for name,value in receipt['original_registry_hashes'].items():check('original_registry_'+name,sha(ROOT/name)==value)
    source=pd.read_parquet(DATA/'europe_source_wheat_pixels.parquet')
    evidence=pd.read_parquet(DATA/'mirca_native_wheat_season_evidence.parquet')
    native=pd.read_parquet(DATA/'europe_native_seasonal_allocation_sensitivity.parquet')
    wide=pd.read_parquet(DATA/'europe_wheat_cells_025_seasonal_sensitivity.parquet')
    old=pd.read_parquet(DATA/'europe_wheat_cells_025.parquet')
    check('all_original_cell_fields_preserved',wide[old.columns].equals(old))
    check('one_cell_record_each',wide.cell_id.is_unique and len(wide)==14941)
    check('all_season_flags_sensitivity',wide.reference_sensitivity_only.all() and not wide.verified_genetic_winter_spring.any())
    check('no_duplicate_native_season_keys',not evidence.duplicated(['grid_code','water_system','provider_subcrop']).any())
    r=source.source_row.to_numpy(int);c=source.source_col.to_numpy(int)
    r0,r1=int(r.min()),int(r.max())+1;c0,c1=int(c.min()),int(c.max())+1
    for system in ['irrigated','rainfed']:
        for subcrop in ['Wheat1','Wheat2']:
            code='ir' if system=='irrigated' else 'rf'
            p=DATA/f'mirca_monthly_wheat_2020/2020/MIRCA-OS_{subcrop}_2020_{code}.nc'
            with Dataset(p) as d:
                v=d.variables['harvested_area'];v.set_auto_mask(False)
                raw=np.asarray(v[:,r0:r1,c0:c1])[:,r-r0,c-c0]
                lat=np.asarray(d.variables['latitude'][:]);lon=np.asarray(d.variables['longitude'][:])
                check('grid_alignment_'+system+'_'+subcrop,np.max(abs(lat-(90-(np.arange(2160)+.5)/12)))<2e-12 and np.max(abs(lon-(-180+(np.arange(4320)+.5)/12)))<2e-12)
            selected=evidence.loc[evidence.water_system.eq(system)&evidence.provider_subcrop.eq(subcrop)].set_index('grid_code').loc[source.grid_code]
            maxarea=np.where(np.isfinite(raw),raw,0).max(axis=0)
            check('all_native_maxima_'+system+'_'+subcrop,np.array_equal(maxarea,selected.season_support_max_ha.to_numpy()))
            check('all_native_month_counts_'+system+'_'+subcrop,np.array_equal((raw>0).sum(axis=0),selected.source_months_positive.to_numpy()))
            check('source_byte_hash_column_'+system+'_'+subcrop,selected.monthly_area_sha256.eq(sha(p)).all())
            cp=DATA/f'mirca_spatial/NetCDF/{"Irrigated" if system=="irrigated" else "Rainfed"}/{subcrop}.nc'
            with Dataset(cp) as d:
                pm=np.asarray(d.variables['Planting_month'][:]);mm=np.asarray(d.variables['Maturity_month'][:])
            check('raw_calendar_'+system+'_'+subcrop,np.array_equal(pm[r,c],selected.planting_month,equal_nan=True) and np.array_equal(mm[r,c],selected.maturity_month,equal_nan=True))
    lookup=source.set_index('grid_code')
    for system in ['irrigated','rainfed']:
        group=native.loc[native.water_system.eq(system)]
        sums=group.groupby('grid_code').provider_season_fraction.sum()
        required=lookup.index[lookup['harvested_'+system+'_ha']>0]
        check('native_fraction_partition_'+system,set(sums.index)==set(required) and abs(sums-1).max()<1e-12)
        expected=lookup.loc[group.grid_code,'harvested_'+system+'_ha'].to_numpy()*group.provider_season_fraction.to_numpy()*group.europe_fraction.to_numpy()
        check('native_system_hectares_'+system,np.max(abs(expected-group.allocated_system_harvested_ha.to_numpy()))<1e-8)
        totals=group.groupby('cell_id').allocated_system_harvested_ha.sum()
        check('each_original_system_cell_denominator_'+system,np.max(abs(wide.cell_id.map(totals).fillna(0)-wide['harvested_'+system+'_ha']))<1e-8)
    celltotals=native.groupby('cell_id').allocated_total_harvested_ha.sum()
    check('original_total_cell_denominator',np.max(abs(wide.cell_id.map(celltotals)-wide.harvested_total_ha))<1e-8)
    groups=['autumn_sowing','spring_sowing','ambiguous','unallocated']
    for group in groups:
        h=native.loc[native.sowing_group.eq(group)].groupby('cell_id').allocated_total_harvested_ha.sum()
        check('wide_group_agreement_'+group,np.max(abs(wide.cell_id.map(h).fillna(0)-wide[group+'_harvested_ha']))<1e-8)
    check('autumn_calendar_window',native.loc[native.sowing_group.eq('autumn_sowing'),'planting_month'].isin([8,9,10,11,12]).all())
    check('spring_calendar_window',native.loc[native.sowing_group.eq('spring_sowing'),'planting_month'].isin([2,3,4,5,6]).all())
    check('conflicting_country_support_is_ambiguous',native.loc[native.country_season_support_conflict,'sowing_group'].eq('ambiguous').all())
    check('Russia_limitation_retained',native.loc[native.ADM0_NAME.eq('Russian Federation'),'Russia_seasonal_allocation_limitation'].all())
    check('native_area_nonnegative',native[['allocated_system_harvested_ha','allocated_total_harvested_ha']].ge(0).all().all())
    detail=pd.read_parquet(DATA/'europe_wheat_seasonal_calendar_area_sensitivity.parquet')
    check('detail_total_conserved',abs(detail.allocated_total_harvested_ha.sum()-old.harvested_total_ha.sum())<1e-7)
    check('wide_fraction_partition',np.max(abs(wide[[g+'_fraction' for g in groups]].sum(axis=1)-1))<1e-12)
    final=dict(status='passed',completed_utc=datetime.now(timezone.utc).isoformat(),checks_passed=len(results),checks=results,
        regional_output_sha256=sha(DATA/'europe_wheat_cells_025_seasonal_sensitivity.parquet'),
        original_registry_sha256=sha(DATA/'europe_wheat_cells_025.parquet'),disease_outcomes_used=False,
        archive_verification='Four source member CRC32 and SHA256 checks; original packed byte-range hashes; entire remote archive checksum unverified')
    (HERE/'seasonal_allocation_independent_verification.json').write_text(json.dumps(final,indent=2)+'\n')
    print(json.dumps({k:final[k] for k in ['status','checks_passed','regional_output_sha256','original_registry_sha256']},indent=2))

if __name__=='__main__':main()
