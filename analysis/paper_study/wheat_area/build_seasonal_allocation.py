"""Transfer provider seasonal spatial support without duplicating SPAM hectares."""
from pathlib import Path
import hashlib, json, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import xarray as xr

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
DATA=ROOT/'data/paper_study/wheat_area'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    policy=json.loads((HERE/'seasonal_allocation_policy.json').read_text())
    native_path=DATA/'europe_source_wheat_pixels.parquet';registry_path=DATA/'europe_wheat_cells_025.parquet'
    before={str(p.relative_to(ROOT)):sha(p) for p in [native_path,registry_path]}
    source=pd.read_parquet(native_path);registry=pd.read_parquet(registry_path)
    r=source.source_row.to_numpy(int);c=source.source_col.to_numpy(int)
    r0,r1=int(r.min()),int(r.max())+1;c0,c1=int(c.min()),int(c.max())+1
    evidence=[];records=[];checks=[];provenance=[]
    for code,system in [('ir','irrigated'),('rf','rainfed')]:
        parts=[]
        for subcrop in ['Wheat1','Wheat2']:
            area_path=DATA/f'mirca_monthly_wheat_2020/2020/MIRCA-OS_{subcrop}_2020_{code}.nc'
            cal_path=DATA/f'mirca_spatial/NetCDF/{"Irrigated" if code=="ir" else "Rainfed"}/{subcrop}.nc'
            with xr.open_dataset(area_path) as d:
                np.testing.assert_allclose(d.latitude,89.95833333333333-np.arange(2160)/12,atol=2e-12,rtol=0)
                np.testing.assert_allclose(d.longitude,-179.95833333333334+np.arange(4320)/12,atol=2e-12,rtol=0)
                assert list(d.month.values)==list(range(1,13)) and int(d.attrs['year'])==2020
                a=d.harvested_area.isel(latitude=slice(r0,r1),longitude=slice(c0,c1)).values[:,r-r0,c-c0].astype(float)
            finite=np.isfinite(a);support=np.max(np.where(finite,a,0),axis=0)
            assert (support>=0).all() and not (a[finite]<0).any()
            minimum=np.min(np.where(a>0,a,np.inf),axis=0)
            varying=(support>0)&(abs(support-minimum)>1e-5)
            with xr.open_dataset(cal_path) as d:
                # Monthly hectares use exact 5-minute footprints. Categorical
                # calendars have a small provider coordinate drift, but every
                # monthly-cell centre remains inside the same calendar cell.
                lat=d.lat.values;lon=d.lon.values;dy=float(lat[0]-lat[1]);dx=float(lon[1]-lon[0])
                actual_lat=89.95833333333333-r/12;actual_lon=-179.95833333333334+c/12
                cr=np.floor((lat[0]+dy/2-actual_lat)/dy).astype(int)
                cc=np.floor((actual_lon-(lon[0]-dx/2))/dx).astype(int)
                assert np.array_equal(cr,r) and np.array_equal(cc,c)
                planting=d.Planting_month.values[r,c];maturity=d.Maturity_month.values[r,c]
            calendar_valid=np.isfinite(planting)&np.isfinite(maturity)&(planting>=1)&(planting<=12)&(maturity>=1)&(maturity<=12)
            months=np.arange(1,13)[:,None]
            expected=((months-planting)%12)<=((maturity-planting)%12)
            disagreement=(support>0)&calendar_valid&np.any((a>0)!=expected,axis=0)
            part=dict(subcrop=subcrop,support=support,planting=planting,maturity=maturity,
                calendar_valid=calendar_valid,varying=varying,disagreement=disagreement)
            parts.append(part)
            ev=source[['grid_code','source_row','source_col','row','col','ADM0_NAME','europe_fraction']].copy()
            ev['water_system']=system;ev['provider_subcrop']=subcrop;ev['season_support_max_ha']=support
            ev['source_months_finite']=finite.sum(axis=0);ev['source_months_positive']=(a>0).sum(axis=0)
            ev['positive_monthly_area_constant']=~varying;ev['calendar_month_pattern_agrees']=calendar_valid&~disagreement
            ev['planting_month']=planting;ev['maturity_month']=maturity;ev['calendar_valid']=calendar_valid
            ev['monthly_area_sha256']=sha(area_path);ev['calendar_sha256']=sha(cal_path)
            evidence.append(ev)
            provenance.append(dict(water_system=system,provider_subcrop=subcrop,area_file=str(area_path.relative_to(ROOT)),
                area_sha256=sha(area_path),calendar_file=str(cal_path.relative_to(ROOT)),calendar_sha256=sha(cal_path)))
        denominator=parts[0]['support']+parts[1]['support']
        ambiguous=parts[0]['varying']|parts[1]['varying']|parts[0]['disagreement']|parts[1]['disagreement']
        transferable=(denominator>0)&~ambiguous
        system_area=source['harvested_'+system+'_ha'].to_numpy(float)
        source_tech_sum=source.harvested_irrigated_ha.to_numpy()+source.harvested_rainfed_ha.to_numpy()
        assert (source_tech_sum>0).all()
        total_assigned=source.harvested_total_ha.to_numpy()*system_area/source_tech_sum
        for part in parts:
            ratio=np.divide(part['support'],denominator,out=np.zeros(len(source)),where=transferable)
            mask=(ratio>0)&(system_area>0)
            rec=source.loc[mask,['grid_code','source_row','source_col','row','col','ADM0_NAME','europe_fraction']].copy()
            rec['water_system']=system;rec['provider_subcrop']=part['subcrop'];rec['provider_season_fraction']=ratio[mask]
            rec['planting_month']=part['planting'][mask];rec['maturity_month']=part['maturity'][mask]
            rec['calendar_crosses_year']=rec.planting_month>rec.maturity_month
            group=np.full(len(source),'ambiguous',dtype=object)
            group[np.isin(part['planting'],policy['autumn_sowing_months'])]='autumn_sowing'
            group[np.isin(part['planting'],policy['spring_sowing_months'])]='spring_sowing'
            group[~part['calendar_valid']]='unallocated'
            reason=np.where(~part['calendar_valid'],'missing_calendar',np.where(np.isin(part['planting'],[1,7]),'outside_declared_two_sowing_windows','supported_calendar_window'))
            rec['sowing_group']=group[mask];rec['allocation_reason']=reason[mask]
            rec['allocated_system_harvested_ha']=system_area[mask]*ratio[mask]*rec.europe_fraction
            rec['allocated_total_harvested_ha']=total_assigned[mask]*ratio[mask]*rec.europe_fraction
            records.append(rec)
        mask=(~transferable)&(system_area>0)
        rec=source.loc[mask,['grid_code','source_row','source_col','row','col','ADM0_NAME','europe_fraction']].copy()
        rec['water_system']=system;rec['provider_subcrop']='unresolved';rec['provider_season_fraction']=1.
        rec['planting_month']=np.nan;rec['maturity_month']=np.nan;rec['calendar_crosses_year']=False
        rec['sowing_group']=np.where(ambiguous[mask],'ambiguous','unallocated')
        rec['allocation_reason']=np.where(ambiguous[mask],'variable_monthly_support_or_pattern','no_positive_MIRCA_season_support')
        rec['allocated_system_harvested_ha']=system_area[mask]*rec.europe_fraction
        rec['allocated_total_harvested_ha']=total_assigned[mask]*rec.europe_fraction
        records.append(rec)
    native=pd.concat(records,ignore_index=True)
    country_stats=pd.read_csv(DATA/'mirca_wheat_subcrop_country_totals.csv')
    native['MIRCA_country_key']=native.ADM0_NAME.replace({'Turkey':'Türkiye','Netherlands':'Netherlands (Kingdom of the)'})
    native=native.merge(country_stats[['Country','Crop','water_system','harvested_ha']],
        left_on=['MIRCA_country_key','provider_subcrop','water_system'],right_on=['Country','Crop','water_system'],
        how='left',validate='many_to_one').drop(columns=['Country','Crop']).rename(columns={'harvested_ha':'MIRCA_country_subcrop_system_ha'})
    native['country_season_support_conflict']=native.MIRCA_country_subcrop_system_ha.eq(0)&native.provider_subcrop.ne('unresolved')
    native.loc[native.country_season_support_conflict,'sowing_group']='ambiguous'
    native.loc[native.country_season_support_conflict,'allocation_reason']='positive_raster_support_but_zero_country_season_statistic'
    native['cell_id']=[f'g025_r{r:03d}_c{c:04d}' for r,c in zip(native.row,native.col)]
    native['Russia_seasonal_allocation_limitation']=native.ADM0_NAME.eq('Russian Federation')
    native['verified_genetic_winter_spring']=False;native['observed_2020_season_share_verified']=False
    native['reference_sensitivity_only']=True
    pd.concat(evidence,ignore_index=True).to_parquet(DATA/'mirca_native_wheat_season_evidence.parquet',index=False)
    native.to_parquet(DATA/'europe_native_seasonal_allocation_sensitivity.parquet',index=False)
    keys=['cell_id','water_system','provider_subcrop','sowing_group','planting_month','maturity_month','allocation_reason']
    detail=native.groupby(keys,dropna=False)[['allocated_system_harvested_ha','allocated_total_harvested_ha']].sum().reset_index()
    detail['reference_sensitivity_only']=True;detail['verified_genetic_winter_spring']=False
    detail['country_season_support_conflict']=detail.allocation_reason.eq('positive_raster_support_but_zero_country_season_statistic')
    detail.to_parquet(DATA/'europe_wheat_seasonal_calendar_area_sensitivity.parquet',index=False)
    detail.to_csv(DATA/'europe_wheat_seasonal_calendar_area_sensitivity.csv.gz',index=False)
    wide=registry.copy()
    for group in ['autumn_sowing','spring_sowing','ambiguous','unallocated']:
        areas=native.loc[native.sowing_group.eq(group)].groupby('cell_id').allocated_total_harvested_ha.sum()
        wide[group+'_harvested_ha']=wide.cell_id.map(areas).fillna(0)
        wide[group+'_fraction']=wide[group+'_harvested_ha']/wide.harvested_total_ha
        for system in ['irrigated','rainfed']:
            areas=native.loc[native.sowing_group.eq(group)&native.water_system.eq(system)].groupby('cell_id').allocated_system_harvested_ha.sum()
            key=group+'_'+system+'_harvested_ha';wide[key]=wide.cell_id.map(areas).fillna(0)
            wide[group+'_'+system+'_fraction']=np.divide(wide[key],wide['harvested_'+system+'_ha'],out=np.full(len(wide),np.nan),where=wide['harvested_'+system+'_ha']>0)
    wide['seasonal_allocation_basis']='MIRCAOS2020_monthly_support_ratio_transfer_sensitivity'
    wide['reference_sensitivity_only']=True;wide['verified_genetic_winter_spring']=False
    wide['Russia_seasonal_allocation_limitation']=wide.cell_id.isin(native.loc[native.Russia_seasonal_allocation_limitation,'cell_id'])
    wide['two_sowing_windows_fully_allocated']=wide.ambiguous_harvested_ha.add(wide.unallocated_harvested_ha).lt(1e-8)
    wide['country_season_support_conflict']=wide.cell_id.isin(native.loc[native.country_season_support_conflict,'cell_id'])
    wide.to_parquet(DATA/'europe_wheat_cells_025_seasonal_sensitivity.parquet',index=False)
    wide.to_csv(DATA/'europe_wheat_cells_025_seasonal_sensitivity.csv',index=False)
    country=native.groupby(['ADM0_NAME','water_system','sowing_group'])[['allocated_system_harvested_ha','allocated_total_harvested_ha']].sum().reset_index()
    country.to_csv(DATA/'europe_country_seasonal_allocation_sensitivity.csv',index=False)
    reasons=native.groupby(['water_system','allocation_reason'])[['allocated_system_harvested_ha','allocated_total_harvested_ha']].sum().reset_index()
    reasons.to_csv(HERE/'seasonal_allocation_reason_totals.csv',index=False)
    for system in ['total','irrigated','rainfed']:
        columns=[g+('_harvested_ha' if system=='total' else '_'+system+'_harvested_ha') for g in ['autumn_sowing','spring_sowing','ambiguous','unallocated']]
        difference=wide[columns].sum(axis=1)-wide['harvested_'+system+'_ha']
        assert difference.abs().max()<1e-8
        checks.append(dict(denominator=system,max_cell_error_ha=float(difference.abs().max()),total_error_ha=float(difference.sum())))
    assert all(sha(ROOT/name)==value for name,value in before.items())
    totals={g:float(wide[g+'_harvested_ha'].sum()) for g in ['autumn_sowing','spring_sowing','ambiguous','unallocated']}
    receipt=dict(status='conserved_reference_sensitivity',completed_utc=datetime.now(timezone.utc).isoformat(),
        cells=len(wide),source_footprints=len(source),native_allocation_rows=len(native),totals_ha=totals,
        conservation_checks=checks,source_files=provenance,original_registry_hashes=before,
        policy_sha256=sha(HERE/'seasonal_allocation_policy.json'),original_registry_preserved=True,
        units_source='MIRCA provider README__monthly_growing_area.txt: hectares; source NetCDF area variable lacks units attribute',
        no_disease_outcomes_used=True,genetic_type_identified=False,Russia_share_limitation_preserved=True,
        denominator_note='Total and I/R system denominators conserved separately; original source floating discrepancy remains')
    (HERE/'seasonal_allocation_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status=receipt['status'],cells=len(wide),totals_ha=totals,conservation=checks),indent=2))

if __name__=='__main__':main()
