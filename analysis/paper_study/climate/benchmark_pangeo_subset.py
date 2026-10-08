"""Benchmark one public native CMIP6 regional daily subset and bilinear remap."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,threading,time
import gcsfs,numpy as np,pandas as pd,xarray as xr
import asyncio
from zarr.storage import FsspecStore,WrapperStore

ROOT=Path(__file__).resolve().parents[3]
OUT=ROOT/'analysis/paper_study/climate'
DATA=ROOT/'data/paper_study/climate/pangeo_candidate/benchmarks'
class Counted(WrapperStore):
    def __init__(self,store):super().__init__(store);self.reads=[];self.lock=threading.Lock()
    async def get(self,key,prototype,byte_range=None):
        started=time.monotonic();value=await self._store.get(key,prototype,byte_range)
        if value is not None:
            with self.lock:self.reads.append({'key':key,'bytes':len(value),'seconds':time.monotonic()-started})
        return value
    async def get_partial_values(self,prototype,key_ranges):
        return await asyncio.gather(*(self.get(key,prototype,byte_range) for key,byte_range in key_ranges))

def main():
    proof=json.loads((OUT/'pangeo_ensemble_source_verification.json').read_text())
    source=next(r for r in proof['stores'] if r['model']=='ACCESS-CM2' and r['scenario']=='ssp126' and r['variable']=='tas')
    cells=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet').sort_values(['row','col']).reset_index(drop=True)
    fs=gcsfs.GCSFileSystem(token='anon',access='read_only');store=Counted(FsspecStore.from_mapper(fs.get_mapper(source['zstore']),read_only=True))
    started=time.monotonic()
    with xr.open_zarr(store,consolidated=True,chunks={},create_default_indexes=True) as ds:
        temporal=ds.tas.sel(time=slice('2030-01-01','2030-12-31'))
        # Native stores use full-global spatial chunks; slice in the graph and
        # measure actual encoded-object traffic, rather than assuming efficient spatial reads.
        array=temporal.assign_coords(lon=(temporal.lon+180)%360-180).sortby('lon').sortby('lat')
        margin_lat=float(np.diff(array.lat.values).max());margin_lon=float(np.diff(array.lon.values).max())
        subset=array.sel(lat=slice(float(cells.latitude.min()-margin_lat),float(cells.latitude.max()+margin_lat)),lon=slice(float(cells.longitude.min()-margin_lon),float(cells.longitude.max()+margin_lon))).load(scheduler='threads',num_workers=4)
        loaded=time.monotonic()-started
        dates=pd.DatetimeIndex(subset.time.values).strftime('%Y-%m-%d')
        assert dates.tolist()==pd.date_range('2030-01-01','2030-12-31').strftime('%Y-%m-%d').tolist()
        mapped=subset.interp(lat=xr.DataArray(cells.latitude.to_numpy(),dims='cell'),lon=xr.DataArray(cells.longitude.to_numpy(),dims='cell'),method='linear').transpose('time','cell')
        values=mapped.values.astype('float32')-np.float32(273.15)
        assert np.isfinite(values).all()
        attributes=dict(ds.attrs)
    DATA.mkdir(parents=True,exist_ok=True)
    frame=pd.DataFrame({'date':np.repeat(dates,len(cells)),'cell_id':np.tile(cells.cell_id.values,len(dates)),'tmean_c':values.ravel()})
    path=DATA/'ACCESS-CM2_ssp126_tas_2030.parquet';frame.to_parquet(path,index=False,compression='zstd')
    native=DATA/'ACCESS-CM2_ssp126_tas_2030_native_europe.nc';subset.to_netcdf(native,engine='h5netcdf')
    result={'checked_utc':datetime.now(timezone.utc).isoformat(),'source':source,'benchmark_year':2030,'native_date_count':len(dates),'target_cells':len(cells),'target_daily_rows':len(frame),'native_regional_shape':list(subset.shape),'regional_load_seconds':loaded,'total_seconds':time.monotonic()-started,'encoded_objects_read':store.reads,'actual_read_bytes':sum(r['bytes'] for r in store.reads),'native_global_attributes':attributes,'bilinear_output_all_finite':True,'output_path':str(path.relative_to(ROOT)),'output_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'native_regional_path':str(native.relative_to(ROOT)),'candidate_status':'Transport and alignment benchmark only; source remains separate from main NEX v2 forcing'}
    (OUT/'pangeo_subset_benchmark.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['native_date_count','target_cells','regional_load_seconds','total_seconds','actual_read_bytes','bilinear_output_all_finite']}),flush=True)
if __name__=='__main__':main()
