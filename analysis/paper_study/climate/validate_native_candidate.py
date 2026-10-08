"""Validate the existing four-field transport sample against independent reads."""
from pathlib import Path
from datetime import datetime
import importlib.util,json
import numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent
SPEC=importlib.util.spec_from_file_location('native_candidate',HERE/'retrieve_native_cmip6.py')
N=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(N)

def main():
    cells=N.C.registry(N.C.REGISTRY)
    proof=json.loads((HERE/'pangeo_ensemble_source_verification.json').read_text())
    sources=[r for r in proof['stores'] if r['model']=='ACCESS-CM2' and r['scenario']=='ssp126']
    assert len(sources)==4
    coordinate_checks=[];corner_checks=[]
    for source in sources:
        prepared=N.prepare_source(source,cells)
        lat,lon=N.GRIDS[source['model']]
        coordinate_checks.append({'variable':source['variable'],'version':source['zstore'].rstrip('/').rsplit('/',1)[-1],
            'shape':[len(lat),len(lon)],'shared_coordinates_verified':True,
            'native_time_strictly_daily':bool(np.all(np.diff(prepared['native_time'])==1))})
        variable=source['variable'];array=prepared['array'];attrs=prepared['metadata']['time/.zattrs']
        start=N.cftime.date2num(datetime(2030,1,1,12),attrs['units'],attrs['calendar'])
        stop=N.cftime.date2num(datetime(2031,1,1,12),attrs['units'],attrs['calendar'])
        indices=np.flatnonzero((prepared['native_time']>=start)&(prepared['native_time']<stop))
        markers=[array.get('fill_value'),prepared['metadata'][variable+'/.zattrs'].get('missing_value'),prepared['metadata'][variable+'/.zattrs'].get('_FillValue')]
        checked=0
        for chunk in np.unique(indices//array['chunks'][0]):
            content,_=N.cached(source,f'{variable}/{chunk}.0.0')
            native=N.decode(content,array).reshape(array['chunks'],order=array['order'])
            positions=indices[indices//array['chunks'][0]==chunk]-chunk*array['chunks'][0]
            for row,col in zip([prepared['i0'],prepared['i1'],prepared['i0'],prepared['i1']],[prepared['j0'],prepared['j0'],prepared['j1'],prepared['j1']]):
                values=native[positions[:,None],row[None,:],col[None,:]]
                assert np.isfinite(values).all()
                for marker in markers:
                    assert marker is None or not np.any(values==np.asarray(marker,dtype=values.dtype))
                checked+=values.size
        corner_checks.append({'variable':variable,'checked_native_corner_values':checked,'native_missing_values':0})
    path=N.DATA/'daily/ACCESS-CM2/ssp126/2030.parquet'
    assert N.C.done(path,N.C.FIELDS)
    frame=pd.read_parquet(path)
    quality=N.C.quality(frame,N.C.FIELDS)
    expected_dates=pd.date_range('2030-01-01','2030-12-31').strftime('%Y-%m-%d')
    assert frame.date.to_numpy().tolist()==np.repeat(expected_dates,len(cells)).tolist()
    assert np.array_equal(frame.cell_id.to_numpy(),np.tile(cells.cell_id.to_numpy(),len(expected_dates)))
    assert not frame.filter(regex='_native_missing$').to_numpy().any()
    benchmark=N.C.DATA/'pangeo_candidate/benchmarks/ACCESS-CM2_ssp126_tas_2030.parquet'
    reference=pd.read_parquet(benchmark)
    assert frame[['date','cell_id']].equals(reference[['date','cell_id']])
    difference=np.abs(frame.tmean_c.to_numpy()-reference.tmean_c.to_numpy())
    assert difference.max()<.0001
    receipt={'checked_utc':N.C.utc(),'candidate_status':'Separate transport candidate; no main-source substitution',
        'model':'ACCESS-CM2','scenario':'ssp126','year':2030,'member_id':'r1i1p1f1',
        'coordinates':coordinate_checks,'quality':quality,'date_cell_combinations_complete':True,
        'native_corner_checks':corner_checks,
        'independent_reader':'xarray linear interpolation from independently retrieved native regional array',
        'temperature_comparison_rows':len(frame),'max_abs_temperature_difference_C':float(difference.max()),
        'tolerance_C':.0001,'parquet_path':str(path.relative_to(N.C.ROOT)),
        'parquet_sha256':N.C.sha256(path),'reference_sha256':N.C.sha256(benchmark),'passed':True}
    N.C.write_json(HERE/'native_cmip6_four_field_validation.json',receipt)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()
