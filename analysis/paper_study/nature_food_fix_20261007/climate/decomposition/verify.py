"""Independent current-kernel four-corner attribution arithmetic and provenance."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json
import numpy as np
import pandas as pd

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[4]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def variance(v,draws):
    return float(sum(float(g.area_mean_weight.sum())**2*np.var(v[g.index.to_numpy()],ddof=1)/len(g) for _,g in draws.groupby('stratum')))


def main():
    reg=json.loads((HERE/'registration_before_decomposition_results.json').read_text());receipt=json.loads((HERE/'completion_receipt.json').read_text())
    for name,h in reg['source_sha256'].items():assert sha(ROOT/name)==h,name
    for name,h in receipt['output_sha256'].items():assert sha(HERE/name)==h,name
    corners=pd.read_parquet(HERE/'four_corner_draw_year_outputs.parquet');draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv')
    gcm=pd.read_parquet(HERE/'decomposition_by_gcm.parquet');ensemble=pd.read_parquet(HERE/'ensemble_decomposition.parquet')
    assert len(corners)==23040 and not corners.duplicated(['corner','model','reference_year','spatial_draw_id']).any()
    assert corners.groupby(['corner','model','reference_year']).size().eq(64).all()
    weights=draws.area_mean_weight.to_numpy();errors=[];mcse_errors=[];influences={};identity_max=0.;checks=0
    baseline=pd.read_parquet(HERE.parent/'draw_season_outputs.parquet');baseline=baseline[baseline.setting.eq('baseline')]
    diagonal_count=0
    for corner,future in [('Wreference_Hreference',False),('Wfuture_Hfuture',True)]:
        a=corners[corners.corner.eq(corner)].sort_values(['model','spatial_draw_id','reference_year']);years=range(2071,2101) if future else range(1991,2021)
        b=baseline[baseline.harvest_year.isin(years)].sort_values(['model','spatial_draw_id','harvest_year'])
        for m in reg['metrics']:
            x=a[m].to_numpy();y=b[m].to_numpy();assert np.array_equal(np.isfinite(x),np.isfinite(y));assert np.array_equal(x[np.isfinite(x)],y[np.isfinite(y)])
            diagonal_count+=int(np.isfinite(x).sum())
    for model in reg['models']:
        arrays={c:g.sort_values(['spatial_draw_id','reference_year'])[reg['metrics']].to_numpy(float).reshape(64,30,len(reg['metrics']))
            for c,g in corners[corners.model.eq(model)].groupby('corner')}
        a=arrays['Wreference_Hreference'];b=arrays['Wfuture_Hreference'];c=arrays['Wreference_Hfuture'];d=arrays['Wfuture_Hfuture']
        valid=np.isfinite(a)&np.isfinite(b)&np.isfinite(c)&np.isfinite(d)
        # Alternate expression: main contrast plus half interaction.
        interaction=(d-b)-(c-a)
        components={'total_change':d-a,'weather_alone':b-a,'host_alone':c-a,'interaction':interaction,
            'weather_shapley':b-a+interaction/2,'host_shapley':c-a+interaction/2}
        identity_max=max(identity_max,float(np.max(abs(np.where(valid,components['weather_shapley']+components['host_shapley']-components['total_change'],0)))))
        for j,metric in enumerate(reg['metrics']):
            mask=valid[:,:,j];z=mask.sum(axis=1)/30;coverage=float(weights@z)
            for name,value in components.items():
                x=np.where(mask,value[:,:,j],0).sum(axis=1)/30;mean=float(weights@x)/coverage
                influence=(x-mean*z)/coverage;mcse=np.sqrt(variance(influence,draws))
                row=gcm[gcm.model.eq(model)&gcm.metric.eq(metric)&gcm.component.eq(name)].iloc[0]
                errors.extend([abs(mean-row['mean']),abs(coverage-row.common_four_corner_area_time_coverage)])
                mcse_errors.append(abs(mcse-row.spatial_mcse));assert int(mask.sum())==row.common_draw_year_pairs
                influences.setdefault((metric,name),[]).append(influence);checks+=1
    for (metric,name),vectors in influences.items():
        row=ensemble[ensemble.metric.eq(metric)&ensemble.component.eq(name)].iloc[0]
        mcse_errors.append(abs(np.sqrt(variance(np.mean(vectors,axis=0),draws))-row.spatial_mcse_gcm_mean))
    assert max(errors)<1e-10 and max(mcse_errors)<1e-10 and identity_max<1e-10
    highT=0;active=0;unavailable=0;annual_checks=0
    for p in (HERE/'paired_outputs').rglob('*.json'):
        r=json.loads(p.read_text());assert sha(p.with_suffix('.parquet'))==r['output_sha256'];assert r['registration_sha256']==sha(HERE/'registration_before_decomposition_results.json')
        annual_checks+=1
        for v in r['checks'].values():
            highT+=v['disease_weather_meanT_above40_active_days'];active+=v['active_target_days'];unavailable+=v['unavailable_active_weather_days']
    assert annual_checks==90 and unavailable==0
    result=dict(status='verified',verified_utc=datetime.now(timezone.utc).isoformat(),independent_GCM_component_checks=checks,
        diagonal_numerical_values_identical=diagonal_count,maximum_mean_or_coverage_error=max(errors),maximum_spatial_mcse_error=max(mcse_errors),
        maximum_alternate_formula_additivity_error=identity_max,paired_receipts_checked=annual_checks,
        hybrid_active_disease_weather_days=active,hybrid_disease_weather_meanT_above40_days=highT,unavailable_active_source_weather_days=unavailable,
        order_invariant_weather_host_attribution=True,source_and_output_sha256_verified=True,code_sha256=sha(Path(__file__)),
        registration_sha256=sha(HERE/'registration_before_decomposition_results.json'))
    (HERE/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
