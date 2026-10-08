"""Independent reconstruction of scenario differences and endpoint fit diagnostics."""
from pathlib import Path
import hashlib, itertools, json, sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
from analysis.paper_study.run_regional import MODELS, SCENARIOS
from analysis.paper_study.aggregate_projection_uncertainty import PERIODS


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    review=ROOT/'analysis/paper_study/manuscript_review_20261006'
    draws=pd.read_csv(ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv')
    weights=draws.area_mean_weight.to_numpy();strata=draws.stratum.to_numpy()
    points=pd.read_csv(ROOT/'data/paper_study/publication/regional_period_comparison.csv')
    reconstructed={};checks=0;maximum=0.
    def close(a,b):
        nonlocal checks,maximum
        a=np.asarray(a);b=np.asarray(b)
        np.testing.assert_allclose(a,b,rtol=0,atol=2e-10)
        checks+=a.size;maximum=max(maximum,float(np.max(np.abs(a-b))))
    def manual_mcse(values):
        variances=np.zeros(values.shape[:-1])
        for label in sorted(set(strata)):
            ix=np.where(strata==label)[0];assert len(ix)==4
            share=sum(weights[ix]);x=values[...,ix];centre=x.sum(axis=-1)/4
            sum_squares=((x-centre[...,None])**2).sum(axis=-1)
            variances+=share**2*sum_squares/3/4
        return np.sqrt(variances)
    for model,scenario,period in itertools.product(MODELS,SCENARIOS,PERIODS):
        source=ROOT/'data/paper_study/projection_uncertainty_reporting/parameter_periods'/model/scenario/f'{period}.npz'
        receipt=json.loads(source.with_suffix('.json').read_text());assert sha(source)==receipt['output_sha256']
        record=points[points.model.eq(model)&points.scenario.eq(scenario)&points.period.eq(period)].iloc[0]
        close(receipt['full_point_mean'],record.severity)
        annual=[]
        for year in PERIODS[period]:
            path=ROOT/'data/paper_study/regional_parameter_uncertainty/annual'/model/scenario/f'{year}.npz'
            assert sha(path)==json.loads(path.with_suffix('.json').read_text())['output_sha256']
            annual.append(np.load(path)['cell_damage_percent'])
        stack=np.stack(annual,axis=1);valid=np.isfinite(stack[0])
        assert np.array_equal(np.isfinite(stack),np.broadcast_to(valid,stack.shape))
        perturb=np.where(valid[None,:,:],stack-stack[0:1],0).sum(axis=1)/30/receipt['full_valid_fraction']
        means=record.severity+np.sum(perturb*weights[None,:],axis=1)
        saved=dict(np.load(source));close(perturb,saved['contributions']);close(means,saved['means'])
        close(manual_mcse(perturb),saved['mcse'])
        reconstructed[model,scenario,period]=(means,perturb)
    contrasts=pd.read_csv(ROOT/'data/paper_study/publication/paired_scenario_contrasts.csv')
    stored=dict(np.load(review/'paired_scenario_draws.npz'))
    for high,low,period,quantity in itertools.product(SCENARIOS,SCENARIOS,list(PERIODS)[1:],
            ['future_severity_difference','difference_in_baseline_relative_changes']):
        selected=contrasts[contrasts.scenario_high.eq(high)&contrasts.scenario_low.eq(low)&contrasts.period.eq(period)&contrasts.quantity.eq(quantity)]
        if selected.empty:continue
        values=[];influences=[]
        for model in MODELS:
            hm,hi=reconstructed[model,high,period];lm,li=reconstructed[model,low,period]
            value=hm-lm;influence=hi-li
            if quantity=='difference_in_baseline_relative_changes':
                bhm,bhi=reconstructed[model,high,'baseline1991_2020'];blm,bli=reconstructed[model,low,'baseline1991_2020']
                value=value-bhm+blm;influence=influence-bhi+bli
            values.append(value);influences.append(influence)
        for model,value,influence in zip(list(MODELS)+['three_model_mean'],
                values+[sum(values)/3],influences+[sum(influences)/3]):
            row=selected[selected.model.eq(model)].iloc[0]
            key='__'.join([model,high,low,period,quantity]);close(value,stored[key+'__means']);close(influence,stored[key+'__contributions'])
            close(value[0],row.point_estimate_pp)
            close(np.percentile(value[1:],[2.5,97.5]),[row.conditional_parameter_p025_pp,row.conditional_parameter_p975_pp])
            mcse=manual_mcse(influence);close(mcse[0],row.point_spatial_mcse_pp);close(mcse[1:].max(),row.bootstrap_spatial_mcse_max_pp)
            if model=='three_model_mean':close([min(v[0] for v in values),max(v[0] for v in values)],[row.climate_model_point_min_pp,row.climate_model_point_max_pp])
    # A fixed set of complete cells supplies a separate denominator sensitivity.
    cells=pd.read_parquet(ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet')
    mapped=pd.read_csv(ROOT/'data/paper_study/publication/fig4_climate_change_cells.csv')
    map_group=mapped[mapped.scenario.eq('ssp126')&mapped.period.eq('midcentury2031_2060')]
    assert map_group.cell_id.to_list()==cells.cell_id.to_list()
    mask=map_group.all_comparisons_common_complete.to_numpy()
    selected_weight=cells.harvested_total_ha.to_numpy()[mask];den=selected_weight.sum()
    common_points={}
    for model,scenario,period in itertools.product(MODELS,SCENARIOS,PERIODS):
        annual=[]
        for year in PERIODS[period]:
            path=ROOT/'analysis/paper_study/regional_v1/annual/nasa'/model/scenario/'adjusted/winter_wheat_rainfed_sow+0_stage85'/f'{year}.parquet'
            assert sha(path)==json.loads(path.with_suffix('.json').read_text())['parquet_sha256']
            frame=pd.read_parquet(path,columns=['cell_id','status','upper3_final_damage_percent'])
            assert frame.cell_id.to_list()==cells.cell_id.to_list()
            assert frame.status.to_numpy()[mask].tolist()==['complete']*int(mask.sum())
            annual.append(np.dot(selected_weight,frame.upper3_final_damage_percent.to_numpy()[mask])/den)
        common_points[model,scenario,period]=sum(annual)/30
    common_summary=pd.read_csv(ROOT/'data/paper_study/publication/common_support_scenario_changes.csv')
    for row in common_summary.itertuples():
        difference=sum(common_points[m,row.scenario,row.period]-common_points[m,row.scenario,'baseline1991_2020'] for m in MODELS)/3
        close(difference,row.common_support_area_weighted_change_pp)
        assert abs(den-row.common_harvested_ha)<1e-6
        checks+=1
    common_contrasts=pd.read_csv(ROOT/'data/paper_study/publication/common_support_scenario_contrasts.csv')
    for row in common_contrasts.itertuples():
        expected=sum((common_points[m,row.scenario_high,row.period]-common_points[m,row.scenario_high,'baseline1991_2020'])-(common_points[m,row.scenario_low,row.period]-common_points[m,row.scenario_low,'baseline1991_2020']) for m in MODELS)/3
        close(expected,row.common_support_difference_in_changes_pp)
    # Reconstruct membership and hierarchical weights without the scoring helper.
    predictions=pd.read_parquet(ROOT/'data/paper_study/empirical_benchmarks/frozen_all_predictions.parquet')
    diagnostics=pd.read_csv(ROOT/'data/paper_study/publication/field_absolute_fit_diagnostics.csv')
    archived=pd.read_csv(ROOT/'analysis/paper_study/empirical_benchmarks/severity_metrics.csv')
    for row in diagnostics.itertuples():
        f=predictions[predictions.partition.eq(row.partition)&predictions.model.eq(row.model)].copy()
        if row.partition=='external':f=f[f.strict_location_disjoint]
        if row.leaf_scope=='upper_three':f=f[f.leaf_index<3]
        latest=f.groupby(['field_id','endpoint_series']).date.transform('max');f=f[f.date.eq(latest)].copy()
        assert not f.duplicated(['field_id','endpoint_series']).any() and len(f)==row.n
        ids=f.coordinate_year.unique();record_weights=np.zeros(len(f))
        for cluster in ids:
            cluster_ix=np.flatnonzero(f.coordinate_year.eq(cluster).to_numpy());fields=f.iloc[cluster_ix].field_id.unique()
            for field in fields:
                ix=np.flatnonzero(f.field_id.eq(field).to_numpy())
                record_weights[ix]=1/len(ids)/len(fields)/len(ix)
        close(record_weights.sum(),1.)
        y=f.value.to_numpy();pred=f.predicted_percent.to_numpy();error=pred-y
        ym=sum(record_weights*y);pm=sum(record_weights*pred)
        mse=sum(record_weights*error**2);yv=sum(record_weights*(y-ym)**2);pv=sum(record_weights*(pred-pm)**2)
        rmse=np.sqrt(mse);mae=sum(record_weights*abs(error));bias=sum(record_weights*error)
        correlation=sum(record_weights*(y-ym)*(pred-pm))/np.sqrt(yv*pv)
        close([rmse,mae,bias,1-mse/yv,correlation,ym,pm],
            [row.rmse,row.mae,row.bias,row.weighted_r_squared,row.weighted_pearson_r,row.observed_weighted_mean,row.predicted_weighted_mean])
        old=archived[archived.partition.eq(row.partition)&archived.model.eq(row.model)&archived.leaf_scope.eq(row.leaf_scope)&archived.endpoint.eq('final_leaf_assessment')]
        if row.partition=='external':old=old[old.geography.eq('strict_location_disjoint')]
        old=old.iloc[0];close([rmse,mae,bias],[old.rmse,old.mae,old.bias])
    frozen=ROOT/'analysis/paper_study/seasonal_calibration_v1/frozen_main_fit.json'
    assert sha(frozen)=='e5feed4c5520184d2128dd9c1bf7c0a0c1c9ad92d25f6180590b1e825e4b3850'
    result=dict(status='passed',numerical_checks=checks,maximum_absolute_difference=maximum,
        source_annual_parameter_partitions=810,paired_scenario_summary_rows=len(contrasts),
        source_annual_full_grid_partitions_for_common_support=810,common_support_cells=int(mask.sum()),
        field_diagnostic_rows=len(diagnostics),frozen_field_parameters_unchanged=True,
        absolute_diagnostics_are_validation_data_descriptions=True,
        source_code_sha256=sha(Path(__file__)))
    (review/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
