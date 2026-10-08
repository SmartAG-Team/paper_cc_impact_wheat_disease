"""Domain-labelled reproduction of published canopy-loss/yield evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from calibration.yield_transfer import fit_group_balanced_slope,leave_group_out

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    source=HERE/'stb_independent/foulkes2006_table2_paired_predicted_means.csv'
    frame=pd.read_csv(source)
    # Compute contrasts from printed means without propagating transcription rounding.
    for kind,protected,affected,absolute in [
            ('HAD','healthy_HAD_GLAI_day','diseased_HAD_GLAI_day','derived_HAD_loss_GLAI_day'),
            ('yield','healthy_yield_t_ha_15pct_moisture','diseased_yield_t_ha_15pct_moisture','derived_yield_loss_t_ha')]:
        frame[absolute]=frame[protected]-frame[affected]
        frame[f'derived_{kind}_loss_fraction']=frame[absolute]/frame[protected]
    frame['genetic_background']=frame.genotype.str.split().str[0].replace({'M.':'M. Huntsman'})
    assert len(frame)==8 and frame.genetic_background.nunique()==5
    weights=1/frame.groupby('genetic_background').genetic_background.transform('size')
    weights=weights/weights.sum()
    outcomes=[]
    for mode,xcol,ycol,unit in [
            ('absolute','derived_HAD_loss_GLAI_day','derived_yield_loss_t_ha','t ha-1'),
            ('relative','derived_HAD_loss_fraction','derived_yield_loss_fraction','fraction of protected yield')]:
        x=frame[xcol].to_numpy(float);y=frame[ycol].to_numpy(float)
        held=leave_group_out(x,y,frame.genetic_background.to_numpy())
        frame[f'{mode}_held_out_prediction']=held['prediction']
        frame[f'{mode}_training_only_slope']=held['slope']
        frame[f'{mode}_training_backgrounds']=held['training_groups']
        frame[f'{mode}_residual']=held['prediction']-y
        mean=np.sum(weights*y);sse=np.sum(weights*(held['prediction']-y)**2)
        tss=np.sum(weights*(y-mean)**2)
        slope=fit_group_balanced_slope(x,y,frame.genetic_background.to_numpy())
        # Independent fold reconstruction, without calling calibration functions.
        for background in frame.genetic_background.unique():
            train=frame.genetic_background.ne(background).to_numpy();test=~train
            group_sizes=frame.loc[train].groupby('genetic_background').genetic_background.transform('size').to_numpy()
            w=1/group_sizes
            expected=np.dot(w*x[train],y[train])/np.dot(w*x[train],x[train])
            np.testing.assert_allclose(frame.loc[test,f'{mode}_training_only_slope'],expected,rtol=0,atol=1e-15)
            np.testing.assert_allclose(frame.loc[test,f'{mode}_held_out_prediction'],expected*x[test],rtol=0,atol=1e-15)
        outcomes.append(dict(response=mode,contrasts=8,genetic_backgrounds=5,
            evaluation='leave-one-genetic-background-out; published aggregate treatment means',
            units=unit,group_balanced_MAE=float(np.sum(weights*np.abs(held['prediction']-y))),
            group_balanced_RMSE=float(np.sqrt(sse)),group_balanced_bias=float(np.sum(weights*(held['prediction']-y))),
            group_balanced_R2=float(1-sse/tss),full_data_slope=slope,
            slope_units='t ha-1 per top-five GLAI-day' if mode=='absolute' else 'relative yield loss per relative top-five HAD loss',
            raw_plot_validation=False,current_top3_yield_calibration=False))
    frame['evaluation_weight']=weights
    frame.to_csv(HERE/'foulkes2006_group_separated_predictions.csv',index=False)
    pd.DataFrame(outcomes).to_csv(HERE/'published_yield_response_metrics.csv',index=False)
    rust=[]
    for year,n,had,haa in [(1994,60,.63,.80),(1995,52,.73,.92)]:
        for predictor,value in [('Healthy area duration',had),('Healthy area absorption',haa)]:
            rust.append(dict(source_key='Bryson1997',doi='10.1016/S1161-0301(97)00025-7',
                disease='yellow rust',year=year,reported_treatment_combinations=n,predictor=predictor,
                metric='reported within-year R2',value=value,leaf_scope='canopy green area; fixed leaf-rank subset unverified',
                time_scope='postflowering',raw_data_available_for_reanalysis=False))
    pd.DataFrame(rust).to_csv(HERE/'bryson1997_reported_yield_fits.csv',index=False)
    leaf_rust=[]
    for year,diseased,control in [('1986-87',.84,.91),('1987-88',.67,.88)]:
        for crop,value in [('leaf rust affected',diseased),('control',control)]:
            leaf_rust.append(dict(source_key='SubbaRao1989',doi='10.1094/Phyto-79-1233',disease='leaf rust',
                crop_season=year,crop_status=crop,predictor='leaf-specific relative healthy area duration',
                outcome='tiller grain weight',metric='reported adjusted R2',value=value,
                leaf_scope='flag, penultimate, antepenultimate and fourth leaf',
                raw_data_available_for_reanalysis=False))
    pd.DataFrame(leaf_rust).to_csv(HERE/'subbarao1989_reported_yield_fits.csv',index=False)
    scope=pd.DataFrame([
        dict(source='Parker2004',disease='STB',measured_or_fitted='117 yield contrasts; 25 cultivar slopes',
            leaf_scope='top three leaves',time_scope='GS31 onward; exact terminal rule unavailable',
            predictor_units='GLAI × calendar day',yield_units='t ha-1',use='primary top-three response coefficient',
            limitation='published slopes and treatment means; original plot-level HAD-yield pairs unavailable'),
        dict(source='Foulkes2006',disease='STB and yellow rust',measured_or_fitted='8 genotype-by-treatment mean pairs; 4 experiments',
            leaf_scope='top five leaves',time_scope='postanthesis to complete senescence',
            predictor_units='GLAI × calendar day',yield_units='t ha-1 at 85% dry matter',
            use='separate aggregate transfer benchmark',limitation='unbalanced mixed-model predicted means; cannot calibrate top-three coefficient'),
        dict(source='Bryson1997',disease='yellow rust',measured_or_fitted='60 and 52 treatment combinations',
            leaf_scope='canopy green area; fixed rank subset unverified',time_scope='postflowering',
            predictor_units='HAD or absorbed total radiation',yield_units='grain dry matter',
            use='published HAD-versus-radiation comparison',limitation='within-year fit summaries; no reconstructed raw pairs'),
        dict(source='SubbaRao1989',disease='leaf rust',measured_or_fitted='10 leaf-retention treatments in 2 seasons',
            leaf_scope='top four individual leaves',time_scope='leaf-specific relative duration',
            predictor_units='normalized relative HAD',yield_units='tiller grain weight',
            use='published leaf-rank physiological evidence',limitation='one cultivar, controlled leaf retention; not a stand-scale slope')])
    scope.to_csv(HERE/'published_yield_domain_compatibility.csv',index=False)
    inputs=[Path(__file__),source,HERE/'parker2004_table4_yield_loss_long.csv',HERE/'parker2004_table5_yield_slopes.csv',
        ROOT/'calibration/yield_transfer.py',ROOT/'tests/test_published_yield_transfer.py']
    receipt=dict(status='verified',published_crop_responses_only=True,pathogen_timing_fits_added=False,
        response_calibration_outside_model=True,current_runtime_and_frozen_disease_parameters_changed=False,
        raw_or_aggregate_status_preserved=True,top_three_and_top_five_not_pooled=True,
        reported_R2_is_not_independent_validation=True,group_separated_fold_coefficients_independently_reproduced=True,
        paired_genotype_contrasts=8,background_folds=5,metrics=outcomes,
        sources={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        generated_data_sha256={p.name:sha(p) for p in sorted(HERE.glob('*.csv'))})
    (HERE/'published_response_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(outcomes))


if __name__=='__main__':main()
