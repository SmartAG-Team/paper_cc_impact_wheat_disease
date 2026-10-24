"""Independent arithmetic and source-row audit for every current article figure.

No fitting, outcome deletion or scientific-source mutation is performed.
Passing establishes faithful displays and calculations, not external model validity.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from .figure_registry import ROLE_NAMES
from .findings_audit import audit_findings

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'paper'


def close(a,b):
    np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True)


def score(frame,weights=None):
    w=frame.scientific_weight.to_numpy() if weights is None else np.asarray(weights)
    y=frame.observed_percent.to_numpy();p=frame.prediction_percent.to_numpy()
    error=p-y;mean=np.average(y,weights=w);mse=np.average(error**2,weights=w)
    variance=np.average((y-mean)**2,weights=w)
    return dict(rmse=np.sqrt(mse),mae=np.average(abs(error),weights=w),
        bias=np.average(error,weights=w),r2=1-mse/variance if variance>0 else np.nan,
        observed_mean=mean)


def check_score(row,frame):
    for metric,value in score(frame).items():
        if metric in row:close(row[metric],value)
    if 'n' in row:assert row['n']==len(frame)
    if 'weight_sum' in row:close(row['weight_sum'],frame.scientific_weight.sum())


def run():
    configuration=OUT/'figure_inputs.json'
    inputs=json.loads(configuration.read_text()) if configuration.exists() else {}
    def selected(key,default):return ROOT/inputs[key] if key in inputs else default
    sources=[]
    def csv(path):
        path=ROOT/path if not isinstance(path,Path) else path
        sources.append(path);return pd.read_csv(path)
    old_roles={'fig1':'workflow','fig2':'data_overview','fig3':'attribution_design',
        'fig4':'monitoring','fig5':'diagnostics','fig7':'climate_change',
        'fig8':'attribution_results','fig9':'production_exposure','figs1':'support'}
    def data(name):
        directory,filename=name.split('/',1)
        directory='monitoring_diagnostics' if directory=='fig6' else ROLE_NAMES.get(old_roles.get(directory,''),directory)
        return csv(OUT/'data'/directory/filename)
    p=data('fig4/predictions.csv')
    original=csv('data/final_monitoring/reference/data/fig1/predictions.csv')
    pd.testing.assert_frame_equal(original,p[original.columns],check_exact=True)
    records={}
    for name in ['fig1','fig3']:
        scheme=json.loads((OUT/f'data/{name}/schematic.json').read_text())
        assert scheme['qualitative_positions'] and scheme['font_pt']==12
        if name=='fig1':
            edges={(r['source'],r['target']) for r in scheme['edges']}
            assert ('host_trajectory','symptom_integral') in edges
            assert ('weather_routing','historical_statistical_fit') in edges
            assert ('stb_progression','host_trajectory') not in edges
            assert ('monitoring_evaluation','climate_outputs') not in edges
        else:
            assert scheme['factorial']['corners']=={'L00':[0,0],'L10':[1,0],'L01':[0,1],'L11':[1,1]}
            support=scheme['integration_support'];a=support['historical_interval_position'];b=support['future_interval_position']
            close(support['common_interval_position'],[max(a[0],b[0]),min(a[1],b[1])])
        records[name]=dict(status='passed',type='model dependencies and illustrative integration support',
            qualification='schematic positions are not measured or simulated dates')

    densities=data('fig4/density.csv');scores=data('fig4/scores.csv')
    all_sites=sorted(p.site_id.unique());indices={s:i for i,s in enumerate(all_sites)}
    multiplicities=np.random.default_rng(20261024).multinomial(len(all_sites),
        np.full(len(all_sites),1/len(all_sites)),size=2000)
    resamples=data('fig4/metric_resamples.csv')
    for design,part in p.groupby('evaluation',sort=False):
        # Independently reconstruct the equal-coordinate-year/field/code/record hierarchy.
        expected=1/(part.groupby(['site_id','season_year']).field_id.transform('nunique')*
            part.groupby('field_id').leaf_index.transform('nunique')*
            part.groupby(['field_id','leaf_index']).global_target_index.transform('size'))
        close(part.scientific_weight,expected)
        row=scores[scores.evaluation.eq(design)].iloc[0];check_score(row,part)
        ix=np.minimum(np.floor(part.observed_percent/2.5).astype(int),39)
        iy=np.minimum(np.floor(part.prediction_percent/2.5).astype(int),39)
        count=np.zeros((40,40));weight=np.zeros((40,40))
        np.add.at(count,(ix,iy),1);np.add.at(weight,(ix,iy),part.scientific_weight)
        shown=densities[densities.evaluation.eq(design)].sort_values(['observed_lower','predicted_lower'])
        close(shown.assessments,count.ravel())
        close(shown.scientific_mass_percent,100*weight.ravel()/weight.sum())
        pos=part.site_id.map(indices).to_numpy()
        draws=resamples[resamples.evaluation.eq(design)].set_index('replicate')
        for replicate,m in enumerate(multiplicities):
            weights=part.scientific_weight.to_numpy()*m[pos]
            result=score(part,weights)
            for metric in ['rmse','r2']:close(draws.loc[replicate,metric],result[metric])
        for metric in ['rmse','r2']:
            close([row[metric+'_lower95'],row[metric+'_upper95']],draws[metric].quantile([.025,.975]))
    records['fig4']=dict(status='passed',assessments_per_design=2017,
        density_formula='100 × sum of evaluation weights in bin / total evaluation weight',
        bin_width_percentage_points=2.5,metric_resamples_recomputed_from_individual_records=4000,
        qualification='fixed forecasts after adaptive internal development; not external validation')

    coverage=data('fig4/error_coverage.csv')
    for design,part in p.groupby('evaluation',sort=False):
        shown=coverage[coverage.evaluation.eq(design)].sort_values('threshold_pp')
        errors=abs(part.prediction_percent-part.observed_percent).to_numpy()
        weights=part.scientific_weight.to_numpy();order=np.argsort(errors)
        thresholds=np.unique(np.r_[np.arange(101,dtype=float),errors])
        close(shown.threshold_pp,thresholds)
        cumulative=np.r_[0,np.cumsum(weights[order])]
        expected=cumulative[np.searchsorted(errors[order],thresholds,side='right')]/weights.sum()*100
        close(shown.covered_weight_percent,expected)
        assert shown.assessments.eq(2017).all() and shown.sites.eq(121).all()
        assert shown.bootstrap_replicates.eq(2000).all() and shown.bootstrap_seed.eq(20261024).all()
        assert shown.panel.eq('c').all() and np.diff(shown.covered_weight_percent).min()>=-1e-10
        pos=part.site_id.map(indices).to_numpy()
        resampled_weights=multiplicities[:,pos]*weights[None,:]
        for threshold in [0,10,15,20,25,50,100]:
            draws=100*(resampled_weights@(errors<=threshold))/resampled_weights.sum(axis=1)
            row=shown[shown.threshold_pp.eq(threshold)].iloc[0]
            close([row.lower95,row.upper95],np.quantile(draws,[.025,.975]))
        assert shown.threshold_pp.iloc[0]==0 and shown.threshold_pp.iloc[-1]==100
        close(shown.covered_weight_percent.iloc[-1],100)
    records['fig4']['error_coverage_checks']=['every empirical jump and integer0–100 threshold','original assessment weights','paired-site intervals at seven thresholds per design']
    records['fig4']['density_display']='temporal primary; both designs retained in numerical density data'

    # Coverage includes every original F1 season; precision is phase-separated.
    onset_source=csv(OUT/'data/onset_evaluation/onset_records.csv')
    onset_drawn=data('fig4/onset_records.csv')
    pd.testing.assert_frame_equal(onset_source,onset_drawn,check_exact=True)
    all_flags=csv(OUT/'data/process_f1/all115_F1_records_with_replay.csv')
    registry=csv(OUT/'data/process_f1/field55_calendar_membership.csv').set_index('field_id')
    histories=data('fig4/onset_field_histories.csv').set_index('field_id')
    assert len(histories)==55 and set(histories.index)==set(all_flags.field_id)
    independently_derived=[]
    for field,group in all_flags.groupby('field_id',sort=True):
        group=group.sort_values('date');positive=group[group.observed_percent.gt(0)]
        sowing=pd.Timestamp(registry.loc[field,'sowing_date'])
        phase='calibration' if group.partition.iloc[0]=='calibration' else 'evaluation'
        lower=upper=np.nan
        if positive.empty:
            category='no_positive';lower=(pd.Timestamp(group.date.iloc[-1])-sowing).days+1
        else:
            upper=(pd.Timestamp(positive.date.iloc[0])-sowing).days+1
            zero=group[group.date.lt(positive.date.iloc[0])&group.observed_percent.eq(0)]
            if zero.empty:category='positive_at_first'
            else:
                category='zero_to_positive';lower=(pd.Timestamp(zero.date.iloc[-1])-sowing).days+1
        shown=histories.loc[field]
        assert shown.phase==phase and shown.observation_status==category
        predicted=group.frozen_symptom_day.unique();assert len(predicted)==1 and predicted[0]==shown.model_state_day
        early=max(lower+1-predicted[0],0) if np.isfinite(lower) else 0
        late=max(predicted[0]-upper,0) if np.isfinite(upper) else 0
        close(shown.constraint_excess_days,early+late)
        assert shown.observational_constraint_compatible==((early+late)==0)
        if category=='zero_to_positive':close(shown.signed_interval_excess_days,late-early)
        independently_derived.append(dict(field_id=field,phase=phase,season_year=group.season_year.iloc[0],
            observation_status=category,positive=not positive.empty,
            signed_error=late-early if category=='zero_to_positive' else np.nan,
            width=upper-lower if category=='zero_to_positive' else np.nan))
    independent=pd.DataFrame(independently_derived)
    assert independent.positive.sum()==52
    coverage=data('fig4/onset_coverage_by_year.csv')
    for row in coverage.itertuples():
        group=independent[independent.phase.eq(row.phase)&independent.season_year.eq(row.season_year)]
        assert row.field_seasons==group.observation_status.eq(row.observation_status).sum() and row.year_total==len(group)
    bracketed=data('fig4/onset_bracketed_histories.csv')
    assert len(bracketed)==7 and set(bracketed.field_id)==set(independent.loc[independent.observation_status.eq('zero_to_positive'),'field_id'])
    metrics=data('fig4/onset_phase_metrics.csv').set_index('phase')
    for phase,n in [('calibration',3),('evaluation',4)]:
        group=independent[independent.phase.eq(phase)&independent.observation_status.eq('zero_to_positive')]
        shown=bracketed[bracketed.phase.eq(phase)]
        assert len(group)==len(shown)==n
        values=group.set_index('field_id').signed_error.reindex(shown.field_id).to_numpy()
        close(shown.plot_y,values)
        close(shown.plot_x,(0 if phase=='calibration' else 1)+np.linspace(-.16,.16,n))
        close(metrics.loc[phase,'mean_interval_excess_days'],np.mean(abs(values)))
        assert metrics.loc[phase,'inside_bracket']==int((values==0).sum())
    close(metrics.loc[['calibration','evaluation'],'mean_interval_excess_days'],[1.,3.75])
    records['fig4'].update(F1_field_seasons=55,positive_F1_field_seasons=52,
        onset_calibration_histories=3,onset_evaluation_histories=4,
        onset_checks=['all55 original F1 seasons and observed censoring classes','all7 bracketed signed errors','phase denominators and timing-point coordinates'],
        onset_scope='calibration and original2019 evaluation kept separate; no pooled validation or exact-date MAE')

    eligibility=data('fig4/onset_eligibility_2019.csv')
    flag=csv(OUT/'data/process_f1/evaluation70_F1_records_with_replay.csv')
    cohort=data('fig2/assessments.csv');cohort=cohort[cohort.season_year.eq(2019)]
    assert len(eligibility)==45 and set(eligibility.field_id)==set(cohort.field_id)
    assert set(flag.global_target_index)==set(cohort[cohort.leaf_index.eq(0)].global_target_index)
    independently_classified={}
    for field,group in cohort.groupby('field_id'):
        history=flag[flag.field_id.eq(field)].sort_values('date')
        positives=history[history.observed_percent.gt(0)]
        if history.empty:status='no_qualified_F1_observation'
        elif positives.empty:status='no_positive_F1_assessment'
        elif history[history.date.lt(positives.date.iloc[0])&history.observed_percent.eq(0)].empty:
            status='F1_positive_at_first_assessment'
        else:status='two_sided_F1_onset'
        independently_classified[field]=status
    assert eligibility.set_index('field_id').onset_eligibility.to_dict()==independently_classified
    assert set(eligibility.loc[eligibility.onset_eligibility.eq('two_sided_F1_onset'),'field_id'])==set(onset_drawn.field_id)
    records['fig4']['onset_cohort_reconciliation']=dict(all_2019_fields=45,qualified_F1_fields=33,
        status_counts=eligibility.onset_eligibility.value_counts().to_dict(),selection='observed chronology only')

    chronology=csv('data/final_monitoring/cohort/training_observation_ledger.csv')
    overview=data('fig2/assessments.csv')
    pd.testing.assert_frame_equal(chronology,overview,check_exact=False,atol=1e-10,rtol=1e-12)
    overview_sites=data('fig2/sites.csv');overview_fields=data('fig2/fields.csv')
    assert len(overview_sites)==128 and len(overview_fields)==218
    overview_years=data('fig2/years.csv').set_index(['source','season_year'])
    for key,group in chronology.groupby(['source','season_year']):
        assert overview_years.loc[key,'assessments']==len(group)
    overview_hist=data('fig2/severity.csv')
    for source,group in chronology.groupby('source'):
        expected,_=np.histogram(group.observed_percent,bins=np.arange(0,101,10))
        close(overview_hist[overview_hist.source.eq(source)].sort_values('lower_percent').assessments,expected)
    overview_visits=data('fig2/visits.csv').set_index(['source','visits'])
    for row in overview_fields.itertuples():
        group=chronology[chronology.field_id.eq(row.field_id)]
        assert row.visits==group.date.nunique() and row.assessments==len(group)
    for key,group in overview_fields.groupby(['source','visits']):
        assert overview_visits.loc[key,'field_seasons']==len(group)
    records['fig2']=dict(status='passed',assessments=2304,field_seasons=218,sites=128,
        checks=['unique retained identities','source/year counts','native-score histogram','unique dates per field-season','original coordinates'],
        qualification='descriptive sample counts; native endpoint differences retained; only untreated zero-spray controls')
    dates=pd.to_datetime(chronology.date);start=dates.groupby(chronology.field_id).min();end=dates.groupby(chronology.field_id).max()
    close(p.first_visit,pd.to_datetime(p.date).eq(p.field_id.map(start)))
    close(p.last_visit,pd.to_datetime(p.date).eq(p.field_id.map(end)))
    for name,groupkey in [('fig5/field_metrics.csv','field_id'),('fig6/site_metrics.csv','site_id')]:
        shown=data(name).set_index(['evaluation',groupkey])
        for key,part in p.groupby(['evaluation',groupkey]):check_score(shown.loc[key],part)
    contexts=data('fig5/visit_context.csv');bias=data('fig5/severity_bias.csv');calibration=data('fig4/calibration.csv')
    assert calibration.panel.eq('b').all()
    rng=np.random.default_rng(20261023)
    for design,part in p.groupby('evaluation',sort=False):
        masks={'First visit':part.first_visit,'Last visit':part.last_visit,
            'Intermediate':~part.first_visit&~part.last_visit,'Only visit':part.first_visit&part.last_visit}
        for label,mask in masks.items():
            row=contexts[contexts.evaluation.eq(design)&contexts.visit_context.eq(label)].iloc[0]
            check_score(row,part[mask])
        for row in bias[bias.evaluation.eq(design)].to_dict('records'):
            mask=part.observed_percent.le(row['bin_upper']) & (part.observed_percent.gt(row['bin_lower']) if row['bin_lower'] else part.observed_percent.ge(0))
            check_score(row,part[mask])
        sites=sorted(part.site_id.unique());mapping={s:i for i,s in enumerate(sites)}
        mult=rng.multinomial(len(sites),np.full(len(sites),1/len(sites)),size=1000)
        for row in calibration[calibration.evaluation.eq(design)].to_dict('records'):
            group=part[np.minimum(np.floor(part.prediction_percent/10).astype(int),9).eq(row['bin_lower']//10)]
            w=group.scientific_weight.to_numpy();y=group.observed_percent.to_numpy()
            close(row['predicted_mean'],np.average(group.prediction_percent,weights=w));close(row['observed_mean'],np.average(y,weights=w))
            assert row['n']==len(group) and row['sites']==group.site_id.nunique()
            if row['sites']>=4:
                rw=mult[:,group.site_id.map(mapping).to_numpy()]*w
                means=np.divide(rw@y,rw.sum(axis=1),out=np.full(1000,np.nan),where=rw.sum(axis=1)>0)
                close([row['lower95'],row['upper95']],np.nanquantile(means,[.025,.975]))
    records['fig4']['forecast_calibration_checks']=['twenty fixed prediction bins','original weights','1000-site-resample percentile intervals']
    records['fig5']=dict(status='passed',field_seasons_per_design=192,
        checks=['outcome-band bias','complete-chronology visit groups','within-field RMSE'],
        qualification='field ECDF assigns one vote per field; pooled headline scores use evaluation weights')
    path=selected('climate_members',ROOT/'data/final_study/evidence/full_grid_climate/results/full_grid_paired_changes_by_gcm.parquet');sources.append(path)
    members=pd.read_parquet(path,filters=[('metric','==','GS65_85_lost_had3'),('period','==','2071-2100')])
    grid=data('fig7/grid.csv').set_index(['cell_id','scenario']);models=sorted(members.model.unique())
    for column,shown in [('change','mean_change'),('reference_on_common','reference_on_common'),('future_on_common','future_on_common')]:
        matrix=members.pivot(index=['cell_id','scenario'],columns='model',values=column).reindex(grid.index)[models]
        complete=np.isfinite(matrix).all(axis=1)
        close(grid[shown],matrix.mean(axis=1).where(complete))
        if column=='change':
            close(grid.available_gcms,np.isfinite(matrix).sum(axis=1))
            close(grid.positive_gcm_count,matrix.gt(0).sum(axis=1));close(grid.negative_gcm_count,matrix.lt(0).sum(axis=1))
    records['fig7']=dict(status='passed',native_cells=14941,
        checks=['equal three-member change/reference/future means','strict three-member availability','individual member direction counts'],
        qualification='normalized affected-tissue days, not percentages, physiological loss or grain loss; agreement is not probability')

    path=selected('attribution',ROOT/'data/final_analyses/reference/data/attribution')/'attribution_pairs.parquet';sources.append(path)
    pairs=pd.read_parquet(path);corners=pairs[['y00','y10','y01','y11']].to_numpy()
    finite=np.isfinite(corners).all(axis=1)
    identity=['model','reference_year','future_year','spatial_draw_id']
    flags=pd.Series(finite,index=pairs.index).groupby([pairs[k] for k in identity]).transform('all')
    assert np.array_equal(flags,pairs.common_across_settings)
    selected=pairs[flags].copy()
    selected['ind_weather']=.5*((selected.y10-selected.y00)+(selected.y11-selected.y01))
    selected['ind_host']=.5*((selected.y01-selected.y00)+(selected.y11-selected.y10))
    selected['ind_net']=selected.y11-selected.y00
    summary=data('fig8/attribution.csv').set_index(['setting','model'])
    for key,group in selected.groupby(['setting','model']):
        for metric in ['weather','host','net']:close(summary.loc[key,metric+'_had_days'],np.average(group['ind_'+metric],weights=group.area_mean_weight))
        requested=pairs[pairs.setting.eq(key[0])&pairs.model.eq(key[1])]
        close(summary.loc[key,'common_area_time_fraction'],group.area_mean_weight.sum()/requested.area_mean_weight.sum())
    for setting,group in summary.reset_index().groupby('setting'):
        for metric in ['weather_had_days','host_had_days','net_had_days']:
            close(group.loc[group.model.eq('ensemble'),metric],group.loc[~group.model.eq('ensemble'),metric].mean())
        assert int(group.loc[group.model.eq('ensemble'),'valid_pairs'].iloc[0])==4117
    records['fig8']=dict(status='passed',checks=['four-corner Shapley arithmetic','identical cross-setting valid pairs','area-weighted model estimates','equal-member means'],
        qualification='three-model minimum–maximum ranges are not confidence intervals; whole/shared-time integration are distinct estimands')
    support=data('figs1/support.csv')
    for row in support.to_dict('records'):close(row['common_area_time_fraction'],summary.loc[(row['setting'],row['model']),'common_area_time_fraction'])
    records['figs1']=dict(status='passed',qualification='weighted design support is distinct from unweighted pair retention and effect sizes')

    production=csv('data/final_study/evidence/food_security_exposure/production_grid_input.csv').set_index('cell_id')
    thresholds=data('fig9/thresholds.csv');regions=data('fig9/all_regions.csv')
    columns={'unanimous_increase_tonnes':'up','unanimous_decrease_tonnes':'down','mixed_or_zero_tonnes':'mixed','unavailable_tonnes':'missing'}
    for row in pd.concat([thresholds,regions]).to_dict('records'):
        subset=members[members.scenario.eq(row['scenario'])]
        values=subset.pivot(index='cell_id',columns='model',values='change').reindex(production.index)[models]
        n=subset.pivot(index='cell_id',columns='model',values='valid_year_pairs').reindex(production.index)[models]
        complete=np.isfinite(values).all(axis=1)&n.ge(1).all(axis=1)
        up=complete&values.gt(row['threshold_days']).all(axis=1);down=complete&values.lt(-row['threshold_days']).all(axis=1)
        masks=dict(up=up,down=down,mixed=complete&~up&~down,missing=~complete)
        weights=production.production_total_tonnes.copy()
        if row['domain_type']=='environment_region':weights=weights.where(production.environment_region.eq(row['domain']),0)
        close(row['baseline_production_tonnes'],weights.sum())
        for col,mask in columns.items():close(row[col],weights[masks[mask]].sum())
        close(row['unanimous_increase_share_percent'],100*weights[up].sum()/weights.sum())
    display=data('fig9/display_regions.csv')
    close(display.baseline_production_tonnes.sum(),production.production_total_tonnes.sum())
    close(display[list(columns)].sum(axis=1),display.baseline_production_tonnes)
    records['fig9']=dict(status='passed',checks=['strict per-member thresholds','complete fixed production denominator','region categories and mass balance'],
        qualification='baseline production in affected cells, not future production or tonnes of grain lost')
    graphic=data('graphical_abstract/attribution_summary.csv')
    for row in graphic.to_dict('records'):close(row['host_had_days'],summary.loc[(row['setting'],'ensemble'),'host_had_days'])
    records['graphical_abstract']=dict(status='passed',qualification='schematic intervals; numerical host contributions use identical common-pair support')

    raw_path=ROOT/'data/final_study/inputs/stb_collection_20261009/sources/other_field_sources/corteva-2014-2018/corteva-wheat-diseases.txt';sources.append(raw_path)
    raw_source=pd.read_csv(raw_path,sep='\t',low_memory=False)
    corteva=chronology[chronology.source.eq('Corteva')]
    for row in corteva.itertuples():
        raw_rows=raw_source.iloc[[int(float(i))-2 for i in row.source_rows.split(';')]]
        assert raw_rows.Field.eq(str(row.source_trial_id)).all() and raw_rows.Date.eq(row.date).all()
        assert raw_rows.NumSprays.eq(0).all() and raw_rows.Treatment.eq('Untreated').all()
        assert raw_rows.PestCode.eq('SEPTTR').all() and raw_rows.EvaluationType.eq('INFECT').all()
        close(raw_rows.Value.mean(),row.observed_percent)
    basf_path=ROOT/'data/gcb_preparation_20261023/source_snapshots/basf-wheat-diseases.txt';sources.append(basf_path)
    basf_raw=pd.read_csv(basf_path,sep='\t')
    basf=basf_raw.loc[basf_raw.Organism.eq('SEPTTR')&basf_raw.Parameter.eq('INFECT')&
        basf_raw.Treatment.eq('Untreated')&basf_raw.NumSprays.eq(0)].copy()
    basf['leaf_index']=pd.to_numeric(basf.PlantPart.str.extract(r'^LEAF,\s*(\d+)',expand=False),errors='coerce')-1
    basf['field_id']='basf-wheat-diseases|'+basf.TrialId.astype(str)+'|'+pd.to_datetime(basf.Date).dt.year.astype(str)
    basf_means=basf.groupby(['field_id','Date','leaf_index']).Value.mean()
    for row in chronology[chronology.source.eq('BASF')].itertuples():
        close(basf_means.loc[(row.field_id,row.date,row.leaf_index)],row.observed_percent)
    inventory=data('fig2/treatment_inventory.csv')
    for source,raw,organism,parameter,leaf in [('BASF',basf_raw,'Organism','Parameter','PlantPart'),
            ('Corteva',raw_source,'PestCode','EvaluationType','SamplingUnit')]:
        eligible_raw=raw.loc[raw[organism].eq('SEPTTR')&raw[parameter].eq('INFECT')&raw[leaf].str.startswith('LEAF',na=False)]
        for (label,sprays),group in eligible_raw.groupby(['Treatment','NumSprays']):
            row=inventory[inventory.source.eq(source)&inventory.treatment_label.eq(label)&inventory.num_sprays.eq(sprays)].iloc[0]
            assert row.raw_leaf_INFECT_rows==len(group)
            assert row.eligible_treatment_label==(label=='Untreated' and sprays==0)
    records={ROLE_NAMES[old_roles.get(name,name)]:value for name,value in records.items()}
    additional,additional_sources=audit_findings(OUT)
    records.update(additional);sources.extend(additional_sources)
    assert set(records)==set(ROLE_NAMES.values())
    receipt=dict(status='passed',figures=records,raw_Corteva_assessments_reconstructed=len(corteva),
        raw_BASF_assessments_reconstructed=584,treatment_eligibility_verified_for_all_retained_records=True,
        auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        source_row_convention='one-based physical text line including header; pandas row index = source line minus two',
        source_sha256={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(set(sources))},
        validation_boundary='Source and arithmetic fidelity of figures; no independent external confirmation, physiological-loss calibration or model refit is implied')
    (OUT/'verification/figure_scientific_audit.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(status='passed',figures=list(records),raw_Corteva_assessments_reconstructed=len(corteva)),indent=2))
    return receipt


if __name__=='__main__':run()
