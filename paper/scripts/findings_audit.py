"""Independently reconstruct the new climate, support and yield figure arithmetic."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
from .figure_registry import ROLE_NAMES

ROOT=Path(__file__).resolve().parents[2]
LOSS='GS65_85_lost_had3'
TIMING='F1_symptom_relative_anthesis_days'
KEY=['cell_id','model','scenario','period']
CENSUS=ROOT/'data/final_study/evidence/full_grid_climate/results'


def audit_findings(package,census=None):
    census=Path(census) if census is not None else CENSUS
    sources=[Path(__file__)]
    def read(path):
        path=Path(path);sources.append(path)
        return pd.read_parquet(path) if path.suffix=='.parquet' else pd.read_csv(path)
    def close(a,b):np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True)
    def folder(role):return package/'data'/ROLE_NAMES[role]
    raw=read(census/'full_grid_paired_changes_by_gcm.parquet')
    cells=read(census/'full_landuse_cell_registry.csv')
    damage=raw[raw.metric.eq(LOSS)].merge(cells[['cell_id','latitude','harvested_total_ha','environment_region']],on='cell_id',validate='many_to_one')
    damage['weight']=damage.harvested_total_ha*damage.valid_year_pairs/30
    damage['latitude_bin_lower']=np.floor(damage.latitude).astype(int)
    shown=read(folder('latitude_environment')/'latitude_by_gcm.csv')
    shown=shown[shown.metric.eq(LOSS)].set_index(['model','scenario','period','latitude_bin_lower'])
    for key,part in damage.groupby(['model','scenario','period','latitude_bin_lower']):
        finite=part.change.notna()&part.weight.gt(0)
        expected=np.average(part.loc[finite,'change'],weights=part.loc[finite,'weight']) if finite.any() else np.nan
        close(shown.loc[key,'change'],expected)
    env=read(folder('latitude_environment')/'environment_contributions_by_gcm.csv').set_index(['model','scenario','period','environment_region'])
    shares=[]
    for key,part in damage.groupby(['model','scenario','period']):
        positive=(part.weight*part.change.clip(lower=0)).sum()
        negative=(part.weight*(-part.change).clip(lower=0)).sum()
        for region,g in part.groupby('environment_region'):
            row=env.loc[(*key,region)]
            close(row.positive_contribution_share_percent,100*(g.weight*g.change.clip(lower=0)).sum()/positive)
            close(row.negative_contribution_share_percent,100*(g.weight*(-g.change).clip(lower=0)).sum()/negative)
        if key[1:] == ('ssp585','2071-2100'):
            band=part[part.latitude.ge(50)&part.latitude.lt(60)]
            shares.append(100*(band.weight*band.change.clip(lower=0)).sum()/positive)
    close(np.mean(shares),68.59231722935579)
    records={ROLE_NAMES['latitude_environment']:dict(status='passed',checks=['all1degree weighted latitude means','all environmental directional contribution shares','fixed-domain denominators'],latitude50_60_positive_share_percent=float(np.mean(shares)))}
    # Independently validate matched year-pair support and timing/response bins.
    timing=raw[raw.metric.eq(TIMING)][KEY+['change','valid_year_pairs']].rename(columns={'change':'timing','valid_year_pairs':'timing_pairs'})
    joint=damage.merge(timing,on=KEY,validate='one_to_one')
    assert joint.timing_pairs.le(joint.valid_year_pairs).all()
    assert (joint.timing_pairs.ne(joint.valid_year_pairs)).sum()==24114
    joint=joint[joint.timing_pairs.eq(joint.valid_year_pairs)&joint.valid_year_pairs.gt(0)].copy()
    joint['timing_bin_lower_days']=2*np.floor(joint.timing/2)
    curves=read(folder('climate_phenology')/'timing_cochange_bins_by_gcm.csv')
    future=joint[joint.scenario.eq('ssp585')&joint.period.eq('2071-2100')]
    for row in curves.itertuples():
        g=future[future.model.eq(row.model)&future.timing_bin_lower_days.eq(row.timing_bin_lower_days)]
        close(row.loss_change_mean_had3,np.average(g.change,weights=g.weight))
        close(row.timing_mean_days,np.average(g.timing,weights=g.weight))
    regimes=read(folder('climate_phenology')/'response_regime_cells.csv').set_index(['cell_id','scenario','period'])
    signs=damage.pivot(index=['cell_id','scenario','period'],columns='model',values='change')
    labels=np.select([signs.notna().all(axis=1)&signs.gt(0).all(axis=1),signs.notna().all(axis=1)&signs.lt(0).all(axis=1),signs.notna().all(axis=1)],['Increase','Decrease','Mixed'],default='Unavailable')
    np.testing.assert_array_equal(regimes.reindex(signs.index).response_regime,labels)
    profiles=read(folder('climate_phenology')/'response_regime_profiles_by_gcm.csv')
    base=damage[KEY+['weight']].merge(regimes.reset_index(),on=['cell_id','scenario','period'],validate='many_to_one')
    weather=raw[raw.metric.isin(['mean_temperature_grain_fill_c','rainfall_grain_fill_mm','grain_fill_days'])].merge(base,on=KEY,validate='many_to_one')
    table=profiles.set_index(['model','scenario','period','response_regime','metric'])
    for key,g in weather.groupby(['model','scenario','period','response_regime','metric']):
        finite=g.change.notna()&g.weight.gt(0)
        value=np.average(g.loc[finite,'change'],weights=g.loc[finite,'weight']) if finite.any() else np.nan
        close(table.loc[key,'change'],value)
    records[ROLE_NAMES['climate_phenology']]=dict(status='passed',checks=['timing pair counts subset and exact matched-year selection','all plotted timing-bin means','independent three-member response categories','all window climate/regime means'],qualification='within-model associations; timing enters endpoint; no causal driver identification')
    # Reconstruct both adaptation supports and every member-level summary.
    path=folder('adaptation')
    receipt=json.loads((path/'receipt.json').read_text())
    for filename,expected in receipt['output_hashes'].items():
        original=path/filename;sources.append(original)
        assert hashlib.sha256(original.read_bytes()).hexdigest()==expected, filename
    pairs=read(path/'adaptation_pairs.parquet')
    bydraw=pairs.groupby(['reference_year','spatial_draw_id']).valid_complete_season.agg(['all','size'])
    broad=bydraw['all']&bydraw['size'].eq(30)
    assert broad.sum()==1902
    index=pd.MultiIndex.from_frame(pairs[['reference_year','spatial_draw_id']])
    pairs['complete_season']=broad.reindex(index).to_numpy()
    strict=pairs.groupby(['reference_year','spatial_draw_id']).shared_interval_nonempty.agg(['all','size'])
    strict_panel=strict['all']&strict['size'].eq(30)
    assert strict_panel.sum()==1656
    close(pairs.common_three_gcm_panel,strict_panel.reindex(index).to_numpy())
    pairs['common_calendar']=pairs.common_three_gcm_panel
    fixed=read(path/'support_sensitivity_fixed_shifts.csv')
    own=['whole_proxy_days','whole_reference_days','grain_fill_days','grain_fill_mean_temperature_c','grain_fill_tmax_above30_days','grain_fill_rainfall_mm']
    shared=['shared_proxy_days','shared_reference_days','shared_window_days']
    for row in fixed[~fixed.model.eq('ensemble')].itertuples():
        g=pairs[pairs[row.support]&pairs.model.eq(row.model)&pairs.period.eq(row.period)&pairs.shift_days.eq(row.shift_days)]
        if row.environment_region!='Sampled design':g=g[g.environment_region.eq(row.environment_region)]
        assert len(g)==row.retained_draw_years
        for q in own+(shared if row.support=='common_calendar' else []):
            for column in [q,'delta_'+q]:close(getattr(row,column),np.average(g[column],weights=g.area_mean_weight))
    choices=read(path/'support_sensitivity_policy_selection.csv')
    for row in choices.itertuples():
        train=fixed[fixed.support.eq('complete_season')&fixed.period.eq('future')&fixed.environment_region.eq(row.environment_region)&~fixed.model.isin(['ensemble',row.held_model])]
        objectives=train.groupby('shift_days').whole_proxy_days.mean()
        eligible=train.unique_cells.min()>=3 and train.retained_draw_years.min()>=30 and train.whole_proxy_days.notna().all()
        expected=min(objectives.index,key=lambda shift:(objectives[shift],abs(shift),shift)) if eligible else 0
        assert expected==row.selected_shift_days
    assert choices.loc[choices.environment_region.eq('Steppic'),'selected_shift_days'].eq(-14).all()
    policy=read(path/'support_sensitivity_same_policy.csv')
    for row in policy[~policy.model.eq('ensemble')].itertuples():
        shifts=choices[choices.held_model.eq(row.model)].set_index('environment_region').selected_shift_days
        g=pairs[pairs[row.support]&pairs.period.eq('future')&pairs.model.eq(row.model)&pairs.shift_days.eq(pairs.environment_region.map(shifts))]
        if row.environment_region!='Sampled design':g=g[g.environment_region.eq(row.environment_region)]
        for q in own+(shared if row.support=='common_calendar' else []):
            for column in [q,'delta_'+q]:close(getattr(row,column),np.average(g[column],weights=g.area_mean_weight))
    for table,groups in [(fixed,['support','environment_region','period','shift_days']),(policy,['support','environment_region','period'])]:
        for key,g in table.groupby(groups):
            ensemble=g[g.model.eq('ensemble')].iloc[0];members=g[~g.model.eq('ensemble')]
            assert len(members)==3
            for q in own+(shared if key[0]=='common_calendar' else []):
                for column in [q,'delta_'+q]:
                    close([ensemble[column],ensemble[column+'_gcm_min'],ensemble[column+'_gcm_max']], [members[column].mean(),members[column].min(),members[column].max()])
    domain=policy[policy.environment_region.eq('Sampled design')&policy.model.eq('ensemble')].set_index('support')
    close(domain.loc['complete_season','delta_whole_proxy_days'],-1.891190123089872)
    records[ROLE_NAMES['adaptation']]=dict(status='passed',raw_records=len(pairs),complete_draw_years=1902,shared_calendar_draw_years=1656,checks=['both raw support masks','all fixed-shift member quantities and contrasts','training-only regional choices','same regional policy on both supports','all equal-GCM means and member ranges'],qualification='bounded sampled-design proxy sensitivity, not agronomic or yield optimum')
    annual=read(folder('yield_context')/'annual_yield_gap.csv')
    assert annual.year.tolist()==list(range(2004,2026))
    # Independent transcription checked against the archived institutional chart.
    labels=np.array([11.5,16.1,14.5,24.2,27.1,20.1,9,8.3,24.6,17.1,21.7,18.1,25,12.4,16.5,11.5,7.8,13.8,11.1,16,22.6,12.2])
    close(annual.yield_gap_t_ha,labels/10)
    assert annual.annual_trial_count.isna().all() and annual.sampling_variance_t2_ha2.isna().all()
    close(annual.yield_gap_t_ha.mean(),1.6418181818181818)
    pdf=package/'data/yield_source_evidence/arvalis_performance_2025_champagne_ardenne.pdf';sources.append(pdf)
    assert hashlib.sha256(pdf.read_bytes()).hexdigest()=='4ed9b15d8785e762123d12a1b2f1ea6483f1e225e8d8c4c52ecff366bf29b423'
    records[ROLE_NAMES['yield_context']]=dict(status='passed',annual_rows=22,checks=['all direct printed labels','unit conversion','missing trial counts and variance retained','original primary source PDF hash'],qualification='mixed foliar-disease treatment response; no new meta-analysis or STB-to-yield conversion')
    return records,sources
