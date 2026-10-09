"""Distinct displays of prediction support, climate response, exposure and timing."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
import geopandas as gpd
from .data import ROOT, GRID, DERIVED, CANOPY, ONSET, FREQUENCY, SEVERITY, SCENARIOS, REGIONS, pooled, grid_data
from .figures import (BLUE, GREEN, GOLD, GREY, COLORS, LABELS, export, style,
                      clean, map_axes, discrete_scale, onset_change_scale, raster)
from .timing import read as timing_data


def prediction_support(destination):
    fig, axes = plt.subplots(2, 2, figsize=(9.4, 6.8))
    fig.subplots_adjust(left=.11, right=.97, bottom=.12, top=.93, hspace=.60, wspace=.55)
    phenology = pd.read_csv(ROOT/'analysis/paper_study/full_validation_20261007/german_stage_metrics_all.csv')
    phenology = phenology[phenology.cohort.eq('testing') & phenology.denominator.eq('three_model_shared')]
    ax = axes[0, 0]
    stages=['10','31','51','85']
    for offset,model,color,label in [(-.13,'T-P-V',BLUE,'T–P–V model'),(.13,'GDD',GREY,'Thermal time')]:
        part=phenology[phenology.model.eq(model)].set_index('stage').loc[stages]
        ax.scatter(part.mae_days,np.arange(4)+offset,color=color,s=28,label=label,zorder=3)
    counts=phenology[phenology.model.eq('T-P-V')].set_index('stage').loc[stages,'matched']
    labels=['Emergence','Stem elongation','Heading','Soft dough']
    ax.set_yticks(np.arange(4),[f'{label} · BBCH {stage}\nn={int(n):,}' for label,stage,n in zip(labels,stages,counts)])
    ax.invert_yaxis(); ax.set_xlim(0,22)
    ax.legend(frameon=False,fontsize=8,loc='lower right')
    ax.set_xlabel('Stage-date MAE (days)')
    ax.set_title('a   Crop phenology · station test', loc='left', weight='bold', fontsize=10, pad=12)
    clean(ax)
    sources = [phenology.assign(panel='a', endpoint='phenology')]
    for ax, endpoint, keys, title, unit, letter in [
        (axes[0, 1], 'symptom_onset', ['model', 'benchmark'], 'b   Symptom-onset prediction', 'Mean interval-excess error (days)', 'b'),
        (axes[1, 0], 'severity', ['model_rmse', 'benchmark_rmse'], 'c   Disease-severity prediction', 'RMSE (percentage points)', 'c')]:
        records = []
        for y, (key, color) in enumerate(zip(keys, [BLUE, GREY])):
            r = pooled(endpoint, key); records.append(r.to_dict())
            ax.errorbar(r.value, y, xerr=[[r.value-r.lower95], [r.upper95-r.value]],
                marker='o', color=color, capsize=4, ms=6, lw=1.2)
            ax.text(r.value, y-.22, f'{r.value:.2f}', ha='center', color=color)
        ax.set_yticks([0, 1], ['Seasonal model', 'Benchmark'])
        ax.set_ylim(1.5, -.6); ax.set_xlim(left=0)
        ax.set_xlabel(unit, fontsize=9); clean(ax)
        ax.set_title(title, loc='left', weight='bold', fontsize=10, pad=12)
        sources.append(pd.DataFrame(records).assign(panel=letter))
    nordic = pd.read_csv(ROOT/'analysis/paper_study/nordic_yield_validation_20261009/model_comparison_metrics.csv')
    n = nordic[nordic.domain.eq('endpoint') & nordic.model.eq('affine_ridge')].iloc[0]
    tunisia = pd.read_csv(ROOT/'analysis/paper_study/crop_damage_validation_20261009/model_comparison_metrics.csv')
    t = tunisia[tunisia.domain.eq('absolute_year_transfer') & tunisia.model.eq('two_leaf_ridge')].iloc[0]
    german = pd.read_csv(ROOT/'analysis/paper_study/crop_damage_validation_20261009/briwecs_model_comparison_metrics.csv')
    g = german[german.domain.eq('unprotected_endpoint') & german.validation.eq('leave_location_out') & german.model.eq('severity_ridge')].iloc[0]
    swiss=pd.read_csv(ROOT/'analysis/paper_study/yield_transfer_audit_20261009/swiss_model_comparison.csv')
    swiss=swiss[swiss.split.eq('leave_one_site_out')].set_index('model')
    response = pd.DataFrame([
        dict(dataset='Nordic · 5 trials', model_rmse=n.RMSE_pp, benchmark_rmse=n.training_mean_RMSE_pp, n=int(n.contrasts), units='percentage points'),
        dict(dataset='Tunisia · 2 years', model_rmse=t.RMSE, benchmark_rmse=t.baseline_RMSE, n=int(t.n), units='t ha-1'),
        dict(dataset='Germany · 5 locations', model_rmse=g.RMSE_pp, benchmark_rmse=g.baseline_RMSE_pp, n=int(g.n), units='percentage points'),
        dict(dataset='Switzerland · 5 locations',model_rmse=swiss.loc['stb_only','rmse_environment_equal_t_ha'],
             benchmark_rmse=swiss.loc['training_mean','rmse_environment_equal_t_ha'],n=724,units='t ha-1')])
    response['rmse_ratio'] = response.model_rmse/response.benchmark_rmse
    ax = axes[1, 1]
    ax.scatter(response.rmse_ratio, np.arange(4), color=BLUE, s=35, zorder=3)
    ax.axvline(1, color=GREY, ls='--', lw=1)
    for y, r in enumerate(response.itertuples()):
        ax.text(r.rmse_ratio, y-.22, f'{r.rmse_ratio:.3f}', ha='center', fontsize=9)
    ax.set_yticks(np.arange(4), response.dataset); ax.set_ylim(3.5, -.6)
    ax.set_xlim(.77, 1.08); ax.set_xticks([.8, .9, 1.]); clean(ax)
    ax.set_xlabel('Yield-response RMSE / benchmark RMSE', fontsize=9)
    ax.set_title('d   Severity-to-yield transfer', loc='left', weight='bold', fontsize=10, pad=12)
    ax.text(.5, -.30, 'Below 1: lower error; above 1: higher error', transform=ax.transAxes, ha='center', fontsize=8, color=GREY)
    sources.append(response.assign(panel='d', endpoint='severity_yield'))
    stem = 'fig1_predictive_support'
    export(fig, destination, stem)
    pd.concat(sources, ignore_index=True).to_csv(destination/f'{stem}.csv', index=False)
    return stem, ('Figure 1 | Predictive support for crop, disease and yield-response components. '
        '(a) Stage-specific mean absolute error on 40,878 matched German station-test events. T–P–V denotes temperature, photoperiod and vernalization; years are not held out. '
        '(b) Symptom-onset interval-excess error on 136 histories. (c) Severity RMSE on 2,033 assessments. '
        'Disease comparisons exclude calibration records; whiskers are 95% paired coordinate-year bootstrap intervals for fixed predictions. '
        '(d) Severity-based yield-response RMSE divided by the corresponding held-out training-mean RMSE: '
        '62 Nordic treatment contrasts, 82 Tunisian plot-season yields, 3,264 German cultivar–management comparisons and 724 Swiss plot harvests. '
        'Validation excludes entire trials, years or locations, respectively. Nordic and German outcomes are relative management-associated yield responses; '
        'Tunisia and Switzerland use absolute yield. The Swiss analysis is exploratory and retains ordinal ratings as categories. '
        'These regressions assess transferability of severity–yield associations and do not validate the canopy-to-yield conversion. '
        'The panels use distinct populations and endpoints; lower errors in one component do not validate the full crop–disease–yield chain.')


def climate_response(destination, data, regions, countries, cells):
    selected = data[data.metric.eq(CANOPY) & data.scenario.eq('ssp585') & data.period.eq('2071-2100')].copy()
    fig = plt.figure(figsize=(9.5, 7.7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], left=.09, right=.97, bottom=.10, top=.94, hspace=.60, wspace=.23)
    ax = fig.add_subplot(gs[0, 0]); map_axes(ax, countries, cells)
    norm, cmap, ticks = discrete_scale(selected.mean_change.to_numpy(), [.5, 1, 2, 5, 10], 2)
    im = raster(ax, selected, 'mean_change', cmap=cmap, norm=norm)
    ax.set_title('a   Canopy damage · late-century SSP5–8.5', loc='left', weight='bold', fontsize=10, pad=12)
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.045, pad=.15, shrink=.96)
    cb.set_ticks(ticks); cb.set_label('Change in normalized HAD loss (days)', fontsize=8.5)
    ax = fig.add_subplot(gs[0, 1]); map_axes(ax, countries, cells, mask_color='#697780')
    selected['agreement'] = np.where(selected.positive_gcm_count.eq(3), 1, np.where(selected.negative_gcm_count.eq(3), -1, 0)).astype(float)
    selected.loc[selected.available_gcms.ne(3), 'agreement'] = np.nan
    im = raster(ax, selected, 'agreement', cmap=ListedColormap([BLUE, '#d9d9d9', GOLD]), vmin=-1.5, vmax=1.5)
    ax.set_title('b   Agreement among three climate models', loc='left', weight='bold', fontsize=10, pad=12)
    ax.set_yticklabels([])
    cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.045, pad=.15, shrink=.96, ticks=[-1, 0, 1])
    cb.ax.set_xticklabels(['All decrease', 'Mixed / zero', 'All increase'], fontsize=8)
    cb.set_label('Direction of canopy-damage change', fontsize=8.5)
    ax = fig.add_subplot(gs[1, :])
    european = regions[regions.environment_region.eq('Europe') & regions.metric.eq(CANOPY)]
    for i, (scenario, label, color) in enumerate(zip(SCENARIOS, LABELS, COLORS)):
        q = european[european.scenario.eq(scenario)].sort_values('period')
        x = np.arange(2)+(i-1)*.16
        ax.errorbar(x, q.mean_change, yerr=[q.mean_change-q.gcm_min, q.gcm_max-q.mean_change],
                    marker=['o', 's', '^'][i], ls='none', color=color, capsize=4, label=label)
    ax.axhline(0, color=GREY, ls='--', lw=.8)
    ax.set_xticks([0, 1], ['2031–2060', '2071–2100']); ax.set_xlim(-.55, 1.55)
    ax.set_ylabel('Change in normalized\nHAD loss (days)', fontsize=9)
    ax.grid(axis='y', color='#e8ecee', lw=.65); ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=3, loc='upper left', fontsize=9)
    ax.set_title('c   European mean and climate-model range', loc='left', weight='bold', fontsize=10, pad=12)
    stem = 'fig2_climate_damage_and_agreement'
    export(fig, destination, stem)
    selected.to_csv(destination/f'{stem}_grids.csv.gz', index=False, compression={'method': 'gzip', 'mtime': 0})
    european.to_csv(destination/f'{stem}_europe.csv', index=False)
    return stem, ('Figure 2 | Projected canopy damage, climate-model agreement and emissions pathways. '
        '(a) Three-model mean change in normalized loss of healthy-area duration (HAD) under SSP5–8.5 in 2071–2100 relative to 1991–2020. '
        'HAD loss is divided by maximum reference upper-three-leaf LAI and expressed in equivalent days of canopy function lost during flowering to soft dough. '
        '(b) Direction of change across climate models; grey indicates differing signs or zero, and dark-grey wheat cells lack complete estimates. '
        '(c) Harvested-area-weighted European changes under three emissions pathways in two future periods. '
        'Points show three-model means; whiskers span model estimates, not confidence intervals. '
        'The fixed all-wheat mask contains 14,941 cells with 14,932 imposed winter-wheat rainfed calendars. '
        'The values describe conditional canopy damage; neither model agreement nor the scenario contrast establishes harvested-yield loss. '
        'Supplementary Fig. S25 contains all six scenario-period maps.')


def production_exposure(destination):
    from analysis.paper_study.food_security_exposure_20261009.exposure import read
    data = read()['domain_summary'].query("analysis_mode == 'primary_any_paired_year'")
    late = data[data.period.eq('2071-2100')]
    regions = late[late.domain_type.eq('environment_region') & late.scenario.eq('ssp585') & late.domain.isin(REGIONS)]
    regions = regions.sort_values('baseline_production_tonnes', ascending=False).copy()
    y = np.arange(len(regions))
    fig = plt.figure(figsize=(10.4, 7.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.7, 1], left=.13, right=.97, bottom=.10, top=.94, hspace=.67, wspace=.23)
    ax = fig.add_subplot(gs[0, 0])
    ax.errorbar(regions.production_weighted_change, y,
        xerr=[regions.production_weighted_change-regions.gcm_min, regions.gcm_max-regions.production_weighted_change],
        color=BLUE, marker='o', ls='none', capsize=3, ms=5)
    ax.set_yticks(y, regions.domain); ax.set_ylim(len(regions)-.5,-.5); clean(ax)
    ax.axvline(0, color=GREY, ls='--', lw=.8)
    ax.set_xlabel('Production-weighted HAD-loss change (days)', fontsize=8.5)
    ax.set_title('a   Regional impact intensity', loc='left', weight='bold', fontsize=10, pad=13)
    ax = fig.add_subplot(gs[0, 1]); left = np.zeros(len(regions))
    regions['other_model_sign_production_tonnes'] = regions.baseline_production_tonnes - regions.all3positive_production_tonnes - regions.all3negative_production_tonnes - regions.unavailable_production_tonnes
    stacks = [('all3positive_production_tonnes', GOLD, 'All models: increase'),
              ('other_model_sign_production_tonnes', '#d9d9d9', 'Mixed signs / zero'),
              ('all3negative_production_tonnes', BLUE, 'All models: decrease'),
              ('unavailable_production_tonnes', '#697780', 'Unavailable')]
    for col, color, label in stacks:
        values = regions[col].to_numpy()/1e6
        ax.barh(y, values, left=left, height=.62, color=color, label=label)
        left += values
    ax.set_yticks(y, []); ax.set_ylim(len(regions)-.5,-.5); clean(ax)
    ax.set_xlabel('Baseline wheat production (million tonnes)', fontsize=8.6)
    ax.set_title('b   Production exposed to each response', loc='left', weight='bold', fontsize=10, pad=13)
    ax.legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(.5, -.20), fontsize=8)
    ax = fig.add_subplot(gs[1, :])
    europe = late[late.domain_type.eq('Europe')].set_index('scenario').loc[SCENARIOS].reset_index()
    x = np.arange(3)
    ax.bar(x-.18, europe.all3positive_production_tonnes/1e6, width=.32, color=GOLD, label='All three models increase')
    ax.bar(x+.18, europe.ensemble_increasing_production_tonnes/1e6, width=.32, color='#d1b789', label='Ensemble mean increases')
    for offset, column in [(-.18, 'all3positive_production_tonnes'), (.18, 'ensemble_increasing_production_tonnes')]:
        for pos, value in zip(x+offset, europe[column]/1e6):
            ax.text(pos, value+2, f'{value:.1f}', ha='center', fontsize=9)
    ax.set_xticks(x, LABELS); ax.set_ylim(0, max(europe.ensemble_increasing_production_tonnes/1e6)*1.27)
    ax.set_ylabel('Production exposed to increasing\ndamage (million tonnes)', fontsize=8.8)
    ax.set_title('c   European exposure across emissions pathways', loc='left', weight='bold', fontsize=10, pad=12)
    ax.legend(frameon=False, ncol=2, loc='upper left', fontsize=8.5)
    ax.grid(axis='y', color='#e8ecee', lw=.65); ax.set_axisbelow(True)
    stem = 'fig4_wheat_production_exposure'
    export(fig, destination, stem)
    regions.to_csv(destination/f'{stem}_regions.csv', index=False)
    europe.to_csv(destination/f'{stem}_europe.csv', index=False)
    return stem, ('Figure 4 | Production exposure changes the regional interpretation of disease impacts. '
        '(a) Production-weighted change in normalized HAD loss under SSP5–8.5 in 2071–2100 relative to 1991–2020. '
        'Points are equal three-climate-model means after production and valid-season weighting; whiskers span model estimates. '
        '(b) Fixed SPAM2020 production in the same environmental regions, partitioned by agreement in the sign of cell-level changes. '
        'Regions are ordered by baseline production. Mixed signs and zero responses are combined for display and retained separately in Source Data. '
        '(c) European baseline production in cells with positive changes in all three models or in their ensemble mean under each emissions pathway. '
        'The two criteria overlap and are shown as separate bars, not additive categories. '
        'All amounts are baseline production located in exposed cells, not estimated tonnes lost or future production. '
        'The fixed all-wheat baseline is represented by imposed winter-wheat rainfed calendars; missing responses remain explicit in the denominators. '
        'Eight named regions are displayed; European totals also include outside-region and unassigned cells.')


def common_timing_figure(destination):
    timeline = timing_data(); europe = timeline[timeline.environment_region.eq('Europe')].set_index('event')
    source = ROOT/'analysis/paper_study/nature_food_fix_20261007/climate/decomposition/supported_forcing_ensemble_decomposition.csv'
    components = pd.read_csv(source).set_index('component')
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={'width_ratios': [1.15, 1]})
    fig.subplots_adjust(left=.13, right=.97, top=.84, bottom=.22, wspace=.60)
    events = ['symptom_days', 'flowering_days', 'soft_dough_days']
    ax = axes[0]
    for y, event in enumerate(events):
        row = europe.loc[event]
        ax.annotate('', xy=(row.future, y), xytext=(row.reference, y), arrowprops=dict(arrowstyle='->', color='#a6aeb2', lw=2))
        ax.scatter(row.reference, y, color=GREY, s=40, marker='o', zorder=3)
        ax.scatter(row.future, y, color=BLUE, s=40, marker='s', zorder=3)
        ax.text((row.reference+row.future)/2, y-.23, f'{row.change:+.2f} days', ha='center', fontsize=9)
    ax.set_yticks([0, 1, 2], ['Flag-leaf symptoms', 'Flowering', 'Soft dough']); ax.set_ylim(2.5, -.7)
    ax.set_xlim(200, 300); ax.set_xlabel('Elapsed days after sowing', fontsize=9); clean(ax)
    ax.set_title('a   Crop and disease timing', loc='left', weight='bold', fontsize=10, pad=12)
    ax.legend(handles=[Line2D([], [], marker='o', color=GREY, ls='none', label='1991–2020'),
                       Line2D([], [], marker='s', color=BLUE, ls='none', label='2071–2100')],
              loc='upper center', bbox_to_anchor=(.5, 1.25), frameon=False, ncol=2, fontsize=8.5)
    ax.text(.5, -.29, f'Symptoms relative to flowering: {europe.loc["symptom_relative_flowering_days", "change"]:+.2f} days',
            transform=ax.transAxes, ha='center', fontsize=9, color=BLUE)
    ax = axes[1]
    for y, key, color in zip(range(3), ['weather_shapley', 'host_shapley', 'total_change'], [GOLD, GREEN, BLUE]):
        row = components.loc[key]
        ax.plot([row.gcm_min, row.gcm_max], [y, y], color='#b1b6ba', lw=2.4)
        ax.errorbar(row.gcm_mean, y, xerr=row.spatial_mcse_gcm_mean, marker='o', color=color, capsize=3, ms=6)
        ax.text(row.gcm_mean, y-.22, f'{row.gcm_mean:+.2f}', ha='center', color=color)
    ax.set_yticks([0, 1, 2], ['Weather', 'Wheat phenology', 'Net change']); ax.set_ylim(2.5, -.7)
    ax.axvline(0, color=GREY, ls='--', lw=.8); clean(ax)
    ax.set_xlabel('Contribution to normalized HAD loss (days)', fontsize=8.7)
    ax.set_title('b   Opposing contributions', loc='left', weight='bold', fontsize=10, pad=12)
    offset = -100*components.loc['host_shapley', 'gcm_mean']/components.loc['weather_shapley', 'gcm_mean']
    ax.text(.5, -.29, f'Phenology offsets {offset:.1f}% of the weather contribution', transform=ax.transAxes, ha='center', fontsize=9, color=GREEN)
    stem = 'figS26_common_pair_timing'
    export(fig, destination, stem)
    timeline.to_csv(destination/f'{stem}_timing.csv', index=False)
    components.reset_index().to_csv(destination/f'{stem}_components.csv', index=False)
    return stem, ('Figure S26 | Common-population calendar timing and weather–phenology contributions. '
        '(a) European mean dates under SSP5–8.5. Symptom appearance, flowering and soft dough use identical historical–future season pairs '
        'with all dates recorded in the full-grid outputs. Fixed harvested-area weighting precedes equal climate-model averaging. '
        'Arrows connect period means; the horizontal axis shows elapsed calendar days after sowing. '
        '(b) Weather, wheat-phenology and net contributions to normalized HAD loss in the separate 62-cell diagnostic sample '
        '(64 area-proportional draws; 5,731 valid season pairs). Grey segments span climate-model means; coloured whiskers show one spatial Monte Carlo standard error. '
        'Each Shapley contribution includes half the weather–phenology interaction. The 70.6% offset is the ratio of ensemble contributions. '
        'The full-grid timing population in a differs from the sampled decomposition in b. These comparisons concern climate-driven timing shifts; '
        'they do not estimate benefits of changing sowing date, cultivar or crop protection.')


def timing_mechanism(destination, data, countries, cells):
    part=data[data.metric.eq(ONSET)&data.scenario.eq('ssp585')&data.period.eq('2071-2100')].copy()
    directory=ROOT/'analysis/paper_study/nature_food_submission_20261008/derived/spatial'
    regional=pd.read_csv(directory/'region_supported_decomposition.csv')
    by_model=pd.read_csv(directory/'region_supported_decomposition_by_gcm.csv')
    source=ROOT/'analysis/paper_study/nature_food_fix_20261007/climate/decomposition/supported_forcing_ensemble_decomposition.csv'
    components=pd.read_csv(source).set_index('component')
    fig=plt.figure(figsize=(10.2,7.8))
    gs=fig.add_gridspec(2,2,height_ratios=[1.6,1.1],left=.12,right=.97,bottom=.10,top=.94,hspace=.58,wspace=.46)
    ax=fig.add_subplot(gs[0,0]);map_axes(ax,countries,cells)
    norm,cmap,ticks=discrete_scale(part.mean_change.to_numpy(),[.5,1,2,5,10,20],2)
    im=raster(ax,part,'mean_change',cmap=cmap,norm=norm)
    ax.set_title('a   First symptoms relative to flowering',loc='left',weight='bold',fontsize=10,pad=12)
    cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.15,shrink=.97)
    cb.set_ticks(ticks);cb.set_label('Change in crop-relative symptom timing (days)',fontsize=8.3)
    ax=fig.add_subplot(gs[0,1])
    for y,key,color in zip(range(3),['weather_shapley','host_shapley','total_change'],[GOLD,GREEN,BLUE]):
        r=components.loc[key]
        ax.plot([r.gcm_min,r.gcm_max],[y,y],color='#b1b6ba',lw=2.4)
        ax.errorbar(r.gcm_mean,y,xerr=r.spatial_mcse_gcm_mean,marker='o',color=color,capsize=3,ms=6)
        ax.text(r.gcm_mean,y-.19,f'{r.gcm_mean:+.2f}',ha='center',color=color)
    ax.set_yticks([0,1,2],['Weather','Wheat phenology','Net change']);ax.set_ylim(2.5,-.65)
    ax.axvline(0,color=GREY,ls='--',lw=.8);clean(ax)
    ax.set_xlabel('Contribution to normalized HAD loss (days)',fontsize=8.3)
    ax.set_title('b   Opposing European contributions',loc='left',weight='bold',fontsize=10,pad=12)
    ax.text(.5,-.23,'Phenology offset: 70.6%',transform=ax.transAxes,ha='center',fontsize=9,color=GREEN)
    ax=fig.add_subplot(gs[1,:])
    names=['Atlantic','Continental','Boreal','Mediterranean','Steppic']
    region=regional.set_index('environment_region').loc[names]
    y=np.arange(len(names))
    for i,(key,color,label) in enumerate([('weather',GOLD,'Weather'),('host',GREEN,'Wheat phenology'),('net',BLUE,'Net change')]):
        limits=by_model[by_model.environment_region.isin(names)].groupby('environment_region')[key].agg(['min','max']).loc[names]
        ax.errorbar(region[key],y+(i-1)*.20,xerr=[region[key]-limits['min'],limits['max']-region[key]],
            color=color,marker=['s','^','o'][i],ls='none',capsize=2,ms=4,label=label)
    ax.set_yticks(y,[f'{n} (n={int(region.loc[n,"sample_draws"])})' for n in names]);ax.invert_yaxis()
    ax.axvline(0,color=GREY,ls='--',lw=.8);clean(ax)
    ax.set_xlabel('Contribution to normalized HAD loss (days)',fontsize=9)
    ax.set_title('c   Regional contributions in the diagnostic sample',loc='left',weight='bold',fontsize=10,pad=34)
    ax.legend(loc='lower left',bbox_to_anchor=(0,1.005),ncol=3,frameon=False,fontsize=8.5)
    stem='fig3_crop_relative_timing_and_mechanism'
    export(fig,destination,stem)
    part.to_csv(destination/f'{stem}_timing_map.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    regional.to_csv(destination/f'{stem}_regions.csv',index=False)
    by_model.to_csv(destination/f'{stem}_regions_by_gcm.csv',index=False)
    components.reset_index().to_csv(destination/f'{stem}_components.csv',index=False)
    return stem,('Figure 3 | Crop-relative timing and opposing weather–phenology contributions. '
        'All panels concern SSP5–8.5 in 2071–2100 relative to 1991–2020. '
        '(a) Full-grid three-model mean change in first symptoms relative to flowering; negative values indicate earlier symptoms within crop development. '
        'Timing requires symptoms in both paired seasons. (b) Weather, phenology and net contributions to normalized HAD loss in the separate diagnostic sample '
        '(64 area-proportional draws at 62 cells; 5,731 valid pairs). Grey segments span climate-model means; coloured whiskers show one spatial Monte Carlo standard error. '
        'Each Shapley contribution includes half the interaction; the 70.6% offset is a ratio of ensemble contributions. '
        '(c) Contributions within the five regions with at least four diagnostic draws; n denotes draws, not independent field experiments. '
        'Points are model means and whiskers model ranges. Pannonian, Alpine and Black Sea regions lack sufficient sampled support for this display. '
        'Regional offsets and spatial sampling errors are supplied in Source Data. Full-grid and sampled estimates have different populations. '
        'These projections do not estimate benefits of sowing-date or cultivar interventions. Supplementary Fig. S26 shows dates after sowing on identical season pairs.')


def supplementary_diagnostics(destination):
    data,_,_=grid_data();cells=data.drop_duplicates('cell_id')
    countries=gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    fig,axes=plt.subplots(1,3,figsize=(11.2,4.3))
    fig.subplots_adjust(left=.05,right=.98,top=.90,bottom=.15,wspace=.17)
    sources=[]
    specs=[(FREQUENCY,100,'a   Symptom frequency','Frequency change (percentage points)'),
           (SEVERITY,100,'b   Relative HAD loss','Relative HAD-loss change (percentage points)'),
           ('F1_symptom_day_after_sowing',1,'c   Symptoms after sowing','Change in days from sowing to symptoms')]
    for i,(ax,(metric,scale,title,unit)) in enumerate(zip(axes,specs)):
        p=data[data.metric.eq(metric)&data.scenario.eq('ssp585')&data.period.eq('2071-2100')].copy()
        p['display']=scale*p.mean_change;sources.append(p);map_axes(ax,countries,cells)
        if metric=='F1_symptom_day_after_sowing':norm,cmap,ticks=onset_change_scale(p.display)
        else:norm,cmap,ticks=discrete_scale(p.display.to_numpy(),[.1,1,5,10,20],.1 if metric==FREQUENCY else 5)
        im=raster(ax,p,'display',cmap=cmap,norm=norm)
        ax.set_title(title,loc='left',weight='bold',fontsize=10,pad=12)
        if i:ax.set_yticklabels([])
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.045,pad=.16,shrink=.97)
        cb.set_ticks(ticks);cb.set_label(unit,fontsize=7.6);cb.ax.tick_params(labelsize=8)
    stem='figS27_frequency_canopy_and_calendar_diagnostics'
    export(fig,destination,stem)
    pd.concat(sources).to_csv(destination/f'{stem}.csv.gz',index=False,compression={'method':'gzip','mtime':0})
    return stem,('Figure S27 | Complementary symptom and canopy diagnostics under late-century SSP5–8.5. '
        'Three-model mean changes in 2071–2100 relative to 1991–2020. (a) Frequency of flag-leaf symptoms before soft dough. '
        '(b) Relative HAD loss during grain filling; this is the fraction of assumed canopy function lost, not observed lesion percentage. '
        '(c) Elapsed days from the fixed sowing date to first symptoms. Its continuous linear symmetric colour scale spans every finite change, '
        'retaining distinctions among ordinary advances and rare extremes. Date arithmetic is checked against recorded sowing and symptom dates; '
        'this consistency check does not validate biological plausibility of extreme projections. Timing requires detected symptoms in both paired seasons, '
        'whereas frequency includes complete symptom-free seasons. Grey wheat cells have unavailable estimates.')


def main(destination):
    style(); data, regions, _ = grid_data()
    countries = gpd.read_file(ROOT/'data/geography/ne_110m_admin_0_countries.zip')
    cells = data.drop_duplicates('cell_id')
    captions = dict([prediction_support(destination), climate_response(destination, data, regions, countries, cells),
                     timing_mechanism(destination,data,countries,cells),production_exposure(destination)])
    (destination/'captions.json').write_text(json.dumps(captions, indent=2)+'\n')
    return captions
