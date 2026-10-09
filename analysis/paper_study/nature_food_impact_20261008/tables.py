"""Climate-impact tables and pooled evidence with explicit denominators."""
import re
import numpy as np
import pandas as pd
from .data import ROOT,HERE,GRID,DERIVED,CANOPY,SEVERITY,ONSET,FREQUENCY,YIELD,SCENARIOS,REGIONS,pooled,grid_data
from analysis.paper_study.nature_food_submission_20261008.build import _curated_tables as legacy_tables


def number(value,digits=2,sign=False):
    if pd.isna(value):return 'Unavailable'
    return f'{value:+.{digits}f}' if sign else f'{value:.{digits}f}'


def main_tables():
    _,regions,_=grid_data()
    p=regions[regions.scenario.eq('ssp585')&regions.period.eq('2071-2100')]
    rows=[['Environmental region','Δ normalized HAD loss (days)','Δ symptom timing (days)',
           'Δ relative HAD loss (pp)','Estimated Δ yield (kg ha⁻¹ per unit LAI)']]
    for name in ['Europe']+REGIONS:
        q=p[p.environment_region.eq(name)].set_index('metric')
        rows.append([name,number(q.loc[CANOPY,'mean_change'],sign=True),number(q.loc[ONSET,'mean_change'],sign=True),
                     number(100*q.loc[SEVERITY,'mean_change'],sign=True),number(-1000*q.loc[YIELD,'mean_change'],1,True)])
    return [(rows,('Table 1 | Late-century climate impacts across European wheat environmental regions. '
        'Values are harvested-area-weighted three-climate-model mean paired changes under SSP5–8.5 in 2071–2100 relative to 1991–2020. '
        'Positive normalized HAD-loss and relative HAD-loss changes indicate greater accumulated loss of assumed canopy function; negative timing changes indicate earlier symptoms relative to flowering. '
        'Estimated disease-related yield change uses the fixed coefficient 0.018 t ha⁻¹ per GLAI-day and a nominal upper-three-leaf reference LAI of one. '
        'The yield column reports a disease-related index, not validated harvested yield. Model ranges, exact regional areas and valid-season coverage are supplied in Supplementary Table S13.'))]


def supplementary_tables():
    legacy=legacy_tables();tables=[]
    retained={'S1a','S1b','S1c','S5a','S5b','S6a','S6b','S6c','S7','S8b','S10c','S11a','S12','S13a','S13b'}
    remap={'S10c':'S4','S11a':'S10','S12':'S11','S13a':'S12a','S13b':'S12b'}
    for rows,caption in legacy:
        m=re.match(r'Table (S\d+[a-z]?) \|',caption)
        if not m or m.group(1) not in retained:continue
        old=m.group(1);new=remap.get(old,old)
        caption=caption.replace(f'Table {old} |',f'Table {new} |',1)
        rows=[[str(v).replace('BASF','Field').replace('Corteva','Field') for v in row] for row in rows]
        if old=='S11a':
            caption=('Table S10 | Sensitivity of late-century SSP5–8.5 model responses to fourteen prespecified settings. '
                     'The diagnostic sample retains 64 spatial draws at 62 cells. Metric-specific common paired populations and coverage are retained. '
                     'Climate-model ranges and spatial sampling error refer to this diagnostic sample separately from the full-grid analysis. '
                     'Yield quantities are conditional canopy-conversion products.')
        if old=='S8b':
            extra=pd.read_csv(ROOT/'analysis/paper_study/nature_food_revision_20261007/basf_crop_response/year_transfer_metrics.csv')
            rows[0][3]='Environments / fields'
            for family in ['linear','exponential']:
                r=extra[extra.domain.eq('all_three_separate_windows')&extra.function.eq(family)].iloc[0]
                rows.append(['2019 year transfer; upper three leaves',family.capitalize(),str(int(r.evaluation_contrasts)),
                    str(int(r.evaluation_fields)),number(r.RMSE_pp),number(r.baseline_RMSE_pp),
                    number(1-(r.RMSE_pp/r.baseline_RMSE_pp)**2,3)])
            caption=('Table S8 | Treatment-response prediction under held-out trial and year transfer. '
                'Trial-held-out endpoint and common-date window models retain their respective 67- and 50-contrast populations across five trial environments. '
                'Upper-three-leaf year transfer fits 2017–2018 treatment contrasts and evaluates 12 contrasts in 11 fields in 2019. '
                'Each model is compared with its own training-mean prediction; the populations and disease-percentage predictors retain distinct definitions. '
                'These comparisons evaluate management-associated relative yield responses rather than cause-specific STB yield loss or measured healthy-area duration.')
        tables.append((rows,caption))
    field=pd.read_csv(DERIVED/'pooled_field_stage_records.csv')
    metrics=pd.read_csv(DERIVED/'pooled_field_stage_metrics.csv')
    rows=[['Cohort','BBCH','Fields','Two-sided n','Left n','Right n','Unavailable n','Mean excess (days)']]
    for (cohort,event),part in field.groupby(['cohort','event']):
        q=metrics[metrics.cohort.eq(cohort)&metrics.event.eq(event)].iloc[0]
        rows.append([cohort,str(event),str(part.field_id.nunique()),str(int(q.bracketed)),
          str(int(part.censoring.eq('left').sum())),str(int(part.censoring.eq('right').sum())),
          str(int((~part.eligible_constraint).sum())),number(q.mean_interval_excess_days)])
    tables.append((rows,'Table S2 | Pooled field developmental-stage observation coverage and interval errors. '
       'Calibration and evaluation remain separate; eligible evaluation records from both archives are combined. '
       'Mean excess uses equal coordinate-year and field weighting on genuinely two-sided intervals. '
       'Unavailable intervals remain distinct from zero prediction error.'))
    score=pd.read_csv(DERIVED/'pooled_evaluation_metrics.csv')
    rows=[['Endpoint','Score','Estimate','95% interval','Records','Field seasons','Coordinate-years']]
    for r in score.itertuples():
        label=r.metric.replace('_',' ')
        scale=100 if r.endpoint=='symptom_detection' else 1
        rows.append([r.endpoint.replace('_',' '),label,number(scale*r.value,3),
                     f'{scale*r.lower95:.3f} to {scale*r.upper95:.3f}' if np.isfinite(r.lower95) else 'Not estimated',
                     str(r.records),str(r.fields),str(r.coordinate_years)])
    tables.append((rows,'Table S3 | Pooled disease evaluation and paired benchmark comparisons. '
      'Onset values are days outside two-sided symptom intervals; severity values are percentage points and detection values are percentages. '
      'Calibration records are excluded. Scores preserve the coordinate-year, field, source-leaf and assessment hierarchy. '
      'Intervals use 20,000 paired coordinate-year bootstrap resamples conditional on fixed predictions; source measurement conventions remain explicit in the underlying data.'))
    _,regions,countries=grid_data()
    rows=[['SSP','Period','Δ normalized HAD loss (days)','Δ timing (days)','Δ frequency (pp)',
           'Δ relative HAD loss (pp)','Estimated Δ yield (kg ha⁻¹ per unit LAI)','Valid HAD area-time (%)']]
    p=regions[regions.environment_region.eq('Europe')]
    for (scenario,period),part in p.groupby(['scenario','period']):
        q=part.set_index('metric')
        rows.append([scenario.upper(),period,number(q.loc[CANOPY,'mean_change'],sign=True),
          number(q.loc[ONSET,'mean_change'],sign=True),number(100*q.loc[FREQUENCY,'mean_change'],3,True),
          number(100*q.loc[SEVERITY,'mean_change'],sign=True),number(-1000*q.loc[YIELD,'mean_change'],1,True),
          number(100*q.loc[CANOPY,'minimum_valid_area_time_fraction'],3)])
    tables.append((rows,'Table S9 | Full-grid European climate impacts by emissions pathway and harvest period. '
      'Entries use fixed harvested-area weights over the valid historical–future season pairs for each metric. '
      'Timing is conditional on detected symptoms; occurrence includes symptom-free complete seasons. '
      'Relative HAD loss is HAD loss divided by reference HAD. Normalized HAD loss is HAD loss divided by maximum reference upper-canopy LAI, with units of days. Estimated disease-related yield change uses the reference coefficient and nominal reference LAI.'))
    for letter,metric,label,scale,unit in [('a',CANOPY,'normalized HAD loss',1,'days'),
      ('b',SEVERITY,'relative HAD loss',100,'percentage points'),('c',ONSET,'symptom timing',1,'days relative to flowering')]:
        rows=[['Region','SSP','Mean change','Three-model range','Eligible cells','Reference area (Mha)','Valid area-time (%)']]
        p=regions[regions.metric.eq(metric)&regions.period.eq('2071-2100')]
        for r in p.itertuples():
            rows.append([r.environment_region,r.scenario.upper(),number(scale*r.mean_change,3,True),
                f'{scale*r.gcm_min:+.3f} to {scale*r.gcm_max:+.3f}',str(r.calendar_eligible_cells),
                number(r.reference_area_ha/1e6,3),number(100*r.minimum_valid_area_time_fraction,3)])
        tables.append((rows,f'Table S13{letter} | Full-grid late-century environmental-region changes in {label} ({unit}). '
          'Historical and future means use identical valid season pairs. Model ranges span three deterministic climate-model estimates. '
          'Unassigned and outside-region groups remain separate from the eight named EEA regions. All wheat grids are simulated, so these estimates have no spatial sampling error.'))
    rows=[['Country group','Eligible cells','Reference area (Mha)','Δ normalized HAD loss (days)',
           'Δ timing (days)','Δ relative HAD loss (pp)','Estimated Δ yield (kg ha⁻¹ per unit LAI)']]
    p=countries[countries.scenario.eq('ssp585')&countries.period.eq('2071-2100')]
    names=p.drop_duplicates('country').sort_values('reference_area_ha',ascending=False).country
    for name in names:
        q=p[p.country.eq(name)].set_index('metric');r=q.loc[CANOPY]
        rows.append([name.replace('United Kingdom of Great Britain and Northern Ireland','United Kingdom'),
          str(int(r.calendar_eligible_cells)),number(r.reference_area_ha/1e6,3),number(r.mean_change,sign=True),
          number(q.loc[ONSET,'mean_change'],sign=True),number(100*q.loc[SEVERITY,'mean_change'],sign=True),
          number(-1000*q.loc[YIELD,'mean_change'],1,True)])
    tables.append((rows,'Table S14 | Country-group climate impacts under late-century SSP5–8.5. '
      'Groups follow the dominant SPAM source-country assignment of quarter-degree wheat cells within the study domain. '
      'Values are exact harvested-area-weighted model scenario responses rather than observed national averages. '
      'Cross-border cells are not partitioned, and the Russian Federation and Turkey entries cover only represented domain cells. '
      'All SSP-specific estimates, climate-model ranges and valid area-time coverage remain in Source Data.'))
    sign=pd.read_csv(GRID/'canopy_change_sign_area_shares.csv')
    rows=[['SSP','Period','Positive ensemble (%)','All models positive (%)','All models negative (%)','Mixed signs (%)','Complete-model area (%)']]
    for (scenario,period),part in sign.groupby(['scenario','period']):
        q=part.set_index('classification')
        rows.append([scenario.upper(),period,*[number(q.loc[c,'landuse_area_share_percent'],2) for c in
          ['positive_ensemble','positive_all_three_gcm','negative_all_three_gcm','mixed_gcm_sign']],
          number(100*q.complete_gcm_landuse_area_fraction.iloc[0],3)])
    tables.append((rows,'Table S15 | Exact harvested-area shares by climate-model canopy-response sign. '
      'Shares use all cells with complete three-model estimates under each scenario–period comparison. '
      'Positive ensemble means and unanimous positive signs are distinct classifications. '
      'Mixed signs include at least one positive and one negative model change; any zero-only combinations remain in Source Data.'))
    registry=pd.read_csv(GRID/'full_landuse_cell_registry.csv')
    rows=[['Environmental region','Land-use cells','Eligible calendars','Reference wheat area (Mha)']]
    for name,part in registry.groupby('environment_region'):
        rows.append([name,str(len(part)),str(int(part.calendar_valid.sum())),number(part.harvested_total_ha.sum()/1e6,4)])
    tables.append((rows,'Table S16 | Full-grid wheat-land-use and environmental-region coverage. '
      'Areas sum the fixed SPAM2020 all-wheat harvested-area registry. Calendar eligibility uses the imposed GGCMI winter-wheat rainfed scenario. '
      'Environmental assignment follows cell centroids; missing and outside-region assignments remain explicit.'))
    def key(item):
        m=re.match(r'Table S(\d+)([a-z]?)',item[1]);return int(m.group(1)),m.group(2)
    def wording(value):
        for old,new in [('days per nominal LAI','days'),('days per reference LAI','days'),
                        ('days / nominal LAI','days'),('days / reference LAI','days'),
                        ('days / LAI','days'),('days per nominal upper-three LAI','days'),
                        ('HAD deficit (days)','Normalized HAD loss (days)'),
                        ('HAD3 is GLAI-days per unit nominal upper-three-leaf LAI','Normalized HAD loss is expressed in days'),
                        ('common paired support','identical valid season pairs'),
                        ('Disease-weather contribution','Weather contribution'),
                        ('Wheat-development','Wheat-phenology'),
                        ('wheat-development','wheat-phenology'),
                        ('Wheat development','Wheat phenology'),
                        ('wheat development','wheat phenology'),
                        ('Host-development','Wheat-phenology'),
                        ('Crop-development','Wheat-phenology'),
                        ('supported-forcing population','seasons meeting the weather input rules')]:
            value=value.replace(old,new)
        return value
    return [([[wording(v) for v in row] for row in rows],wording(cap)) for rows,cap in sorted(tables,key=key)]
