"""Complete supplementary validation tables, with cohort and model identity."""
import pandas as pd
from .evidence import HERE,PARTITIONS,LABELS,EVENTS


def number(value,digits=2):
    return '—' if pd.isna(value) else f'{value:.{digits}f}'


def percent(value):
    return number(100*value,1)


def supplementary_tables():
    tables=[]
    g=pd.read_csv(HERE/'german_stage_metrics_all.csv',dtype={'stage':str})
    for letter,cohort in zip('abc',['calibration','validation','testing']):
        frame=g.loc[g.cohort.eq(cohort)&g.denominator.eq('three_model_shared')]
        rows=[['Model','BBCH','Matched n','Observed n','Coverage (%)','MAE (d)','RMSE (d)','Bias (d)','Missing n']]
        for model in ['GDD','T-P','T-P-V']:
            for stage in ['10','31','51','85','overall']:
                r=frame.loc[frame.model.eq(model)&frame.stage.eq(stage)].iloc[0]
                rows.append([model,stage,str(r.matched),str(r.observed),percent(r.coverage_fraction),
                    number(r.mae_days),number(r.rmse_days),number(r.bias_days),str(r.missing_predictions)])
        tables.append((rows,f'Table S1{letter} | German station {cohort} event-date evaluation. '
            'All process models use identical station/sowing-cycle/stage matches. Coverage includes observed events lacking shared predictions; '
            'Missing n counts unavailable predictions for the individual model. The distinct own-matched denominator is retained in Source Data. '
            'The station split does not withhold calendar years. Errors describe realized-weather, known-sowing hindcasts.',
            [44,37,49,49,63,45,47,45,46]))
    coverage=pd.read_csv(HERE/'field_stage_coverage_all.csv')
    metrics=pd.read_csv(HERE/'field_stage_metrics_all.csv')
    for letter,partition in zip('abcd',PARTITIONS):
        rows=[['BBCH','Registered n','Constrained n','Two-sided n','Left n','Right n','Unavailable n']]
        for r in coverage.loc[coverage.partition.eq(partition)].itertuples():
            rows.append([str(r.event),str(r.registered_fields),str(r.constrained_fields),str(r.genuine_two_sided),
                str(r.left_censored),str(r.right_censored),str(r.unavailable_or_rejected_fields)])
        tables.append((rows,f'Table S2{letter} | {LABELS[partition]}: observation coverage for every retained stage. '
            'Unavailable n includes absent or rejected stage evidence. Left- and right-censored records are one-sided; '
            'a stage with no genuine two-sided intervals lacks direct interval timing validation. '
            +('The full Corteva cohort overlaps the location-disjoint subset and is not pooled with it.' if partition=='reused_external_full' else ''),
            [40,68,75,72,50,50,70]))
        rows=[['BBCH','Two-sided n','Median width (d)','Mean excess (d)','Signed excess (d)','Compatible (%)']]
        selected=metrics.loc[metrics.forcing_scope.eq('full_season')&metrics.partition.eq(partition)&metrics.scope.eq('genuine_two_sided')]
        for r in selected.itertuples():
            rows.append([str(r.event),str(r.n_field_events),number(r.median_bracket_width_days,1),
                number(r.mean_distance_days),number(r.mean_signed_distance_days),percent(r.compatible_fraction)])
        tables.append((rows,f'Table S3{letter} | {LABELS[partition]}: full-season stage predictions versus genuine two-sided observations. '
            'Distances are calendar days outside the daily admissible interval, with negative signed distance for early prediction. '
            'Interval widths use the original last-below/first-at-or-above assessment-date separation. '
            'Missing stages and both observation-horizon and full-season one-sided scores remain explicit in Source Data.',
            [36,62,93,90,92,82]))
    names={'calibration':'BASF17–18','reused_BASF2019':'BASF19','reused_strict_Corteva':'Corteva strict'}
    models={'overwinter_source_model':'Seasonal','phenology_only':'Phenology'}
    detection=pd.read_csv(HERE/'disease_assessment_detection_metrics_baseline.csv')
    rows=[['Set','Model','Leaf scope','Assessment n','Sensitivity (%)','Specificity (%)','Balanced (%)']]
    for r in detection.itertuples():
        rows.append([names[r.partition],models[r.model],'F1–F3' if r.leaf_scope=='top3' else 'All ranks',
            str(r.n),percent(r.sensitivity),percent(r.specificity),percent(r.balanced_accuracy)])
    tables.append((rows,'Table S4a | Baseline disease detection in every cohort and both leaf scopes. '
        'Calibration is fitted evidence; BASF2019 and Corteva are reused validation. Equal coordinate-year, field and leaf weighting '
        'is retained. All ordinal ranks and F1–F3 overlap and are not pooled.',[58,56,60,60,75,75,64]))
    onset=pd.read_csv(HERE/'disease_onset_bracket_metrics_baseline.csv')
    for letter,kind in zip('bcd',['two_sided','left','right']):
        rows=[['Set','Model','Leaf scope','History n','Mean excess (d)','Compatible (%)','Missed positive n']]
        for r in onset.loc[onset.censoring.eq(kind)].itertuples():
            rows.append([names[r.partition],models[r.model],'F1–F3' if r.leaf_scope=='top3' else 'All ranks',
                str(r.n),number(r.distance_days),percent(r.compatible_fraction),str(r.missed_positive)])
        tables.append((rows,f'Table S4{letter} | Baseline symptom-onset evaluation: {kind.replace("_","-")} histories. '
            'Only genuinely two-sided histories directly bracket visible onset. One-sided compatibility does not establish exact-date '
            'accuracy or whole-season disease absence. Actual infection dates are unobserved.',[58,56,60,52,87,77,80]))
    occurrences=pd.read_csv(HERE/'disease_observed_window_occurrence_metrics_baseline.csv')
    rows=[['Set','Model','Scope','Level','Window n','Positive n','Negative n','Sensitivity (%)','Specificity (%)']]
    for r in occurrences.itertuples():
        rows.append([names[r.partition],models[r.model],'F1–F3' if r.leaf_scope=='top3' else 'All ranks',
            r.level,str(r.n),str(r.positive_count),str(r.negative_count),percent(r.sensitivity),percent(r.specificity)])
    tables.append((rows,'Table S4e | Occurrence within recorded assessment windows for all cohorts, models and leaf scopes. '
        'Field and field-leaf summaries overlap. An observed-zero window does not establish a disease-free complete season; '
        'the counts constrain the evidential basis of occurrence specificity.',[53,48,46,67,41,47,47,58,59]))
    archived=pd.read_csv(HERE/'archived_seasonal_model_results.csv')
    for letter,source in zip('ab',[
            'analysis/paper_study/seasonal_calibration_v1/severity_metrics.csv',
            'analysis/paper_study/corteva_external_seasonal_v1/severity_metrics.csv']):
        selected=archived.loc[archived.source_path.eq(source)&archived.endpoint.eq('final_numeric_assessment')]
        rows=[['Cohort','Archived model','Leaf scope','Endpoint n','RMSE (pp)','MAE (pp)','Bias (pp)']]
        for r in selected.to_dict('records'):
            cohort=r.get('partition') if pd.notna(r.get('partition')) else r.get('geography')
            scope=r.get('subset') if pd.notna(r.get('subset')) else r.get('leaf_scope')
            rows.append([str(cohort),str(r['model']).replace('_',' '),'F1–F3' if scope=='top3' else str(scope),
                str(int(r['n'])),number(r['rmse']),number(r['mae']),number(r['bias'])])
        tables.append((rows,f'Table S5{letter} | Archived seasonal-v1 final-severity comparisons. '
            'Percentage-point errors belong to the earlier frozen severity model and its comparators. '
            'These records preserve earlier dataset outcomes and are separate from validation of the current seasonal-source model. '
            'All assessments, alternative endpoints and archived onset comparisons remain in Source Data.',[85,94,63,46,57,57,57]))
    for letter,version in zip('ab',['v1','v2']):
        frame=pd.read_csv(HERE/f'french_transfer_{version}.csv')
        rows=[['Year','Archived model','Targets n','Series n','RMSE (pp)','MAE (pp)','Bias (pp)']]
        for r in frame.itertuples():
            label=r.model.removesuffix('_BASF_frozen').replace('_',' ')
            rows.append([str(r.year),label,str(r.n_targets),str(r.n_series),number(r.rmse_pp),number(r.mae_pp),number(r.bias_pp)])
        tables.append((rows,f'Table S6{letter} | French plot-leaf transfer in2018–2019, archived hidden-state {version} and frozen comparators. '
            'The383 future assessments comprise168 and215 targets from72 series in each year, at one representative station. '
            'Predictions are conditioned on initial pycnidial observations and realized weather; French targets are not fitted. '
            'The pycnidial observation mapping and initial conditioning differ from natural field-onset validation and from the current model. '
            'The two versions share observations and are not independent replication.',[35,172,50,47,57,57,57]))
    return sorted(tables,key=lambda table:table[1].split(' |')[0])
