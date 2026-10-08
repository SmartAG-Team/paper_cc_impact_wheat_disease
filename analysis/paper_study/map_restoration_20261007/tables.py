"""Native supplemental evidence for epidemic and empirical crop-response checks."""
from pathlib import Path
import pandas as pd
from .prepare import ROOT
NEW=ROOT/'analysis/paper_study/nature_food_revision_20261007'


def supplemental_tables():
    data=pd.read_csv(NEW/'epidemic_evaluation/severity_trajectory_metrics.csv')
    rows=[['Set','Target','Assessment n','Fields','RMSE (pp)','Bias (pp)','R²']]
    names={'calibration':'BASF17–18','reused_BASF2019':'BASF19','reused_strict_Corteva':'Corteva'}
    for part in names:
        for endpoint in ['all_assessments','final_observed_leaf']:
            selected=data.loc[data.partition.eq(part)&data.endpoint.eq(endpoint)&data.leaf_scope.eq('all_numbered_leaves')
                &data.comparison_population.eq('all_eligible_assessments')&data.model.eq('frozen_raw_damage')]
            if selected.empty:
                selected=data.loc[data.partition.eq(part)&data.endpoint.str.contains('final')&data.leaf_scope.eq('all_numbered_leaves')
                    &data.comparison_population.eq('all_eligible_assessments')&data.model.eq('frozen_raw_damage')]
            assert len(selected)==1,(part,endpoint)
            r=selected.iloc[0]
            rows.append([names[part],'All dates' if endpoint=='all_assessments' else 'Final leaf',str(int(r['n'])),str(int(r.fields)),
                f'{r.rmse:.2f}',f'{r.bias:+.2f}',f'{r.weighted_R2:.3f}'])
    tables=[(rows,'Table S12a | Frozen current-model numerical severity assessment, with original calibration and retrospective validation membership. '
        'Scores use equal coordinate-year, field and source-numbered-leaf hierarchy; no numerical disease score enters the redacted prediction boundary. '
        'All ordinal leaves retain their source meanings. Negative R² and systematic underprediction constrain epidemic and yield extrapolation. '
        'Observed-stage-covered and top-three comparisons, temporal increments and representative trajectories remain in the accompanying evidence.',[65,63,70,46,70,70,60])]
    data=pd.read_csv(NEW/'yield_response/model_comparison_metrics.csv')
    select=data.loc[data.domain.str.startswith(('endpoint67_','integral50_','four_disease131_'))]
    rows=[['Domain','Model','Contrasts','Trials','RMSE (pp)','Baseline (pp)','MSE skill']]
    for r in select.itertuples():
        label='Endpoint' if r.domain.startswith('endpoint') else 'Window integral' if r.domain.startswith('integral') else 'Four diseases'
        rows.append([label,'Linear' if r.family=='linear' else 'Exponential',str(r.rows),str(r.independent_trials),
            f'{r.model_rmse_pp:.3f}',f'{r.baseline_rmse_pp:.3f}',f'{r.mse_skill:.3f}'])
    tables.append((rows,'Table S12b | Independent-trial prediction of signed relative treatment-associated yield response in verified public Nordic trial records. '
        'All comparisons sharing a control remain in one fold; trials receive equal total weights. The baseline is estimated from training trials only. '
        'Endpoint and observed-window severity-integral populations preserve recorded disease-free observations. Units are percentage points of relative crop response; '
        'integrals are severity percentage-days, not HAD. The four-disease domain has incomplete biological attribution. These models do not establish absolute disease-free yield loss. '
        'Primary records: Nordic Field Trial System, https://nfts.dlbr.dk/Forms/Dokumentation.aspx?KardexID=51831&applLangID=sv.',[75,64,52,43,66,77,63]))
    return tables
