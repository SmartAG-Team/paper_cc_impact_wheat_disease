"""Paired scenario contrasts with shared disease-parameter and spatial draws."""
from pathlib import Path
import hashlib, itertools, json, sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.paper_study.run_regional import MODELS, SCENARIOS
from analysis.paper_study.aggregate_projection_uncertainty import PERIODS
from model.seasonal_septoria.spatial_estimation import stratified_mcse

DATA = ROOT / 'data/paper_study/publication'
SOURCE = ROOT / 'data/paper_study/projection_uncertainty_reporting/parameter_periods'
REVIEW = ROOT / 'analysis/paper_study/manuscript_review_20261006'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    draws = pd.read_csv(ROOT / 'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv')
    arrays, hashes = {}, {}
    for model, scenario, period in itertools.product(MODELS, SCENARIOS, PERIODS):
        path = SOURCE / model / scenario / f'{period}.npz'
        assert sha(path) == json.loads(path.with_suffix('.json').read_text())['output_sha256']
        arrays[model, scenario, period] = dict(np.load(path))
        hashes[str(path.relative_to(ROOT))] = sha(path)
    rows, saved = [], {}
    for low, high in itertools.combinations(SCENARIOS, 2):
        for period in list(PERIODS)[1:]:
            for quantity in ['future_severity_difference', 'difference_in_baseline_relative_changes']:
                means, influences = [], []
                for model in MODELS:
                    hi, lo = arrays[model, high, period], arrays[model, low, period]
                    value = hi['means'] - lo['means']
                    influence = hi['contributions'] - lo['contributions']
                    if quantity == 'difference_in_baseline_relative_changes':
                        bh, bl = arrays[model, high, 'baseline1991_2020'], arrays[model, low, 'baseline1991_2020']
                        value -= bh['means'] - bl['means']
                        influence -= bh['contributions'] - bl['contributions']
                    means.append(value); influences.append(influence)
                for model, value, influence in zip(list(MODELS) + ['three_model_mean'],
                        means + [np.mean(means, axis=0)], influences + [np.mean(influences, axis=0)]):
                    mcse = stratified_mcse(influence, draws.area_mean_weight, draws.stratum)
                    p025, p975 = np.quantile(value[1:], [.025, .975])
                    rows.append(dict(model=model, scenario_high=high, scenario_low=low, period=period,
                        quantity=quantity, point_estimate_pp=value[0], conditional_parameter_p025_pp=p025,
                        conditional_parameter_p975_pp=p975, point_spatial_mcse_pp=mcse[0],
                        bootstrap_spatial_mcse_max_pp=mcse[1:].max(),
                        climate_model_point_min_pp=min(x[0] for x in means) if model == 'three_model_mean' else np.nan,
                        climate_model_point_max_pp=max(x[0] for x in means) if model == 'three_model_mean' else np.nan,
                        shared_parameter_and_spatial_draws=True, probabilistic_climate_interval=False))
                    key = '__'.join([model, high, low, period, quantity])
                    saved[key + '__means'] = value
                    saved[key + '__contributions'] = influence
    frame = pd.DataFrame(rows)
    frame.to_csv(DATA / 'paired_scenario_contrasts.csv', index=False)
    np.savez_compressed(REVIEW / 'paired_scenario_draws.npz', **saved)
    # Country comparisons cover all scenarios and both periods, irrespective of sign.
    country = pd.read_csv(DATA / 'country_production_weighted_periods.csv')
    keys = ['model', 'scenario', 'country', 'reference_production_tonnes']
    base = country[country.period.eq('baseline1991_2020')][keys + ['production_weighted_mean_severity_percent']]
    base = base.rename(columns={'production_weighted_mean_severity_percent': 'baseline_severity_percent'})
    future = country[~country.period.eq('baseline1991_2020')].merge(base, on=keys, validate='many_to_one')
    future['change_pp'] = future.production_weighted_mean_severity_percent - future.baseline_severity_percent
    assert len(future) == 3 * 3 * 2 * 41
    summary = future.groupby(['scenario', 'period', 'country'], as_index=False).agg(
        reference_production_tonnes=('reference_production_tonnes', 'first'),
        mean_change_pp=('change_pp', 'mean'), climate_model_min_pp=('change_pp', 'min'),
        climate_model_max_pp=('change_pp', 'max'), climate_models=('model', 'nunique'))
    assert summary.climate_models.eq(3).all()
    summary.to_csv(DATA / 'country_all_scenario_comparisons.csv', index=False)
    mapped=pd.read_csv(DATA/'fig4_climate_change_cells.csv')
    hashes['data/paper_study/publication/fig4_climate_change_cells.csv']=sha(DATA/'fig4_climate_change_cells.csv')
    common=mapped[mapped.all_comparisons_common_complete]
    common_rows=[]
    for (scenario,period),group in common.groupby(['scenario','period']):
        common_rows.append(dict(scenario=scenario,period=period,
            common_support_area_weighted_change_pp=np.average(group.three_model_mean_change_pp,weights=group.harvested_total_ha),
            common_harvested_ha=group.harvested_total_ha.sum(),common_cells=len(group)))
    common_summary=pd.DataFrame(common_rows)
    common_summary.to_csv(DATA/'common_support_scenario_changes.csv',index=False)
    common_contrasts=[]
    for period in list(PERIODS)[1:]:
        values=common_summary[common_summary.period.eq(period)].set_index('scenario')
        for low,high in itertools.combinations(SCENARIOS,2):
            common_contrasts.append(dict(scenario_high=high,scenario_low=low,period=period,
                common_support_difference_in_changes_pp=values.loc[high,'common_support_area_weighted_change_pp']-values.loc[low,'common_support_area_weighted_change_pp'],
                conditional_parameter_interval_computed=False))
    pd.DataFrame(common_contrasts).to_csv(DATA/'common_support_scenario_contrasts.csv',index=False)
    (REVIEW / 'scenario_contrast_provenance.json').write_text(json.dumps(dict(
        source_hashes=hashes, source_code_sha256=sha(Path(__file__)),
        definitions={'future_severity_difference': 'Higher minus lower SSP final severity in the same future period',
        'difference_in_baseline_relative_changes': '(Higher future - higher reference) - (lower future - lower reference)'},
        baseline='Historical 1991-2014 plus scenario-specific 2015-2020 continuation',
        conditional_uncertainty='100 shared calibration draws; fixed structure, host parameters and latent delay',
        source_rows=len(frame), all_three_ssps=True, field_parameters_modified=False), indent=2) + '\n')
    print(frame[frame.model.eq('three_model_mean')].to_string(index=False))


if __name__ == '__main__':
    main()
