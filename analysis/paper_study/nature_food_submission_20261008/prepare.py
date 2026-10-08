"""Derived publication summaries from the frozen current-model evidence."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
EVIDENCE = ROOT / 'analysis/paper_study/nature_food_fix_20261007'
STUDY = ROOT / 'analysis/paper_study/overwinter_leaf_model_20261006'


def prepare():
    from .spatial import prepare as prepare_spatial
    dest = HERE / 'derived'
    dest.mkdir(exist_ok=True)
    source = EVIDENCE / 'climate/decomposition/supported_forcing_decomposition_by_gcm.csv'
    raw = pd.read_csv(source)
    wide = raw.pivot(index='model', columns='component', values='mean')
    rows = []
    for model, r in wide.iterrows():
        rows.append(dict(climate_model=model, weather_contribution=r.weather_shapley,
                         host_contribution=r.host_shapley, net_change=r.total_change,
                         interaction=r.interaction,
                         host_offset_percent=100 * (-r.host_shapley) / r.weather_shapley))
    ensemble = wide.mean()
    rows.append(dict(climate_model='Three-model ensemble',
                     weather_contribution=ensemble.weather_shapley,
                     host_contribution=ensemble.host_shapley,
                     net_change=ensemble.total_change,
                     interaction=ensemble.interaction,
                     host_offset_percent=100 * (-ensemble.host_shapley) / ensemble.weather_shapley))
    offset = pd.DataFrame(rows)
    if not np.allclose(offset.weather_contribution + offset.host_contribution,
                       offset.net_change, rtol=0, atol=1e-12):
        raise ValueError('Weather and host contributions do not reconcile.')
    offset.to_csv(dest / 'weather_host_offset.csv', index=False)

    paired_path = STUDY / 'climate/paired_draw_period_moments.csv'
    paired = pd.read_csv(paired_path)
    paired = paired[paired.metric.eq('GS65_85_lost_had3')].copy()
    paired['value'] = np.divide(paired.numerator, paired.denominator,
                               out=np.full(len(paired), np.nan),
                               where=paired.denominator.to_numpy() > 0)
    draw_path = ROOT / 'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv'
    draws = pd.read_csv(draw_path)[['spatial_draw_id', 'cell_id', 'latitude', 'longitude']]
    paired = paired.merge(draws, on='spatial_draw_id', validate='many_to_one')
    groupkeys = ['model', 'scenario', 'future_period', 'cell_id']
    for _, g in paired.groupby(groupkeys):
        if not np.allclose(g.value, np.full(len(g), g.value.iloc[0]),
                           rtol=0, atol=1e-12, equal_nan=True):
            raise ValueError('Duplicated spatial draw values differ.')
    unique = paired.drop_duplicates(groupkeys)
    maps = []
    for key, g in unique.groupby(['scenario', 'future_period', 'cell_id']):
        scenario, period, cell = key
        complete = g[g.value.notna()]
        maps.append(dict(scenario=scenario, period=period, cell_id=cell,
                         latitude=g.latitude.iloc[0], longitude=g.longitude.iloc[0],
                         n_gcm=len(complete), value=complete.value.mean() if len(complete) == 3 else np.nan,
                         gcm_min=complete.value.min() if len(complete) == 3 else np.nan,
                         gcm_max=complete.value.max() if len(complete) == 3 else np.nan,
                         minimum_valid_year_pairs=int(g.valid_year_pairs.min()),
                         maximum_valid_year_pairs=int(g.valid_year_pairs.max())))
    pd.DataFrame(maps).to_csv(dest / 'map_canopy_changes.csv', index=False)
    provenance = {
        'model_status': 'Frozen current model; no refitting or new disease simulation',
        'host_offset_definition': '-100 * host contribution / weather contribution',
        'ensemble_definition': 'Ratio of ensemble contributions, not mean of individual ratios',
        'map_definition': 'Arithmetic three-GCM mean at actual sampled cells; duplicates collapsed for display only',
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in [source, paired_path, draw_path]},
        'host_offset_percent': float(offset.iloc[-1].host_offset_percent),
        'individual_model_offset_range': [float(offset.iloc[:-1].host_offset_percent.min()),
                                        float(offset.iloc[:-1].host_offset_percent.max())],
        'source_model_count': len(wide),
    }
    (dest / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    prepare_spatial()
    return provenance


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
