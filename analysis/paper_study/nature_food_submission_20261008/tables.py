"""Compact evidence table and descriptive weather–host contribution ratios."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent


def main_tables():
    rows = [
        ['Endpoint', 'Evidence population', 'Principal result', 'Scope of production inference'],
        ['Retained crop-stage dates', '40,878 German station-held-out test events',
         'MAE 8.02 days; RMSE 11.75 days',
         'Evaluates four retained stages under known sowing and realized weather; added leaf stages remain sparsely observed.'],
        ['First-symptom intervals', '13 BASF and 123 Corteva histories; 8 and 54 coordinate-years',
         'Model / benchmark error: 5.06 / 3.00 and 5.36 / 2.75 days',
         'Retrospective timing does not establish improved epidemic prediction or cancellation of error under future climates.'],
        ['Numerical disease severity', '330 BASF and 1,703 Corteva assessments',
         'Model / benchmark RMSE: 26.63 / 22.99 and 32.63 / 31.86 percentage points',
         'Source-defined disease percentages are not measured functional green area; quantitative disease-to-canopy transfer remains unresolved.'],
        ['Published canopy–yield transfer', 'Eight aggregate contrasts; five genetic backgrounds; mixed disease exposure',
         'Relative RMSE 5.16 percentage points; absolute RMSE 0.617 t ha⁻¹',
         'Evaluates treatment-mean relationships for measured top-five canopy area, not current-model upper-three-leaf yield loss.'],
        ['Paired climate–canopy response', 'Three climate models; 64 spatial draws at 62 cells',
         'Supported net +2.64 HAD days per reference LAI; host offset 70.6%',
         'Identifies opposing contributions within a fixed-management model; actual yield effects and adaptation efficacy are unvalidated.'],
    ]
    caption = ('Table 1 | Evidence supporting the crop–disease–production pathway and its inferential boundaries. '
               'BASF results refer to 2019; Corteva results use location-disjoint source-numbered leaves. '
               'Symptom benchmarks use phenology alone; severity benchmarks use calibration-leaf means. '
               'The climate row uses the common supported-forcing decomposition population under late-century SSP5–8.5. '
               'HAD denotes healthy-area duration; reference LAI is a nominal upper-three-leaf area index of one. '
               'Different rows evaluate different quantities and cannot be combined into a single overall model-accuracy score.')
    return [(rows, caption)]


def offset_table():
    data = pd.read_csv(HERE / 'derived/weather_host_offset.csv')
    rows = [['Climate model', 'Weather ΔHAD', 'Host ΔHAD', 'Net ΔHAD', 'Host offset (%)']]
    for r in data.itertuples():
        rows.append([r.climate_model, f'{r.weather_contribution:+.3f}',
                     f'{r.host_contribution:+.3f}', f'{r.net_change:+.3f}', f'{r.host_offset_percent:.1f}'])
    caption = ('Table S13b | Host development offsets part of the weather contribution to modeled canopy damage. '
               'HAD changes are days per unit nominal reference LAI on the common supported-forcing population. '
               'Host offset is −100 × host/weather; the ensemble percentage is the ratio of the three-model mean '
               'contributions, not the mean of the model-specific percentages. The interaction is already allocated '
               'equally to the weather and host terms. Percentages are descriptive model ratios without probability '
               'intervals and do not measure the benefit of an adaptation intervention.')
    return rows, caption
