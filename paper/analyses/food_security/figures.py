"""Standalone scientific figures bound to executed extension tables."""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .exposure import MODELS

COLORS = ['#245979', '#867047', '#577b70']


def _save(figure, directory, name):
    paths = []
    for suffix in ['png', 'pdf', 'svg']:
        path = directory / (name + '.' + suffix)
        figure.savefig(path, dpi=220, bbox_inches='tight', facecolor='white')
        paths.append(path.name)
    plt.close(figure)
    return paths


def render(output, *, directory='figures'):
    output = Path(output)
    if not directory or Path(directory).name != directory or directory in {'.', '..'}:
        raise ValueError('Figure directory must be a distinct child-directory name')
    destination = output / directory
    destination.mkdir(exist_ok=False)
    paths = []
    with plt.rc_context({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False}):
        source = output / 'exposure/production_exposure.csv'
        if source.is_file():
            frame = pd.read_csv(source)
            selected = frame.loc[frame.domain_type.eq('Europe') & frame.weight_basis.eq('total')
                                & frame.period.eq('2071-2100') & frame.minimum_paired_seasons.eq(1)]
            selected.to_csv(destination / 'exposure_figure_source.csv', index=False)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), layout='constrained')
            styles = ['-', '--', ':']
            for i, scenario in enumerate(['ssp126', 'ssp245', 'ssp585']):
                part = selected.loc[selected.members.eq('|'.join(MODELS)) & selected.scenario.eq(scenario)].sort_values('threshold_days')
                axes[0].plot(part.threshold_days, part.unanimous_increase_share_percent,
                             marker=['o', 's', '^'][i], color=COLORS[i], linestyle=styles[i],
                             label=['SSP1–2.6', 'SSP2–4.5', 'SSP5–8.5'][i])
            axes[0].set(xlabel='Required HAD-loss increase in every model (days)',
                        ylabel='Baseline production exposed (%)', ylim=(0, 100),
                        title='a  Magnitude threshold')
            axes[0].legend(frameon=False)
            subsets = [MODELS, MODELS[1:], [MODELS[0], MODELS[2]], MODELS[:2]]
            labels = ['All three', 'Without\nACCESS', 'Without\nMPI', 'Without\nMRI']
            x = np.arange(4)
            for i, scenario in enumerate(['ssp126', 'ssp585']):
                heights = []
                for members in subsets:
                    row = selected.loc[selected.members.eq('|'.join(members)) & selected.scenario.eq(scenario)
                                       & selected.threshold_days.eq(0)]
                    if len(row) != 1:
                        raise ValueError('Incomplete unique climate-composition figure support')
                    heights.append(float(row.ensemble_increase_share_percent.iloc[0]))
                axes[1].bar(x + (i - .5) * .36, heights, width=.36, color=COLORS[i * 2],
                            edgecolor='white', hatch='' if i == 0 else '//',
                            label=['SSP1–2.6', 'SSP5–8.5'][i])
            axes[1].set(xticks=x, xticklabels=labels, ylabel='Baseline production in positive-mean cells (%)',
                        ylim=(0, 100), title='b  Climate-model composition')
            axes[1].legend(frameon=False, loc='lower left', bbox_to_anchor=(0., 1.02), ncol=2)
            axes[1].set_title('b  Climate-model composition', pad=38)
            for ax in axes:
                ax.grid(axis='y', color='#e5e5e5', zorder=0)
                ax.set_axisbelow(True)
            fig.suptitle('Late-century wheat production exposure — fixed circa-2020 baseline', fontsize=13)
            paths += _save(fig, destination, 'production_exposure_robustness')

        source = output / 'benchmark/summary.csv'
        if source.is_file():
            frame = pd.read_csv(source)
            pooled = frame.loc[frame.source.eq('pooled') & frame.test_year.eq(0)]
            names = ['training_leaf_mean', 'calendar_ridge', 'weather_calendar_rf']
            labels = ['Training leaf mean', 'Calendar ridge', 'Weather/calendar RF']
            fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout='constrained')
            used = []
            for ax, endpoint, unit in zip(axes, ['assessment', 'window_integral'],
                                           ['percentage points', 'percentage-point days']):
                part = pooled.loc[pooled.endpoint.eq(endpoint)].set_index('model')
                if part.index.duplicated().any() or not set(names) <= set(part.index):
                    raise ValueError('Incomplete unique autonomous benchmark figure support')
                values = part.loc[names, 'rmse'].to_numpy(float)
                finite = np.isfinite(values)
                if np.any(values[finite] < 0):
                    raise ValueError('RMSE cannot be negative')
                scale = max(float(values[finite].max()), 1.) if finite.any() else 1.
                ax.barh(np.arange(3)[finite], values[finite],
                        color=np.asarray(COLORS)[finite].tolist(), height=.6)
                ax.set(yticks=np.arange(3), yticklabels=labels, xlabel='RMSE (' + unit + ')',
                       title='Assessment severity' if endpoint == 'assessment' else 'Observed-window integral')
                ax.invert_yaxis()
                ax.set_xlim(0, scale * 1.17)
                for y, value in enumerate(values):
                    ax.text(value + scale * .02 if np.isfinite(value) else scale * .03,
                            y, f'{value:.2f}' if np.isfinite(value) else 'Unavailable', va='center')
                used.append(part.loc[names].reset_index())
            pd.concat(used).to_csv(destination / 'benchmark_figure_source.csv', index=False)
            fig.suptitle('Fresh autonomous candidates — earlier-year and held-site evaluation', fontsize=13)
            fig.supxlabel('Retrospective comparisons with realized weather; missing scores remain unavailable', fontsize=9)
            paths += _save(fig, destination, 'autonomous_disease_benchmark')

        source = output / 'attribution/attribution_summary.csv'
        if source.is_file():
            frame = pd.read_csv(source)
            part = frame.loc[frame.model.eq('ensemble') & frame.support.eq('across_settings')].copy()
            within = frame.loc[frame.model.eq('ensemble') & frame.support.eq('within_setting')].copy()
            if part.setting.duplicated().any():
                raise ValueError('Duplicate ensemble attribution setting')
            order = ['baseline', 'anthesis_early_5pct', 'anthesis_late_5pct', 'terminal_early_5pct',
                     'terminal_late_5pct', 'reference_linear_senescence', 'field_ungated_damage', 'fixed_common_overlap']
            labels = ['Baseline', 'Earlier flowering', 'Later flowering', 'Earlier terminal stage',
                      'Later terminal stage', 'Reference senescence', 'Ungated damage', 'Common overlap window']
            part = part.set_index('setting').loc[order].reset_index()
            within = within.set_index('setting').loc[order].reset_index()
            pd.concat([part, within]).to_csv(destination / 'attribution_figure_source.csv', index=False)
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), layout='constrained')
            y = np.arange(len(part))
            for i, component in enumerate(['weather', 'host', 'net']):
                axes[0].plot(part[component + '_had_days'], y + (i - 1) * .16,
                             ['o', 's', '^'][i], color=COLORS[i], label=['Weather', 'Host', 'Net'][i])
            axes[0].axvline(0, color='#555555', linewidth=.8)
            axes[0].set(yticks=y, yticklabels=labels, xlabel='Normalized HAD-loss contribution (days)',
                        title='a  Contributions on shared setting support')
            axes[0].invert_yaxis()
            axes[0].legend(frameon=False, loc='lower left', bbox_to_anchor=(0., 1.02), ncol=3)
            axes[0].set_title('a  Contributions on shared setting support', pad=38)
            axes[1].barh(y - .16, 100 * within.common_area_time_fraction, color=COLORS[0], height=.3,
                         label='Within setting')
            axes[1].barh(y + .16, 100 * part.common_area_time_fraction, color='#c2c7ca', height=.3,
                         label='Across all settings')
            axes[1].set(yticks=y, yticklabels=labels, xlim=(0, 100),
                        xlabel='Valid weighted area–time support (%)', title='b  Integration-window support')
            axes[1].invert_yaxis()
            axes[1].legend(frameon=False, loc='lower left', bbox_to_anchor=(0., 1.02), ncol=2)
            axes[1].set_title('b  Integration-window support', pad=38)
            fig.suptitle('Sampled SSP5–8.5 attribution — diagnostic settings and explicit support', fontsize=13)
            fig.supxlabel('Uncalibrated sensitivity scenarios; point estimates without uncertainty bands', fontsize=9)
            paths += _save(fig, destination, 'attribution_sensitivity')
    return {'files': paths, 'source_csvs': sorted(path.name for path in destination.glob('*source.csv')),
            'intervals_shown': False, 'scope': 'Point estimates and support; no calibrated uncertainty bands'}
