"""Standalone source-backed German station-held-out BBCH validation figure."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot(directory):
    directory = Path(directory).resolve()
    source = directory / 'heldout_test_figure_data.csv'
    rows = pd.read_csv(source)
    assert set(rows.denominator) == {'three_model_shared'}
    assert set(rows.cohort) == {'testing'}
    stages = [10, 31, 51, 85]
    models = ['GDD', 'T-P', 'T-P-V']
    assert len(rows) == 12 and not rows.duplicated(['model', 'stage']).any()
    sample_counts = rows.groupby('stage').matched.first().reindex(stages).astype(int)
    assert rows.groupby('stage').matched.nunique().eq(1).all()
    palette = {'GDD': '#A0A7AF', 'T-P': '#4776A8', 'T-P-V': '#32856C'}
    hatches = {'GDD': '', 'T-P': '//', 'T-P-V': '..'}
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.labelsize': 11, 'axes.edgecolor': '#32383E', 'text.color': '#263039',
        'axes.labelcolor': '#263039', 'xtick.color': '#263039', 'ytick.color': '#263039',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'figure.facecolor': 'white',
        'axes.facecolor': 'white'})
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.2), sharey=True)
    x = np.arange(len(stages))
    width = .23
    handles = []
    for axis, metric, label, panel in zip(axes, ['mae_days', 'rmse_days'],
            ['MAE (days)', 'RMSE (days)'], ['a', 'b']):
        for number, model in enumerate(models):
            values = rows.loc[rows.model == model].set_index('stage').reindex(stages)[metric]
            marks = axis.bar(x + (number - 1) * width, values, width,
                label=model.replace('-', '–'), color=palette[model],
                edgecolor='#263039', linewidth=.45, hatch=hatches[model], zorder=3)
            if metric == 'mae_days':
                handles.append(marks)
        axis.set_xticks(x, [f'BBCH {stage}\nn = {sample_counts.loc[stage]:,}' for stage in stages])
        axis.set_ylim(0, 30)
        axis.set_yticks(np.arange(0, 31, 5))
        axis.set_ylabel(label)
        axis.tick_params(axis='x', length=0, pad=8)
        axis.tick_params(axis='y', length=3, labelleft=True)
        axis.set_axisbelow(True)
        axis.grid(axis='y', color='#DEE2E6', linewidth=.55)
        axis.spines[['top', 'right']].set_visible(False)
        axis.text(0, 1.025, f'({panel})', transform=axis.transAxes,
            va='bottom', ha='left', fontweight='bold', fontsize=11)
    fig.legend(handles, ['GDD', 'T–P', 'T–P–V'], frameon=False,
        ncol=3, loc='upper center', bbox_to_anchor=(.5, .985), columnspacing=2.2)
    fig.text(.5, .028, 'German station holdout • 1985–2015 • realized-weather hindcasts',
        ha='center', fontsize=9.5)
    fig.subplots_adjust(left=.085, right=.985, bottom=.22, top=.82, wspace=.28)
    png = directory / 'german_station_heldout_validation.png'
    pdf = directory / 'german_station_heldout_validation.pdf'
    fig.savefig(png, dpi=600, facecolor='white')
    fig.savefig(pdf, facecolor='white')
    plt.close(fig)
    caption = ('German winter-wheat phenology event-date errors on held-out stations. '
        '(a) MAE and (b) RMSE in calendar days for GDD, T–P and T–P–V. '
        'The frozen parameters were fitted on 187 training stations; testing uses 1,140 '
        'eligible stations from 1,223 reserved stations, with 83 scoring exclusions. '
        'All models use identical matched station/sowing-cycle/BBCH event keys: '
        '13,471 at BBCH10, 12,109 at BBCH31, 12,541 at BBCH51 and 2,757 at BBCH85. '
        'These 40,878 shared events represent 94.68% of 43,177 observed physiological events. '
        'Sowing is a supplied management anchor and is excluded from error metrics. '
        'BBCH85 denotes soft dough. Both training and testing contain observations in every '
        'calendar year from 1985 to 2015; the holdout concerns stations, not years. '
        'Realized full-season weather conditions these retrospective hindcasts. '
        'Missing predictions and incomplete cycles remain in coverage, while MAE/RMSE '
        'are conditional on shared matched events. German phenology evaluation does not '
        'establish European Septoria disease-model validation. '
        'Source: AGC-PhenFormer-v2-final-results-20260926/process_models/testing, '
        'process-only evaluation refreshed 2026-09-28.\n')
    (directory / 'german_station_heldout_validation_caption.txt').write_text(caption)
    receipt = dict(source_file=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        rows=12, units='calendar days', denominator='three-model shared matched events',
        stages=stages, shared_matched_by_stage=sample_counts.to_dict(),
        holdout='station; no year holdout', BBCH85='soft dough',
        png_pixels=[5040, 2520], pdf_inches=[8.4, 4.2],
        artifacts={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in [png, pdf]},
        note='Point error metrics; no uncertainty intervals or significance claims.')
    (directory / 'figure_provenance.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plot(args.directory), indent=2))
