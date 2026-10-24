"""Render the restored monitoring predictions as a standalone working Figure 1."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.final_study.metrics import hierarchical_weights
from .innovation_common import ROOT
from .inputs import sha

SOURCE = ROOT/'data/final_monitoring/evaluation'
OUT = ROOT/'data/final_monitoring/reference'


def render():
    if OUT.exists(): raise FileExistsError('Preserve an existing working figure')
    receipt = json.loads((SOURCE/'verification.json').read_text()); assert receipt['passed']
    for name, digest in receipt['artifacts_sha256'].items(): assert sha(SOURCE/name) == digest
    OUT.mkdir(parents=True); data_dir = OUT/'data/fig1'; data_dir.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':12, 'axes.labelsize':12,
        'xtick.labelsize':12, 'ytick.labelsize':12, 'legend.fontsize':12, 'svg.fonttype':'none',
        'pdf.fonttype':42, 'axes.spines.right':False, 'axes.spines.top':False})
    fig, axes = plt.subplots(1, 2, figsize=(19/2.54, 9.2/2.54), sharex=True, sharey=True,
        layout='constrained')
    fig.set_constrained_layout_pads(w_pad=.03, h_pad=.03, wspace=.04, hspace=.02)
    frames, paths = [], []
    for ax, panel, design, filename in zip(axes, ['a','b'], ['temporal','spatial_50km'],
            ['assessment_predictions.csv','spatial_predictions.csv']):
        path = SOURCE/filename; paths.append(path)
        frame = pd.read_csv(path, float_precision='round_trip')
        x, y = frame.observed_percent.to_numpy(), frame.prediction_percent.to_numpy()
        assert len(frame) == 2017 and np.isfinite(np.r_[x,y]).all()
        ax.scatter(x, y, s=7, color='#0072B2', alpha=.28, edgecolors='none', rasterized=True)
        ax.plot([0,100],[0,100],color='#6B6B6B',linestyle='--',linewidth=1.0,zorder=0)
        ax.set(xlim=(-2,102),ylim=(-2,102),xticks=[0,25,50,75,100],yticks=[0,25,50,75,100],
            xlabel='Observed severity (%)')
        ax.set_box_aspect(1)
        ax.text(.02,1.035,f'({panel})',transform=ax.transAxes,fontsize=12,ha='left',va='bottom')
        ax.tick_params(pad=3)
        part = frame[['global_target_index','field_id','site_id','source','leaf_index','season_year',
            'date','observed_percent','prediction_percent','outer_fold']].copy()
        part.insert(0,'panel',panel); part.insert(1,'evaluation',design)
        part['scientific_weight'] = hierarchical_weights(frame)
        frames.append(part)
    axes[0].set_ylabel('Predicted severity (%)')
    fig.canvas.draw()
    from matplotlib.text import Text
    for artist in fig.findobj(Text):
        if artist.get_text(): assert artist.get_fontsize() == 12
    for extension in ['png','pdf','svg']:
        fig.savefig(OUT/f'fig1.{extension}',dpi=300,facecolor='white')
    plt.close(fig)
    pd.concat(frames,ignore_index=True).to_csv(data_dir/'predictions.csv',index=False)
    units = dict(panel='Figure panel', evaluation='Original validation design', global_target_index='Original target identity',
        field_id='Original field-season identity', site_id='Original location identity', source='Original data source',
        leaf_index='Nominal source-native endpoint code; not a common anatomical rank', season_year='Harvest year',
        date='Assessment date', observed_percent='Source-reported severity percentage',
        prediction_percent='Predicted severity percentage', outer_fold='Original outer fold',
        scientific_weight='Original within-design coordinate-year/field/native-endpoint/record relative mass')
    (data_dir/'variables.json').write_text(json.dumps(dict(variables=units,
        uncertainty='No uncertainty band is plotted; dots are individual assessments, not independent replicates',
        display='Scatterpoints have equal marker size; metric weights are retained separately'),indent=2)+'\n')
    caption = ('Figure 1. Restored rolling monitoring of wheat disease severity. '
        '(a) Predictions from five historical temporal splits. (b) Predictions from 15 chronology-plus-50 km spatial splits. '
        'Each panel retains the original 2,017 assessment records from 192 field-seasons at 121 locations. '
        'Blue points show individual assessments; dashed grey lines show 1:1 agreement. '
        'Predictions use the restored logit field-discrepancy formulation and strictly earlier same-field observations, '
        'with all query-day disease observations withheld together. Earlier observations in held spatial locations may '
        'be available at prediction time. Native endpoint codes do not establish common anatomical leaf ranks. '
        'These repeatedly inspected internal comparisons describe observation-conditioned monitoring; they do not '
        'validate observation-free climate projections. Points have equal marker sizes; reported performance metrics '
        'use the separately retained scientific hierarchy weights.\n')
    (OUT/'fig1_caption.txt').write_text(caption)
    result = dict(passed=True, figure='fig1', scope='Standalone working disease-validation figure; original integrated manuscript figure retained',
        width_cm=19., height_cm=9.2, font_pt=12, panels={'a':'temporal','b':'spatial_50km'},
        selected_model='discrepancy_logit', numeric_performance_annotations=False,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [*paths,SOURCE/'verification.json',Path(__file__)]},
        artifacts_sha256={str(p.relative_to(OUT)):sha(p) for p in OUT.rglob('*') if p.is_file()})
    (OUT/'figure_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    return OUT


if __name__ == '__main__': print(render())
