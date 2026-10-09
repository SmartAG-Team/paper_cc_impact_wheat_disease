"""Inspectable native-stage comparison and compact deterministic checkpoint replay."""
from pathlib import Path
import argparse
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from .train import load_inputs,restore,sha
from .evaluate import predict_family

HERE=Path(__file__).resolve().parent


def export(inputs):
    out=HERE/'results';a,env,m=load_inputs(inputs)
    selection=json.loads((out/'selection_before_test.json').read_text())
    decision=json.loads((out/'decision.json').read_text())
    common=pd.read_csv(out/'common_stage_metrics.csv');own=pd.read_csv(out/'stage_metrics.csv')
    labels={10:'First leaf (GS10)',31:'First node (GS31)',51:'Heading (GS51)',85:'Soft dough (GS85)'}
    colors={'T-P-V':'#52697d','lstm':'#8a7446','tcn':'#237b81'}
    fig,axes=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
    pos=np.arange(4)
    for i,model in enumerate(colors):
        q=common.query("cohort=='testing' and model==@model").sort_values('stage')
        axes[0].barh(pos+(i-1)*.22,q.mae_days,height=.2,color=colors[model],label=model.upper() if model!='T-P-V' else model)
        for y,value in zip(pos+(i-1)*.22,q.mae_days):axes[0].text(value+.1,y,f'{value:.2f}',va='center',fontsize=8)
    axes[0].set(yticks=pos,yticklabels=[labels[s] for s in labels],xlabel='Mean absolute date error (days)',xlim=(0,13.6))
    axes[0].invert_yaxis();axes[0].legend(frameon=False,loc='lower right',fontsize=8)
    axes[0].set_title('(a) Same test events for every model',loc='left',fontsize=10)
    for i,model in enumerate(colors):
        q=own.query("cohort=='testing' and model==@model").sort_values('stage')
        missing=q.observed-q.matched
        axes[1].barh(pos+(i-1)*.22,missing,height=.2,color=colors[model])
        for y,value in zip(pos+(i-1)*.22,missing):axes[1].text(value+4,y,str(value),va='center',fontsize=8)
    axes[1].set(yticks=pos,yticklabels=[labels[s] for s in labels],xlabel='Observed events without a predicted median',xlim=(0,550))
    axes[1].invert_yaxis();axes[1].set_title('(b) Coverage on all eligible test events',loc='left',fontsize=10)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=9)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'Wheat_DL_phenology_comparison.{ext}',dpi=300)
    plt.close(fig)
    caption=('Wheat-fitted adaptations of the maize_phen_dl LSTM and causal temporal-convolution architectures. '
        'Panel (a) uses the identical intersection of native-stage test events for all three models. '
        'Panel (b) includes every weather-qualified observed event, retaining unreached medians. '
        'The TCN family was selected on validation stations before its test evaluation. '
        'The three-seed ensembles are trained only on wheat calibration stations. '
        'Station holdouts are retrospective and do not establish temporal or future-climate transfer. '
        'GS39 and GS65 are not native German calibration targets.')
    (out/'Wheat_DL_phenology_comparison.caption.txt').write_text(caption+'\n')
    # A source-key-selected panel, independent of model errors, supports public CPU replay.
    ids=np.flatnonzero(env.cohort.eq('testing').to_numpy())[:32]
    with np.load(Path(inputs)/'arrays.npz') as z:
        np.savez_compressed(HERE/'replay_inputs.npz',X=z['X'][ids],G=z['G'][ids],Y=z['Y'][ids],lengths=z['lengths'][ids],indices=ids)
    predictions={}
    for family in ['lstm','tcn']:
        paths=[HERE/'fits'/f'{family}_s{seed}.pt' for seed in [17,29,43]]
        pred=predict_family(a,m,ids,paths)
        predictions.update({family+'_'+key:value for key,value in pred.items()})
    np.savez_compressed(HERE/'replay_expected.npz',**predictions)
    env.iloc[ids].to_csv(HERE/'replay_environments.csv',index=False)
    record={'selection':'First 32 test environments in source station/sowing order; no error-based selection',
            'scope':'Inference replay only; full fitting requires the hashed external daily source',
            'inputs_sha256':sha(HERE/'replay_inputs.npz'),'expected_sha256':sha(HERE/'replay_expected.npz'),
            'environments_sha256':sha(HERE/'replay_environments.csv'),'input_metadata_sha256':sha(HERE/'input_metadata.json')}
    (HERE/'replay_manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    summary=[]
    for model,g in common.query("cohort=='testing'").groupby('model'):
        summary.append(dict(model=model,common_events=int(g.matched.sum()),pooled_mae=float(np.average(g.mae_days,weights=g.matched)),macro_mae=float(g.mae_days.mean())))
    pd.DataFrame(summary).to_csv(out/'common_overall_metrics.csv',index=False)
    # Extend, rather than replace, the completed evaluation's content manifest.
    receipt=json.loads((out/'receipt.json').read_text())
    receipt['artifact_generator_sha256']=sha(__file__)
    receipt['output_sha256']={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file() and p.name!='receipt.json'}
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path)
    args=p.parse_args();torch.set_num_threads(2);export(args.inputs)
