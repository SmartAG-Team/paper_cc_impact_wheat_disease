"""Native three-seed wheat AGC recipe using immutable retained prepared inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from analysis.final_study.metrics import RetainedFiles
from . import adapter
from .device import mixed_neural_device
from .common import crps_values, json_write, file_record, verify_records, digest

HERE = None
PROTOCOL = None
ORIGINAL = 'analysis/paper_study/agc_wheat_backbone_20261012/'
CURRENT = 'analysis/paper_study/phenology_transfer_20261010/'


SETTINGS = dict(strategy='process_residual', encoder='gated_delta', d_model=32,
                n_heads=4, layers=1, dropout=.1)


SEEDS = [17, 29, 43]


OPTIMIZATION = dict(max_epochs=20, patience=5, batch_size=32, learning_rate=.0005,
                    weight_decay=.0001, duration_weight=.05, monitor_duration_weight=.25,
                    monitor_cycles=512, threads=4, neural_device='mps')


def supervision(targets, horizon=330):
    y = np.asarray(targets)
    if y.ndim != 2 or y.shape[1] != 4 or y.dtype.kind not in 'iu' or np.any(y < -1):
        raise ValueError('Four integer event-day columns; missing value is −1')
    within = (y >= 0) & (y < horizon)
    beyond = y >= horizon
    observed = np.full((len(y), horizon+1), -1, np.int64)
    observed[:, 0] = 0
    for i, j in zip(*np.nonzero(within)):
        observed[i, int(y[i, j])+1] = j+1
    return dict(event_kind=np.where(within, 1, np.where(beyond, 2, 0)).astype(np.int64),
                event_lower=np.where(within, y, 0).astype(np.int64),
                event_upper=np.where(within, y+1, 0).astype(np.int64),
                event_limit=np.full(y.shape, horizon, np.int64),
                cycle_weight=np.ones(len(y), np.float32), observed_state=observed)


def monitor_membership(cycles, validation, maximum=512):
    subset = cycles.iloc[np.asarray(validation)].copy()
    if not subset.partition.eq('val').all() or not subset.harvest_year.le(2015).all():
        raise ValueError('Epoch monitoring is restricted to historical validation')
    subset['_index'] = np.asarray(validation)
    subset['_rank'] = [hashlib.sha256(f'{r.PEP_ID}:{r.SOWING_DATE}:20261012'.encode()).hexdigest()
                       for r in subset.itertuples(index=False)]
    chosen = subset.sort_values('_rank').drop_duplicates('PEP_ID').head(maximum)
    return np.sort(chosen['_index'].to_numpy(int))


def slice_batch(batch, positions):
    return {key: value[positions] for key, value in batch.items()}


def duration_loss(cdf, y):
    observed = torch.as_tensor(y, dtype=torch.long)
    valid = (observed[:, 2:] >= 0).all(1) & (observed[:, 2:] < cdf.shape[2]).all(1)
    if not valid.any():
        return cdf.sum()*0.
    predicted = (cdf[:, 2]-cdf[:, 3]).sum(1)
    return (predicted[valid]-(observed[:,3]-observed[:,2])[valid]).abs().mean()


def predict(model, batch, batch_size=64):
    model.eval()
    values=[]
    with torch.no_grad():
        for start in range(0, len(batch['static']), batch_size):
            part=slice_batch(batch,slice(start,start+batch_size))
            values.append(adapter.model_outputs_cdf(model(part)).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(values)


def monitor_score(cdf, targets, duration_weight=.25):
    crps=float(np.nanmean(crps_values(cdf,targets),axis=0).mean())
    known=(targets[:,2:]>=0).all(1)&(targets[:,2:]<cdf.shape[2]).all(1)
    mean_duration=(cdf[:,2]-cdf[:,3]).sum(1)
    duration=float(np.abs(mean_duration[known]-(targets[:,3]-targets[:,2])[known]).mean())
    return crps+duration_weight*duration


def _fit_one(seed, protocol, arrays, cycles, train, monitor):
    folder=HERE/'fits'/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'receipt.json').exists():
        receipt=json.loads((folder/'receipt.json').read_text());verify_records(receipt['outputs'])
        assert receipt['protocol_sha256']==digest(PROTOCOL)
        return
    torch.manual_seed(seed);np.random.seed(seed)
    settings=protocol['settings'];options=protocol['optimization']
    model=adapter.make_model(settings)
    if options['neural_device']!='cpu':model=mixed_neural_device(model,options['neural_device'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=options['learning_rate'],weight_decay=options['weight_decay'])
    preprocessing=json.loads((HERE/'preprocessing.json').read_text())
    train_batch=adapter.input_batch(arrays['weather'],arrays['p'],arrays['inc'],cycles,train,
                                    preprocessing,supervision(arrays['y']))
    monitor_batch=adapter.input_batch(arrays['weather'],arrays['p'],arrays['inc'],cycles,monitor,preprocessing)
    targets=np.asarray(arrays['y'][train]);monitor_targets=np.asarray(arrays['y'][monitor])
    rng=np.random.default_rng(seed);best=float('inf');stale=0;log=[]
    for epoch in range(1,options['max_epochs']+1):
        started=time.monotonic();model.train();losses=[]
        permutation=rng.permutation(len(train))
        for start in range(0,len(train),options['batch_size']):
            indices=permutation[start:start+options['batch_size']]
            batch=slice_batch(train_batch,indices);optimizer.zero_grad(set_to_none=True)
            output=model(batch);cdf=adapter.model_outputs_cdf(output)
            loss=adapter.model_loss(output,batch)['loss']+options['duration_weight']*duration_loss(cdf,targets[indices])
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite canonical wheat training loss')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step();losses.append(float(loss.detach()))
        cdf=predict(model,monitor_batch,options['batch_size'])
        score=monitor_score(cdf,monitor_targets,options['monitor_duration_weight'])
        if not np.isfinite(score):raise FloatingPointError('Nonfinite historical monitor score')
        better=score<best-1e-6
        if better:
            best=score;stale=0
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                            settings=settings,seed=seed,epoch=epoch,monitor_score=score,
                            preprocessing=preprocessing,protocol_sha256=digest(PROTOCOL)),folder/'selected.pt')
        else:stale+=1
        row=dict(seed=seed,epoch=epoch,training_loss=float(np.mean(losses)),monitor_score=score,
                 selected=better,seconds=time.monotonic()-started)
        log.append(row);json_write(folder/'epochs.json',log);print(json.dumps(row),flush=True)
        if stale>=options['patience']:break
    cycles.iloc[train].to_csv(folder/'training_membership.csv',index=False)
    cycles.iloc[monitor].to_csv(folder/'monitor_membership.csv',index=False)
    json_write(folder/'receipt.json',dict(status='fitted',seed=seed,protocol_sha256=digest(PROTOCOL),
        selected_monitor_score=best,epochs=len(log),fit_targets_end_in2015=True,
        outputs=[file_record(folder/name) for name in ['selected.pt','epochs.json','training_membership.csv','monitor_membership.csv']]))


def load_prepared():
    files = RetainedFiles()
    cycles = files.csv(CURRENT + 'cycles.csv')
    arrays = dict(weather=np.load(files.resolve(ORIGINAL + 'data/historical_weather.npy'), mmap_mode='r'))
    for key, name in [('p','progress'),('inc','increments'),('y','targets')]:
        arrays[key] = np.load(files.resolve(CURRENT + 'arrays/' + name + '.npy'), mmap_mode='r')
    metadata = files.json(CURRENT + 'model_metadata.json')
    train = np.flatnonzero(cycles.partition.eq('train'))
    validation = np.flatnonzero(cycles.partition.eq('val'))
    monitor = files.csv(ORIGINAL + 'epoch_monitor.csv').source_index.to_numpy(int)
    protocol = files.json(ORIGINAL + 'fit_protocol.json')
    np.testing.assert_array_equal(monitor, monitor_membership(cycles, validation, protocol['optimization']['monitor_cycles']))
    if len(train) != protocol['train_cycles'] or not cycles.iloc[train].harvest_year.le(2015).all():
        raise ValueError('Frozen native training membership changed')
    return arrays, cycles, metadata, train, monitor, protocol


def _prepare_output(output, protocol):
    global HERE, PROTOCOL
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    files = RetainedFiles()
    original = files.json(ORIGINAL + 'fit_protocol.json')
    if protocol != original:
        raise ValueError('Native fitting recipe must match the frozen selected experiment')
    shutil.copyfile(files.resolve(ORIGINAL + 'fit_protocol.json'), folder / 'fit_protocol.json')
    shutil.copyfile(files.resolve(ORIGINAL + 'preprocessing.json'), folder / 'preprocessing.json')
    HERE, PROTOCOL = folder, folder / 'fit_protocol.json'


def fit_one(seed, protocol, arrays, cycles, train, monitor, *, output):
    if seed not in protocol['seeds']:
        raise ValueError('Seed is outside the frozen native experiment')
    _prepare_output(output, protocol)
    return _fit_one(seed, protocol, arrays, cycles, train, monitor)


def run(output):
    arrays, cycles, metadata, train, monitor, protocol = load_prepared()
    _prepare_output(output, protocol)
    torch.set_num_threads(protocol['optimization']['threads'])
    for seed in protocol['seeds']:
        _fit_one(seed, protocol, arrays, cycles, train, monitor)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
