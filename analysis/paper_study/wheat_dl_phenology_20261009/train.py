"""Fixed-budget wheat fitting; station-test labels never select a checkpoint."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import torch
from model.wheat_phenology_dl.network import WheatDevelopmentModel, observed_crps, quantile_days

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f,'sha256').hexdigest()


def load_inputs(path):
    path=Path(path);m=json.loads((path/'metadata.json').read_text())
    if sha(HERE/'protocol.json')!=m['protocol_sha256']:raise ValueError('Protocol changed after input preparation')
    if sha(path/'arrays.npz')!=m['input_arrays_sha256']:raise ValueError('Input array hash mismatch')
    if sha(path/'environments.csv')!=m['environments_sha256']:raise ValueError('Environment identity hash mismatch')
    with np.load(path/'arrays.npz') as z:a={k:z[k] for k in z.files}
    a['X']=((a['X']-np.asarray(m['mean'],np.float32))/np.asarray(m['std'],np.float32)).astype(np.float32)
    env=pd.read_csv(path/'environments.csv')
    assert len(env)==len(a['X']) and len(env)==len(a['Y'])
    return a,env,m


def batch_cdf(model,a,ids,m,device,batch=256):
    model.eval()
    with torch.no_grad():
        for k in range(0,len(ids),batch):
            ix=ids[k:k+batch]
            x=torch.as_tensor(a['X'][ix],device=device);g=torch.as_tensor(a['G'][ix],device=device)
            _,cdf=model(x,g,m['thermal_scale'])
            yield ix,cdf.detach().cpu().numpy()


def crps_by_stage(cdf,y,lengths):
    days=np.arange(1,cdf.shape[2]+1)
    truth=days[None,None,:]>=y[:,:,None]
    errors=np.square(cdf-truth)*(days[None,None,:]<=lengths[:,None,None])
    return np.where((y>0)&(y<=lengths[:,None]),errors.sum(2),np.nan)


def validation_score(model,a,ids,m,device):
    sums=np.zeros(4);counts=np.zeros(4)
    for ix,cdf in batch_cdf(model,a,ids,m,device):
        s=crps_by_stage(cdf,a['Y'][ix],a['lengths'][ix])
        sums+=np.nansum(s,0);counts+=np.isfinite(s).sum(0)
    if (counts==0).any():raise ValueError('No validation support at a native stage')
    return float(np.mean(sums/counts))


def restore(checkpoint,device='cpu'):
    b=torch.load(checkpoint,map_location='cpu',weights_only=False)
    m=b['metadata'];c=b['config']
    model=WheatDevelopmentModel(m['thresholds'],len(m['features']),encoder=c['encoder'],width=c['width'])
    model.load_state_dict(b['state_dict']);model.to(device);model.eval()
    return model,b


def fit(a,env,m,encoder,seed,out,device):
    protocol=json.loads((HERE/'protocol.json').read_text());out=Path(out);out.mkdir(parents=True,exist_ok=True)
    tag=f'{encoder}_s{seed}';report_path=out/(tag+'.json');checkpoint=out/(tag+'.pt')
    fit_ids=np.flatnonzero(env.cohort.eq('calibration').to_numpy()&(a['Y']>0).any(1))
    tune_ids=np.flatnonzero(env.cohort.eq('validation').to_numpy()&(a['Y']>0).any(1))
    assert len(fit_ids) and len(tune_ids) and not set(env.iloc[fit_ids].PEP_ID)&set(env.iloc[tune_ids].PEP_ID)
    config=dict(encoder=encoder,seed=seed,width=protocol['width'],maximum_epochs=protocol['maximum_epochs'],
                learning_rate=protocol['learning_rate'],patience=protocol['patience'],batch_size=protocol['batch_size'])
    request=dict(config=config,protocol_sha256=sha(HERE/'protocol.json'),input_arrays_sha256=m['input_arrays_sha256'],
                 network_sha256=sha(ROOT/'model/wheat_phenology_dl/network.py'),trainer_sha256=sha(__file__),
                 training_ids_sha256=hashlib.sha256(fit_ids.tobytes()).hexdigest(),
                 tuning_ids_sha256=hashlib.sha256(tune_ids.tobytes()).hexdigest())
    if report_path.exists():
        report=json.loads(report_path.read_text())
        if any(report[k]!=v for k,v in request.items()) or sha(checkpoint)!=report['checkpoint_sha256']:
            raise ValueError('Existing fit differs from current fixed request')
        return report
    torch.manual_seed(seed);rng=np.random.default_rng(seed)
    model=WheatDevelopmentModel(m['thresholds'],len(m['features']),encoder=encoder,width=config['width']).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],weight_decay=1e-4)
    counts=(a['Y'][fit_ids]>0).sum(0)
    if (counts==0).any():raise ValueError('Missing calibration stage')
    weights=torch.tensor(len(fit_ids)/(4*counts),dtype=torch.float32,device=device)
    best=np.inf;wait=0;history=[]
    for epoch in range(1,config['maximum_epochs']+1):
        tick=time.time();model.train();order=rng.permutation(fit_ids);total=0.
        for k in range(0,len(order),config['batch_size']):
            ix=order[k:k+config['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            _,cdf=model(torch.as_tensor(a['X'][ix],device=device),torch.as_tensor(a['G'][ix],device=device),m['thermal_scale'])
            loss=observed_crps(cdf,torch.as_tensor(a['Y'][ix],device=device),weights,torch.as_tensor(a['lengths'][ix],device=device))
            if not torch.isfinite(loss):raise FloatingPointError('Nonfinite training objective')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            total+=float(loss.detach().cpu())*len(ix)
        value=validation_score(model,a,tune_ids,m,device)
        if not np.isfinite(value):raise FloatingPointError('Nonfinite validation objective')
        history.append(dict(epoch=epoch,training_crps=total/len(order),validation_crps=value,seconds=time.time()-tick))
        if value<best-1e-4:
            best=value;best_epoch=epoch;wait=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(dict(request,metadata=m,state_dict=state,best_epoch=epoch,validation_crps=value),checkpoint)
        else:wait+=1
        pd.DataFrame(history).to_csv(out/(tag+'_history.csv'),index=False)
        print(json.dumps(dict(fit=tag,**history[-1])),flush=True)
        if wait>=config['patience']:break
    report=dict(request,status='complete',best_epoch=best_epoch,epochs_run=len(history),validation_crps=best,
                training_environments=len(fit_ids),tuning_environments=len(tune_ids),test_labels_used=False,
                device=device,torch_version=torch.__version__,elapsed_seconds=sum(x['seconds'] for x in history),
                checkpoint_sha256=sha(checkpoint),parameter_count=sum(p.numel() for p in model.parameters()))
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--device',choices=['cpu','mps'],default='cpu');p.add_argument('--encoder',choices=['lstm','tcn']);p.add_argument('--seed',type=int)
    args=p.parse_args();torch.set_num_threads(2)
    if args.device=='mps':torch.mps.set_per_process_memory_fraction(.25)
    a,env,m=load_inputs(args.inputs);protocol=json.loads((HERE/'protocol.json').read_text())
    for encoder in [args.encoder] if args.encoder else protocol['candidates']:
        for seed in [args.seed] if args.seed is not None else protocol['seeds']:
            fit(a,env,m,encoder,seed,args.output,args.device)


if __name__=='__main__':main()
