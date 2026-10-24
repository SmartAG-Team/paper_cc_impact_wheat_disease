"""Original nine-fit wheat benchmark and historical CRPS family selection."""
import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np
import pandas as pd
import torch
from analysis.final_study.metrics import RetainedFiles
from analysis.methods.wheat_phenology.wheat_transfer import HORIZON, STAGES, WheatPhenology, predict_batches, digest, validate_splits

HERE = None
ORIGINAL = 'analysis/paper_study/phenology_transfer_20261010/'


SEEDS = [17, 29, 43]


MODES = ["ordered_process", "ordered_lstm", "stage_attention"]


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def crps_values(cdf, targets):
    truth = np.arange(HORIZON)[None, None, :] >= np.asarray(targets)[:, :, None]
    scores = np.sum(np.square(cdf - truth), axis=2)
    return np.where(targets >= 0, scores, np.nan)


def torch_loss(cdf, targets, weights):
    truth = torch.arange(HORIZON)[None, None, :] >= targets[:, :, None]
    scores = ((cdf - truth.double()) ** 2).sum(2)
    valid = targets >= 0
    return ((scores * valid) * weights[None]).sum() / (valid * weights[None]).sum()


def _train(epochs=24, patience=5, device="cpu", batch=128):
    torch.set_num_threads(2)
    a, records, metadata = load_arrays()
    train_ix = np.flatnonzero(records.partition.eq("train"))
    val_ix = np.flatnonzero(records.partition.eq("val"))
    folder = HERE / "checkpoints"; folder.mkdir(exist_ok=True)
    protocol = dict(seeds=SEEDS, models=MODES, maximum_epochs=epochs, patience=patience, batch_size=batch,
                    device=device, threads=2, width=32, learning_rates=dict(ordered_process=.01, ordered_lstm=.001, stage_attention=.001),
                    weight_decay=.0001, gradient_clip=1., correction_bound=1.,
                    objective="stage-equal restricted-horizon CRPS", selection="Mean of three seed-specific historical-validation CRPS minima; family selection never reads later-period errors",
                    train_indices_sha256=__import__("hashlib").sha256(train_ix.tobytes()).hexdigest(),
                    val_indices_sha256=__import__("hashlib").sha256(val_ix.tobytes()).hexdigest(),
                    train_station_ids=sorted(records.loc[train_ix, "PEP_ID"].unique().tolist()),
                    validation_station_count=int(records.loc[val_ix, "PEP_ID"].nunique()),
                    source_code_sha256=original_recipe()['source_code_sha256'])
    frozen = HERE / "fit_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text()) != protocol:
        raise ValueError("A different fitting protocol is already recorded")
    write_json(frozen, protocol)
    reports = []
    counts = (a["y"][train_ix] >= 0).sum(0)
    weights = torch.tensor(counts.max() / counts, dtype=torch.float64)
    for mode in MODES:
        for seed in SEEDS:
            tag = f"{mode}_s{seed}"
            reportpath = folder / (tag + ".json")
            if reportpath.exists():
                reports.append(json.loads(reportpath.read_text())); continue
            torch.manual_seed(seed); rng = np.random.default_rng(seed)
            model = WheatPhenology(metadata, mode).set_neural_device(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=protocol["learning_rates"][mode], weight_decay=.0001)
            history = []; best = np.inf; wait = 0; best_epoch = 0; start = time.time(); best_state = None
            for epoch in range(1, epochs + 1):
                model.train(); order = rng.permutation(train_ix); total = 0; tick = time.time()
                for offset in range(0, len(order), batch):
                    ix = order[offset:offset + batch]
                    x, p, inc, y = [torch.as_tensor(np.asarray(a[key][ix])) for key in ["x", "p", "inc", "y"]]
                    optimizer.zero_grad(set_to_none=True)
                    cdf = model(x, p, inc)
                    loss = torch_loss(cdf, y, weights)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Nonfinite loss at {tag} epoch {epoch}")
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.); optimizer.step()
                    total += float(loss.detach()) * len(ix)
                prediction = predict_batches(model, a["x"], a["p"], a["inc"], val_ix, batch=256)
                score = float(np.nanmean(np.nanmean(crps_values(prediction, a["y"][val_ix]), axis=0)))
                history.append(dict(epoch=epoch, train_crps=total / len(train_ix), validation_crps=score, seconds=time.time() - tick))
                if score < best - 1e-5:
                    best = score; best_epoch = epoch; wait = 0
                    best_state = {key:value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    wait += 1
                print(f"{tag} epoch={epoch} validation_CRPS={score:.5f} best={best:.5f} elapsed={time.time()-start:.1f}s", flush=True)
                if wait >= patience:
                    break
            checkpoint = folder / (tag + ".pt")
            torch.save(dict(schema_version=1, mode=mode, metadata=metadata, state_dict=best_state, seed=seed,
                            best_epoch=best_epoch, maize_weights_used=False, stage_codes=STAGES,
                            data_manifest_sha256=digest(HERE / "data_manifest.json"), fit_protocol_sha256=digest(frozen)), checkpoint)
            report = dict(model=mode, seed=seed, best_epoch=best_epoch, validation_crps=best, epochs_run=len(history),
                          seconds=time.time()-start, history=history, checkpoint_sha256=digest(checkpoint),
                          later_test_scores_inspected=False)
            write_json(reportpath, clean_json(report)); reports.append(report)
    validation = pd.DataFrame([{k:r[k] for k in ["model", "seed", "best_epoch", "validation_crps", "seconds"]} for r in reports])
    validation.to_csv(HERE / "validation_selection.csv", index=False)
    mean = validation.groupby("model").validation_crps.mean().sort_values()
    # Selection is frozen before evaluation(), which reads later-period errors.
    selected = str(mean.index[0])
    decision = dict(selected_family=selected, mean_validation_crps=mean.to_dict(), seeds=SEEDS,
                    selected_checkpoint_policy="All three selected-family checkpoints; pooled CDF equal-weight ensemble",
                    test_error_access_before_selection=False, retrospective_reused_data=True,
                    limitation="Architecture shortlist informed by an earlier maize study; wheat outcomes used only within declared training/validation partitions")
    write_json(HERE / "selection.json", decision)
    return decision


def load_arrays():
    files = RetainedFiles()
    arrays = {key:np.load(files.resolve(ORIGINAL + 'arrays/' + name + '.npy'), mmap_mode='r')
              for key, name in [('x','features'),('p','progress'),('inc','increments'),('y','targets')]}
    records = files.csv(ORIGINAL + 'cycles.csv')
    validate_splits(records)
    return arrays, records, files.json(ORIGINAL + 'model_metadata.json')


def original_recipe():
    return RetainedFiles().json(ORIGINAL + 'fit_protocol.json')


def train(output, *, epochs=None, patience=None, device=None, batch=None):
    global HERE
    recipe = original_recipe()
    expected = [recipe['maximum_epochs'], recipe['patience'], recipe['device'], recipe['batch_size']]
    arguments = [epochs, patience, device, batch]
    if any(value is not None and value != fixed for value, fixed in zip(arguments, expected)):
        raise ValueError('Only the frozen established crop-fitting recipe is retained')
    folder = Path(output).resolve()
    folder.mkdir(parents=True, exist_ok=False)
    files = RetainedFiles()
    shutil.copyfile(files.resolve(ORIGINAL + 'data_manifest.json'), folder / 'data_manifest.json')
    HERE = folder
    return _train(*expected)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    train(parser.parse_args().output)
