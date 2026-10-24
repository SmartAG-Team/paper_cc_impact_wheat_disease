"""Learn a weighted state-space metric from coordinate-held-out analog forecasts."""
import numpy as np
import pandas as pd
import torch


def _categories(frame):
    return (frame.source.astype(str) + '|' + frame.leaf_index.astype(str)).to_numpy()


def training_neighbor_mask(frame):
    category = _categories(frame)
    if {'latitude', 'longitude'}.issubset(frame.columns):
        coordinate = (frame.latitude.astype(str) + '|' + frame.longitude.astype(str)).to_numpy()
    else:
        coordinate = frame.field_id.astype(str).to_numpy()
    field = frame.field_id.astype(str).to_numpy()
    return ((category[:, None] == category[None, :]) &
        (coordinate[:, None] != coordinate[None, :]) & (field[:, None] != field[None, :]))


def _probabilities(query, library, metric, bandwidth, mass, mask):
    distance = ((query.square() * metric).sum(1)[:, None] +
        (library.square() * metric).sum(1)[None, :] - 2 * (query * metric) @ library.T)
    logits = torch.log(mass)[None, :] - distance.clamp_min(0) / (2 * bandwidth.square())
    # Finite sentinel keeps rows without eligible support differentiable.
    probability = torch.softmax(logits.masked_fill(~mask, -1e9), dim=1) * mask
    probability = probability / probability.sum(1, keepdim=True).clamp_min(1e-12)
    ess = 1 / probability.square().sum(1).clamp_min(1e-12)
    coefficient = torch.where(mask.any(1), ess / (ess + 2), torch.zeros_like(ess))
    return probability, coefficient


def fit_analogs(x, frame, baseline, weights, formulation='direct', epochs=300):
    if formulation not in ['direct', 'residual']:
        raise ValueError('Registered direct or residual formulation required')
    torch.set_num_threads(1)
    frame = frame.reset_index(drop=True)
    x, baseline, weights = np.asarray(x, float), np.asarray(baseline, float), np.asarray(weights, float)
    if not np.isfinite(x).all() or not np.isfinite(weights).all() or (weights <= 0).any():
        raise ValueError('Finite predictors and positive scientific weights required')
    x = np.column_stack([x, baseline / 100])
    w = weights / weights.sum()
    center = np.average(x, axis=0, weights=w)
    scale = np.maximum(.01, np.sqrt(np.average((x-center)**2, axis=0, weights=w)))
    xx = torch.tensor(np.clip((x-center)/scale, -8, 8), dtype=torch.float32)
    y = frame.observed_percent.to_numpy(float) / 100
    target = torch.tensor(y, dtype=torch.float32)
    base = torch.tensor(baseline / 100, dtype=torch.float32)
    value = target if formulation == 'direct' else target - base
    mass = torch.tensor(w, dtype=torch.float32)
    mask = torch.tensor(training_neighbor_mask(frame))
    metric_logits = torch.nn.Parameter(torch.zeros(x.shape[1]))
    log_bandwidth = torch.nn.Parameter(torch.zeros(()))
    optimizer = torch.optim.Adam([metric_logits, log_bandwidth], lr=.04)
    trace = []
    for epoch in range(epochs):
        optimizer.zero_grad()
        metric = torch.softmax(metric_logits, dim=0)
        bandwidth = log_bandwidth.clamp(-3.5, 3.).exp()
        probability, coefficient = _probabilities(xx, xx, metric, bandwidth, mass, mask)
        retrieved = probability @ value
        prediction = ((1-coefficient)*base + coefficient*retrieved if formulation == 'direct'
            else base + coefficient*retrieved)
        loss = (mass*(prediction-target).square()).sum()
        objective = loss + .0005*metric_logits.square().mean() + .0001*log_bandwidth.square()
        if not torch.isfinite(objective):
            raise RuntimeError('Nonfinite analog metric objective')
        objective.backward()
        optimizer.step()
        if epoch % 50 == 0 or epoch + 1 == epochs:
            trace.append(dict(epoch=epoch+1, coordinate_heldout_training_rmse=float(100*loss.detach().sqrt()),
                bandwidth=float(bandwidth.detach())))
    metric = torch.softmax(metric_logits.detach(), dim=0).numpy().astype(float)
    return dict(center=center, scale=scale, x=xx.numpy(), metric=metric,
        bandwidth=float(log_bandwidth.detach().clamp(-3.5, 3.).exp()), mass=w,
        values=value.numpy(), formulation=formulation, categories=_categories(frame),
        training_ids=frame.global_target_index.astype(int).tolist(),
        training_fields=frame.field_id.astype(str).tolist(),
        training_neighbor_mask=mask.numpy(), scientific_weight=weights, trace=trace,
        training_baseline=baseline, epochs=epochs)


def predict_analogs(fitted, x, frame, baseline):
    torch.set_num_threads(1)
    baseline = np.asarray(baseline, float)
    x = np.column_stack([np.asarray(x, float), baseline/100])
    query = torch.tensor(np.clip((x-fitted['center'])/fitted['scale'], -8, 8), dtype=torch.float32)
    library = torch.tensor(fitted['x'], dtype=torch.float32)
    mask = torch.tensor(_categories(frame)[:, None] == fitted['categories'][None, :])
    with torch.no_grad():
        probability, coefficient = _probabilities(query, library,
            torch.tensor(fitted['metric'], dtype=torch.float32), torch.tensor(fitted['bandwidth']),
            torch.tensor(fitted['mass'], dtype=torch.float32), mask)
        retrieved = (probability @ torch.tensor(fitted['values'])).numpy()
        coefficient = coefficient.numpy()
    result = ((1-coefficient)*baseline + coefficient*100*retrieved
        if fitted['formulation'] == 'direct' else baseline + coefficient*100*retrieved)
    result[~mask.numpy().any(axis=1)] = baseline[~mask.numpy().any(axis=1)]
    return np.clip(result, 0, 100)
