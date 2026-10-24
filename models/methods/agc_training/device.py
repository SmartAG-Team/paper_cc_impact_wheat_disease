"""Original mixed neural-device path; double-precision process stays on CPU."""
from types import MethodType
import torch


def _stage_features_on_device(self, batch):
    prepared = dict(batch)
    for name in ['known_future', 'future_valid', 'static']:
        prepared[name] = batch[name].to(self._neural_compute_device)
    features, valid = self._canonical_stage_features(prepared)
    return features.to('cpu'), valid.to('cpu')


def mixed_neural_device(model, device='mps'):
    """The source model/loss forward code and double-precision arrays stay on CPU."""
    if device == 'cpu':
        return model
    if device != 'mps' or not torch.backends.mps.is_available():
        raise ValueError('Available MPS or CPU device required')
    if hasattr(model, '_canonical_stage_features'):
        raise ValueError('Neural device is already configured')
    model._canonical_stage_features = model.stage_features
    model._neural_compute_device = torch.device(device)
    for name in ['seasonal_projection', 'context', 'dropout', 'encoder', 'cross_attention', 'query_norm']:
        getattr(model, name).to(device)
    model.stage_embedding = torch.nn.Parameter(model.stage_embedding.detach().to(device))
    model.stage_features = MethodType(_stage_features_on_device, model)
    return model

