"""Persist and verify the data and station partition used to fit a checkpoint."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path


@lru_cache(maxsize=16)
def _file_hash(path, size, mtime_ns):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def file_hash(path):
    path = Path(path).resolve(); stat = path.stat()
    return _file_hash(str(path), stat.st_size, stat.st_mtime_ns)


def data_contract(dataset):
    split = dataset.station_split
    files = [dataset.root_path / dataset.data_path, dataset.sim_path]
    calibration = dataset.root_path / 'calibration.json'
    manifest_path = dataset.root_path / 'manifest.json'
    if not calibration.is_file() or not manifest_path.is_file():
        raise ValueError('Verified calibration.json and manifest.json are required; regenerate inputs with process_model/calibrate.py')
    basis = dataset.evaluation_protocol.get('data_manifest_sha256')
    if basis is not None and basis != file_hash(manifest_path):
        raise ValueError('Evaluation eligibility was established on different calibrated data')
    parameters = json.loads(calibration.read_text())
    if parameters.get('schema_version') != 2 or parameters.get('observed_stage_conditioned') is not False:
        raise ValueError('Causal process calibration is required; archived observed-stage features cannot be used')
    manifest = json.loads(manifest_path.read_text())
    if manifest.get('status') != 'complete' or manifest.get('split_sha256') != dataset.split_hash:
        raise ValueError('Calibration manifest is incomplete or uses a different split')
    if parameters['split_sha256'] != dataset.split_hash:
        raise ValueError('Calibration split differs from model station split')
    training = sorted(split.loc[split.split == 'train', 'PEP_ID'].astype(int).tolist())
    if parameters['fit_station_ids'] != training:
        raise ValueError('Calibration training sites differ from model training sites')
    files.append(calibration)
    for path in files:
        if manifest.get('artifacts', {}).get(path.name) != file_hash(path):
            raise ValueError(f'Calibrated artifact differs from completed regeneration: {path.name}')
    return dict(schema_version=2, split_sha256=dataset.split_hash,
        input_stage_source='simulated_tppv_stage_code',
        observed_target_usage='future supervision only; excluded from encoder and decoder context',
        forecast_issue_time='End of the final encoder day',
        process_state_policy=parameters['process_state_policy'],
        sowing_policy=parameters['sowing_policy'],
        weather_policy=parameters['weather_policy'],
        evaluation_sha256=dataset.evaluation_hash,
        evaluated_station_ids=dataset.evaluation_protocol['station_ids'],
        excluded_test_station_ids=dataset.evaluation_protocol['excluded_station_ids'],
        **{f'{group}_station_ids': sorted(split.loc[split.split == group, 'PEP_ID'].astype(int).tolist())
           for group in ['train', 'val', 'test']},
        data_sha256={p.name: file_hash(p) for p in files},
        seq_len=dataset.seq_len, label_len=dataset.label_len, pred_len=dataset.pred_len,
        target=dataset.target, features=dataset.features,
        feature_order=dataset.df_full_with_pep.columns.drop(['date', 'PEP_ID']).tolist(),
        scaler_mean=dataset.scaler.mean_.tolist() if dataset.scale else None,
        scaler_scale=dataset.scaler.scale_.tolist() if dataset.scale else None)


def verify_contract(path, actual):
    path = Path(path)
    if not path.is_file():
        raise ValueError('Checkpoint lacks data_contract.json; archived checkpoints require separate historical evaluation')
    if json.loads(path.read_text()) != actual:
        raise ValueError('Checkpoint data/split contract differs from evaluation inputs')


def verify_checkpoint(directory, actual):
    directory = Path(directory)
    verify_contract(directory/'data_contract.json', actual)
    identity_path = directory/'checkpoint_identity.json'
    if not identity_path.is_file():
        raise ValueError('Checkpoint lacks verified weight identity')
    identity = json.loads(identity_path.read_text())
    if not all((directory/name).is_file() for name in ['training_args.json', 'training_source.json']):
        raise ValueError('Checkpoint lacks training arguments/source provenance; use a fresh corrected run')
    if identity != dict(checkpoint_sha256=file_hash(directory/'checkpoint.pth'),
                        contract_sha256=file_hash(directory/'data_contract.json'),
                        training_args_sha256=file_hash(directory/'training_args.json'),
                        training_source_sha256=file_hash(directory/'training_source.json')):
        raise ValueError('Checkpoint weights or provenance changed after training')
