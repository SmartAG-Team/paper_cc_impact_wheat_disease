"""Score frozen process models on the approved shared evaluation-site group."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from process_model._support.station_split import load_station_split, split_identity, load_evaluation_sites, evaluation_identity
from process_model._support.data_contract import file_hash
from process_model.calibrate import INDICES, simulate_frame


def evaluate(data, selection, output):
    data = Path(data).resolve(); output = Path(output).resolve()
    split = load_station_split(data/'station_split.json')
    protocol = load_evaluation_sites(selection, split)
    manifest = json.loads((data/'manifest.json').read_text())
    if manifest.get('status') != 'complete' or manifest.get('split_sha256') != split_identity(split):
        raise ValueError('Completed calibration with the same reserved station split is required')
    if protocol.get('data_manifest_sha256') != file_hash(data/'manifest.json'):
        raise ValueError('Evaluation-site eligibility belongs to different calibrated inputs')
    for name in ['data_cleaned.csv', 'calibration.json']:
        if file_hash(data/name) != manifest.get('artifacts', {}).get(name):
            raise ValueError(f'Calibrated artifact changed: {name}')
    parameters = json.loads((data/'calibration.json').read_text())
    if parameters.get('schema_version') != 2 or parameters.get('observed_stage_conditioned') is not False:
        raise ValueError('Causal process calibration is required for current evaluation')
    if parameters['split_sha256'] != split_identity(split) or parameters['fit_station_ids'] != sorted(
            split.loc[split.split == 'train', 'PEP_ID'].astype(int).tolist()):
        raise ValueError('Process parameters were not fitted on the designated training stations')
    if output.exists():
        raise FileExistsError(f'Use a new evaluation directory: {output}')
    target = data/'evaluation_sites.json'
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError('Dataset already has a different evaluation-site protocol')
    selected = set(protocol['station_ids'])
    sums = np.zeros(3); counts = np.zeros(3, dtype=np.int64); seen = set(); header = True
    output.mkdir(parents=True)
    columns = ['PEP_ID', 'DATE', 'CODE_new'] + INDICES
    for chunk in pd.read_csv(data/'data_cleaned.csv', usecols=columns, chunksize=200000):
        rows = chunk.loc[chunk.PEP_ID.isin(selected)]
        if rows.empty:
            continue
        predictions = simulate_frame(rows, parameters)
        predictions['CODE_new'] = rows.CODE_new
        predictions.to_csv(output/'simulation_test.csv', mode='w' if header else 'a', header=header, index=False)
        header = False; seen.update(rows.PEP_ID)
        for index, column in enumerate(['gdd_simulate', 'tpp_simulate', 'tppv_simulate']):
            codes = predictions[column].map({0: 0, 10: 1, 31: 2, 51: 3, 85: 4})
            valid = rows.CODE_new.notna() & codes.notna()
            sums[index] += (codes[valid] - rows.loc[valid, 'CODE_new']).abs().sum()
            counts[index] += valid.sum()
    if seen != selected or (counts == 0).any():
        raise ValueError('Process evaluation lacks approved sites or scored targets')
    report = dict(split_sha256=split_identity(split), evaluation_sha256=evaluation_identity(protocol),
        test_station_ids=sorted(selected), excluded_test_station_ids=protocol['excluded_station_ids'],
        exclusion_reason=protocol.get('exclusion_reason'), observed_stage_conditioned=False,
        known_sowing_conditioned=True, weather_policy=parameters['weather_policy'],
        metrics={model: dict(MAE_stage_code=float(sums[i]/counts[i]), rows=int(counts[i]))
                 for i, model in enumerate(['GDD', 'T-P', 'T-P-V'])})
    (output/'process_test_metrics.json').write_text(json.dumps(report, indent=2)+'\n')
    audit = dict(status='complete', split_sha256=split_identity(split),
        evaluation_sha256=evaluation_identity(protocol),
        calibration_manifest_sha256=file_hash(data/'manifest.json'),
        artifacts={name: file_hash(output/name) for name in ['process_test_metrics.json','simulation_test.csv']})
    (output/'manifest.json').write_text(json.dumps(audit, indent=2)+'\n')
    # Preserve calibration artifacts; install only the separate scoring protocol.
    target.write_text(json.dumps(protocol, indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'dataset/calibrated_causal')
    parser.add_argument('--evaluation', type=Path, default=ROOT/'configs/evaluation_sites.json')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    report = evaluate(args.data, args.evaluation, args.output or args.data/'common_evaluation')
    print(json.dumps(dict(test_sites=len(report['test_station_ids']),
                         excluded_sites=len(report['excluded_test_station_ids']),
                         metrics=report['metrics']), indent=2))


if __name__ == '__main__':
    main()
