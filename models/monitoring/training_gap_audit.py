"""Compare fitted and later-year errors without refitting or selecting models."""
import argparse
import importlib.util
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .evaluation import score_subset
from .inputs import sha

ROOT = Path(__file__).resolve().parents[2]


def run(root, output):
    root, output = Path(root), Path(output)
    if output.exists():
        raise FileExistsError('A new output directory is required')
    source = root/'outputs/disease_improvement/history_anchor_20261022'
    obs = pd.read_csv(source/'observation_ledger.csv')
    test = pd.read_csv(source/'assessment_predictions.csv')
    arrays = np.load(source/'predictors.npz')
    np.testing.assert_array_equal(arrays['target_ids'], obs.global_target_index)
    names = json.loads((root/'outputs/disease_improvement/quality_weather_20261021_v2/feature_schema.json').read_text())['history_names']
    history = pd.DataFrame(arrays['history'], columns=names)
    code = source/'code_snapshot/analysis/disease_improvement/models.py'
    spec = importlib.util.spec_from_file_location('frozen_gap_models', code)
    frozen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(frozen)
    rows, records, paths = [], [], [code, source/'observation_ledger.csv',
        source/'assessment_predictions.csv', source/'predictors.npz']
    for year, part in test.groupby('season_year'):
        positions = np.flatnonzero(obs.season_year.lt(year))
        train = obs.iloc[positions].copy()
        path = source/f'{year}_shared_replay.joblib'
        model = joblib.load(path)
        assert set(model['training_ids']) == set(train.global_target_index)
        train['shared_replay'] = frozen.predict_candidate(model, arrays['shared_replay'][positions],
            history.iloc[positions], train.drop(columns='observed_percent'))
        train['evaluation_year'] = year
        records.append(train)
        paths.append(path)
        for population, frame in [('training', train), ('later_year', part)]:
            for stratum, mask in [('all', np.ones(len(frame), dtype=bool)),
                                   ('severe_above75', frame.observed_percent.gt(75))]:
                rows.append(dict(evaluation_year=int(year), population=population,
                    stratum=stratum, **score_subset(frame, mask, 'shared_replay')))
    output.mkdir(parents=True)
    pd.DataFrame(rows).to_csv(output/'training_gap.csv', index=False)
    pd.concat(records, ignore_index=True).to_csv(output/'fitted_training_predictions.csv', index=False)
    (output/'manifest.json').write_text(json.dumps(dict(
        source_sha256={str(p.relative_to(root)):sha(p) for p in paths},
        code_sha256=sha(Path(__file__)), fitting_or_tuning=False,
        interpretation='Fitted/later-year gap is descriptive and confounded with temporal distribution shift.'), indent=2)+'\n')
    return pd.DataFrame(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/disease_improvement/training_gap_20261022')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        print(run(ROOT, args.output).to_string(index=False))
