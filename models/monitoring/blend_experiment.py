"""Reproduce and verify the recorded, retrospective equal-weight blends.

This script was exported after the original calculation. The original
registration records the earlier weight decision; neither is preregistration
of independent validation or evidence that the components were chosen blindly.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .evaluation import score_predictions, score_subset, paired_interval
from .inputs import sha

ROOT = Path(__file__).resolve().parents[2]


def calculate(root, registration):
    root = Path(root)
    for path, digest in registration['source_sha256'].items():
        if sha(root/path) != digest:
            raise ValueError('Source hash mismatch: '+path)
    source = root/'outputs/disease_improvement/history_anchor_20261022'
    frames = {}
    metrics, intervals, diagnostics = [], [], []
    for label, filename in [('temporal', 'assessment_predictions.csv'),
                            ('spatial_50km', 'spatial_predictions.csv')]:
        frame = pd.read_csv(source/filename)
        for name, recipe in registration['recipes'].items():
            weights = np.asarray(recipe['weights'])
            if not np.allclose(weights, 1/len(weights)):
                raise ValueError('Only the recorded equal-weight recipes are allowed')
            frame[name] = frame[recipe['components']].to_numpy() @ weights
            metrics.append(dict(evaluation=label, model=name, **score_predictions(frame, name)))
            for reference in ['reference_replay', 'shared_replay', 'bagged_shared']:
                ci = paired_interval(frame, name, reference)
                intervals.append(dict(evaluation=label, model=name, reference=reference,
                                      lower=ci[0], upper=ci[1]))
        if label == 'temporal':
            populations = dict(cold=~frame.prior_assessment_available,
                warm=frame.prior_assessment_available, BASF=frame.source.eq('BASF'),
                Corteva=frame.source.eq('Corteva'), severe_above75=frame.observed_percent.gt(75),
                new_site=frame.new_site)
            for population, mask in populations.items():
                for name in registration['recipes']:
                    diagnostics.append(dict(population=population, model=name,
                                            **score_subset(frame, mask, name)))
        frames[filename] = frame
    frames['metrics.csv'] = pd.DataFrame(metrics)
    frames['paired_intervals.csv'] = pd.DataFrame(intervals)
    frames['subgroup_metrics.csv'] = pd.DataFrame(diagnostics)
    return frames


def verify(root, output):
    output = Path(output)
    registration = json.loads((output/'registration.json').read_text())
    expected = calculate(root, registration)
    for filename, frame in expected.items():
        saved = pd.read_csv(output/filename)
        for column in saved:
            if pd.api.types.is_numeric_dtype(saved[column]):
                frame[column] = pd.to_numeric(frame[column])
        # Column order is not scientific content; all values and rows must agree.
        pd.testing.assert_frame_equal(frame[saved.columns], saved,
            check_dtype=False, atol=1e-9, rtol=1e-10)
    result = dict(verified=True, assessments=len(expected['assessment_predictions.csv']),
        spatial_assessments=len(expected['spatial_predictions.csv']),
        metrics_recomputed=len(expected['metrics.csv']),
        intervals_recomputed=len(expected['paired_intervals.csv']),
        script_exported_after_original_calculation=True,
        adaptive_retrospective_development=True, independent_external_validation=False,
        publication_model_promoted=False, verifier_sha256=sha(Path(__file__)))
    (output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('registered_output', type=Path)
    parser.add_argument('--output', type=Path, help='Reproduce into a new directory')
    args = parser.parse_args()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=False)
        registration = json.loads((args.registered_output/'registration.json').read_text())
        (args.output/'registration.json').write_text(json.dumps(registration, indent=2)+'\n')
        for filename, frame in calculate(ROOT, registration).items():
            frame.to_csv(args.output/filename, index=False)
        target = args.output
    else:
        target = args.registered_output
    print(json.dumps(verify(ROOT, target), indent=2))
