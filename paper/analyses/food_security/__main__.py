"""Run the separately versioned climate/food-security extension."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform

from .exposure import MODELS
from .provenance import ROOT, digest, output_hashes, prepare_destination, write_json


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True,
                        help='New run directory inside this repository outputs/')
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--stages', nargs='+', choices=['benchmark', 'attribution', 'exposure', 'evidence'],
                        default=['benchmark', 'attribution', 'exposure', 'evidence'])
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    years = parser.add_mutually_exclusive_group()
    years.add_argument('--reference-years', type=int, nargs='+', choices=range(1991, 2021))
    years.add_argument('--full-period', action='store_true', help='All 1991–2020 / 2071–2100 pairs')
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    args = parser.parse_args(argv)
    args.reference_years = list(range(1991, 2021)) if args.full_period else (args.reference_years or [1991, 2005, 2020])
    if (len(set(args.models)) != len(args.models) or len(set(args.stages)) != len(args.stages)
            or len(set(args.reference_years)) != len(args.reference_years) or args.bootstrap_draws < 1):
        parser.error('Models, stages and years must be unique; bootstrap draws must be positive')
    return args


def frozen_snapshot(root):
    paths = ['repository_manifest.json', 'data/final_study/path_manifest.json',
             'data/final_climate/scientific_manifest.json', 'data/final_models/manifest.json',
             'paper/Manuscript.docx',
             'publication/european_wheat_stb/Supplementary_Information.docx']
    return {path: digest(root / path) for path in paths}


def main(argv=None):
    args = parse_args(argv)
    root = args.root.resolve()
    output = prepare_destination(args.output, root)
    frozen = frozen_snapshot(root)
    code = {path.relative_to(root).as_posix(): digest(path)
            for path in sorted((root / 'paper/analyses/food_security').glob('*.py'))}
    protocol = dict(schema_version=1, started_utc=datetime.now(timezone.utc).isoformat(),
                    stages=args.stages, models=args.models, reference_years=args.reference_years,
                    attribution_period='1991–2020 to 2071–2100' if args.full_period else 'selected-year diagnostic',
                    bootstrap_draws=args.bootstrap_draws, python=platform.python_version(),
                    frozen_baseline_sha256=frozen, implementation_sha256=code,
                    retrospective_outcomes_previously_examined=True,
                    manuscript_or_frozen_model_update_authorized=False)
    write_json(output / 'execution_protocol.json', protocol)
    results = {}
    try:
        for stage in args.stages:
            print('Running ' + stage, flush=True)
            if stage == 'benchmark':
                from .benchmark import run
                result = run(output / stage, root=root, bootstrap_draws=args.bootstrap_draws)
            elif stage == 'attribution':
                from .attribution import run
                result = run(output / stage, root=root, models=args.models, reference_years=args.reference_years)
            elif stage == 'exposure':
                from .exposure import run
                result = run(output / stage, root=root)
            else:
                from .evidence import run
                result = run(output / stage, root=root)
            results[stage] = result
            write_json(output / 'stage_status.json', {'completed_stages': list(results), 'results': results})
            print('Completed ' + stage, flush=True)
        if frozen != frozen_snapshot(root):
            raise ValueError('Frozen publication or source manifests changed during execution')
        for path, expected in code.items():
            if digest(root / path) != expected:
                raise ValueError('Extension implementation changed during execution: ' + path)
        from .figures import render
        figures = render(output)
        receipt = dict(status='completed', completed_stages=list(results), requested_stages=args.stages,
                       frozen_baseline_unchanged=True, implementation_unchanged=True,
                       results=results, figures=figures, outputs_sha256=output_hashes(output),
                       model_adopted_for_projections=False, grain_loss_estimated=False,
                       food_security_scope='Conditional baseline production exposure; physiological losses unresolved',
                       execution_protocol_sha256=digest(output / 'execution_protocol.json'))
        write_json(output / 'receipt.json', receipt)
        print(json.dumps({'status': receipt['status'], 'output': str(output),
                          'completed_stages': list(results), 'grain_loss_estimated': False}), flush=True)
        return receipt
    except Exception as exc:
        write_json(output / 'failure.json', dict(status='incomplete', completed_stages=list(results),
                                               error_type=type(exc).__name__, error=str(exc)))
        raise


if __name__ == '__main__':
    main()
