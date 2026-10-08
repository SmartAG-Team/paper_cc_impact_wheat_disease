"""Command-line execution from explicit configuration and weather CSV inputs."""

import argparse
import hashlib
import json
from pathlib import Path

from .config import EngineConfig
from .engine import WheatSTBSimulation
from .weather import WeatherProvider


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--weather', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', type=Path)
    parser.add_argument('--days', type=int)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Output directory must be new; existing results are preserved.')
    simulation = WheatSTBSimulation(EngineConfig.from_json(args.config), WeatherProvider.from_csv(args.weather))
    if args.resume is not None:
        simulation.restore(json.loads(args.resume.read_text()))
    simulation.run(args.days)
    checkpoint = simulation.snapshot()
    results = simulation.finalize()
    dest = results.write(args.output)
    (dest/'checkpoint.json').write_text(json.dumps(checkpoint, indent=2, allow_nan=False)+'\n')
    (dest/'effective_configuration.json').write_text(json.dumps(simulation.config.to_dict(), indent=2, allow_nan=False)+'\n')
    files = {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in dest.iterdir() if path.is_file()}
    receipt = dict(status='complete', days_simulated=len(results.daily),
        model_version=results.summary['model_version'], configuration_sha256=simulation.config.identity,
        weather_exhausted=results.summary['weather_exhausted'], output_sha256=files,
        actual_field_yield_forecast=False, observations_assimilated=False)
    (dest/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(dest.resolve()), days_simulated=len(results.daily),
                         yield_mode=simulation.config.yield_model.mode)))
    return 0
