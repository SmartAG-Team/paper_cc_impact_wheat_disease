"""External conversion of frozen stage/process fits to explicit runtime cases."""

import argparse
import hashlib
import json
from pathlib import Path

from model.seasonal_septoria.overwinter import OverwinterParameters
from model.wheat_stb.config import EngineConfig, LeafParameters, TPVParameters, YieldParameters


def make_configuration(*, tpv_fit, stage_fit, latitude, sowing_date,
                       disease_fit=None, yield_model=None, field_id='field', crop_end_date=None,
                       detection_fraction=.001):
    tpv_fit, stage_fit = Path(tpv_fit), Path(stage_fit)
    tpv_raw, stage_raw = json.loads(tpv_fit.read_text()), json.loads(stage_fit.read_text())
    if stage_raw.get('q') != 1. or stage_raw.get('disease_labels_used', False):
        raise ValueError('An independent q=1 stage-only calibration is required.')
    phenology = TPVParameters(**{key: float(tpv_raw[key]) for key in TPVParameters.__dataclass_fields__})
    thresholds = tuple((int(key), float(value)) for key, value in stage_raw['all_stage_thresholds'].items())
    leaf = LeafParameters(thresholds, threshold_source_sha256=hashlib.sha256(stage_fit.read_bytes()).hexdigest(),
        threshold_status='stage-only calibrated thresholds; weak identification and explicit spacing scenario')
    values = dict(latitude=latitude, sowing_date=sowing_date, phenology=phenology, leaf=leaf,
        yield_model=yield_model or YieldParameters(), field_id=field_id, crop_end_date=crop_end_date,
        detection_fraction=detection_fraction)
    if disease_fit is not None:
        path = Path(disease_fit)
        record = json.loads(path.read_text())
        if (record.get('status') != 'fit_complete' or record.get('disease_magnitude_fit', False)
                or record.get('validation_outcomes_used', False)):
            raise ValueError('A frozen calibration-only sign/timing fit is required.')
        try:
            fitted = record['fitted']
            preprocessing = fitted['weather_preprocessing']
            if preprocessing['weather_operator'] != 'duration_proxy':
                raise ValueError('Unsupported fitted weather operator.')
            values.update(disease=OverwinterParameters(**fitted['parameters']),
                detection_fraction=record.get('primary_detection_fraction',
                    fitted.get('detection_fraction', detection_fraction)),
                leaf=LeafParameters(leaf.stage_thresholds, fitted['rank_spacing_units'],
                    leaf.threshold_source_sha256, leaf.threshold_status,
                    fitted.get('juvenile_policy', record.get('juvenile_policy', 'handover_31_39'))),
                rain_rate_mm_hour=preprocessing['rain_rate_mm_hour'],
                background_imported_pressure=fitted['constant_imported_pressure'],
                initial_local_source=1., time_step=.25,
                disease_fit_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parameter_status='sign-only timing fit on original 28 fields; ecological rates weakly identified; conditional on initial source=1 and time step=0.25')
        except KeyError as error:
            raise ValueError('The frozen fit is missing a required selected parameter.') from error
    return EngineConfig(**values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tpv-fit', type=Path, required=True)
    parser.add_argument('--stage-fit', type=Path, required=True)
    parser.add_argument('--disease-fit', type=Path)
    parser.add_argument('--latitude', type=float, required=True)
    parser.add_argument('--sowing-date', required=True)
    parser.add_argument('--field-id', default='field')
    parser.add_argument('--detection-fraction', type=float, default=.001)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError('Explicit case configuration output must be new.')
    config = make_configuration(tpv_fit=args.tpv_fit, stage_fit=args.stage_fit,
        disease_fit=args.disease_fit, latitude=args.latitude, sowing_date=args.sowing_date,
        field_id=args.field_id, detection_fraction=args.detection_fraction)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(config.to_dict(), indent=2, allow_nan=False)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
