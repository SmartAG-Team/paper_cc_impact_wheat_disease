"""Exact sowing-cycle event comparison for the whole-season weather protocol.

Lead one is the sowing day. Neural dates are first crossings of the expected
ordinal stage at 0.5, 1.5, 2.5, and 3.5. No date is interpolated or extrapolated.
Process dates are supplied by frozen calibrated process simulations on the
same coverage records. Missing observations provide no evidence of absence.
"""
from collections import Counter
from datetime import date, timedelta
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path
import csv
import json
import math

import numpy as np


PROTOCOL = 'whole-season-known-weather-midpoint-v1'
STAGES = (10, 31, 51, 85)
PROCESS_MODELS = ('GDD', 'T-P', 'T-P-V')
_FIELDS = ('PEP_ID', 'SOWING_DATE', 'END_DATE', 'BBCH', 'model', 'eligible',
           'reason', 'true_date', 'predicted_date', 'error_days', 'status', 'common_matched')
_ARTIFACTS = ('coverage.json', 'event_dates.csv', 'metrics.json')


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'Unsupported JSON type: {type(value).__name__}')


def _json(value):
    return json.dumps(value, sort_keys=True, indent=2, allow_nan=False, default=_json_default) + '\n'


def _hash(path):
    digest = sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _date(value, field, nullable=False):
    if value is None and nullable:
        return None
    if not isinstance(value, str):
        raise ValueError(f'{field} must be an ISO date string')
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f'Invalid {field}: {value!r}') from exc
    if result.isoformat() != value:
        raise ValueError(f'{field} must use YYYY-MM-DD')
    return result


def _integer(value, field):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f'{field} must be an integer')
    return int(value)


def _name(value):
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError('model must be a nonempty trimmed string')
    return value


def _validate_coverage(coverage):
    records, keys = [], set()
    for source in coverage:
        record = dict(source)
        try:
            station = _integer(record['PEP_ID'], 'PEP_ID')
            sow = _date(record['SOWING_DATE'], 'SOWING_DATE')
            end = _date(record['END_DATE'], 'END_DATE')
            length = _integer(record['length'], 'length')
            eligible = record['eligible']
            true_dates, process_dates = record['true_dates'], record['process_dates']
        except KeyError as exc:
            raise ValueError(f'Missing coverage field: {exc.args[0]}') from exc
        if not isinstance(eligible, (bool, np.bool_)):
            raise ValueError('eligible must be boolean')
        calendar_length = (end-sow).days+1
        if calendar_length < 1 or length < 0 or length > calendar_length:
            raise ValueError('Invalid cycle length or end before sowing')
        if eligible and length != calendar_length:
            raise ValueError('Eligible length must equal inclusive cycle calendar length')
        if not isinstance(record.get('reason', ''), str):
            raise ValueError('reason must be a string')
        key = (station, sow.isoformat())
        if key in keys:
            raise ValueError(f'Duplicate coverage cycle: {key}')
        keys.add(key)
        if len(true_dates) != 4 or set(process_dates) != set(PROCESS_MODELS):
            raise ValueError('Coverage needs four true dates and all three process models')
        for label, values in [('true_dates', true_dates)] + list(process_dates.items()):
            if len(values) != 4:
                raise ValueError(f'{label} needs four stage dates')
            for value in values:
                parsed = _date(value, label, nullable=True)
                if parsed is not None and not sow <= parsed <= end:
                    raise ValueError(f'{label} outside cycle boundary: {key}')
        record.update(PEP_ID=station, length=length, eligible=bool(eligible), reason=record.get('reason', ''))
        records.append(record)
    previous = {}
    for record in sorted(records, key=lambda r: (r['PEP_ID'], r['SOWING_DATE'])):
        station = record['PEP_ID']
        if station in previous and previous[station] >= record['SOWING_DATE']:
            raise ValueError(f'Overlapping sowing cycles for PEP_ID {station}')
        previous[station] = record['END_DATE']
    return records


def _row(record, index, model, predicted):
    return dict(PEP_ID=record['PEP_ID'], SOWING_DATE=record['SOWING_DATE'],
                END_DATE=record['END_DATE'], BBCH=STAGES[index], model=model,
                eligible=record['eligible'], reason=record['reason'],
                true_date=record['true_dates'][index], predicted_date=predicted)


def prediction_rows(records, stage_means, model_name):
    """Return four date dictionaries per input cycle, in input/stage order.

    Each dictionary has PEP_ID, SOWING_DATE, END_DATE, BBCH, model, eligible,
    reason, true_date and predicted_date. Dates are ISO strings or None. Supply
    one finite unpadded one-dimensional array per eligible cycle. A scalar
    regression may leave [0,4]; thresholds apply directly without clipping.
    Ineligible cycles accept None and always return missing predictions.
    """
    records = _validate_coverage(records)
    model_name = _name(model_name)
    rows, sentinel = [], object()
    for record, values in zip_longest(records, stage_means, fillvalue=sentinel):
        if record is sentinel or values is sentinel:
            raise ValueError('stage_means count must match cycle count exactly')
        if not record['eligible']:
            if values is not None:
                raise ValueError('Ineligible cycles cannot receive predicted trajectories')
            rows.extend(_row(record, k, model_name, None) for k in range(4))
            continue
        if hasattr(values, 'detach'):
            values = values.detach().cpu().numpy()
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (record['length'],) or not np.isfinite(values).all():
            raise ValueError('stage_means must be finite unpadded vectors matching each cycle length')
        sow = _date(record['SOWING_DATE'], 'SOWING_DATE')
        for k, threshold in enumerate((.5, 1.5, 2.5, 3.5)):
            positions = np.flatnonzero(values >= threshold)
            predicted = (sow + timedelta(days=int(positions[0]))).isoformat() if len(positions) else None
            rows.append(_row(record, k, model_name, predicted))
    return rows


def _event_key(row):
    try:
        key = (_integer(row['PEP_ID'], 'PEP_ID'), row['SOWING_DATE'], _integer(row['BBCH'], 'BBCH'))
    except KeyError as exc:
        raise ValueError(f'Missing prediction key: {exc.args[0]}') from exc
    _date(key[1], 'SOWING_DATE')
    if key[2] not in STAGES:
        raise ValueError('Only physiological BBCH 10/31/51/85 can be scored')
    return key


def _assemble(records, model_dates):
    models = [_name(name) for name in model_dates]
    if any(name in PROCESS_MODELS for name in models):
        raise ValueError('Neural model names cannot replace frozen process models')
    lookup = {(r['PEP_ID'], r['SOWING_DATE'], stage): (r, k)
              for r in records for k, stage in enumerate(STAGES)}
    all_rows = []
    for model in models:
        provided = {}
        for row in model_dates[model]:
            key = _event_key(row)
            if key in provided:
                raise ValueError(f'Duplicate prediction key for {model}: {key}')
            if key not in lookup:
                raise ValueError(f'Unknown prediction key for {model}: {key}')
            record, k = lookup[key]
            expected = _row(record, k, model, None)
            for field in ('model', 'END_DATE', 'true_date', 'eligible'):
                if field not in row or row[field] != expected[field]:
                    raise ValueError(f'Inconsistent {field} for {model}: {key}')
            if 'predicted_date' not in row:
                raise ValueError(f'Missing predicted_date for {model}: {key}')
            predicted = _date(row['predicted_date'], 'predicted_date', nullable=True)
            if predicted is not None:
                if not record['eligible']:
                    raise ValueError('Ineligible cycle predictions must be missing')
                if not record['SOWING_DATE'] <= predicted.isoformat() <= record['END_DATE']:
                    raise ValueError(f'Prediction outside cycle boundary for {model}: {key}')
            provided[key] = row['predicted_date']
        for key, (record, k) in lookup.items():
            if record['eligible'] and key not in provided:
                raise ValueError(f'Missing eligible prediction key for {model}: {key}')
            all_rows.append(_row(record, k, model, provided.get(key)))
    for model in PROCESS_MODELS:
        all_rows.extend(_row(record, k, model, record['process_dates'][model][k] if record['eligible'] else None)
                        for record in records for k in range(4))
    models += list(PROCESS_MODELS)
    common = Counter()
    for row in all_rows:
        true, predicted = row['true_date'], row['predicted_date']
        if true is not None and predicted is not None:
            row['error_days'] = (_date(predicted, 'predicted_date') - _date(true, 'true_date')).days
            row['status'] = 'matched'
            common[_event_key(row)] += 1
        else:
            row['error_days'] = None
            row['status'] = ('missing_prediction' if true is not None else
                             'prediction_without_observation' if predicted is not None else 'neither_observed_nor_predicted')
        row['common_matched'] = False
    for row in all_rows:
        row['common_matched'] = common[_event_key(row)] == len(models)
    all_rows.sort(key=lambda r: (r['PEP_ID'], r['SOWING_DATE'], r['BBCH'], r['model']))
    return all_rows, models


def _summary(rows):
    observed = sum(r['true_date'] is not None for r in rows)
    errors = [r['error_days'] for r in rows if r['error_days'] is not None]
    matched = len(errors)
    return dict(events=len(rows), observed=observed, matched=matched,
                predicted=sum(r['predicted_date'] is not None for r in rows),
                missing_predictions=observed-matched,
                predictions_without_observation=sum(r['status'] == 'prediction_without_observation' for r in rows),
                neither_observed_nor_predicted=sum(r['status'] == 'neither_observed_nor_predicted' for r in rows),
                coverage=matched/observed if observed else None,
                mae_days=sum(abs(e) for e in errors)/matched if matched else None,
                rmse_days=math.sqrt(sum(e*e for e in errors)/matched) if matched else None,
                bias_days=sum(errors)/matched if matched else None)


def _metrics(records, rows, models):
    own, paired = {}, {}
    for model in models:
        selected = [r for r in rows if r['model'] == model]
        own[model] = dict(overall=_summary(selected), by_stage={str(s): _summary([r for r in selected if r['BBCH'] == s]) for s in STAGES})
        paired[model] = [r for r in selected if r['common_matched']]
    common_overall = dict(matched=len(paired[models[0]]), models={m: _summary(paired[m]) for m in models})
    common_stage = {}
    for stage in STAGES:
        stage_rows = {m: [r for r in paired[m] if r['BBCH'] == stage] for m in models}
        common_stage[str(stage)] = dict(matched=len(stage_rows[models[0]]), models={m: _summary(stage_rows[m]) for m in models})
    eligible = sum(r['eligible'] for r in records)
    return dict(
        protocol=PROTOCOL, stages=list(STAGES), count_unit='station/sowing-cycle/physiological-stage',
        error_definition='predicted_date minus true_date in calendar days',
        coverage_denominator='all recorded physiological events including ineligible cycles',
        coverage=dict(cycles=len(records), eligible_cycles=eligible, ineligible_cycles=len(records)-eligible,
                      sites=len({r['PEP_ID'] for r in records}),
                      ineligible_reasons=dict(Counter(r['reason'] for r in records if not r['eligible']))),
        models=own, common_matched=dict(overall=common_overall, by_stage=common_stage),
    )


def evaluate_seasons(coverage, model_dates, output, provenance):
    """Write exact-key event CSV, coverage, metrics and an integrity manifest.

    ``coverage`` contains all cycles, including input-ineligible ones.
    ``model_dates`` maps each neural model to prediction_rows output; all four
    keys for every eligible cycle are mandatory, even for missing crossings.
    Ineligible keys may be omitted. The returned dictionary is metrics.json.
    The manifest preserves caller-supplied data/calibration/split/configuration/
    source/weight provenance without interpreting or fabricating those values.
    """
    records = _validate_coverage(coverage)
    rows, models = _assemble(records, model_dates)
    metrics = _metrics(records, rows, models)
    # Validate serialization before creating artifacts.
    coverage_text, metrics_text, provenance_text = _json(records), _json(metrics), _json(provenance)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'coverage.json').write_text(coverage_text, encoding='utf-8')
    (output / 'metrics.json').write_text(metrics_text, encoding='utf-8')
    with (output / 'event_dates.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    manifest = dict(schema=1, protocol=PROTOCOL, models=models, cycles=len(records), event_rows=len(rows),
                    provenance=json.loads(provenance_text), source_sha256=_hash(__file__),
                    files={name: _hash(output / name) for name in _ARTIFACTS})
    (output / 'manifest.json').write_text(_json(manifest), encoding='utf-8')
    (output / 'manifest.sha256').write_text(_hash(output / 'manifest.json')+'  manifest.json\n', encoding='utf-8')
    verify_seasonal_evaluation(output)
    return metrics


def verify_seasonal_evaluation(output):
    """Verify hashes and recompute the exact-key table and all scoring metrics.

    Checksums detect accidental mutation, not adversarial replacement of both
    artifacts and checksums. The caller binds external inputs and checkpoints
    to the recorded provenance before accepting a production run.
    """
    output = Path(output)
    try:
        checksum = (output / 'manifest.sha256').read_text(encoding='utf-8').strip().split()
        if len(checksum) != 2 or checksum[1] != 'manifest.json' or checksum[0] != _hash(output / 'manifest.json'):
            raise ValueError('Manifest SHA-256 integrity mismatch')
        manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('schema') != 1 or manifest.get('protocol') != PROTOCOL:
            raise ValueError('Unsupported seasonal manifest schema/protocol')
        if manifest.get('source_sha256') != _hash(__file__):
            raise ValueError('Evaluator source hash mismatch')
        if set(manifest.get('files', {})) != set(_ARTIFACTS):
            raise ValueError('Incomplete seasonal artifact hash manifest')
        for name, expected in manifest['files'].items():
            if _hash(output / name) != expected:
                raise ValueError(f'Artifact hash mismatch: {name}')
        records = _validate_coverage(json.loads((output / 'coverage.json').read_text(encoding='utf-8')))
        models = manifest['models']
        if len(models) != len(set(models)) or models[-3:] != list(PROCESS_MODELS):
            raise ValueError('Invalid model set in manifest')
        neural = {model: [] for model in models[:-3]}
        with (output / 'event_dates.csv').open(encoding='utf-8', newline='') as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(_FIELDS):
                raise ValueError('Invalid event CSV columns')
            actual = list(reader)
        for raw in actual:
            if raw['model'] not in models:
                raise ValueError('Unknown model in event CSV')
            if raw['eligible'] not in ('True', 'False'):
                raise ValueError('Invalid eligible flag in event CSV')
            if raw['model'] in neural:
                row = dict(raw, PEP_ID=int(raw['PEP_ID']), BBCH=int(raw['BBCH']),
                           eligible=raw['eligible'] == 'True', true_date=raw['true_date'] or None,
                           predicted_date=raw['predicted_date'] or None)
                neural[raw['model']].append(row)
        expected, rebuilt_models = _assemble(records, neural)
        normalized = [{key: '' if r[key] is None else str(r[key]) for key in _FIELDS} for r in expected]
        if actual != normalized or rebuilt_models != models:
            raise ValueError('Event CSV inconsistent with exact cycle keys and coverage')
        if manifest.get('cycles') != len(records) or manifest.get('event_rows') != len(expected):
            raise ValueError('Manifest cycle/event counts disagree')
        metrics = json.loads((output / 'metrics.json').read_text(encoding='utf-8'))
        if metrics != _metrics(records, expected, models):
            raise ValueError('Metrics inconsistent with event CSV')
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'Seasonal evaluation integrity failure: {exc}') from exc
    return manifest
