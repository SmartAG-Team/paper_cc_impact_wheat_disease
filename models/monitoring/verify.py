"""Independently reconstruct saved metrics and audit source/split identities."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .inputs import ARCHIVE, INPUT, sha


def _weights(frame):
    fields = frame[['coordinate_year', 'field_id']].drop_duplicates().groupby('coordinate_year').size()
    leaves = frame[['coordinate_year', 'field_id', 'leaf_index']].drop_duplicates().groupby(['coordinate_year', 'field_id']).size()
    counts = frame.groupby(['coordinate_year', 'field_id', 'leaf_index']).size()
    return np.array([1/(fields.loc[r.coordinate_year]*leaves.loc[(r.coordinate_year, r.field_id)]
                        * counts.loc[(r.coordinate_year, r.field_id, r.leaf_index)])
                     for r in frame.itertuples(index=False)])


def _integral_rmse(frame, name):
    rows = []
    keys = ['partition', 'field_id', 'coordinate_year', 'source', 'leaf_index', 'endpoint_series']
    for identity, group in frame.groupby(keys, sort=False):
        group = group.sort_values('date')
        time = (pd.to_datetime(group.date)-pd.Timestamp(group.date.iloc[0])).dt.days.to_numpy()
        if len(time) < 2 or time[-1] == 0:
            continue
        error = (group[name]-group.observed_percent).to_numpy()
        rows.append(dict(zip(keys, identity), integral_error=float(np.sum(np.diff(time)*(error[:-1]+error[1:])/2))))
    if not rows:
        return np.nan
    windows = pd.DataFrame(rows)
    return float(np.sqrt(np.average(windows.integral_error**2, weights=_weights(windows))))


def verify_run(root, output, *, check_code=False):
    root, output = Path(root), Path(output)
    registration = json.loads((output/'registration.json').read_text())
    for relative, expected in registration['source_provenance']['source_sha256'].items():
        if sha(root/relative) != expected:
            raise ValueError('Source hash differs: '+relative)
    for relative, expected in registration['code_sha256'].items():
        if sha(output/'code_snapshot'/relative) != expected:
            raise ValueError('Code snapshot hash differs: '+relative)
        if check_code and sha(root/relative) != expected:
            raise ValueError('Current code differs from run: '+relative)
    original = pd.read_csv(root/INPUT/'observations.csv')
    original['season_year'] = pd.to_datetime(original.date).dt.year
    expected_ids = set(original.loc[original.season_year.isin(registration['years']), 'global_target_index'])
    excluded_ids = set(registration.get('excluded_target_ids', []))
    expected_ids -= excluded_ids
    if registration.get('quality_policy') == 'quarantine':
        exclusions = pd.read_csv(output/'excluded_observations.csv')
        if set(exclusions.global_target_index) != excluded_ids:
            raise ValueError('Exclusion ledger differs from registration')
    predictions = pd.read_csv(output/'assessment_predictions.csv')
    if predictions.global_target_index.duplicated().any() or set(predictions.global_target_index) != expected_ids:
        raise ValueError('Incomplete or duplicate prediction support')
    original = original.set_index('global_target_index')
    matched = original.loc[predictions.global_target_index]
    for name in ['observed_percent', 'field_id', 'site_id', 'date', 'endpoint_series', 'leaf_index']:
        if not np.array_equal(matched[name].to_numpy(), predictions[name].to_numpy()):
            raise ValueError('Prediction identity differs: '+name)
    archive = pd.read_csv(root/ARCHIVE).set_index('global_target_index')
    if not np.allclose(archive.loc[predictions.global_target_index, 'selected_pipeline'], predictions.current, atol=1e-12):
        raise ValueError('Frozen benchmark differs')
    model_names = ['current', 'persistence', *registration['candidate_registry'], 'selected_candidate']
    if 'refit_current' in predictions:
        model_names.append('refit_current')
    values = predictions[model_names].to_numpy()
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 100)):
        raise ValueError('Nonfinite or unbounded predictions')
    masks = dict(all=np.ones(len(predictions), bool),
        cold=~predictions.prior_assessment_available, warm=predictions.prior_assessment_available,
        BASF=predictions.source.eq('BASF'), Corteva=predictions.source.eq('Corteva'),
        new_sites=predictions.new_site, observed_above_75=predictions.observed_percent > 75,
        observed_at_most_5=predictions.observed_percent <= 5,
        observed_increase_above_40=predictions.observed_change_pp > 40)
    global_weights = _weights(predictions)
    checks = 0
    for row in pd.read_csv(output/'metrics.csv').itertuples(index=False):
        mask = np.asarray(masks[row.population]); part = predictions.loc[mask]
        diagnostic = row.population.startswith('observed_')
        weight = global_weights[mask] if diagnostic else _weights(part)
        error = part[row.model].to_numpy()-part.observed_percent.to_numpy()
        for name, value in dict(rmse=np.sqrt(np.average(error**2, weights=weight)),
            mae=np.average(abs(error), weights=weight), bias=np.average(error, weights=weight)).items():
            if not np.isclose(value, getattr(row, name), atol=1e-9, rtol=0):
                raise ValueError(f'Saved metric differs: {row.population}/{row.model}/{name}')
            checks += 1
        integral = np.nan if diagnostic else _integral_rmse(part, row.model)
        if not ((pd.isna(integral) and pd.isna(row.integral_rmse))
                or np.isclose(integral, row.integral_rmse, atol=1e-8, rtol=0)):
            raise ValueError(f'Saved integral metric differs: {row.population}/{row.model}')
    grouped = {}
    for entry in json.loads((output/'split_membership.json').read_text()):
        if excluded_ids & set(entry['target_ids']):
            raise ValueError('Quarantined observation entered training or validation')
        key = (entry['outer_fold'], entry.get('inner_fold'))
        grouped.setdefault(key, {})[entry['role']] = entry['target_ids']
    for (_, inner), roles in grouped.items():
        train = original.loc[roles['train']]
        test = original.loc[roles['validation' if inner else 'test']]
        if set(train.index) & set(test.index) or train.season_year.max() >= test.season_year.min():
            raise ValueError('Temporal split leakage')
        if inner and set(train.site_id) & set(test.site_id):
            raise ValueError('Inner site leakage')
        if inner and registration.get('inner_spatial_buffer_km', 0):
            from .evaluation import distances_km
            fields = pd.read_csv(root/INPUT/'fields.csv').set_index('field_id')
            first = fields.loc[train.field_id, ['latitude', 'longitude']].drop_duplicates()
            second = fields.loc[test.field_id, ['latitude', 'longitude']].drop_duplicates()
            if distances_km(first, second).min()+1e-8 < registration['inner_spatial_buffer_km']:
                raise ValueError('Inner spatial buffer violated')
    selections = json.loads((output/'selection.json').read_text())
    if not all(row['selected'] in registration['primary_candidates'] for row in selections):
        raise ValueError('Primary selection used a nonregistered crop-conditioned recipe')
    if 'weather_cutoff_days' in registration:
        provenance = pd.read_csv(output/'feature_provenance.csv')
        dates = original.loc[provenance.global_target_index, 'date'].to_numpy()
        if not (pd.to_datetime(provenance.weather_end).to_numpy()
                <= (pd.to_datetime(dates)-pd.Timedelta(days=registration['weather_cutoff_days'])).to_numpy()).all():
            raise ValueError('Weather extends past declared cutoff')
        if not provenance.rain_variable.eq('rain_mm').all():
            raise ValueError('Rainfall predictors do not use liquid rain')
        for row in json.loads((output/'history_provenance.json').read_text()):
            if excluded_ids & set(row['prior_target_ids']):
                raise ValueError('Quarantined observation entered prediction history')
    spatial = output/'spatial_predictions.csv'
    if spatial.exists():
        from .evaluation import distances_km
        fields = pd.read_csv(root/INPUT/'fields.csv').set_index('field_id')
        for entry in json.loads((output/'spatial_membership.json').read_text()):
            train, test = original.loc[entry['train']], original.loc[entry['test']]
            if train.season_year.max() >= test.season_year.min() or set(train.site_id) & set(test.site_id):
                raise ValueError('Spatial temporal/site leakage')
            first = fields.loc[train.field_id, ['latitude', 'longitude']].drop_duplicates()
            second = fields.loc[test.field_id, ['latitude', 'longitude']].drop_duplicates()
            if distances_km(first, second).min()+1e-8 < registration['spatial_buffer_km']:
                raise ValueError('Spatial buffer violated')
        support = json.loads((output/'spatial_support.json').read_text())
        s = pd.read_csv(spatial)
        if len(s) != support['scored_assessments'] or s.global_target_index.duplicated().any():
            raise ValueError('Spatial support differs')
    promotion = json.loads((output/'promotion.json').read_text())
    if not registration['independent_validation'] and promotion['promote']:
        raise ValueError('Adaptive evidence was promoted')
    return dict(verified=True, assessments=len(predictions), independent_metric_checks=checks,
        source_hashes_checked=len(registration['source_provenance']['source_sha256']),
        code_snapshots_checked=len(registration['code_sha256']),
        independent_validation=False, promoted=promotion['promote'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--check-code', action='store_true')
    args = parser.parse_args()
    result = verify_run(args.root, args.output, check_code=args.check_code)
    (args.output/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
