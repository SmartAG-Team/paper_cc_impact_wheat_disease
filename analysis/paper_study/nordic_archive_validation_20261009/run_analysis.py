"""Standalone Nordic/Baltic management-grain analysis; reads only bundled public sources."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
from xml.etree import ElementTree
from zipfile import ZipFile
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
KEY = ['year', 'country', 'trial_id', 'treatment_code']
TRIAL = KEY[:3]
MODELS = ('training_mean', 'management_mean', 'crop_management_mean')
SCHEMES = ('leave_country_year_out', 'leave_country_out', 'leave_year_out', 'forward_year')
SEED, BOOTSTRAPS = 20261009, 2000
PAIR_COLUMNS = ['pair_id', 'trial_key', 'control_id', 'environment_group', 'year', 'country',
    'trial_id', 'crop', 'treatment_code', 'treatment_count', 'station_raw', 'region',
    'control_source_row', 'treated_source_row', 'control_yield_t_ha', 'treated_yield_t_ha',
    'gain_t_ha', 'gain_pct_untreated', 'contrast_pct_treated', 'reported_gain_kg_ha',
    'gain_audit_difference_kg_ha', 'gain_inconsistent', 'trial_weight']


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def verify_bundle(bundle, manifest):
    contents = Path(bundle).read_bytes()
    spec = json.loads(Path(manifest).read_text())
    if sha256(contents) != spec['sha256']:
        raise ValueError('Source bundle SHA-256 mismatch')
    result = {}
    with ZipFile(io.BytesIO(contents)) as z:
        expected = {x['bundle_member'] for x in spec['files']}
        if set(z.namelist()) != expected or len(z.namelist()) != len(expected):
            raise ValueError('Missing, duplicate or unexpected source member')
        for e in spec['files']:
            b = z.read(e['bundle_member'])
            if len(b) != e['size_bytes'] or sha256(b) != e['sha256']:
                raise ValueError('Source member checksum mismatch')
            md5 = hashlib.md5(b).hexdigest()
            if md5 != e['md5'] or md5 != e.get('publisher_md5', md5):
                raise ValueError('Public source MD5 mismatch')
            result[Path(e['bundle_member']).name] = b
    return result


def clean_cells(df):
    return df.map(lambda v: np.nan if isinstance(v, str) and v.strip() in ('', '.')
                  else v.strip() if isinstance(v, str) else v)


def read_sources(data):
    y = pd.read_excel(io.BytesIO(data['Yield.xlsx']), header=3)
    expected = ['Year', 'Country', 'SpotIT_ID', 'Fung_treated', 'Weather_station',
                'Region', 'Crop*', 'Treatment_code**', 'YieldKgHa15%',
                'YieldGainKgHa', 'Yield gain 1 trt', 'Yield gain 2 trt']
    if y.columns[:12].tolist() != expected:
        raise ValueError('Incompatible grain-yield schema or moisture convention')
    y = clean_cells(y.iloc[:, :12])
    y.insert(0, 'source_row', np.arange(len(y)) + 5)
    y = y[y.Year.notna()].copy()
    y.rename(columns=dict(zip(expected, ['year', 'country', 'trial_id', 'treated',
        'station_raw', 'region', 'crop', 'treatment_code', 'yield_kg_ha',
        'reported_gain_kg_ha', 'reported_gain_1_kg_ha', 'reported_gain_2_kg_ha'])), inplace=True)
    y['source_file'], y['source_sheet'] = 'Yield.xlsx', 'Yield data'
    y['crop_raw'] = y['crop']; y['crop'] = y.crop.str.upper()
    for c in ('year', 'treated', 'yield_kg_ha', 'reported_gain_kg_ha',
              'reported_gain_1_kg_ha', 'reported_gain_2_kg_ha'):
        y[c] = pd.to_numeric(y[c], errors='raise')
    y['year'] = y.year.astype(int)
    for c in TRIAL + ['crop', 'treatment_code', 'station_raw', 'region']:
        if y[c].isna().any(): raise ValueError('Missing primary yield identity: ' + c)
    sources = []
    for name, sheet in [('Winter_Wheat_sasdataset_extr.xls', 'Winter wheat'),
                        ('Spring_Wheat_sasdataset_extr.xls', 'Spring wheat')]:
        d = clean_cells(pd.read_excel(io.BytesIO(data[name]), header=5))
        if d.columns[:17].tolist() != ['Line', 'Wst_ID', 'year', 'country', 'Group',
            'SpotID', 'FungTrt', 'WstName', 'WstYrNo', 'Region', 'Crop', 'TrtCode*',
            'yield', 'Gain**', 'Gain1', 'Gain2', 'WstnSite']:
            raise ValueError('Incompatible SAS schema')
        d.insert(0, 'source_row', np.arange(len(d)) + 7)
        d = d[d['Line'].notna()].copy()
        d.rename(columns={'SpotID': 'trial_id', 'FungTrt': 'treated',
            'WstName': 'station_raw', 'Region': 'region', 'Crop': 'crop',
            'TrtCode*': 'treatment_code', 'yield': 'yield_kg_ha',
            'Gain**': 'reported_gain_kg_ha'}, inplace=True)
        d['crop_raw'] = d['crop']; d['crop'] = d.crop.str.upper()
        d['source_file'], d['source_sheet'] = name, sheet
        for c in ('year', 'treated', 'yield_kg_ha', 'reported_gain_kg_ha'):
            d[c] = pd.to_numeric(d[c], errors='raise')
        d['year'] = d.year.astype(int)
        sources.append(d)
    return y.reset_index(drop=True), pd.concat(sources, ignore_index=True)


def audit_sas_join(yields, sas):
    """Attach only unique matches: one audit record per primary input record."""
    s = sas.dropna(subset=KEY).copy()
    counts = s.groupby(KEY, dropna=False).size().rename('sas_match_count').reset_index()
    primary = yields.copy()
    primary['primary_key_ambiguous'] = primary.duplicated(KEY, keep=False)
    out = primary.merge(counts, on=KEY, how='left', validate='many_to_one')
    unique = s.loc[~s.duplicated(KEY, keep=False)].copy()
    keep = [c for c in ['yield_kg_ha', 'reported_gain_kg_ha', 'source_row', 'source_file',
                       'crop', 'region', 'station_raw', 'WstnSite', 'Wst_ID', 'Group'] if c in unique]
    unique = unique[KEY + keep].rename(columns={c: 'sas_' + c for c in keep})
    out = out.merge(unique, on=KEY, how='left', validate='many_to_one')
    if len(out) != len(yields): raise ValueError('SAS attachment expanded primary records')
    out['sas_match_count'] = out.sas_match_count.fillna(0).astype(int)
    out['join_status'] = np.select([out.primary_key_ambiguous, out.sas_match_count.gt(1),
        out.sas_match_count.eq(0)], ['ambiguous_primary_key', 'ambiguous_sas_key', 'no_sas_match'],
        default='unique_match')
    ambiguous = out.primary_key_ambiguous | out.sas_match_count.gt(1)
    for c in ('sas_yield_kg_ha', 'sas_reported_gain_kg_ha'):
        if c in out: out.loc[ambiguous, c] = np.nan
    out['yield_difference_kg_ha'] = out.yield_kg_ha - out.sas_yield_kg_ha
    out['numeric_agreement'] = out.yield_difference_kg_ha.abs().le(.51) & out.join_status.eq('unique_match')
    return out


def build_pairs(records):
    results, exclusions = [], []
    def reject(frame, reason):
        for row in frame.to_dict('records'):
            row['reason'] = reason; exclusions.append(row)
    for keys, group in records.groupby(TRIAL, sort=True, dropna=False):
        year, country, trial = keys
        if not (2012 <= year <= 2016):
            reject(group, 'outside_2012_2016'); continue
        if group.treatment_code.duplicated().any():
            reject(group, 'ambiguous_treatment_key'); continue
        metadata = ['crop', 'station_raw', 'region']
        if group[metadata].isna().any().any() or group[metadata].nunique(dropna=False).gt(1).any():
            reject(group, 'inconsistent_trial_metadata'); continue
        if (not group.treatment_code.isin(['A', 'B', 'C']).all() or
            not group.crop.isin(['WW', 'SW']).all() or
            not (group.treated == group.treatment_code.ne('A').astype(int)).all()):
            reject(group, 'invalid_management_codes'); continue
        control = group[group.treatment_code.eq('A')]
        if len(control) != 1:
            reject(group, 'missing_control'); continue
        c = control.iloc[0]
        if not np.isfinite(c.yield_kg_ha) or c.yield_kg_ha <= 0:
            reject(group, 'invalid_control_yield'); continue
        treated = group[group.treatment_code.isin(['B', 'C'])]
        if treated.empty:
            reject(group, 'no_treated_comparison'); continue
        trial_key = f'{country}:{int(year)}:{trial}'
        trial_pairs = []
        for t in treated.sort_values('treatment_code').itertuples():
            if not np.isfinite(t.yield_kg_ha) or t.yield_kg_ha <= 0:
                reject(group.loc[group.source_row.eq(t.source_row)], 'invalid_treated_yield'); continue
            gain = float(t.yield_kg_ha - c.yield_kg_ha)
            reported = float(t.reported_gain_kg_ha)
            difference = gain - reported
            trial_pairs.append(dict(pair_id=trial_key + ':' + t.treatment_code + '-A',
                trial_key=trial_key, control_id=trial_key + ':A', environment_group=f'{country}:{int(year)}',
                year=int(year), country=country, trial_id=trial, crop=t.crop,
                treatment_code=t.treatment_code, treatment_count={'B': 1, 'C': 2}[t.treatment_code],
                station_raw=t.station_raw, region=t.region, control_source_row=int(c.source_row),
                treated_source_row=int(t.source_row), control_yield_t_ha=float(c.yield_kg_ha / 1000),
                treated_yield_t_ha=float(t.yield_kg_ha / 1000), gain_t_ha=gain / 1000,
                gain_pct_untreated=100 * gain / c.yield_kg_ha,
                contrast_pct_treated=100 * gain / t.yield_kg_ha,
                reported_gain_kg_ha=reported, gain_audit_difference_kg_ha=difference,
                gain_inconsistent=bool(np.isfinite(difference) and abs(difference) > 1)))
        for p in trial_pairs:
            p['trial_weight'] = 1 / len(trial_pairs); results.append(p)
    pairs = pd.DataFrame(results, columns=PAIR_COLUMNS)
    if not pairs.empty and pairs.pair_id.duplicated().any(): raise ValueError('Duplicate pair identity')
    return pairs, pd.DataFrame(exclusions, columns=list(records.columns) + ['reason'])


def audit_source_groups(sas, yields):
    """Group numbers recur across years; equal yields alone do not identify plots."""
    s = sas.dropna(subset=['trial_id', 'Group']).copy()
    keys = ['source_file', 'country', 'year', 'Group']
    audit = s.groupby(keys, dropna=False).agg(
        trial_count=('trial_id', 'nunique'),
        trial_ids=('trial_id', lambda x: '|'.join(sorted(set(x)))),
        source_records=('trial_id', 'size')).reset_index()
    if audit.trial_count.gt(1).any():
        raise ValueError('Source group maps to multiple trials within country-year')
    if s.groupby(['source_file'] + TRIAL)['Group'].nunique().gt(1).any():
        raise ValueError('Source trial maps to multiple groups')
    c = yields[yields.treatment_code.eq('A') & yields.year.between(2012, 2016)
               & np.isfinite(yields.yield_kg_ha) & yields.yield_kg_ha.gt(0)].copy()
    signature = ['year', 'country', 'station_raw', 'crop', 'yield_kg_ha']
    c = c[c.duplicated(signature, keep=False)].copy()
    c['environment_group'] = c.country + ':' + c.year.astype(str)
    c['possible_control_signature'] = [
        f'{r.country}:{int(r.year)}:{r.station_raw}:{r.crop}:{r.yield_kg_ha:.12g}'
        for r in c.itertuples()]
    c['interpretation'] = 'Same reported value; physical shared-control identity is not established.'
    return audit, c


def compatibility_gate(columns):
    requirements = {
        'observed_severity_yield_pairs': ['observed_severity', 'disease_species', 'assessment_date', 'trial_id'],
        'leaf_stage_repeated_comparison': ['observed_severity', 'disease_species', 'assessment_date',
                                           'leaf_identity', 'observed_crop_stage', 'trial_id'],
        'functional_canopy': ['measured_lai', 'measured_green_area', 'assessment_date', 'trial_id'],
        'healthy_yield_validation': ['observed_healthy_reference_yield', 'verified_disease_free_reference']}
    return {name: dict(available=False, status='not_identifiable_from_fixed_source_release',
            missing_required_fields=sorted(set(required) - set(columns)),
            reason='Fixed source documentation provides no compatible measured disease/canopy observations.')
            for name, required in requirements.items()}


def harmonize_stage_reference(data):
    raw = pd.read_excel(io.BytesIO(data['Development stage.xlsx']), header=None)
    if raw.iloc[1, 2] != 'DC 32' or raw.iloc[1, 7] != 'DC 30':
        raise ValueError('Unexpected development-stage schema')
    rows = []
    for i, row in raw.iloc[3:].iterrows():
        for crop, stages in [('WW', [(32, 2, 4), (71, 3, 5)]), ('SW', [(30, 7, 9), (65, 8, 10)])]:
            for stage, doy_col, date_col in stages:
                d = pd.Timestamp(row[date_col])
                rows.append(dict(source_file='Development stage.xlsx', source_row=int(i + 1),
                    region=str(row[1]).strip(), crop=crop, reference_stage=stage,
                    published_average_day_number=int(row[doy_col]), reference_date_cell=d.date().isoformat(),
                    reference_date_day_of_year=int(d.dayofyear),
                    day_number_discrepancy=int(d.dayofyear - row[doy_col]), actual_trial_observation=False,
                    reference_duration_days=int(row[12 if crop == 'WW' else 13])))
    return pd.DataFrame(rows)


def assert_no_leakage(train, test, forward=False):
    if train.empty or test.empty: raise ValueError('Empty training or test fold')
    for column in ('trial_key', 'control_id', 'environment_group'):
        if set(train[column]) & set(test[column]): raise ValueError('Leakage in ' + column)
    if forward and train.year.max() >= test.year.min(): raise ValueError('Temporal leakage')


def make_splits(pairs, scheme):
    cols = {'leave_country_year_out': 'environment_group', 'leave_country_out': 'country', 'leave_year_out': 'year'}
    if scheme in cols:
        col = cols[scheme]
        for group in sorted(pairs[col].unique()):
            test = pairs[col].eq(group)
            yield str(group), pairs.index[~test], pairs.index[test]
    elif scheme == 'forward_year':
        for year in (2014, 2015, 2016):
            yield str(year), pairs.index[pairs.year.lt(year)], pairs.index[pairs.year.eq(year)]
    else:
        raise ValueError('Unknown holdout scheme')


def weighted_mean(values, weights):
    values, weights = np.asarray(values, dtype=float), np.asarray(weights, dtype=float)
    if (len(values) == 0 or not np.isfinite(values).all() or not np.isfinite(weights).all()
            or (weights <= 0).any()):
        raise ValueError('Invalid weighted estimate inputs')
    return float(np.average(values, weights=weights))


def benchmark_predict(train, test, model):
    """Only training responses and weights enter estimates; test outcomes are ignored."""
    if model not in MODELS: raise ValueError('Unknown benchmark')
    overall = weighted_mean(train.gain_t_ha, train.trial_weight)
    management = {k: weighted_mean(g.gain_t_ha, g.trial_weight) for k, g in train.groupby('treatment_code')}
    crop_management = {k: weighted_mean(g.gain_t_ha, g.trial_weight)
                       for k, g in train.groupby(['crop', 'treatment_code'])}
    pred, levels = [], []
    for crop, trt in test[['crop', 'treatment_code']].itertuples(index=False, name=None):
        if model == 'crop_management_mean' and (crop, trt) in crop_management:
            pred.append(crop_management[(crop, trt)]); levels.append('crop_management')
        elif model != 'training_mean' and trt in management:
            pred.append(management[trt]); levels.append('management')
        else:
            pred.append(overall); levels.append('overall')
    return np.array(pred), np.array(levels)


def evaluate(pairs, schemes=SCHEMES):
    predictions, memberships, folds = [], [], []
    for scheme in schemes:
        for label, tr, te in make_splits(pairs, scheme):
            train, test = pairs.loc[tr], pairs.loc[te]
            assert_no_leakage(train, test, forward=scheme == 'forward_year')
            for role, frame in [('train', train), ('test', test)]:
                for row in frame.itertuples():
                    memberships.append(dict(scheme=scheme, fold=label, role=role, pair_id=row.pair_id,
                        trial_key=row.trial_key, control_id=row.control_id, environment_group=row.environment_group,
                        country=row.country, year=row.year))
            fold = dict(scheme=scheme, fold=label, training_pairs=len(train), test_pairs=len(test),
                training_trials=int(train.trial_key.nunique()), test_trials=int(test.trial_key.nunique()),
                training_environment_groups=int(train.environment_group.nunique()),
                test_environment_groups=int(test.environment_group.nunique()),
                training_year_min=int(train.year.min()), training_year_max=int(train.year.max()),
                test_year_min=int(test.year.min()), test_year_max=int(test.year.max()))
            for model in MODELS:
                pred, levels = benchmark_predict(train, test, model)
                for row, value, level in zip(test.itertuples(), pred, levels):
                    predictions.append(dict(scheme=scheme, fold=label, model=model, pair_id=row.pair_id,
                        trial_key=row.trial_key, control_id=row.control_id, environment_group=row.environment_group,
                        country=row.country, year=row.year, crop=row.crop, treatment_code=row.treatment_code,
                        trial_weight=row.trial_weight, observed_gain_t_ha=row.gain_t_ha,
                        predicted_gain_t_ha=float(value), fitted_level=level, error_t_ha=float(value-row.gain_t_ha)))
                expected = {'training_mean': 'overall', 'management_mean': 'management',
                            'crop_management_mean': 'crop_management'}[model]
                fold[model + '_fallback_count'] = int(sum(levels != expected))
            folds.append(fold)
    pred = pd.DataFrame(predictions)
    if pred.duplicated(['scheme', 'model', 'pair_id']).any(): raise ValueError('Duplicate held-out response')
    return pred, pd.DataFrame(memberships), pd.DataFrame(folds), score_predictions(pred)


def score_predictions(pred):
    rows = []
    for (scheme, model), g in pred.groupby(['scheme', 'model'], sort=True):
        mse = weighted_mean(g.error_t_ha ** 2, g.trial_weight)
        env_mse = [weighted_mean(e.error_t_ha ** 2, e.trial_weight) for _, e in g.groupby('environment_group')]
        rows.append(dict(scheme=scheme, model=model, pairs=len(g), trials=int(g.trial_key.nunique()),
            environment_groups=int(g.environment_group.nunique()), rmse_t_ha=float(np.sqrt(mse)),
            mae_t_ha=weighted_mean(abs(g.error_t_ha), g.trial_weight),
            mean_error_t_ha=weighted_mean(g.error_t_ha, g.trial_weight),
            country_year_balanced_rmse_t_ha=float(np.sqrt(np.mean(env_mse))), mse=mse))
    scores = pd.DataFrame(rows)
    baseline = scores[scores.model.eq('training_mean')].set_index('scheme').mse
    scores['mse_skill_vs_training_mean'] = 1 - scores.mse / scores.scheme.map(baseline)
    return scores.drop(columns='mse')


def descriptive_summary(pairs):
    rows = []; specs = [('all', 'all', pairs)]
    for variable in ['treatment_code', 'crop', 'country', 'year']:
        specs += [(variable, str(value), g) for value, g in pairs.groupby(variable, sort=True)]
    for by, label, g in specs:
        weights = 1 / g.groupby('trial_key').pair_id.transform('size')
        rows.append(dict(group_by=by, group=label, pairs=len(g), trials=int(g.trial_key.nunique()),
            environment_groups=int(g.environment_group.nunique()), mean_gain_t_ha=weighted_mean(g.gain_t_ha, weights),
            mean_gain_pct_untreated=weighted_mean(g.gain_pct_untreated, weights),
            mean_contrast_pct_treated=weighted_mean(g.contrast_pct_treated, weights),
            negative_pairs=int(g.gain_t_ha.lt(0).sum()), zero_pairs=int(g.gain_t_ha.eq(0).sum()),
            negative_pair_fraction_trial_weighted=weighted_mean(g.gain_t_ha.lt(0), weights)))
    return pd.DataFrame(rows)


def bootstrap_intervals(pairs, pred):
    primary = pred[pred.scheme.eq('leave_country_year_out')]
    groups = sorted(pairs.environment_group.unique())
    weights = np.array([pairs.loc[pairs.environment_group.eq(g), 'trial_weight'].sum() for g in groups])
    gains = np.array([(q.trial_weight*q.gain_t_ha).sum() for g in groups
                      for q in [pairs[pairs.environment_group.eq(g)]]])
    sse = {}
    for model in MODELS:
        frame = primary[primary.model.eq(model)]
        sse[model] = np.array([(q.error_t_ha**2*q.trial_weight).sum() for g in groups
                              for q in [frame[frame.environment_group.eq(g)]]])
    sample = np.random.default_rng(SEED).integers(0, len(groups), (BOOTSTRAPS, len(groups)))
    denom = weights[sample].sum(axis=1)
    mean = gains[sample].sum(axis=1) / denom
    rmse = {m: np.sqrt(s[sample].sum(axis=1) / denom) for m, s in sse.items()}
    rows = [dict(estimand='trial_weighted_mean_gain_t_ha',
        estimate=weighted_mean(pairs.gain_t_ha, pairs.trial_weight),
        lower=float(np.quantile(mean, .025)), upper=float(np.quantile(mean, .975)))]
    for m in MODELS[1:]:
        diff = rmse[m] - rmse['training_mean']
        value = np.sqrt(sse[m].sum()/weights.sum()) - np.sqrt(sse['training_mean'].sum()/weights.sum())
        rows.append(dict(estimand=m+'_minus_training_mean_rmse_t_ha', estimate=float(value),
                         lower=float(np.quantile(diff, .025)), upper=float(np.quantile(diff, .975))))
    return dict(seed=SEED, bootstrap_draws=BOOTSTRAPS, resampling_unit='country_year', intervals=rows,
        fitting_uncertainty_included=False,
        interpretation='Evaluation intervals condition on fixed out-of-fold predictions; no refitting.')


def source_document_text(data):
    with ZipFile(io.BytesIO(data['Weather data description.docx'])) as z:
        doc = ElementTree.fromstring(z.read('word/document.xml'))
    paras = [''.join(p.itertext()) for p in doc.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p')]
    parts = ['Weather data description.docx (paragraph text)\n' + '\n'.join(paras)]
    parts.append('SAS script (unmodified text)\n' + data['SAS Script Sens Spec SEWW_0.33_1trt_350.txt'].decode())
    for f in ('figshare-19203377-metadata.json', 'figshare-19203398-metadata.json'):
        m = json.loads(data[f]); parts.append(m['citation']+'\n'+m['description'])
    return '\n\n'.join(parts)+'\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=HERE/'outputs')
    parser.add_argument('--bundle', type=Path, default=HERE/'source_bundle/nordic_public_sources.zip')
    parser.add_argument('--manifest', type=Path, default=HERE/'source_bundle/manifest.json')
    args = parser.parse_args()
    protocol = json.loads((HERE/'receipts/protocol_fixed.json').read_text())
    if sha256((HERE/'PROTOCOL.md').read_bytes()) != protocol['sha256']:
        raise ValueError('Protocol has changed since pre-fit receipt')
    data = verify_bundle(args.bundle, args.manifest)
    y, sas = read_sources(data)
    source_groups, possible_controls = audit_source_groups(sas, y)
    joined = audit_sas_join(y, sas)
    pairs, excluded = build_pairs(y)
    if pairs.empty: raise ValueError('No compatible observed grain pairs; evaluation unavailable')
    compared = joined.join_status.eq('unique_match') & joined.yield_kg_ha.notna() & joined.sas_yield_kg_ha.notna()
    if not joined.loc[compared, 'numeric_agreement'].all():
        raise ValueError('Conflicting primary/SAS yields beyond rounding tolerance')
    pred, membership, folds, scores = evaluate(pairs)
    summary = descriptive_summary(pairs)
    intervals = bootstrap_intervals(pairs, pred)
    inconsistent = set(pairs.loc[pairs.gain_inconsistent, 'trial_key'])
    sensitivity = pairs.loc[~pairs.trial_key.isin(inconsistent)].copy()
    sensitivity['trial_weight'] = 1/sensitivity.groupby('trial_key').pair_id.transform('size')
    spred, smembership, sfolds, sscores = evaluate(sensitivity, ['leave_country_year_out'])
    stages = harmonize_stage_reference(data)
    stations = clean_cells(pd.read_excel(io.BytesIO(data['Weather stations.xlsx'])))
    stations.rename(columns={'Unnamed: 0': 'source_station_index', 'Station': 'station', 'Region': 'region'}, inplace=True)
    stations['station_key'] = stations.station.str.upper()
    stations['source_row'] = np.arange(len(stations))+2
    if stations.station_key.duplicated().any(): raise ValueError('Ambiguous full station ID')
    site_audit = sas[sas.trial_id.notna()].copy()
    site_audit['station_key'] = site_audit.WstnSite.str.upper()
    site_audit = site_audit.merge(stations[['station_key', 'station', 'region']].rename(
        columns={'station': 'matched_full_station', 'region': 'station_region'}),
        on='station_key', how='left', validate='many_to_one')
    site_audit['station_match_status'] = np.where(site_audit.matched_full_station.notna(), 'exact_full_id', 'unresolved')
    stage_keys = stages[['region', 'crop']].drop_duplicates().copy()
    stage_keys['region_join_key'] = stage_keys.region.str.replace(' ', '', regex=False)
    stage_keys = stage_keys[['region_join_key', 'crop']]
    if stage_keys.duplicated().any(): raise ValueError('Ambiguous stage metadata')
    stage_join = pairs[['pair_id', 'region', 'crop']].copy()
    stage_join['region_join_key'] = stage_join.region.str.replace(' ', '', regex=False)
    stage_join = stage_join.merge(stage_keys.assign(regional_reference_available=True),
        on=['region_join_key', 'crop'], how='left', validate='many_to_one')
    stage_join['observed_trial_stage_available'] = False
    sas_only = sas.dropna(subset=KEY).merge(y[KEY].drop_duplicates().assign(in_primary=True),
        on=KEY, how='left', validate='many_to_one')
    sas_only = sas_only[sas_only.in_primary.isna()].drop(columns='in_primary')
    availability = compatibility_gate(list(y.columns)+list(sas.columns))
    availability['management_grain_pairs'] = dict(available=True, pairs=len(pairs),
        unit='reported_trial_treatment_summary', outcome='observed_management_associated_grain_response')
    out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    frames = {'yield_source_records.csv': y, 'sas_source_records.csv': sas,
        'cross_source_join_audit.csv': joined, 'sas_only_records.csv': sas_only,
        'source_group_audit.csv': source_groups, 'possible_reused_controls.csv': possible_controls,
        'station_join_audit.csv': site_audit, 'regional_stage_reference.csv': stages,
        'weather_station_reference.csv': stations, 'regional_stage_join_audit.csv': stage_join,
        'paired_management_grain.csv': pairs, 'excluded_source_records.csv': excluded,
        'descriptive_summary.csv': summary, 'holdout_predictions.csv': pred,
        'fold_membership.csv': membership, 'fold_receipt.csv': folds, 'benchmark_metrics.csv': scores,
        'sensitivity_pairs.csv': sensitivity, 'sensitivity_predictions.csv': spred,
        'sensitivity_fold_membership.csv': smembership, 'sensitivity_fold_receipt.csv': sfolds,
        'sensitivity_benchmark_metrics.csv': sscores}
    for name, frame in frames.items():
        frame.to_csv(out/name, index=False, float_format='%.12g', lineterminator='\n')
    write_json(out/'compatibility.json', availability)
    write_json(out/'uncertainty.json', intervals)
    (out/'source_documentation.txt').write_text(source_document_text(data))
    shared = pairs.groupby('control_id').size()
    receipt = dict(analysis='Nordic/Baltic 2012–2016 archived management-grain response',
        executed_utc=datetime.now(timezone.utc).isoformat(), protocol_fixed_utc=protocol['recorded_utc'],
        protocol_sha256=protocol['sha256'], code_sha256=sha256(Path(__file__).read_bytes()),
        source_bundle_sha256=sha256(args.bundle.read_bytes()), manifest_sha256=sha256(args.manifest.read_bytes()),
        python=platform.python_version(), package_versions={p: importlib.metadata.version(p)
            for p in ('pandas', 'numpy', 'openpyxl', 'xlrd')},
        execution_inputs='Bundled public-source subset only; no original archive or network reads.',
        source_yield_records=len(y), source_sas_records=len(sas), source_sas_trial_records=int(sas.trial_id.notna().sum()),
        year_counts={str(k): int(v) for k,v in y.year.value_counts().sort_index().items()},
        exclusion_counts={str(k): int(v) for k,v in excluded.reason.value_counts().items()},
        eligible_pairs=len(pairs), eligible_trials=int(pairs.trial_key.nunique()),
        country_year_groups=int(pairs.environment_group.nunique()), countries=sorted(pairs.country.unique()),
        crop_counts={str(k): int(v) for k,v in pairs.crop.value_counts().items()},
        shared_controls=int(shared.gt(1).sum()), negative_pairs=int(pairs.gain_t_ha.lt(0).sum()),
        sum_trial_weights=float(pairs.trial_weight.sum()), substantive_gain_inconsistency_trials=sorted(inconsistent),
        unique_cross_source_matches=int(joined.join_status.eq('unique_match').sum()),
        yield_rounding_max_absolute_difference_kg_ha=float(joined.loc[compared, 'yield_difference_kg_ha'].abs().max()),
        cross_source_join_status={str(k): int(v) for k,v in joined.join_status.value_counts().items()},
        sas_only_trial_records=len(sas_only), regional_reference_rows=len(stages),
        regional_reference_calendar_discrepancy_rows=int(stages.day_number_discrepancy.ne(0).sum()),
        full_station_join_status={str(k): int(v) for k,v in site_audit.station_match_status.value_counts().items()},
        source_group_ids_with_verified_country_year_scope=len(source_groups),
        possible_reused_control_records=len(possible_controls),
        possible_reused_control_signatures=int(possible_controls.possible_control_signature.nunique()),
        observed_disease_records=0, observed_trial_stage_records=0, observed_plot_identifiers=0,
        disease_species='not_recorded', recommendation_evaluation='not_performed; source-derived fields preserved for provenance only',
        headline_descriptive=summary.iloc[0].to_dict(),
        primary_metrics=scores[scores.scheme.eq('leave_country_year_out')].to_dict('records'),
        claim_boundary='Retrospective management-grain contrasts and benchmark transport; no disease-severity, healthy-yield, STB-specific, physiological or climate-response validation.',
        output_sha256={name: sha256((out/name).read_bytes()) for name in
            sorted([*frames,'uncertainty.json','compatibility.json','source_documentation.txt'])})
    write_json(out/'analysis_receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ['eligible_pairs','eligible_trials','country_year_groups',
        'shared_controls','negative_pairs','observed_disease_records']}, indent=2))
    print(scores.to_string(index=False))


if __name__ == '__main__':
    main()
