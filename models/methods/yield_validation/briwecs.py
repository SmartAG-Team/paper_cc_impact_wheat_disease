"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/crop_damage_validation_20261009')
import numpy as np
import pandas as pd
from analysis.methods.yield_validation.run import COLLECTION
SOURCE = COLLECTION / 'sources/public_septoria/briwecs-2025/extracted/Briwecs_data'
DISEASES = ['Stripe_rust', 'Powdery_mildew', 'Leaf_rust', 'DTR', 'Fusarium']

def prepare():
    frames = []
    invalid = 0
    for path in sorted((SOURCE / 'data/final_study/inputs/locations').glob('*.csv')):
        try:
            content = path.read_text(encoding='utf-8-sig')
            encoding = 'utf-8-sig'
        except UnicodeDecodeError:
            content = path.read_text(encoding='cp1252')
            encoding = 'cp1252'
        sep = ';' if ';' in content.splitlines()[0] else ','
        frame = pd.read_csv(path, sep=sep, encoding=encoding)
        if not {'Septoria', 'Seedyield', 'BRISONr', 'Treatment', 'Location', 'Year'} <= set(frame):
            continue
        numeric = pd.to_numeric(frame.Septoria, errors='coerce')
        invalid += int((frame.Septoria.notna() & numeric.isna()).sum())
        frame['Septoria'] = numeric
        for col in ['Seedyield', 'BBCH59', 'BBCH87', *DISEASES]:
            frame[col] = pd.to_numeric(frame[col], errors='coerce') if col in frame else np.nan
        assert frame.Septoria.dropna().between(0, 100).all()
        frame['source_file'] = path.name
        frame['source_row'] = np.arange(len(frame)) + 2
        d = frame.Treatment.str.endswith(('_D', '_DD'))
        irrigated = frame.Location.eq('GGE') & frame.Year.isin([2015, 2018, 2019]) & ~d
        frame['water_regime'] = np.where(irrigated, 'IR', np.where(frame.Location.eq('DKI'), 'RO', 'RF'))
        frame['nitrogen'] = frame.Treatment.str.split('_').str[0]
        frame['protection'] = frame.Treatment.str.split('_').str[1]
        frames.append(frame)
    observations = pd.concat(frames, ignore_index=True)
    eligible = observations[observations.Seedyield.gt(0) & observations.Treatment.isin(['HN_NF', 'HN_WF', 'LN_NF', 'LN_WF'])]
    keys = ['Location', 'Year', 'water_regime', 'nitrogen', 'BRISONr']
    aggregations = {col: 'mean' for col in ['Seedyield', 'Septoria', 'BBCH59', 'BBCH87', *DISEASES]}
    means = eligible.groupby(keys + ['protection'], dropna=False).agg(**{**{col: (col, function) for col, function in aggregations.items()}, 'yield_replicates': ('Seedyield', 'count'), 'severity_replicates': ('Septoria', 'count'), 'source_rows': ('source_row', lambda x: '|'.join(map(str, x))), 'source_file': ('source_file', lambda x: '|'.join(sorted(set(x))))}).reset_index()
    unprotected = means[means.protection.eq('NF') & means.Septoria.notna()]
    protected = means[means.protection.eq('WF')]
    pairs = unprotected.merge(protected, on=keys, suffixes=('_unprotected', '_protected'), validate='one_to_one')
    pairs['severity_unprotected'] = pairs.Septoria_unprotected / 100
    pairs['severity_protected'] = pairs.Septoria_protected / 100
    pairs['severity_reduction'] = pairs.severity_unprotected - pairs.severity_protected
    pairs['response_fraction'] = 1 - pairs.Seedyield_unprotected / pairs.Seedyield_protected
    pairs['site_year'] = pairs.Location + '_' + pairs.Year.astype(str)
    pairs['contrast_id'] = pairs[keys].astype(str).agg('|'.join, axis=1)
    pairs['low_nitrogen'] = pairs.nitrogen.eq('LN').astype(float)
    pairs['irrigated'] = pairs.water_regime.eq('IR').astype(float)
    pairs['heading_doy'] = pairs.BBCH59_unprotected
    pairs['heading_to_ripeness_days'] = pairs.BBCH87_unprotected - pairs.BBCH59_unprotected
    pairs.loc[~pairs.heading_to_ripeness_days.between(1, 100), 'heading_to_ripeness_days'] = np.nan
    for disease in DISEASES:
        pairs[disease + '_unprotected'] /= 100
    assert not pairs.contrast_id.duplicated().any()
    quality = dict(source_files=len(frames), source_rows=len(observations), non_numeric_severity_excluded=invalid, paired_cultivar_management_contrasts=len(pairs), source_site_years=int(pairs.site_year.nunique()), locations=int(pairs.Location.nunique()), negative_responses=int(pairs.response_fraction.lt(0).sum()), pairs_with_protected_severity=int(pairs.severity_protected.notna().sum()), reference_yield_basis='dry mass', grain_unit='Source dt ha-1 at 100% dry mass; relative response cancels common units', severity_unit='Source metadata: percent leaf area, whole-plot assessment; source decimals preserved', moisture_conversion='None; no absolute yields pooled with Tunisia or Nordic 15% moisture data', management='Same source location, year, cultivar, nitrogen and water regime; treatment means, not paired physical plots', limitations=['Disease dates and final-leaf ranks are undocumented in the location tables.', 'Some severity means and harvest means have different replicate coverage.', 'Protection targets multiple diseases; treatment responses do not identify STB-specific losses.', 'Whole-site validation uses five or fewer geographic sites and does not validate stage-dependent canopy physiology.', 'Site-level rating comparability and source-coded water assignment remain transfer limitations; no source rating is rescaled or rounded.'])
    return (pairs, quality)
