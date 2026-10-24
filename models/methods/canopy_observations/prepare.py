"""Scientific methods retained for the final wheat study; original source identities are recorded in source_manifest.json."""
from pathlib import Path
from analysis.final_study.paths import ROOT as PROJECT_ROOT, resolve, evidence_dir
ROOT = PROJECT_ROOT
HERE = evidence_dir('analysis/paper_study/canopy_observations_20261010')
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np
import pandas as pd
from analysis.methods.canopy_observations.observations import numeric, unique, dry_yield_t_ha, green_records, join_plot_observations, observed_windows, protection_contrasts, boundary_qualification
HERE = evidence_dir('analysis/paper_study/canopy_observations_20261010')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def verify_inputs(base=HERE):
    raise RuntimeError('Historical registration/source-snapshot guard is not a final reproduction interface; use analysis.final_study with the retained scientific manifest.')

def read_workbook(base=HERE):
    raw = json.loads(gzip.decompress((Path(base) / 'inputs/han2018_observations.json.gz').read_bytes()))
    columns = [str(c).strip() if c is not None else f'unnamed_{i + 1}' for i, c in enumerate(raw['header'])]
    if len(columns) != len(set(columns)):
        raise ValueError('duplicate workbook header')
    frame = pd.DataFrame([r['values'] for r in raw['rows']], columns=columns)
    frame['source_row'] = [r['source_row'] for r in raw['rows']]
    frame = frame[frame.Probe.notna()].copy()
    return (frame, raw)

def plot_ids(frame):
    return frame.location + '|' + frame.year.astype(int).astype(str) + '|' + frame.treatment + '|' + frame.block + '|' + frame.cultivar_id

def prepare(base=HERE):
    base = Path(base)
    verify_inputs(base)
    inputs = base / 'inputs'
    protocol = json.loads((base / 'protocol.json').read_text())
    source = []
    for year in range(2015, 2019):
        f = inputs / f'HAN_{year}.csv'
        d = pd.read_csv(f, sep=';')
        d['source_row'] = np.arange(len(d)) + 2
        if year < 2018:
            d = d[d.Treatment.eq('HN_WF')].copy()
        d = d.rename(columns={'Year': 'year', 'Location': 'location', 'BRISONr': 'cultivar_id', 'Block': 'block', 'Treatment': 'treatment'})
        d['plot_id'] = plot_ids(d)
        d['nitrogen'] = d.treatment.str.split('_').str[0]
        d['protection'] = d.treatment.str.split('_').str[1]
        d['water_regime'] = 'RF'
        d['yield_dry_t_ha'] = dry_yield_t_ha(d.Seedyield, 'dt_ha')
        d['stb_percent'] = numeric(d.Septoria)
        d['heading_doy'] = numeric(d.BBCH59)
        d['terminal_doy'] = numeric(d.BBCH87)
        d['sowing_source_doy'] = numeric(d.Sowing_date)
        for disease in ['Stripe_rust', 'Powdery_mildew', 'Leaf_rust', 'DTR', 'Fusarium']:
            d[disease] = numeric(d[disease]) if disease in d else np.nan
        d['harvest_source_file'] = f.name
        d['harvest_source_row'] = d.source_row
        source.append(d)
    rawplots = pd.concat(source, ignore_index=True)
    unique(rawplots, ['plot_id'])
    fields = ['plot_id', 'year', 'location', 'cultivar_id', 'block', 'treatment', 'nitrogen', 'protection', 'water_regime', 'yield_dry_t_ha', 'stb_percent', 'heading_doy', 'terminal_doy', 'sowing_source_doy', 'Stripe_rust', 'Powdery_mildew', 'Leaf_rust', 'DTR', 'Fusarium', 'harvest_source_file', 'harvest_source_row']
    plots = rawplots[fields].copy()
    g = pd.read_csv(inputs / 'gb_long.csv')
    g['source_row'] = np.arange(len(g)) + 2
    g['location'] = 'HAN'
    g['treatment'] = 'HN_WF'
    g['cultivar_id'] = 'BRISONr_' + g.cult.astype(int).astype(str)
    g['block'] = 'B' + g.wdh.astype(int).astype(str)
    g['plot_id'] = plot_ids(g)
    oldgreen = green_records(g, value_col='perc_green', date_col='Datum', unit='fraction', source_name='gb_long.csv')
    q = boundary_qualification(g)
    oldgreen = oldgreen.drop(columns=['measurement_eligible', 'record_kind']).merge(q, on=['plot_id', 'source_row'], validate='one_to_one')
    workbook, _ = read_workbook(base)
    w = workbook.rename(columns={'Year': 'year', 'Location': 'location', 'Treatment': 'treatment'}).copy()
    w['cultivar_id'] = 'BRISONr_' + w.BRISO_Nr.astype(int).astype(str)
    w['block'] = w.replication.map(protocol['canopy_2018']['replication_crosswalk'])
    w['plot_id'] = plot_ids(w)
    unique(w, ['plot_id'])
    unique(w, ['Probe'])
    dates = protocol['canopy_2018']['date_columns']
    long = w[['plot_id', 'source_row', *dates]].melt(id_vars=['plot_id', 'source_row'], var_name='source_date', value_name='value')
    long['source_date'] = pd.to_datetime(long.source_date, format='%d.%m.%Y')
    newgreen = green_records(long, value_col='value', date_col='source_date', unit='percent', source_name='HAN_2018.xlsx:Ergebnisse HAN 2018')
    newgreen['record_kind'] = 'workbook_dated_assessment'
    green = pd.concat([oldgreen, newgreen], ignore_index=True)
    unique(green, ['observation_id'])
    unique(green, ['plot_id', 'date'])
    join_plot_observations(green, plots)
    if set(green.plot_id) != set(plots.plot_id):
        raise ValueError('Plot population and canopy population differ')
    m = w[['plot_id', 'source_row', 'Seedyield', 'Septoria', 'BBCH59', 'BBCH87']].merge(plots, on='plot_id', validate='one_to_one')
    comparisons = []
    for left, right, unit, tolerance in [('Seedyield', 'yield_dry_t_ha', 't ha-1 dry mass', 0.00051), ('Septoria', 'stb_percent', 'source percent', 1e-12), ('BBCH59', 'heading_doy', 'source DOY', 0), ('BBCH87', 'terminal_doy', 'source DOY', 0)]:
        a = numeric(m[left])
        b = numeric(m[right])
        delta = a - b
        if (a.isna() != b.isna()).any() or delta.dropna().abs().gt(tolerance).any():
            raise ValueError(f'Original workbook/location disagreement: {left}')
        for i in m.index:
            comparisons.append(dict(plot_id=m.loc[i, 'plot_id'], trait=left, comparison='workbook_minus_original_location', raw_value=a.loc[i], comparison_value=b.loc[i], difference=delta.loc[i], unit=unit))
    wi = w.set_index('plot_id')
    is2018 = plots.year.eq(2018)
    plots.loc[is2018, 'yield_dry_t_ha'] = plots.loc[is2018, 'plot_id'].map(numeric(wi.Seedyield))
    plots['workbook_source_row'] = plots.plot_id.map(wi.source_row)
    plots['yield_source_precision'] = np.where(is2018, 'unrounded 2018 workbook; reconciled to original location table', 'original location table')
    core = pd.read_csv(inputs / 'publication_core.csv', sep=';')
    core = core[core.Location.eq('HAN') & core.Year.between(2015, 2018)].copy()
    core = core.rename(columns={'Year': 'year', 'Location': 'location', 'BRISONr': 'cultivar_id', 'Block': 'block', 'Treatment': 'treatment'})
    core.treatment = core.treatment.str.removesuffix('_RF')
    core['plot_id'] = plot_ids(core)
    unique(core, ['plot_id'])
    c = plots[['plot_id', 'stb_percent']].merge(core[['plot_id', 'Septoria']], on='plot_id', how='left', validate='one_to_one', indicator=True)
    if c['_merge'].ne('both').any():
        raise ValueError('Unmatched publication core identity')
    for row in c.itertuples():
        comparisons.append(dict(plot_id=row.plot_id, trait='Septoria', comparison='original_minus_publication_core', raw_value=row.stb_percent, comparison_value=row.Septoria, difference=row.stb_percent - row.Septoria, unit='source percent'))
    lai = pd.read_csv(inputs / 'laili_merge.csv')
    lai['source_row'] = np.arange(len(lai)) + 2
    lai['year'] = lai.season
    lai['location'] = 'HAN'
    lai['treatment'] = 'HN_WF'
    lai['cultivar_id'] = 'BRISONr_' + lai.cult.astype(int).astype(str)
    lai['block'] = 'B' + lai.wdh.astype(int).astype(str)
    lai['plot_id'] = plot_ids(lai)
    lai['date'] = pd.to_datetime(lai['Date.y'], errors='raise').dt.strftime('%Y-%m-%d')
    lai['source_LAI'] = numeric(lai.LAI)
    lai['source_light_interception'] = numeric(lai.li_mean)
    lai['source_file'] = 'laili_merge.csv'
    lai['measurement_scope'] = 'Source LAI estimate; not independently measured functional upper-three-leaf area'
    lai = lai[['plot_id', 'date', 'source_LAI', 'source_light_interception', 'source_file', 'source_row', 'measurement_scope']]
    join_plot_observations(lai, plots)
    lai['source_record_id'] = lai.source_file + '|' + lai.source_row.astype(str)
    unique(lai, ['source_record_id'])
    lai['plot_date_source_records'] = lai.groupby(['plot_id', 'date']).source_row.transform('size')
    lai['duplicate_plot_date'] = lai.plot_date_source_records.gt(1)
    lai['unique_plot_date_eligible'] = ~lai.duplicate_plot_date
    lai['record_role'] = 'raw_source_audit'
    unique(lai[lai.unique_plot_date_eligible], ['plot_id', 'date'])
    lai_duplicates = lai[lai.duplicate_plot_date].copy()
    eligible_green = green[green.measurement_eligible].copy()
    boundaries = green[~green.measurement_eligible].copy()
    windows = observed_windows(eligible_green)
    primary_dates = protocol['canopy_2018']['common_primary_dates']
    primary = eligible_green[eligible_green.date.isin(primary_dates)]
    pw = observed_windows(primary).rename(columns={'green_fraction_days': 'primary_green_fraction_days'})
    pw['primary_complete'] = pw.observations.eq(len(primary_dates)) & pw.records.eq(len(primary_dates))
    plots = plots.merge(pw[['plot_id', 'primary_green_fraction_days', 'primary_complete']], on='plot_id', how='left', validate='one_to_one')
    plots['primary_complete'] = plots.primary_complete.fillna(False).astype(bool)
    plots['functional_HAD_eligible'] = False
    plots['independent_new_environment'] = False
    plots['disease_date_status'] = 'undated source score; no disease time integral'
    plots['heading_definition'] = 'source BBCH59 heading; not anthesis'
    plots['terminal_definition'] = 'source BBCH87 maturity definition; not assumed equal to DWD phase 21'
    pairs = protection_contrasts(plots[plots.year.eq(2018)])
    byyear = plots.groupby('year', sort=True).agg(plots=('plot_id', 'size'), cultivars=('cultivar_id', 'nunique'), yield_observed=('yield_dry_t_ha', 'count'), stb_score_observed=('stb_percent', 'count'), heading_observed=('heading_doy', 'count'), terminal_observed=('terminal_doy', 'count')).reset_index()
    linked = join_plot_observations(green, plots[['plot_id', 'year']])
    linked['eligible_value'] = linked.green_fraction.where(linked.measurement_eligible)
    linked['unverified_boundary'] = ~linked.measurement_eligible
    gc = linked.groupby('year').agg(canopy_rows=('green_fraction', 'size'), canopy_nonmissing_source_values=('green_fraction', 'count'), canopy_observed=('eligible_value', 'count'), canopy_eligible_rows=('measurement_eligible', 'sum'), unverified_boundary_records=('unverified_boundary', 'sum'), canopy_dates=('date', 'nunique'))
    byyear = byyear.merge(gc, on='year', validate='one_to_one')
    byyear['location'] = 'HAN'
    byyear['environmental_replicates'] = 1
    byyear['new_independent_environments'] = 0
    overlaps = plots[['plot_id', 'year', 'location', 'harvest_source_file', 'harvest_source_row']].copy()
    overlaps['existing_evidence'] = 'BRIWECS original harvest records already available in crop_damage_validation_20261009'
    overlaps['new_measurement'] = 'repeated whole-canopy green fraction; not a new field environment'
    return dict(plots=plots, canopy_observations=green, boundary_records=boundaries, lai_observations=lai, lai_duplicate_records=lai_duplicates, observed_windows=windows, protection_contrasts=pairs, source_disagreement=pd.DataFrame(comparisons), cohort_overlap=overlaps, coverage=byyear)
