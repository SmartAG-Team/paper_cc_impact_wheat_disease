"""Compare crop and symptom dates on one identical population of season pairs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from .data import ROOT, GRID, DERIVED, MODELS, sha

EVENTS = ['symptom_days', 'flowering_days', 'soft_dough_days',
          'symptom_relative_flowering_days', 'grain_fill_elapsed_days']
COLUMNS = ['cell_id', 'valid_complete_season', 'calendar_sowing_date',
           'F1_symptom_date', 'BBCH65_date', 'BBCH85_date',
           'F1_symptom_day_after_sowing', 'F1_symptom_relative_anthesis_days']
ANNUAL = ROOT/'analysis/paper_study/nature_food_revision_20261007/continental_replay/annual_outputs/nasa'


def elapsed_dates(frame):
    if frame.cell_id.duplicated().any():
        raise ValueError('Duplicate cell identities')
    date = {key: pd.to_datetime(frame[key]) for key in COLUMNS if key.endswith('_date')}
    result = pd.DataFrame(index=frame.index)
    for event, key in [('symptom_days', 'F1_symptom_date'),
                       ('flowering_days', 'BBCH65_date'), ('soft_dough_days', 'BBCH85_date')]:
        result[event] = (date[key]-date['calendar_sowing_date']).dt.days
    result['symptom_relative_flowering_days'] = result.symptom_days-result.flowering_days
    result['grain_fill_elapsed_days'] = result.soft_dough_days-result.flowering_days
    for key, recorded in [('symptom_days', 'F1_symptom_day_after_sowing'),
                          ('symptom_relative_flowering_days', 'F1_symptom_relative_anthesis_days')]:
        np.testing.assert_allclose(result[key], frame[recorded], equal_nan=True, atol=0)
    result.loc[~frame.valid_complete_season, :] = np.nan
    finite = result.dropna()
    if (finite.symptom_days < 0).any() or (finite.grain_fill_elapsed_days < 0).any():
        raise ValueError('Invalid within-season date order')
    result.index = frame.cell_id
    return result


def common_pair(past, future):
    a, b = elapsed_dates(past), elapsed_dates(future)
    if set(a.index) != set(b.index):
        raise ValueError('Historical and future cell identities differ')
    b = b.loc[a.index]
    valid = np.isfinite(a).all(axis=1) & np.isfinite(b).all(axis=1)
    return a, b, valid.to_numpy()


def run():
    DERIVED.mkdir(exist_ok=True)
    registry = pd.read_csv(GRID/'full_landuse_cell_registry.csv').set_index('cell_id')
    rows, cells, sources = [], [], {str((GRID/'full_landuse_cell_registry.csv').relative_to(ROOT)): sha(GRID/'full_landuse_cell_registry.csv')}
    for model in MODELS:
        a_sum = np.zeros((len(registry), len(EVENTS)))
        b_sum = np.zeros_like(a_sum)
        counts = np.zeros(len(registry), dtype=int)
        for i in range(30):
            paths = [ANNUAL/model/'ssp585'/f'{year+i}.parquet' for year in [1991, 2071]]
            frames = [pd.read_parquet(path, columns=COLUMNS).set_index('cell_id').loc[registry.index].reset_index() for path in paths]
            for path in paths:
                sources[str(path.relative_to(ROOT))] = sha(path)
            a, b, valid = common_pair(*frames)
            a_sum += np.where(valid[:, None], a.to_numpy(), 0)
            b_sum += np.where(valid[:, None], b.to_numpy(), 0)
            counts += valid
        for name in ['Europe', *sorted(registry.environment_region.fillna('Unassigned').unique())]:
            mask = np.ones(len(registry), bool) if name == 'Europe' else registry.environment_region.fillna('Unassigned').eq(name).to_numpy()
            weights = registry.harvested_total_ha.to_numpy()*mask
            denominator = weights@counts
            if denominator == 0:
                continue
            for j, event in enumerate(EVENTS):
                past = weights@a_sum[:, j]/denominator
                future = weights@b_sum[:, j]/denominator
                rows.append(dict(model=model, environment_region=name, event=event,
                    reference=past, future=future, change=future-past,
                    valid_pairs=int(counts[mask].sum()), valid_cells=int(((counts > 0)&mask).sum()),
                    valid_area_time_fraction=denominator/(30*weights.sum())))
        for j, event in enumerate(EVENTS):
            change = np.divide(b_sum[:, j]-a_sum[:, j], counts,
                               out=np.full(len(counts), np.nan), where=counts > 0)
            cells.append(pd.DataFrame(dict(cell_id=registry.index, model=model, event=event,
                                            change=change, valid_year_pairs=counts)))
    by_model = pd.DataFrame(rows)
    by_model.to_csv(DERIVED/'common_pair_timing_by_gcm.csv', index=False)
    ensemble = by_model.groupby(['environment_region', 'event']).agg(
        reference=('reference', 'mean'), future=('future', 'mean'), change=('change', 'mean'),
        gcm_min=('change', 'min'), gcm_max=('change', 'max'), models=('model', 'nunique'),
        minimum_valid_area_time_fraction=('valid_area_time_fraction', 'min')).reset_index()
    if not ensemble.models.eq(3).all():
        raise ValueError('Three complete climate-model estimates required')
    ensemble.to_csv(DERIVED/'common_pair_timing.csv', index=False)
    cell = pd.concat(cells, ignore_index=True)
    cell.to_parquet(DERIVED/'common_pair_timing_cells.parquet', index=False)
    for frame in [by_model.set_index(['model', 'environment_region', 'event']),
                  ensemble.set_index(['environment_region', 'event'])]:
        for column in ['reference', 'future', 'change']:
            x = frame[column].unstack('event')
            np.testing.assert_allclose(x.symptom_days-x.flowering_days,
                x.symptom_relative_flowering_days, rtol=0, atol=1e-12)
            np.testing.assert_allclose(x.soft_dough_days-x.flowering_days,
                x.grain_fill_elapsed_days, rtol=0, atol=1e-12)
    outputs = ['common_pair_timing_by_gcm.csv', 'common_pair_timing.csv', 'common_pair_timing_cells.parquet']
    receipt = dict(status='passed', scenario='ssp585', reference='1991-2020', future='2071-2100',
        pair_population='Both complete seasons and all five elapsed-date quantities finite; identical population for every event',
        weighting='Fixed harvested area times valid pair count within climate model; equal three-model means',
        grain_fill_definition='Elapsed calendar days between flowering and soft dough, excluding inclusive-day counting',
        not_an_intervention_effect=True, source_sha256=sources,
        outputs_sha256={name: sha(DERIVED/name) for name in outputs})
    (DERIVED/'common_pair_timing_receipt.json').write_text(json.dumps(receipt, indent=2)+'\n')
    print(ensemble[ensemble.environment_region.eq('Europe')].to_string(index=False))


def read():
    receipt = json.loads((DERIVED/'common_pair_timing_receipt.json').read_text())
    for name, digest in receipt['outputs_sha256'].items():
        if sha(DERIVED/name) != digest:
            raise ValueError('Timing evidence changed: '+name)
    return pd.read_csv(DERIVED/'common_pair_timing.csv')


if __name__ == '__main__':
    run()
