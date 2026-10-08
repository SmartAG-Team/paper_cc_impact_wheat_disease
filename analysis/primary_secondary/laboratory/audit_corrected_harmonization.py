"""Keep a separate receipt for the current harmonized Hafeez metadata snapshot."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[3]


def verify():
    scores = pd.read_csv(HERE / 'scores_long.csv.gz', low_memory=False)
    scores['source_row'] = scores.source_row.astype(str)
    scores = scores.rename(columns={'assay': 'source_table'})
    path = PROJECT / 'data/harmonized/observations.parquet'
    current_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    frame = pd.read_parquet(path, filters=[('dataset_id', '==', 'hafeez-2025-infection')])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == current_hash, 'Shared snapshot changed during read'
    joint = scores.merge(frame, on=['source_table', 'source_row', 'source_column'],
        suffixes=('_source', '_harmonized'), validate='one_to_one')
    assert len(joint) == len(scores) == len(frame) == 60828
    np.testing.assert_allclose(joint.score_percent, joint.value, rtol=0, atol=0, equal_nan=True)
    assert joint.quality_status_source.equals(joint.quality_status_harmonized)
    isolate_errors = int(joint.isolate_source.ne(joint.isolate_harmonized).sum())
    accepted = {'primary_seedling_leaf': {'seedling_leaf', 'primary_seedling_leaf'},
        'second_leaf_6cm_section': {'second_leaf', 'second_leaf_section', 'second_leaf_6cm_section'}}
    organ_errors = sum(str(row.organ) not in accepted[row.leaf] for row in joint.itertuples())
    initial = json.loads((HERE / 'initial_harmonized_reconciliation.json').read_text())
    receipt = dict(status='verified' if isolate_errors == organ_errors == 0 else 'metadata_discrepancies_remain',
        completed_utc=datetime.now(timezone.utc).isoformat(), score_cells_checked=len(joint),
        numeric_and_quality_flags_identical=True, current_harmonized_sha256=current_hash,
        initial_harmonized_sha256=initial['harmonized_sha256'],
        changed_from_initial_snapshot=current_hash != initial['harmonized_sha256'],
        remaining_isolate_discrepancies=isolate_errors, remaining_leaf_organ_discrepancies=int(organ_errors),
        initial_metadata_discrepancies=initial['metadata_discrepancies'],
        initial_audit_preserved=True)
    (HERE / 'corrected_harmonized_reconciliation.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
