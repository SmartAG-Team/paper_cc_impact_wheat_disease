"""Preserve additional source annotations and audit joint pycnidia/damage scores."""
import hashlib
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = Path(__file__).resolve().parents[3]
BOOK = PROJECT / 'data/public_septoria/hafeez-2025-infection/Hafeez et al. Stb15, Nature Plants - pathology data.xlsx'


def run():
    annotations = []
    for sheet in pd.ExcelFile(BOOK).sheet_names:
        if sheet == 'ArinaEMS_IPO88004':
            continue
        frame = pd.read_excel(BOOK, sheet_name=sheet)
        for index, row in frame.iterrows():
            original = {str(key): None if pd.isna(value) else str(value)
                for key, value in row.items() if not (str(key)[0:1] in {'p', 'd'} and str(key)[1:].isdigit())}
            annotations.append(dict(assay=sheet, source_row=index + 2,
                plant_id=f'{sheet}|row{index+2}', original_metadata_json=json.dumps(original),
                source_mildew_annotation=row.get('Mildew_Max'),
                source_SN_annotation=row.get('SN', row.get('sn')),
                derived_AUDPC_used_as_replicate=False))
    pd.DataFrame(annotations).to_csv(HERE / 'source_additional_annotations.csv', index=False)
    scores = pd.read_csv(HERE / 'scores_long.csv.gz', low_memory=False)
    pivot = scores.pivot(index=['plant_id', 'dpi'], columns='endpoint', values='score_percent')
    damage = pivot.damage_necrosis_chlorosis.combine_first(pivot.necrosis)
    paired = pivot.pycnidia.notna() & damage.notna()
    inconsistent = paired & (pivot.pycnidia > damage)
    wrong = pivot.loc[inconsistent].copy()
    wrong['source_damage_or_necrosis_percent'] = damage[inconsistent]
    wrong['pycnidia_excess_pp'] = wrong.pycnidia - damage[inconsistent]
    wrong.reset_index().to_csv(HERE / 'paired_score_observation_conflicts.csv', index=False)
    metadata = pd.DataFrame(annotations)
    receipt = dict(status='verified', source_workbook_sha256=hashlib.sha256(BOOK.read_bytes()).hexdigest(),
        preserved_non_score_source_records=len(annotations),
        positive_source_Mildew_Max_annotations=int(pd.to_numeric(metadata.source_mildew_annotation, errors='coerce').gt(0).sum()),
        Mildew_Max_unit='source_annotation_scale_not_defined_in_workbook',
        valid_paired_pycnidia_damage_scores=int(paired.sum()),
        pycnidia_above_damage_scores=int(inconsistent.sum()),
        maximum_pycnidia_excess_pp=float(wrong.pycnidia_excess_pp.max()),
        interpretation='Separate observed endpoints and explicit observation discrepancies; no score correction or forced latent-stage allocation')
    (HERE / 'additional_annotations_audit.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return receipt


if __name__ == '__main__':
    print(json.dumps(run(), indent=2))
