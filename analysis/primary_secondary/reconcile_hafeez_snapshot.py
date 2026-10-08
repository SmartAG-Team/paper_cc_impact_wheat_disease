"""Narrow metadata correction; scores, IDs and original bytes remain immutable."""
from datetime import datetime,timezone
from pathlib import Path
import hashlib
import json
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from analysis.hafeez_metadata import isolate_metadata,leaf_organ_metadata,transgenic_sheet_names

ROOT=Path(__file__).resolve().parents[2]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    path=ROOT/'data/harmonized/observations.parquet'
    table=pq.read_table(path)
    before=sha(path)
    frame=table.to_pandas()
    original=frame.copy()
    subset=frame.dataset_id.eq('hafeez-2025-infection')
    workbook=ROOT/'data/public_septoria/hafeez-2025-infection/Hafeez et al. Stb15, Nature Plants - pathology data.xlsx'
    raw_hash=sha(workbook)
    for sheet in transgenic_sheet_names(pd.ExcelFile(workbook).sheet_names):
        source=pd.read_excel(workbook,sheet_name=sheet)
        mapping=dict(zip((source.index+2).astype(str),isolate_metadata(source,sheet)))
        index=subset&frame.source_table.eq(sheet)
        values=frame.loc[index,'source_row'].map(mapping)
        assert values.notna().all()
        frame.loc[index,'isolate']=values
    frame.loc[subset&frame.source_table.eq('ArinaEMS_IPO88004'),'organ']=leaf_organ_metadata('ArinaEMS_IPO88004')
    untouched=[x for x in frame if x not in ('isolate','organ')]
    pd.testing.assert_frame_equal(frame[untouched],original[untouched])
    changed={c:int(frame[c].fillna('<missing>').ne(original[c].fillna('<missing>')).sum())
             for c in ('isolate','organ')}
    pq.write_table(pa.Table.from_pandas(frame,schema=table.schema,preserve_index=False),path,
                   compression='zstd',use_dictionary=True)
    frame.to_csv(ROOT/'data/harmonized/observations.csv.gz',index=False,
                 compression={'method':'gzip','mtime':0})
    assert sha(workbook)==raw_hash
    manifest_path=ROOT/'data/harmonized/manifest.json'
    manifest=json.loads(manifest_path.read_text())
    for entry in manifest['files']:
        artifact=ROOT/entry['path']
        entry['bytes']=artifact.stat().st_size
        entry['sha256']=sha(artifact)
    manifest['metadata_reconciled_utc']=datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    receipt={'before_sha256':before,'after_sha256':sha(path),'changed_metadata_cells':changed,
        'all_other_cells_identical':True,'original_workbook_immutable':True,
        'workbook_sha256':raw_hash,'records':len(frame)}
    target=ROOT/'analysis/primary_secondary/hafeez_metadata_correction.json'
    target.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
