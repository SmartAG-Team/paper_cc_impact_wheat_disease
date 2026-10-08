"""Append production-reference cells without rewriting existing measurements."""
from pathlib import Path
import json,shutil,sys
import pandas as pd
import pyarrow.parquet as pq
import build_package as package

ROOT=package.ROOT;OUT=package.OUT;HERE=package.HERE


def main():
    manifest=json.loads((OUT/'manifest.json').read_text())
    backup=OUT/'manifest_before_production_exposure.json'
    if not backup.exists():shutil.copyfile(OUT/'manifest.json',backup)
    protected={p:package.sha(OUT/p) for p in manifest['common_schema_partitions']}
    source=ROOT/'data/paper_study/wheat_production/europe_wheat_production_025.parquet'
    production=pd.read_parquet(source)
    name='measurements/wheat_production_metadata/spam2020_europe_wheat_production/part-00000.parquet'
    if name in manifest['common_schema_partitions']:
        print('Existing production partition retained.');return
    country=pd.read_parquet(OUT/'measurements/wheat_area_metadata/spam2020_europe_wheat_area/part-00000.parquet',columns=['record_id','country']).drop_duplicates('record_id').set_index('record_id').country
    frames=[]
    for metric in ['production_total_tonnes','production_irrigated_tonnes','production_rainfed_tonnes']:
        frame=production[['cell_id','latitude','longitude',metric]].copy()
        frame['observation_id']='spam2020_production|'+frame.cell_id+'|'+metric
        frame['dataset_id']='spam2020_europe_wheat_production'
        frame['source_file']=str(source.relative_to(ROOT));frame['source_table']='positive_area_wheat_grid_reference_production'
        frame['source_row']=frame.index.astype(str);frame['source_column']=metric
        frame['record_id']=frame.cell_id;frame['site_id']=frame.cell_id;frame['country']=frame.cell_id.map(country)
        frame['season_year']=2020.;frame['time_basis']='production_reference_year'
        frame['organ']='all_wheat_unallocated_winter_spring';frame['metric']=metric;frame['unit']='tonnes'
        frame['value']=frame[metric];frame['raw_value']=frame[metric].map(repr)
        frame['quality_status']='source_spatial_aggregation; fixed_reference_exposure_not_disease_loss'
        frame['measurement_role']='spatial_wheat_production_reference';frames.append(frame)
    package.emit(pd.concat(frames,ignore_index=True),'wheat_production_metadata')
    assert package.PARTITIONS==[name] and package.COUNTS[0]['measurement_cells']==44823
    for entry in package.METRICS:
        entry['definition']='Fixed2020 source wheat production in metric tonnes, aggregated over European native pixel fractions; not future production or disease loss.'
    count=pd.read_csv(OUT/'measurement_counts.csv');pd.concat([count,pd.DataFrame(package.COUNTS)],ignore_index=True).to_csv(OUT/'measurement_counts.csv',index=False)
    metrics=pd.read_csv(OUT/'common_metric_dictionary.csv');pd.concat([metrics,pd.DataFrame(package.METRICS)],ignore_index=True).to_csv(OUT/'common_metric_dictionary.csv',index=False)
    manifest['common_schema_partitions'].append(name);manifest['views']['wheat_production_metadata']=[name]
    manifest['common_records']+=44823
    manifest['source_sha256'][str(source.relative_to(ROOT))]=package.sha(source)
    manifest['production_exposure_extension']=dict(measurement_cells=44823,grid_cells=14941,source_sha256=package.sha(source),
        historical_measurement_partition_hashes_unchanged=True,source_code_sha256=package.sha(Path(__file__)))
    assert all(package.sha(OUT/p)==h for p,h in protected.items())
    (OUT/'metadata/europe_wheat_production_025.parquet').write_bytes(source.read_bytes())
    readme=OUT/'README.txt'
    text=readme.read_text()+'\n\nThe wheat_production_metadata view adds44,823 metric-tonne cells for total, rainfed and irrigated2020 reference wheat production. The original17 measurement partitions retain their byte hashes. Production exposure is a spatial reference quantity and cannot be pooled with disease observations or interpreted as simulated yield loss. The common package now contains1,355,842 measurement cells in18 exact29-column partitions.\n'
    readme.write_text(text)
    manifest['files']={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=package.sha(p))
        for p in OUT.rglob('*') if p.is_file() and p.name!='manifest.json'}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    check=pd.read_parquet(OUT/name)
    assert pq.read_schema(OUT/name).remove_metadata().equals(package.SCHEMA)
    for metric in check.metric.unique():
        a=check[check.metric.eq(metric)].set_index('record_id').value.reindex(production.cell_id)
        assert (a.to_numpy()==production[metric].to_numpy()).all()
    print(json.dumps(dict(status='passed',common_measurement_cells=manifest['common_records'],
        original_partitions_unchanged=True,production_measurement_cells=len(check))),flush=True)


if __name__=='__main__':main()
