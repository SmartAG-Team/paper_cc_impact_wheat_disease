"""Source-backed geography and archived maps; no biological fitting or simulation."""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
STUDY=ROOT/'analysis/paper_study/overwinter_leaf_model_20261006'
CLIMATE=STUDY/'climate'
ARCHIVE=ROOT/'publication/european_wheat_stb_before_overwinter_oop_20261006'
METRICS=['F1_symptom_relative_anthesis_days','grain_fill_days','conditional_yield_loss_GS65_85_b0180_t_ha_per_unit_lai']
ARCHIVED={'figS15_archived_domain':'fig1_domain_and_model',
    'figS16_archived_era5_baseline':'fig3_era5_baseline',
    'figS17_archived_climate_maps':'fig4_climate_change_maps',
    'figS18_archived_production_exposure':'fig6_production_exposure'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    cells_path=ROOT/'data/paper_study/wheat_area/europe_wheat_cells_025.parquet'
    cells=pd.read_parquet(cells_path)[['cell_id','row','col','latitude','longitude','harvested_total_ha']]
    cells.to_csv(HERE/'map_wheat_domain.csv',index=False)
    membership_path=STUDY/'phenology/original_input_membership.csv'
    fields=pd.read_csv(membership_path)[['field_id','site_id','season_year','latitude','longitude','partition']]
    assert len(fields)==216 and fields.groupby('partition').size().to_dict()=={
        'calibration':28,'reused_development_2019':45,'reused_external_strict':143}
    fields.to_csv(HERE/'map_field_locations.csv',index=False)
    draw_path=ROOT/'data/paper_study/regional_parameter_uncertainty/spatial_draws.csv'
    draws=pd.read_csv(draw_path)[['spatial_draw_id','cell_id','latitude','longitude','area_mean_weight']]
    all_rows=[];ensemble=[]
    for kind,name,periodcol in [('level','draw_period_moments.csv','period'),
            ('change','paired_draw_period_moments.csv','future_period')]:
        source=pd.read_csv(CLIMATE/name)
        frame=source.loc[source.metric.isin(METRICS)].copy()
        frame['mapped_value']=np.divide(frame.numerator,frame.denominator,
            out=np.full(len(frame),np.nan),where=frame.denominator.to_numpy()>0)
        frame['kind']=kind;frame['period']=frame[periodcol]
        frame=frame.merge(draws,on='spatial_draw_id',suffixes=('','_registered'),validate='many_to_one')
        frame['valid_years_or_pairs']=frame['valid_years' if kind=='level' else 'valid_year_pairs']
        all_rows.append(frame)
        for key,group in frame.groupby(['model','scenario','period','metric','cell_id']):
            values=group.mapped_value.to_numpy(float)
            np.testing.assert_allclose(values,np.full(len(values),values[0]),rtol=0,atol=1e-12,equal_nan=True)
        unique=frame.drop_duplicates(['model','scenario','period','metric','cell_id'])
        for key,group in unique.groupby(['scenario','period','metric','cell_id']):
            good=group.loc[group.mapped_value.notna()]
            scenario,period,metric,cell=key
            ensemble.append(dict(kind=kind,scenario=scenario,period=period,metric=metric,cell_id=cell,
                latitude=group.latitude.iloc[0],longitude=group.longitude.iloc[0],
                n_gcm=len(good),value=float(good.mapped_value.mean()) if len(good)==3 else np.nan,
                gcm_min=float(good.mapped_value.min()) if len(good)==3 else np.nan,
                gcm_max=float(good.mapped_value.max()) if len(good)==3 else np.nan,
                minimum_valid_years_or_pairs=int(group.valid_years_or_pairs.min()),
                maximum_valid_years_or_pairs=int(group.valid_years_or_pairs.max()),
                duplicated_draws_collapsed_for_display_only=int(len(frame.loc[frame.scenario.eq(scenario)&frame.period.eq(period)&frame.metric.eq(metric)&frame.cell_id.eq(cell)]))//3))
    pd.concat(all_rows,ignore_index=True).to_csv(HERE/'map_draw_values.csv',index=False)
    ensemble=pd.DataFrame(ensemble);ensemble.to_csv(HERE/'map_ensemble_values.csv',index=False)
    coverage=ensemble.groupby(['kind','scenario','period','metric']).agg(registered_unique_cells=('cell_id','size'),
        mapped_cells=('value','count'),minimum_valid_years_or_pairs=('minimum_valid_years_or_pairs','min')).reset_index()
    coverage.to_csv(HERE/'map_metric_coverage.csv',index=False)
    original={}
    for name in ['field_and_baseline_captions.json','climate_and_production_captions.json']:
        original.update(json.loads((ARCHIVE/'figures'/name).read_text()))
    archives=[]
    for new,old in ARCHIVED.items():
        archives.append(dict(figure=new,original_figure=old,model_version='archived seasonal-v1',
            source_png=str((ARCHIVE/'figures'/f'{old}.png').relative_to(ROOT)),
            source_pdf=str((ARCHIVE/'figures'/f'{old}.pdf').relative_to(ROOT)),
            png_sha256=sha(ARCHIVE/'figures'/f'{old}.png'),pdf_sha256=sha(ARCHIVE/'figures'/f'{old}.pdf'),
            original_caption=original[old],current_model_validation=False))
    (HERE/'archived_map_registry.json').write_text(json.dumps(archives,indent=2)+'\n')
    inputs=[cells_path,membership_path,draw_path,CLIMATE/'draw_period_moments.csv',CLIMATE/'paired_draw_period_moments.csv',
        ROOT/'data/geography/ne_110m_admin_0_countries.zip',Path(__file__)]
    receipt=dict(status='verified',full_reference_wheat_grid_cells=len(cells),current_model_unique_sampled_cells=62,
        trial_field_seasons=216,maps_interpolated=False,duplicate_draws_preserved_in_regional_statistics=True,
        current_maps_full_domain_census=False,archived_model_maps_labelled_separately=True,
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in inputs},
        table_sha256={p.name:sha(p) for p in HERE.glob('map_*.csv')})
    (HERE/'map_data_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))


if __name__=='__main__':prepare()
