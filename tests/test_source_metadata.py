"""Recover only factual source metadata, without target-value dependence."""
import pandas as pd
import pytest


def module():
    from models.monitoring import source_metadata
    return source_metadata


def basf_rows():
    return pd.DataFrame({
        'TrialId': [1]*6, 'Country': ['GERMANY']*6,
        'Latitude': [48.3]*6, 'Longitude': [12.0]*6,
        'Date': ['2017-05-01','2017-05-10','2017-05-10','2017-05-20','2017-05-21','2017-05-22'],
        'GsFrom': [31,37,39,73,45,49], 'GsTo': [31,39,41,65,None,49],
        'NumSprays': [0,0,0,0,0,1], 'Treatment': ['Untreated']*6,
        'Organism': ['PUCCST','SEPTTR','SEPTTR','SEPTTR','SEPTTR','SEPTTR'],
        'Parameter': ['INFECT']*6, 'PlantPart': ['LEAF, 1ST / FLAG LEAF']*6,
        'Value': [90,1,99,3,4,5]})


def targets():
    return pd.DataFrame({'global_target_index':[42,43],
        'field_id':['basf-wheat-diseases|1|2017']*2,
        'endpoint_series':['1|LEAF, 1ST / FLAG LEAF']*2,
        'date':['2017-05-10','2017-05-21']})


def test_source_rows_and_calendar_keep_other_pests_but_reject_bad_ranges():
    m=module(); raw=m.normalize_source(basf_rows(),'BASF','raw.tsv','digest')
    assert 'Value' not in raw.columns
    assert raw.source_row_id.tolist()==[2,3,4,5,6,7]
    assert raw.stage_status.tolist()==['valid','valid','valid','reversed_range','incomplete','valid']
    ledger,calendar=m.recover_metadata(targets(),raw)
    assert ledger.global_target_index.tolist()==[42,43]
    assert ledger.source_row_ids.tolist()==['3;4','6']
    assert ledger.treatment.eq('Untreated').all() and ledger.num_sprays.eq(0).all()
    assert ledger.plot_id.isna().all() and ledger.replicate_id.isna().all()
    assert ledger.anatomical_leaf_status.eq('explicit_flag_leaf').all()
    earlier=calendar[calendar.date.eq('2017-05-01')].iloc[0]
    assert earlier.stage_midpoint==31 and earlier.pest_codes=='PUCCST'
    conflict=calendar[calendar.date.eq('2017-05-10')].iloc[0]
    assert conflict.stage_status=='conflicting_ranges' and pd.isna(conflict.stage_midpoint)
    assert '2017-05-22' not in calendar.date.tolist()
    assert ledger.assessment_stage_use.eq('oracle_diagnostic_only').all()


def test_value_changes_do_not_change_recovery_and_unsupported_columns_rejected():
    m=module(); a=basf_rows(); b=a.copy(); b['Value']=999999
    aa=m.recover_metadata(targets(),m.normalize_source(a,'BASF','p','h'))
    bb=m.recover_metadata(targets(),m.normalize_source(b,'BASF','p','h'))
    for x,y in zip(aa,bb): pd.testing.assert_frame_equal(x,y)
    with pytest.raises(ValueError,match='target values'):
        m.recover_metadata(targets().assign(observed_percent=0),m.normalize_source(a,'BASF','p','h'))


def test_corteva_source_label_does_not_verify_anatomical_flag_leaf():
    m=module(); c=pd.DataFrame({'Field':['2017-1'],'Date':['2017-05-10'],
        'Country':['GERMANY'],'Latitude':[48.3],'Longitude':[12.0],
        'GsFrom':[39],'GsTo':[39],'NumSprays':[0],'Treatment':['Untreated'],
        'PestCode':['SEPTTR'],'EvaluationType':['INFECT'],'SamplingUnit':['LEAF 1'],
        'TreatmentCode':['A'],'Applications':[None],'Value':[5]})
    t=pd.DataFrame({'global_target_index':[1],'field_id':['corteva-2014-2018-external|2017-1|2017'],
        'endpoint_series':['2017-1|LEAF 1'],'date':['2017-05-10']})
    ledger,calendar=m.recover_metadata(t,m.normalize_source(c,'Corteva','p','h'))
    assert ledger.iloc[0].anatomical_leaf_status=='ordinal_reference_unverified'
    assert ledger.iloc[0].treatment_codes=='A'
    assert ledger.iloc[0].source_row_ids=='2'
    assert calendar.iloc[0].stage_from==calendar.iloc[0].stage_to==39


def test_artifact_export_handles_targets_with_no_prior_stage(tmp_path):
    import json
    m=module()
    root=tmp_path
    obs_path=root/'outputs/disease_improvement/station_repairs_20261022_v4/observation_ledger.csv'
    obs_path.parent.mkdir(parents=True)
    obs=targets().assign(global_target_index=[479,43],source='BASF',latitude=48.3,longitude=12.0,observed_percent=[1,4])
    obs.to_csv(obs_path,index=False)
    bpath=root/'basf.tsv'
    basf_rows().iloc[1:].to_csv(bpath,sep='\t',index=False)
    cpath=root/'data/final_study/inputs/stb_collection_20261009/sources/other_field_sources/corteva-2014-2018/corteva-wheat-diseases.txt'
    cpath.parent.mkdir(parents=True)
    pd.DataFrame({'Field':['2017-2'],'Date':['2017-05-10'],'Country':['GERMANY'],
        'Latitude':[48.3],'Longitude':[12.0],'GsFrom':[39],'GsTo':[39],
        'NumSprays':[0],'Treatment':['Untreated'],'PestCode':['SEPTTR'],
        'EvaluationType':['INFECT'],'SamplingUnit':['LEAF 1']}).to_csv(cpath,sep='\t',index=False)
    report=m.run(root,root/'artifacts',bpath)
    loaded=json.loads((root/'artifacts/coverage.json').read_text())
    assert loaded['n_targets']==report['n_targets']==2
    exported=pd.read_csv(root/'artifacts/metadata_ledger.csv')
    assert exported.global_target_index.tolist()==[479,43]
    assert 'observed_percent' not in exported.columns
    assert pd.read_csv(root/'artifacts/strictly_prior_stage_coverage.csv').strictly_prior_stage_available.eq(False).all()
