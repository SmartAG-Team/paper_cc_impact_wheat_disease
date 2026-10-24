"""Independent table-only checks; no extension imports or new simulations."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

CORNERS = ['y00', 'y10', 'y01', 'y11']
KEY = ['model', 'reference_year', 'future_year', 'spatial_draw_id', 'cell_id']
MAP = {'Wreference_Hreference': 'y00', 'Wfuture_Hreference': 'y10',
       'Wreference_Hfuture': 'y01', 'Wfuture_Hfuture': 'y11'}
TERMS = ['net', 'weather', 'host', 'weather_alone', 'host_alone', 'interaction']
WINDOWS = ['reference_window_days', 'future_window_days',
           'reference_host_window_days', 'future_host_window_days']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def weighted(values, weights):
    return math.fsum((np.asarray(values, float)*np.asarray(weights, float)).tolist()) / math.fsum(np.asarray(weights, float).tolist())


def archived_baseline(root):
    relative = 'data/final_study/evidence/review_response/attribution/four_corner_draw_year_outputs.parquet'
    path = root / relative
    manifest = json.loads((root / 'data/final_study/path_manifest.json').read_text())
    matches = [row for row in manifest['files'].values() if row.get('path') == relative]
    assert len(matches) == 1 and sha(path) == matches[0]['sha256'], 'Archive source identity mismatch'
    original = pd.read_parquet(path)
    original = original.loc[original.setting.eq('baseline_shift0')]
    assert len(original) == 4*3*30*64
    assert (original.groupby(KEY).size() == 4).all()
    wide = original.pivot(index=KEY, columns='corner', values='GS65_85_lost_had3').rename(columns=MAP).reset_index()
    details = original.groupby(KEY).agg(area_mean_weight=('area_mean_weight','first'),
        excluded=('excluded_unsupported_hybrid_forcing','any')).reset_index()
    wide = wide.merge(details, on=KEY, validate='one_to_one')
    wide['finite'] = np.isfinite(wide[CORNERS]).all(axis=1)
    wide['supported'] = wide.finite & ~wide.excluded
    return wide


def values(frame):
    a,b,c,d = [frame[x].to_numpy(float) for x in CORNERS]
    return {'net': d-a, 'weather': ((b-a)+(d-c))/2,
            'host': ((c-a)+(d-b))/2, 'weather_alone': b-a,
            'host_alone': c-a, 'interaction': d-b-c+a}


def ratio_summary(frame, flag, models):
    members=[]
    for model in models:
        whole=frame.loc[frame.model.eq(model)]
        selected=whole.loc[whole[flag]]
        row={'model': model, 'requested_pairs':len(whole), 'valid_pairs':len(selected),
             'common_area_time_fraction':math.fsum(selected.area_mean_weight.tolist())/math.fsum(whole.area_mean_weight.tolist())}
        assert len(selected)>0, 'Unsupported member cannot supply an ensemble estimate'
        row.update({term:weighted(v,selected.area_mean_weight) for term,v in values(selected).items()})
        for name in WINDOWS:
            if name in selected:
                row[name]=weighted(selected[name],selected.area_mean_weight)
        members.append(row)
    ensemble={term:math.fsum(row[term] for row in members)/len(models) for term in TERMS}
    for name in WINDOWS:
        if name in frame:
            ensemble[name]=math.fsum(row[name] for row in members)/len(models)
    selected=frame.loc[frame[flag]]
    ensemble.update(model='ensemble',requested_pairs=len(frame),valid_pairs=len(selected),
        common_area_time_fraction=math.fsum(selected.area_mean_weight.tolist())/math.fsum(frame.area_mean_weight.tolist()))
    ensemble['offset_percent']=-100*ensemble['host']/ensemble['weather']
    return members,ensemble


def verify(root, output):
    archive=archived_baseline(root)
    archive_members,archive_ensemble=ratio_summary(archive,'supported',sorted(archive.model.unique()))
    assert len(archive)==5760 and int(archive.finite.sum())==5751 and int(archive.supported.sum())==5731
    assert abs(archive_ensemble['offset_percent']-70.63467958402005)<1e-10
    reference={'total_pairs':len(archive),'finite_pairs':int(archive.finite.sum()),
               'supported_pairs':int(archive.supported.sum()),'model_ratios':archive_members,
               'equal_model_mean':archive_ensemble}
    if output is None:
        return {'archive_reference':reference,'final_verification':'pending'}
    config=json.loads((output/'attribution_registration.json').read_text())
    receipt=json.loads((output/'attribution_receipt.json').read_text())
    assert config['period_estimate'] and config['reference_years']==list(range(1991,2021))
    models=config['models']; settings=[x['name'] for x in config['settings']]
    assert len(models)==3 and len(settings)==8
    p=pd.read_parquet(output/'attribution_pairs.parquet')
    s=pd.read_csv(output/'attribution_summary.csv')
    assert len(p)==8*3*30*64==46080
    assert set(p.model)==set(models) and set(p.setting)==set(settings)
    assert set(p.reference_year)==set(range(1991,2021))
    assert not p.duplicated(['setting',*KEY]).any()
    assert (p.groupby(['setting','model','reference_year']).size()==64).all()
    assert (p.future_year-p.reference_year==80).all()
    assert p.cell_id.nunique()==62 and p.spatial_draw_id.nunique()==64
    retained_draws=pd.read_csv(root/'data/final_climate/spatial_draws.csv').set_index('spatial_draw_id')
    for draw,g in p.groupby('spatial_draw_id'):
        assert g.cell_id.nunique()==1 and g.cell_id.iloc[0]==retained_draws.loc[draw,'cell_id']
        np.testing.assert_allclose(g.area_mean_weight,retained_draws.loc[draw,'area_mean_weight'],rtol=0,atol=1e-15)
    p['independent_common']=np.isfinite(p[CORNERS]).all(axis=1)
    np.testing.assert_array_equal(p.common,p.independent_common)
    across=p.groupby(KEY).independent_common.agg(['all','size'])
    assert across['size'].eq(8).all()
    p=p.merge(across[['all']].rename(columns={'all':'independent_across'}),on=KEY,validate='many_to_one')
    np.testing.assert_array_equal(p.common_across_settings,p.independent_across)
    max_identity=0.
    max_canopy_error=0.
    for corner in CORNERS:
        estimate=p[corner].to_numpy(float); reference_area=p[f'{corner}_reference_had3'].to_numpy(float)
        finite=np.isfinite(estimate)
        np.testing.assert_array_equal(finite,p[f'{corner}_status'].eq('complete').to_numpy())
        assert np.all(estimate[finite]>=-1e-10) and np.all(estimate[finite]<=reference_area[finite]+1e-10)
        assert np.isnan(reference_area[~finite]).all()
    checks=[]
    for setting,part in p.groupby('setting',sort=False):
        parts=values(part)
        common=part.independent_common.to_numpy()
        for term,calculated in parts.items():
            np.testing.assert_allclose(part[term],np.where(common,calculated,np.nan),rtol=0,atol=1e-10,equal_nan=True)
        max_identity=max(max_identity,float(np.max(np.abs(parts['weather'][common]+parts['host'][common]-parts['net'][common]))))
        for label,flag in [('within_setting','independent_common'),('across_settings','independent_across')]:
            members,ensemble=ratio_summary(part,flag,models)
            for row in members+[ensemble]:
                saved=s.loc[s.setting.eq(setting)&s.model.eq(row['model'])&s.support.eq(label)]
                assert len(saved)==1
                saved=saved.iloc[0]
                assert int(saved.valid_pairs)==row['valid_pairs'] and int(saved.requested_pairs)==row['requested_pairs']
                assert abs(saved.common_area_time_fraction-row['common_area_time_fraction'])<1e-12
                for term in TERMS:
                    assert abs(saved[f'{term}_had_days']-row[term])<1e-10
                if row['model']=='ensemble':
                    assert saved.complete_model_ensemble and int(saved.requested_models)==3 and int(saved.available_model_estimates)==3
                    assert abs(saved.offset_percent-row['offset_percent'])<1e-10
                for name in WINDOWS:
                    assert abs(saved[name]-row[name])<1e-10
            checks.append({'setting':setting,'support':label,**ensemble})
    baseline=p.loc[p.setting.eq('baseline')]
    matched=baseline.merge(archive,on=KEY,suffixes=('_new','_archive'),validate='one_to_one')
    assert len(matched)==5760
    np.testing.assert_array_equal(matched.independent_common,matched.supported)
    shared=matched.loc[matched.independent_common]
    maximum_archive_difference={}
    for corner in CORNERS:
        difference=shared[f'{corner}_new']-shared[f'{corner}_archive']
        maximum_archive_difference[corner]=float(np.max(np.abs(difference)))
        assert maximum_archive_difference[corner]<1e-10
    original=next(row for row in checks if row['setting']=='baseline' and row['support']=='within_setting')
    assert abs(original['offset_percent']-70.63467958402005)<1e-10
    overlap=p.loc[p.setting.eq('fixed_common_overlap') & p.independent_common]
    np.testing.assert_array_equal(overlap.reference_window_days,overlap.future_window_days)
    assert (overlap.reference_window_days<=overlap.reference_host_window_days).all()
    assert (overlap.future_window_days<=overlap.future_host_window_days).all()
    for setting in settings:
        part=p.loc[p.setting.eq(setting)&p.independent_common]
        multiplier=.5 if setting=='reference_linear_senescence' else 1.
        for corner,window in [('y00','reference_window_days'),('y10','reference_window_days'),
                              ('y01','future_window_days'),('y11','future_window_days')]:
            difference=part[f'{corner}_reference_had3']-multiplier*part[window]
            max_canopy_error=max(max_canopy_error,float(np.max(np.abs(difference))))
            assert np.all(np.abs(difference)<1e-10)
    mass=receipt['maximum_mass_error']; assert mass<1e-10
    diagonals=pd.read_csv(output/'attribution_baseline_replay.csv')
    assert len(diagonals)==180 and diagonals.matches_frozen_diagonal.all()
    assert diagonals.maximum_difference.max()<1e-10
    for name,expected in receipt['output_hashes'].items():
        assert sha(output/name)==expected,'Output checksum mismatch: '+name
    assert sha(output/'attribution_registration.json')==receipt['registration_sha256']
    return {'archive_reference':reference,'final_verification':'passed','records':len(p),
            'exact_baseline_support_match':True,'maximum_baseline_corner_difference':maximum_archive_difference,
            'maximum_shapley_balance_error':max_identity,'maximum_reference_accounting_error':max_canopy_error,
            'maximum_tissue_mass_error':mass,'baseline_diagonal_checks':len(diagonals),
            'independent_equal_model_summaries':checks}


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path.cwd())
    parser.add_argument('--output',type=Path)
    parser.add_argument('--save',type=Path,default=Path('/tmp/food_security_attribution_independent_verification_20261013.json'))
    args=parser.parse_args()
    result=verify(args.root.resolve(),args.output.resolve() if args.output else None)
    args.save.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))
