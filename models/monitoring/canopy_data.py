"""Bind qualified retrospective canopy prefixes to the unchanged disease cohort."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from .canopy_inputs import canopy_prefix, BANDS
from .innovation_common import ROOT, PARENT, DATA
from .inputs import sha

RAW = ROOT/'data/disease_improvement/canopy_regional_v2_20261022'


def build(output=None):
    out = Path(output or ROOT/'data/disease_improvement/canopy_prefix_20261022').resolve()
    if out.exists():
        raise FileExistsError('Fresh canopy-prefix destination required')
    receipt = json.loads((RAW/'verification.json').read_text())
    assert receipt['passed'] and receipt['source_unchanged']
    for path,digest in receipt['artifacts_sha256'].items():
        assert sha(RAW/path)==digest,path
    sources = [RAW/'registration.json',RAW/'verification.json',RAW/'regional_canopy.csv',
        RAW/'raw/crop_2009.json',PARENT/'training_observation_ledger.csv',DATA/'fields.csv']
    codes = [Path(__file__),Path(__file__).with_name('canopy_inputs.py'),Path(__file__).with_name('gather_regional_canopy.py')]
    out.mkdir(parents=True)
    registration = dict(registered_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
        code_sha256={str(p.relative_to(ROOT)):sha(p) for p in codes},
        transformation='52prefixsummaries at1e4mbuffer, cropmaskGlobCover2009classes11/14, originalsitecoordinates; no fieldboundary/wheatidentity inference',
        qualification='Mainalgorithm/clear/land/no snow,aerosol,cirrus,internalcloud,shadow; allrawLAI/FPAR/SD0..100; region validcropfraction>=.2 and>=20validpixels; all6statisticsfinite',
        timing='Actualcomposite system:time_end +fixed7dayassumedprocessinglag <=querydate; allstatistics limitedtopast84days; no future peak/smoothing',
        missing='Zeroencodingwithavailability/age/coverageflags, not observedzero vegetation; alltargetsretained and model mustfallbackifnoqualifiedpriorcanopy',
        interpretation='Regional retrospective crop-conditionproxy, no verifiedactualfieldstage/cultivar/canopydisease or real-timeproductlatency',
        disease_labels_used_for_transformation=False)
    (out/'registration.json').write_text(json.dumps(registration,indent=2)+'\n')
    for code in codes:
        dest=out/'code_snapshot'/code.relative_to(ROOT);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(code,dest)
    frame = pd.read_csv(RAW/'regional_canopy.csv',float_precision='round_trip')
    frame.period_start = pd.to_datetime(frame.period_start,unit='ms')
    frame.period_end = pd.to_datetime(frame.period_end,unit='ms')
    assert not frame.duplicated(['site_id','image_index']).any()
    duration = (frame.period_end-frame.period_start).dt.days
    assert duration.between(1,4).all()
    assert frame.valid_mean.dropna().between(0,1).all() and frame.crop_fraction.between(0,1).all()
    for band,bound in [('lai_mean',10),('fpar_mean',1),('lai_uncertainty_mean',10),('fpar_uncertainty_mean',1)]:
        assert frame[band].dropna().between(0,bound).all()
    # These four channels use exactly the same pixel-level qualification mask.
    for band in ['fpar','lai_uncertainty','fpar_uncertainty']:
        np.testing.assert_array_equal(frame.lai_count,frame[band+'_count'])
    obs = pd.read_csv(PARENT/'training_observation_ledger.csv',
        usecols=['global_target_index','field_id','site_id','date'])
    assert frame.site_id.nunique() == obs.site_id.nunique() == 128
    by_site = {site:g for site,g in frame.groupby('site_id')}
    features, available, provenance = [], [], []
    for query in obs.itertuples():
        value,names,use = canopy_prefix(by_site[query.site_id],query.date)
        features.append(value);available.append(use)
        age = value[names.index('canopy_age_30d')]*30
        latest_end = pd.Timestamp(query.date)-pd.Timedelta(days=7+round(age)) if use else pd.NaT
        assert not use or latest_end+pd.Timedelta(days=7) <= pd.Timestamp(query.date)
        provenance.append(dict(global_target_index=int(query.global_target_index),field_id=query.field_id,
            site_id=query.site_id,query_date=query.date,qualified_canopy_available=bool(use),
            latest_used_composite_end=str(latest_end.date()) if use else None,
            availability_lag_assumption_days=7,actual_field_canopy_verified=False))
    values = np.asarray(features,float); assert values.shape==(2304,52) and np.isfinite(values).all()
    np.savez_compressed(out/'predictors.npz',canopy=values,available=np.asarray(available,bool),
        target_ids=obs.global_target_index.to_numpy(int))
    pd.DataFrame(provenance).to_csv(out/'query_provenance.csv',index=False)
    frame.groupby('site_id').agg(composites=('image_index','size'),crop_fraction=('crop_fraction','first'),
        valid_composites=('lai_count',lambda x:int(x.ge(20).sum()))).to_csv(out/'site_coverage.csv')
    (out/'feature_schema.json').write_text(json.dumps(dict(feature_names=names,features=52,
        units='LAI:m2/m2;FPAR:fraction; SDs correspond tothoseunits; temporal scaling explicitinname',
        full_available_pixel_mask_shared_across_bands=True,missing_encoded_zero_with_flags=True),indent=2)+'\n')
    verification = dict(passed=True,cohort_rows=len(obs),fields=obs.field_id.nunique(),sites=128,
        qualified_rows=int(np.sum(available)),unavailable_rows=int(np.sum(~np.asarray(available))),
        regional_composites=len(frame),canopy_observation_is_regional=True,
        source_unchanged=all(sha(ROOT/p)==h for p,h in registration['source_sha256'].items()),
        artifacts_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps({k:v for k,v in verification.items() if k!='artifacts_sha256'},indent=2),flush=True)
    return out


if __name__=='__main__':
    build()
