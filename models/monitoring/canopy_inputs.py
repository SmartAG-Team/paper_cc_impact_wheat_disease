"""Quality-qualified regional canopy summaries from complete past composites."""
import numpy as np
import pandas as pd

BANDS = ['lai_mean','fpar_mean','lai_stdDev','fpar_stdDev',
    'lai_uncertainty_mean','fpar_uncertainty_mean']


def canopy_quality(qc, extra, lai, fpar, lai_sd, fpar_sd):
    qc, extra = np.asarray(qc, int), np.asarray(extra, int)
    good = ((qc & 1) == 0) & ((qc & 4) == 0) & (((qc >> 3) & 3) == 0) & (((qc >> 5) & 7) <= 1)
    good &= (extra & 3) == 0
    good &= (extra & 124) == 0  # Snow, aerosol, cirrus, internal cloud, shadow.
    for value in [lai,fpar,lai_sd,fpar_sd]:
        value = np.asarray(value, float)
        good &= np.isfinite(value) & (value >= 0) & (value <= 100)
    return good


def canopy_prefix(series, query_date, lag_days=7):
    query_date = pd.Timestamp(query_date)
    frame = series.copy()
    frame['available_date'] = pd.to_datetime(frame.period_end)+pd.Timedelta(days=lag_days)
    frame = frame.loc[frame.available_date.le(query_date) &
        frame.available_date.gt(query_date-pd.Timedelta(days=84))].sort_values('available_date')
    valid = (frame.valid_mean.ge(.2) & frame.lai_count.ge(20) &
        np.isfinite(frame[BANDS].to_numpy(float)).all(axis=1))
    good = frame.loc[valid]
    names = ['canopy_latest_lai','canopy_latest_fpar','canopy_latest_lai_spatial_sd',
        'canopy_latest_fpar_spatial_sd','canopy_latest_lai_retrieval_sd',
        'canopy_latest_fpar_retrieval_sd','canopy_latest_valid_crop_fraction',
        'canopy_static_crop_fraction','canopy_age_30d','canopy_available']
    latest = good.iloc[-1] if len(good) else None
    crop = float(frame.crop_fraction.iloc[-1]) if len(frame) else 0.
    values = ([float(latest[b]) for b in BANDS]+[float(latest.valid_mean),crop,
        float((query_date-latest.available_date).days/30),1.] if latest is not None else [0.]*7+[crop,2.8,0.])
    for window in [14,42,84]:
        part = good.loc[good.available_date.gt(query_date-pd.Timedelta(days=window))]
        for band in ['lai','fpar']:
            column = band+'_mean'; v = part[column].to_numpy(float)
            t = (part.available_date-query_date).dt.total_seconds().to_numpy()/86400
            slope = float(np.sum((t-t.mean())*(v-v.mean()))/np.sum((t-t.mean())**2)*30) if len(v)>1 and np.ptp(t)>0 else 0.
            values += [float(v.mean()),float(v.min()),float(v.max()),float(v.std()),slope] if len(v) else [0.]*5
            names += [f'canopy_{band}_{stat}_{window}d' for stat in ['mean','min','max','sd','slope_per30d']]
        values += [len(part)/21,float(part.valid_mean.mean()) if len(part) else 0.]
        names += [f'canopy_qualified_count_{window}d_per21',f'canopy_valid_crop_fraction_{window}d']
    for band in ['lai','fpar']:
        peak = float(good[band+'_mean'].max()) if len(good) else 0.
        drop = float(latest[band+'_mean']-peak) if latest is not None else 0.
        values += [peak,drop]; names += [f'canopy_{band}_past84d_peak',f'canopy_{band}_change_from_past84d_peak']
    values += [len(good)/len(frame) if len(frame) else 0.,float(latest.lai_count/1000) if latest is not None else 0.]
    names += ['canopy_qualified_composite_fraction_84d','canopy_latest_valid_pixels_per1000']
    assert len(names) == len(values) == 52
    return np.asarray(values,float), names, latest is not None
