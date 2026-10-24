"""Original monthly alignment formula with retained exact reference statistics."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.special import logit
from model.seasonal_septoria.climate_alignment import RH_EPSILON
from .run import ROOT, DATA, read_manifest, check_file, forcing_path

def alignment_from_monthly_means(reference, historical):
    """Coefficients from independently accumulated full-baseline monthly means."""
    keys = ('tmean_c','rh_logit','precipitation_mm')
    a,b = [{key:np.asarray(source[key],float) for key in keys} for source in (reference,historical)]
    shape = a['tmean_c'].shape
    if (len(shape)!=2 or shape[1]!=12 or not all(x.shape==shape and np.isfinite(x).all()
            for source in (a,b) for x in source.values())
            or np.any(a['precipitation_mm']<0) or np.any(b['precipitation_mm']<0)):
        raise ValueError('Complete finite 12-month climatologies required.')
    if np.any((b['precipitation_mm']==0)&(a['precipitation_mm']>0)):
        raise ValueError('Positive reference rain cannot be aligned from a zero model monthly mean.')
    ratio = np.divide(a['precipitation_mm'],b['precipitation_mm'],
                      out=np.ones(shape),where=b['precipitation_mm']!=0)
    return dict(temperature_offset=a['tmean_c']-b['tmean_c'],
                humidity_logit_offset=a['rh_logit']-b['rh_logit'],
                precipitation_ratio=ratio)


def reference_monthly_means(root=ROOT):
    root = Path(root)
    manifest = read_manifest(root)
    cells = pd.read_parquet(root / DATA / "cells.parquet")
    sums = {key: np.zeros((len(cells), 12)) for key in
            ("tmean_c", "rh_logit", "precipitation_mm")}
    count = np.zeros(12, int)
    for record in manifest["reference_monthly_statistics"]:
        path = root / record["path"]
        check_file(path, record, verify_hash=True)
        with np.load(path) as source:
            count += source["count"]
            for key in sums:
                sums[key] += source[key]
    dates = pd.date_range("1991-01-01", "2020-12-31")
    expected = np.array([(dates.month == month).sum() for month in range(1, 13)])
    if not np.array_equal(count, expected):
        raise ValueError("Reference monthly statistics lack complete 1991–2020 support")
    return {key: value / count[None, :] for key, value in sums.items()}


def model_monthly_means(model, scenario, root=ROOT):
    root = Path(root)
    manifest = read_manifest(root)
    records = {row["path"]: row for row in manifest["forcing"]}
    cells = pd.read_parquet(root / DATA / "cells.parquet")
    sums = {key: np.zeros((len(cells), 12)) for key in
            ("tmean_c", "rh_logit", "precipitation_mm")}
    count = np.zeros(12, int)
    for year in range(1991, 2021):
        relative = forcing_path(model, scenario, year)
        check_file(root / relative, records[relative], verify_hash=True)
        frame = pq.read_table(root / relative,
                              columns=["date", "cell_id", "tmean_c", "rh_mean_pct", "precipitation_mm"]).to_pandas()
        dates = pd.DatetimeIndex(pd.to_datetime(frame.date.unique()))
        if len(frame) != len(cells) * len(dates) or not np.all(
                frame.cell_id.to_numpy().reshape(len(dates), len(cells)) == cells.cell_id.to_numpy()[None, :]):
            raise ValueError("Historical forcing cell/day support differs")
        arrays = {key: frame[key].to_numpy(float).reshape(len(dates), len(cells)).T
                  for key in ("tmean_c", "rh_mean_pct", "precipitation_mm")}
        arrays["rh_logit"] = logit(np.clip(arrays["rh_mean_pct"] / 100, RH_EPSILON, 1 - RH_EPSILON))
        for month in range(1, 13):
            included = dates.month == month
            count[month - 1] += included.sum()
            for key in sums:
                sums[key][:, month - 1] += arrays[key][:, included].sum(axis=1)
    expected_dates = pd.date_range("1991-01-01", "2020-12-31")
    if not np.array_equal(count, [(expected_dates.month == month).sum() for month in range(1, 13)]):
        raise ValueError("Historical forcing monthly support differs")
    return {key: value / count[None, :] for key, value in sums.items()}


def rebuild_alignment(output, root=ROOT):
    """Reproduce nine original coefficient tables; no future observations enter."""
    root, output = Path(root), Path(output).resolve()
    manifest = read_manifest(root)
    cells = pd.read_parquet(root / DATA / "cells.parquet")
    reference = reference_monthly_means(root)
    reports = []
    for model in manifest["models"]:
        for scenario in manifest["scenarios"]:
            coefficients = alignment_from_monthly_means(reference, model_monthly_means(model, scenario, root))
            table = pd.DataFrame({"cell_id": np.repeat(cells.cell_id, 12),
                                  "month": np.tile(np.arange(1, 13), len(cells))})
            for key, value in coefficients.items():
                table[key] = value.ravel()
            expected = pd.read_parquet(root / DATA / "alignment" / model / scenario / "coefficients.parquet")
            pd.testing.assert_frame_equal(table, expected, check_exact=False, rtol=0, atol=1e-10)
            path = output / model / scenario / "coefficients.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                raise FileExistsError(path)
            table.to_parquet(path, index=False, compression="zstd")
            reports.append(dict(model=model, scenario=scenario, cells=len(cells),
                                matches_original_coefficients=True, output=str(path)))
    return dict(status="verified", alignments=reports, reference_period="1991–2020")
