"""Descriptive BASF weather exposures and publication PNG/PDF figures.

Read-only inputs: original BASF data and completed ERA5 archive.
All output writes are restricted to analysis/era5/.
Dependencies: pandas, numpy, pyarrow, matplotlib, pyshp, pyproj, requests.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from pyproj import Transformer
import requests
import shapefile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "era5"
DATA = ROOT / "data" / "era5"
WEATHER_CREDIT = "Weather data: Open-Meteo.com / ERA5 · CC BY 4.0 · https://open-meteo.com/"
YEARS = (2017, 2018, 2019)
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#515960", "axes.linewidth": 0.7,
    "text.color": "#20272d", "axes.labelcolor": "#20272d",
    "xtick.color": "#424b52", "ytick.color": "#424b52",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.facecolor": "white", "figure.facecolor": "white",
})


def write_table(frame: pd.DataFrame, name: str) -> None:
    frame.to_csv(OUT / f"{name}.csv", index=False, float_format="%.8f")
    frame.to_parquet(OUT / f"{name}.parquet", index=False, compression="zstd")


def hash_location(latitude: float, longitude: float) -> str:
    return "era5loc_" + hashlib.sha256(f"{latitude:.6f},{longitude:.6f}".encode()).hexdigest()[:12]


def boundary_shapes():
    archive = OUT / "ne_110m_admin_0_countries.zip"
    if not archive.exists():
        url = "https://naturalearth.s3.amazonaws.com/110m_cultural/ne_110m_admin_0_countries.zip"
        response = requests.get(url, timeout=45)
        response.raise_for_status()
        archive.write_bytes(response.content)
        (OUT / "natural_earth_provenance.json").write_text(json.dumps({
            "url": url, "documentation_url": "https://www.naturalearthdata.com/downloads/110m-cultural-vectors/110m-admin-0-countries/",
            "license": "Public domain", "license_url": "https://www.naturalearthdata.com/about/terms-of-use/",
            "retrieved_utc": datetime.now(timezone.utc).isoformat(),
            "sha256": hashlib.sha256(response.content).hexdigest(), "bytes": len(response.content),
        }, indent=2)+"\n")
    metadata = json.loads((OUT / "natural_earth_provenance.json").read_text())
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == metadata["sha256"]
    with zipfile.ZipFile(archive) as zipped:
        stem = "ne_110m_admin_0_countries"
        reader = shapefile.Reader(shp=BytesIO(zipped.read(stem+".shp")),
                                  shx=BytesIO(zipped.read(stem+".shx")),
                                  dbf=BytesIO(zipped.read(stem+".dbf")))
        return list(reader.iterShapes())


def save_figure(fig, filename: str) -> None:
    fig.savefig(OUT / (filename+".png"), dpi=400)
    fig.savefig(OUT / (filename+".pdf"), metadata={"Creator":"NWAFU research; reproducible Matplotlib output", "Title":filename})
    plt.close(fig)


def rainfall_map(seasonal: pd.DataFrame) -> None:
    projection = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    all_x, all_y = projection.transform(seasonal.requested_longitude.to_numpy(), seasonal.requested_latitude.to_numpy())
    xlim = (float(np.min(all_x)-280_000), float(np.max(all_x)+280_000))
    ylim = (float(np.min(all_y)-250_000), float(np.max(all_y)+250_000))
    shapes = boundary_shapes()
    paths = []
    for shape in shapes:
        if shape.bbox[2] < -16 or shape.bbox[0] > 35 or shape.bbox[3] < 38 or shape.bbox[1] > 62:
            continue
        vertices, codes = [], []
        offsets = list(shape.parts) + [len(shape.points)]
        for start, end in zip(offsets[:-1], offsets[1:]):
            points = np.asarray(shape.points[start:end])
            if len(points) < 3:
                continue
            x, y = projection.transform(points[:,0], points[:,1])
            ring = np.column_stack([x,y])
            vertices.extend(ring.tolist())
            codes.extend([MplPath.MOVETO]+[MplPath.LINETO]*(len(ring)-2)+[MplPath.CLOSEPOLY])
        if vertices:
            paths.append(MplPath(np.asarray(vertices), np.asarray(codes)))
    fig, axes = plt.subplots(1,3,figsize=(10.8,4.7))
    fig.subplots_adjust(left=0.025,right=0.875,bottom=0.21,top=0.77,wspace=0.035)
    norm = Normalize(vmin=400, vmax=1050)
    cmap = plt.colormaps["Blues"]
    for panel, (axis, year) in enumerate(zip(axes, YEARS)):
        for path in paths:
            axis.add_patch(PathPatch(path,facecolor="#f2f2ef",edgecolor="#a0a6a9",linewidth=0.45,zorder=1))
        rows = seasonal.loc[seasonal.season_year == year]
        x,y = projection.transform(rows.requested_longitude.to_numpy(),rows.requested_latitude.to_numpy())
        axis.scatter(x,y,c=rows.rain_mm,cmap=cmap,norm=norm,s=45,edgecolors="#253a4a",linewidths=0.45,zorder=3)
        axis.set_xlim(xlim);axis.set_ylim(ylim);axis.set_aspect("equal");axis.set_axis_off()
        axis.set_title(f"({chr(97+panel)})  {year}\n{len(rows)} coordinate-years · {int(rows.basf_trial_count.sum())} trials",pad=8,fontsize=10.5)
        if panel == 0:
            sx = xlim[0]+140_000; sy = ylim[0]+140_000
            axis.plot([sx,sx+500_000],[sy,sy],color="#30383e",lw=1.6,zorder=5)
            axis.plot([sx,sx],[sy-20_000,sy+20_000],color="#30383e",lw=1,zorder=5)
            axis.plot([sx+500_000,sx+500_000],[sy-20_000,sy+20_000],color="#30383e",lw=1,zorder=5)
            axis.text(sx+250_000,sy+60_000,"500 km",ha="center",va="bottom",fontsize=8)
    color_axis = fig.add_axes([0.900,0.27,0.017,0.43])
    colorbar = fig.colorbar(ScalarMappable(norm=norm,cmap=cmap),cax=color_axis,ticks=[400,600,800,1000])
    colorbar.set_label("Liquid precipitation (mm)",labelpad=8,fontsize=10)
    colorbar.outline.set_linewidth(0.5)
    fig.text(0.03,0.955,"ERA5 rainfall in the fixed coverage intervals",ha="left",va="top",fontsize=14)
    fig.text(0.03,0.885,"1 September in the preceding year–30 September in the observation year (395 days)",ha="left",fontsize=10)
    fig.text(0.03,0.120,"One marker per source coordinate-year; source coordinates rounded to 0.1°. Common colour scale.",fontsize=8.8)
    fig.text(0.03,0.076,"Lambert azimuthal equal-area projection (EPSG:3035) · Boundaries: Natural Earth (public domain)",fontsize=8.5)
    fig.text(0.03,0.033,WEATHER_CREDIT,fontsize=8.5,url="https://open-meteo.com/")
    save_figure(fig,"basf_era5_fixed_interval_rainfall_map")


def exposure_distributions(windows: pd.DataFrame) -> None:
    specifications = [
        ("tmean_30d_c","Mean air temperature (°C)",None),
        ("rain_30d_mm","Liquid precipitation (mm)",0),
        ("rh_hours_ge_90pct_30d","Hours with 2 m RH ≥90%",0),
        ("rain_hours_gt_0_1mm_30d","Hours with rain >0.1 mm",0),
    ]
    fig,axes = plt.subplots(2,2,figsize=(9,7.4))
    fig.subplots_adjust(left=0.105,right=0.97,bottom=0.18,top=0.82,hspace=0.53,wspace=0.31)
    rng = np.random.default_rng(384)
    for panel,(axis,(field,label,lower)) in enumerate(zip(axes.flat,specifications)):
        groups = [windows.loc[windows.season_year.eq(year),field].to_numpy() for year in YEARS]
        box = axis.boxplot(groups,positions=[1,2,3],widths=0.42,patch_artist=True,showfliers=False,
            medianprops=dict(color="#1e3850",lw=1.4),whiskerprops=dict(color="#576570",lw=0.8),
            capprops=dict(color="#576570",lw=0.8),boxprops=dict(edgecolor="#426382",lw=0.9))
        for patch in box["boxes"]:
            patch.set_facecolor("#dbe8f2")
        for position,values in enumerate(groups,1):
            jitter = rng.uniform(-0.15,0.15,len(values))
            axis.scatter(position+jitter,values,s=11,color="#497a9d",alpha=0.45,linewidths=0,zorder=2)
        axis.set_xticks([1,2,3],[f"{year}\nn={len(values)}" for year,values in zip(YEARS,groups)])
        axis.set_ylabel(label)
        axis.set_title(f"({chr(97+panel)})  {label}",loc="left",pad=10,fontsize=10.5)
        axis.yaxis.grid(True,color="#e6eaed",lw=0.6,zorder=0)
        axis.set_axisbelow(True)
        if lower is not None:
            axis.set_ylim(bottom=lower)
        if field == "rh_hours_ge_90pct_30d":
            axis.set_ylim(0,720)
    fig.text(0.105,0.955,"Weather during the 30 completed days before Septoria assessment",fontsize=13,ha="left",va="top")
    fig.text(0.105,0.890,"One exposure per ERA5 grid-cell/date; assessment day excluded",fontsize=10)
    fig.text(0.105,0.120,"Boxes: median and interquartile range; whiskers: 1.5×IQR. Points: distinct grid-cell/date windows.",fontsize=8.6)
    fig.text(0.105,0.079,"RH exposure is atmospheric humidity, not measured leaf wetness. Windows overlap; observations are correlated.",fontsize=8.3)
    fig.text(0.105,0.036,WEATHER_CREDIT,fontsize=8.3,url="https://open-meteo.com/")
    save_figure(fig,"basf_era5_assessment_weather_30d")


def summary_rows(table: pd.DataFrame, fields: list[str], grain: str) -> pd.DataFrame:
    rows=[]
    for year,group in table.groupby("season_year",sort=True):
        for field in fields:
            values = group[field]
            rows.append({"season_year":int(year),"sampling_grain":grain,"variable":field,"n":len(group),
                "mean":float(values.mean()),"sd":float(values.std(ddof=1)),"min":float(values.min()),
                "q25":float(values.quantile(0.25)),"median":float(values.median()),
                "q75":float(values.quantile(0.75)),"max":float(values.max())})
    return pd.DataFrame(rows)


def main() -> None:
    raw = pd.read_csv(ROOT / "data" / "basf-wheat-diseases.txt",sep="\t")
    raw["source_row"] = np.arange(2,len(raw)+2)
    disease = raw.loc[raw.Treatment.eq("Untreated") & raw.Organism.eq("SEPTTR")].copy()
    disease["assessment_date"] = pd.to_datetime(disease.Date).dt.strftime("%Y-%m-%d")
    disease["season_year"] = pd.to_datetime(disease.Date).dt.year
    disease["location_id"] = [hash_location(a,b) for a,b in zip(disease.Latitude,disease.Longitude)]
    disease["observation_id"] = "basf-wheat-diseases|data/basf-wheat-diseases.txt|trials|"+disease.source_row.astype(str)+"|Value"
    disease["disease_response_eligible"] = ~disease.Clarifier.eq("CROP INJURY") & disease.Value.between(0,100) & disease.Parameter.eq("INFECT")
    registry = pd.read_parquet(DATA / "location_registry.parquet")
    daily = pd.read_parquet(DATA / "daily_weather.parquet")
    seasonal = pd.read_parquet(DATA / "seasonal_weather.parquet").query("basf_trial_count > 0").copy()
    rows=[]
    for (loc_id,date), observations in disease.groupby(["location_id","assessment_date"],sort=True):
        end = pd.Timestamp(date)-pd.Timedelta(days=1)
        start = pd.Timestamp(date)-pd.Timedelta(days=30)
        subset = daily.loc[daily.location_id.eq(loc_id) & daily.date.between(start.date().isoformat(),end.date().isoformat())]
        expected = pd.date_range(start,end,freq="D").strftime("%Y-%m-%d").tolist()
        assert sorted(subset.date.tolist()) == expected
        assert subset.hour_count.eq(24).all() and len(subset)==30
        rows.append({"weather_window_id":f"{loc_id}_{date}_lag30d","location_id":loc_id,"assessment_date":date,
            "season_year":int(observations.season_year.iloc[0]),"window_start_date":start.date().isoformat(),
            "window_end_date":end.date().isoformat(),"window_days":len(subset),"available_hours":int(subset.hour_count.sum()),
            "source_observation_rows":len(observations),"source_trial_count":observations.TrialId.nunique(),
            "eligible_disease_rows":int(observations.disease_response_eligible.sum()),
            "tmean_30d_c":float(subset.tmean_c.mean()),"tmin_30d_c":float(subset.tmin_c.min()),"tmax_30d_c":float(subset.tmax_c.max()),
            "rh_mean_30d_pct":float(subset.rh_mean_pct.mean()),"rh_hours_ge_90pct_30d":int(subset.rh_hours_ge_90pct.sum()),
            "dewpoint_mean_30d_c":float(subset.dewpoint_mean_c.mean()),"rain_30d_mm":float(subset.rain_mm.sum()),
            "precipitation_30d_mm":float(subset.precipitation_mm.sum()),"rain_hours_gt_0_1mm_30d":int(subset.rain_hours_gt_0_1mm.sum()),
            "rain_days_gt_1mm_30d":int(subset.rain_mm.gt(1).sum()),"wind_mean_30d_m_s":float(subset.wind_mean_m_s.mean()),
            "shortwave_30d_mj_m2":float(subset.shortwave_mj_m2.sum()),
        })
    windows = pd.DataFrame(rows).merge(registry[["location_id","grid_cell_id","grid_latitude","grid_longitude","requested_to_grid_distance_km"]],on="location_id",validate="many_to_one")
    assert not windows.duplicated(["location_id","assessment_date"]).any()
    assert (pd.to_datetime(windows.window_end_date) < pd.to_datetime(windows.assessment_date)).all()
    assert (pd.to_datetime(windows.assessment_date)-pd.to_datetime(windows.window_start_date)).dt.days.eq(30).all()
    exposure_fields = [c for c in windows if c.endswith("_30d") or c.endswith("_30d_c") or c.endswith("_30d_mm") or c.endswith("_30d_pct") or c.endswith("_30d_m_s") or c.endswith("_30d_mj_m2")]
    assert windows[exposure_fields].notna().all().all()
    assert windows.groupby(["grid_cell_id","assessment_date"])[exposure_fields].nunique().le(1).all().all()
    merged = disease.merge(windows.drop(columns="season_year"),on=["location_id","assessment_date"],validate="many_to_one",how="left")
    assert len(merged)==len(disease)==609 and merged.weather_window_id.notna().all()
    assert merged.observation_id.nunique()==609
    trial_assessments = merged.drop_duplicates(["TrialId","assessment_date"])[["TrialId","location_id","assessment_date","season_year","weather_window_id"]+exposure_fields]
    write_table(windows,"basf_assessment_weather_30d_unique")
    write_table(merged,"basf_untreated_septoria_with_weather_30d")
    write_table(trial_assessments,"basf_trial_assessment_weather_30d")
    eligible = windows.loc[windows.eligible_disease_rows.gt(0)].copy()
    plotted = eligible.drop_duplicates(["grid_cell_id","assessment_date"]).copy()
    write_table(plotted,"basf_weather_30d_figure_source")
    interval_fields=["tmean_c","rh_mean_pct","rh_hours_ge_90pct","rain_mm","precipitation_mm","rain_hours_gt_0_1mm","wind_mean_m_s","shortwave_mj_m2"]
    seasonal_stats=summary_rows(seasonal,interval_fields,"source_coordinate_year")
    write_table(seasonal_stats,"basf_fixed_interval_weather_stats")
    exposure_stats=summary_rows(plotted,exposure_fields,"distinct_ERA5_grid_cell_assessment_date")
    write_table(exposure_stats,"basf_assessment_weather_30d_stats")
    counts=[]
    for year in YEARS:
        d=disease.loc[disease.season_year.eq(year)]
        w=windows.loc[windows.season_year.eq(year)]
        e=eligible.loc[eligible.season_year.eq(year)]
        p=plotted.loc[plotted.season_year.eq(year)]
        s=seasonal.loc[seasonal.season_year.eq(year)]
        counts.append({"season_year":year,"source_rows":len(d),"eligible_disease_rows":int(d.disease_response_eligible.sum()),
            "flagged_crop_injury_rows":int(d.Clarifier.eq("CROP INJURY").sum()),"trials":d.TrialId.nunique(),
            "coordinate_years":len(s),"trial_dates":len(d[["TrialId","assessment_date"]].drop_duplicates()),
            "coordinate_dates":len(w),"eligible_coordinate_dates":len(e),"distinct_eligible_grid_dates":len(p)})
    counts=pd.DataFrame(counts)
    write_table(counts,"basf_weather_analysis_counts")
    rainfall_map(seasonal)
    exposure_distributions(plotted)
    # Independent scalar arithmetic for first and last forcing window at every source location.
    checked=0
    for loc, group in windows.groupby("location_id"):
        for sample in group.sort_values("assessment_date").iloc[[0,-1]].drop_duplicates().to_dict("records"):
            subset=daily.loc[daily.location_id.eq(loc)&daily.date.between(sample["window_start_date"],sample["window_end_date"])]
            assert math.isclose(math.fsum(subset.tmean_c.tolist())/30,sample["tmean_30d_c"],abs_tol=1e-10)
            assert math.isclose(math.fsum(subset.rain_mm.tolist()),sample["rain_30d_mm"],abs_tol=1e-10)
            assert sum(subset.rh_hours_ge_90pct.tolist())==sample["rh_hours_ge_90pct_30d"]
            assert subset.date.max()<sample["assessment_date"]
            checked+=1
    # Cross-check every forcing window against the hourly archive independently of daily aggregations.
    hourly=pd.read_parquet(DATA/"hourly_weather.parquet")
    hourly_checked=0
    for loc,group in windows.groupby("location_id"):
        source=hourly.loc[hourly.location_id.eq(loc)]
        for sample in group.to_dict("records"):
            start=pd.Timestamp(sample["window_start_date"],tz="UTC")
            finish=pd.Timestamp(sample["assessment_date"],tz="UTC")
            h=source.loc[source.time_utc.ge(start)&source.time_utc.lt(finish)]
            assert len(h)==720
            direct={"tmean_30d_c":h.temperature_2m_c.mean(),"tmin_30d_c":h.temperature_2m_c.min(),"tmax_30d_c":h.temperature_2m_c.max(),
                "rh_mean_30d_pct":h.relative_humidity_2m_pct.mean(),"rh_hours_ge_90pct_30d":h.relative_humidity_2m_pct.ge(90).sum(),
                "rain_30d_mm":h.rain_mm.sum(),"precipitation_30d_mm":h.precipitation_mm.sum(),
                "rain_hours_gt_0_1mm_30d":h.rain_mm.gt(0.1).sum(),"wind_mean_30d_m_s":h.wind_speed_10m_m_s.mean(),
                "dewpoint_mean_30d_c":h.dew_point_2m_c.mean(),"shortwave_30d_mj_m2":h.shortwave_radiation_w_m2.sum()*0.0036}
            for field,number in direct.items():
                assert math.isclose(number,sample[field],rel_tol=1e-12,abs_tol=1e-8),(sample["weather_window_id"],field,number,sample[field])
            hourly_checked+=1
    validation={"checked_utc":datetime.now(timezone.utc).isoformat(),"result":"pass", "source_rows_preserved":len(merged),
        "untreated_Septoria_source_rows":len(disease),"eligible_disease_rows":int(disease.disease_response_eligible.sum()),
        "crop_injury_rows_preserved_flagged":int(disease.Clarifier.eq("CROP INJURY").sum()),"trials":disease.TrialId.nunique(),
        "coordinate_years":len(seasonal),"unique_coordinate_dates":len(windows),"trial_dates":len(trial_assessments),
        "eligible_coordinate_dates":len(eligible),"distinct_eligible_ERA5_grid_dates":len(plotted),
        "window_days_all30":bool(windows.window_days.eq(30).all()),"window_hours_all720":bool(windows.available_hours.eq(720).all()),
        "assessment_day_excluded_all_windows":True,"missing_exposure_values":int(windows[exposure_fields].isna().sum().sum()),
        "coordinate_date_duplicate_keys":int(windows.duplicated(["location_id","assessment_date"]).sum()),
        "same_grid_date_exposures_agree":True,"independent_scalar_samples_checked":checked,
        "forcing_windows_checked_against_hourly_archive":hourly_checked,
        "source_observation_id_compatible_with_harmonized_table":True,"counts_by_year":counts.to_dict("records")}
    (OUT/"weather_exposure_validation.json").write_text(json.dumps(validation,indent=2)+"\n")
    write_interpretation(counts,seasonal_stats,exposure_stats)
    artifacts=[]
    for name in ["basf_era5_fixed_interval_rainfall_map.png","basf_era5_fixed_interval_rainfall_map.pdf",
                 "basf_era5_assessment_weather_30d.png","basf_era5_assessment_weather_30d.pdf"]:
        path=OUT/name
        artifacts.append({"path":str(path.relative_to(ROOT)),"bytes":path.stat().st_size,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (OUT/"weather_figures_manifest.json").write_text(json.dumps({"created_utc":datetime.now(timezone.utc).isoformat(),
        "weather_source":"ERA5 via Open-Meteo", "source_archive_manifest":"data/era5/manifest.json",
        "map_source":"Natural Earth 1:110m Admin 0 Countries; public domain", "projection":"EPSG:3035",
        "figure_source_tables":["data/era5/seasonal_weather.parquet","analysis/era5/basf_weather_30d_figure_source.parquet"],
        "figures":artifacts},indent=2)+"\n")
    print(json.dumps(validation,indent=2))


def write_interpretation(counts: pd.DataFrame, interval: pd.DataFrame, exposures: pd.DataFrame) -> None:
    def value(table,year,field,stat):
        return float(table.loc[table.season_year.eq(year)&table.variable.eq(field),stat].iloc[0])
    interval_text="; ".join(f"{year}: {value(interval,year,'rain_mm','median'):.1f} mm ({value(interval,year,'rain_mm','min'):.1f}–{value(interval,year,'rain_mm','max'):.1f} mm)" for year in YEARS)
    temp_text="; ".join(f"{year}: {value(exposures,year,'tmean_30d_c','median'):.1f}°C" for year in YEARS)
    rain_text="; ".join(f"{year}: {value(exposures,year,'rain_30d_mm','median'):.1f} mm" for year in YEARS)
    text=(
        "The BASF network comprises 76 trials at 56 source coordinates and 65 coordinate-years. The observation years 2017, 2018 and 2019 contain 21, 10 and 45 trials, representing 19, 10 and 36 coordinate-years, respectively. ERA5 describes a 0.25° grid-scale weather exposure; the source coordinates are rounded to 0.1°, and multiple trials can share identical weather. The fixed archive intervals extend from 1 September in the preceding year to 30 September in the observation year, spanning 395 days for each BASF coordinate-year.\n\n"
        f"Median liquid precipitation across source coordinate-years was {interval_text}. These totals describe the fixed coverage intervals. Changes in the geographical composition of the network contribute to differences between observation-year distributions; the medians do not estimate a climate trend or a growing-season rainfall total.\n\n"
        "The untreated SEPTTR records comprise 609 observations at 296 trial-dates and 283 source-coordinate/date combinations. Six records annotated CROP INJURY retain their original values and weather links but are unsuitable disease-response observations. The remaining 603 records occupy 279 coordinate-date combinations and 275 distinct ERA5 grid-cell/date combinations. Each exposure window comprises exactly 30 completed UTC days and 720 hourly records, ending on the day before assessment. Weather from the assessment day and later dates is excluded. Daily forcing uses the UTC date of returned hourly timestamps; precipitation and radiation retain the preceding-hour convention of the API. Disease values and organ-specific observations remain separate source records.\n\n"
        f"Among distinct eligible ERA5 grid-cell/date windows, median 30-day mean air temperature was {temp_text}, and median 30-day liquid precipitation was {rain_text}. The distributions include exposures sampled at different locations and assessment dates. Atmospheric humidity exposure is the number of hours with 2 m relative humidity ≥90%; it is not measured leaf wetness. Temporal overlap between windows and shared grid cells induce dependence among exposures.\n\n"
        "The exposure tables support temporally ordered weather-association analyses after accounting for trial, observation date, organ, sampling method and repeated measurements. The heterogeneous infection-percentage annotations do not establish a single uniform disease-severity endpoint. Three selected observation years and an unbalanced trial network do not identify long-term climate-change effects or causal weather responses. Reanalysis also cannot resolve field microclimate, canopy humidity, cultivar susceptibility or management histories absent from the source records. [Weather data by Open-Meteo](https://open-meteo.com/) are supplied under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); country boundaries derive from [Natural Earth public-domain data](https://www.naturalearthdata.com/about/terms-of-use/).\n"
    )
    (OUT/"weather_interpretation.md").write_text(text,encoding="utf-8")
    captions=(
        "Figure 1. ERA5 liquid precipitation during the fixed 395-day coverage intervals for the BASF trial network in 2017, 2018 and 2019. Each marker represents one source coordinate-year; the colour scale is shared across panels. Requested coordinates are rounded to 0.1°, and weather represents the nearest 0.25° ERA5 grid cell. The coverage interval extends from 1 September in the preceding year to 30 September in the observation year and does not imply observed sowing or harvest dates. Country boundaries: Natural Earth, public domain. Weather: [Open-Meteo / ERA5](https://open-meteo.com/), CC BY 4.0.\n\n"
        "Figure 2. Weather exposure during the 30 completed UTC days preceding untreated Septoria assessment. Panels show mean air temperature, accumulated liquid precipitation, hours with 2 m relative humidity ≥90%, and hours with liquid precipitation >0.1 mm. Each point represents one distinct ERA5 grid-cell/assessment-date window linked to an eligible disease observation. Median, interquartile range and 1.5×IQR whiskers describe the distribution; they are not confidence intervals. Six CROP INJURY-annotated source rows are excluded from the eligible disease subset. Overlapping windows and shared grid cells induce dependence. High-humidity hours describe atmospheric exposure rather than measured leaf wetness. Weather: [Open-Meteo / ERA5](https://open-meteo.com/), CC BY 4.0.\n"
    )
    (OUT/"weather_figure_captions.md").write_text(captions,encoding="utf-8")


if __name__=="__main__":
    main()
