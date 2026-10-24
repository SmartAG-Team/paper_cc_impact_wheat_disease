"""Bounded sowing-date sensitivities of the unchanged frozen STB climate model.

The retained 64-draw design (62 unique cells) is a sampled design, not an
independently established unbiased estimator of continental management effects.
Only sowing dates change. No yield, economic, cultivar or fungicide response is
fitted or inferred. Calendar shifts are sensitivity candidates rather than
verified agronomic recommendations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from models.climate.kernel import simulate_draw_seasons
from models.climate.run import DATA, ROOT, cell_registry, digest, load_weather, read_manifest
from paper.analyses.food_security.attribution import _authenticated_inputs
from paper.analyses.food_security.provenance import prepare_destination, write_json
from models.seasonal_septoria.host import sowing_date_from_calendar


SHIFTS = (-14, -7, 0, 7, 14)
MODELS = ("ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0")
REFERENCE_YEARS = tuple(range(1991, 2021))
PERIODS = ("reference", "future")
IDENTITY = ["model", "reference_year", "spatial_draw_id"]
PANEL_IDENTITY = ["reference_year", "spatial_draw_id"]
PRIOR = Path("data/final_analyses/attribution_replay")
REGION_MINIMUM_CELLS = 3
REGION_MINIMUM_DRAW_YEARS = 30
QUANTITIES = (
    "whole_proxy_days", "shared_proxy_days", "whole_reference_days",
    "shared_reference_days", "grain_fill_days", "grain_fill_mean_temperature_c",
    "grain_fill_tmax_above30_days", "grain_fill_rainfall_mm", "shared_window_days",
)


def _registered_inputs(root, manifest):
    """Authenticate frozen inputs and the earlier independently retained replay."""
    records = _authenticated_inputs(root, manifest, list(MODELS), list(REFERENCE_YEARS))
    regions = next(row for row in manifest["inputs"] if row["path"] == str(DATA / "regions.csv"))
    assert digest(root / regions["path"]) == regions["sha256"]
    records.append(dict(path=regions["path"], sha256=regions["sha256"],
                        bytes=(root / regions["path"]).stat().st_size))
    receipt_path = root / PRIOR / "attribution_receipt.json"
    prior_receipt = json.loads(receipt_path.read_text())
    path = root / PRIOR / "attribution_pairs.parquet"
    if digest(path) != prior_receipt["output_hashes"][path.name]:
        raise ValueError("Earlier retained attribution replay checksum changed")
    for p in (receipt_path, path, Path(__file__).resolve()):
        records.append(dict(path=str(p.relative_to(root)), sha256=digest(p), bytes=p.stat().st_size))
    return list({row["path"]: row for row in records}.values())


def _registration(inputs):
    return dict(
        registered_utc=datetime.now(timezone.utc).isoformat(),
        models=list(MODELS), scenario="ssp585", reference_years=list(REFERENCE_YEARS),
        future_years=[year + 80 for year in REFERENCE_YEARS], shifts_days=list(SHIFTS),
        varying_input="calendar sowing date only; identical realized weather for all options within each season",
        immutable="all host, phenology, disease, wetness, alignment and forcing parameters and source files",
        sampled_design=dict(spatial_draws=64, unique_cells=62, weights="original area_mean_weight; duplicate draws retained",
                            inference="sampled-design sensitivities; continental management representativeness not established"),
        calendar=dict(source_file="wwh_rf_ggcmi_crop_calendar_phase3_v1.01.nc4",
            source_url="https://zenodo.org/records/5062513",
            source_facts="static multiyear representative planting/maturity dates at 0.5 degrees; spatial extrapolation and gap filling; winter/rainfed selection retained",
            scenario_bounds="preserve baseline sowing calendar year, actual valid dates, date inside retained forcing, sowing before provider maturity; no clipping or wrapping",
            agronomic_bounds="GGCMI does not provide an admissible local sowing-date range; +/-14 days is a predeclared bounded sensitivity, not an empirically verified feasible window",
            excluded_processes=["soil workability", "soil-moisture establishment failure", "winter kill", "crop rotation constraints", "local cultivar and yield optimization"]),
        endpoint=dict(name="GS65_85_lost_had3", units="normalized affected-tissue days",
            whole="own modeled GS65-85 window for each sowing option",
            shared="intersection of all five GS65-85 windows on actual calendar dates within the same weather season; no extrapolation",
            shared_time_choice="actual calendar dates, not shifted elapsed-sowing age, keep management contrasts under the identical weather sequence",
            interpretation="post-latent symptom-tissue proxy; changes in tissue area and integration timing are not physiological or grain-yield benefits"),
        common_support="all five options valid with nonempty shared interval in both periods, then all three GCMs valid for the identical draw/reference-year; main endpoints use that single strict panel",
        policy=dict(selection="leave one GCM out; one fixed regional shift selected using the equal mean of the other two GCM future whole-window proxy means",
            evaluation="held-out GCM future contrast versus its unchanged sowing date, evaluated on the declared identical common-support panel",
            tie_break="minimum absolute shift, then earlier signed shift; zero wins ties",
            minimum_unique_cells=REGION_MINIMUM_CELLS,
            minimum_draw_years=REGION_MINIMUM_DRAW_YEARS,
            sparse_region_action="retain fixed-shift sensitivity results; policy falls back to unchanged sowing",
            qualification="within-scenario climate-model robustness, not independent field validation or a yield/economic optimum"),
        tradeoffs="modeled grain-fill duration, mean temperature, rainfall and count of Tmax>30 C days; descriptive exposures without a stress or yield response model",
        baseline_check="unchanged 0-day whole-window outputs versus retained original 30-year attribution y00/y11 at absolute tolerance 1e-10",
        input_hashes=inputs,
    )


def _shared_indices(starts, ends, valid):
    """One all-option calendar intersection per draw; empty support stays absent."""
    starts, ends, valid = np.asarray(starts), np.asarray(ends), np.asarray(valid, bool)
    good_stages = (starts >= 0) & (ends >= starts)
    complete = (valid & good_stages).all(axis=0)
    first, last = starts.max(axis=0), ends.min(axis=0)
    nonempty = complete & (last >= first)
    return first, last, complete, nonempty


def _summarize(frame, groups, *, support="common_three_gcm_panel"):
    """Conditional weighted means per GCM; strict equal-member means afterward."""
    rows = []
    for key, group in frame.groupby(groups, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        selected = group.loc[group[support]]
        w = selected.area_mean_weight.to_numpy()
        row = dict(zip(groups, key))
        row.update(requested_draw_years=len(group), retained_draw_years=len(selected),
            unique_cells=selected.cell_id.nunique(), requested_weight=float(group.area_mean_weight.sum()),
            retained_weight=float(w.sum()),
            retained_weight_fraction=float(w.sum() / group.area_mean_weight.sum()))
        for quantity in QUANTITIES:
            row[quantity] = float(np.average(selected[quantity], weights=w)) if len(selected) else np.nan
            delta = "delta_" + quantity
            if delta in selected:
                row[delta] = float(np.average(selected[delta], weights=w)) if len(selected) else np.nan
        rows.append(row)
    members = pd.DataFrame(rows)
    ensemble = []
    without_model = [key for key in groups if key != "model"]
    for key, group in members.groupby(without_model, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(without_model, key));row["model"] = "ensemble"
        complete = set(group.model) == set(MODELS) and group.whole_proxy_days.notna().all()
        row["complete_three_gcm_mean"] = bool(complete)
        row["unique_cells"] = int(group.unique_cells.max())
        row["retained_draw_years"] = int(group.retained_draw_years.sum())
        row["requested_draw_years"] = int(group.requested_draw_years.sum())
        row["retained_weight_fraction"] = float(group.retained_weight.sum() / group.requested_weight.sum())
        for q in QUANTITIES:
            for quantity in (q, "delta_" + q):
                if quantity in group:
                    row[quantity] = float(group[quantity].mean()) if complete else np.nan
                    row[quantity + "_gcm_min"] = float(group[quantity].min()) if complete else np.nan
                    row[quantity + "_gcm_max"] = float(group[quantity].max()) if complete else np.nan
        ensemble.append(row)
    return pd.concat([members, pd.DataFrame(ensemble)], ignore_index=True)


def _select_policy(regional):
    """Choose using training GCM means only; never a cell/year or held-model oracle."""
    future = regional.loc[regional.period.eq("future") & ~regional.model.eq("ensemble")]
    rows = []
    for region, part in future.groupby("environment_region", sort=False):
        for held_model in MODELS:
            training = part.loc[~part.model.eq(held_model)]
            held = part.loc[part.model.eq(held_model)].set_index("shift_days")
            if set(training.model) != set(MODELS) - {held_model} or set(held.index) != set(SHIFTS):
                raise ValueError("Every option and declared GCM required for policy evaluation")
            eligible = (training.unique_cells.min() >= REGION_MINIMUM_CELLS
                        and training.retained_draw_years.min() >= REGION_MINIMUM_DRAW_YEARS
                        and training.whole_proxy_days.notna().all())
            if eligible:
                candidates = training.groupby("shift_days").whole_proxy_days.mean()
                chosen = min(SHIFTS, key=lambda shift: (float(candidates.loc[shift]), abs(shift), shift))
                objective = float(candidates.loc[chosen])
            else:
                chosen, objective = 0, np.nan
            selection = held.loc[chosen]
            row = dict(environment_region=region, held_model=held_model,
                selected_shift_days=chosen, eligible_region=bool(eligible),
                training_models="|".join(model for model in MODELS if model != held_model),
                training_whole_proxy_days=objective,
                held_unique_cells=int(selection.unique_cells),
                held_draw_years=int(selection.retained_draw_years),
                retained_weight_fraction=float(selection.retained_weight_fraction))
            for q in QUANTITIES:
                row["held_" + q] = selection[q]
                row["held_delta_" + q] = selection["delta_" + q]
            rows.append(row)
    return pd.DataFrame(rows)


def _self_check():
    a,b,complete,nonempty = _shared_indices([[2,8],[4,10]], [[7,9],[6,11]], [[True,True],[True,True]])
    np.testing.assert_array_equal(a,[4,10]);np.testing.assert_array_equal(b,[6,9])
    assert complete.all() and nonempty.tolist() == [True,False]
    # A held-GCM value cannot enter its own training objective.
    rows=[]
    for m in MODELS:
        for shift in SHIFTS:
            r=dict(environment_region="test",period="future",model=m,shift_days=shift,
                   unique_cells=5,retained_draw_years=50,retained_weight_fraction=1.)
            for q in QUANTITIES:r[q]=(shift+7)**2;r["delta_"+q]=r[q]-49
            rows.append(r)
    first=_select_policy(pd.DataFrame(rows))
    changed=pd.DataFrame(rows)
    changed.loc[changed.model.eq(MODELS[0]),"whole_proxy_days"] = np.arange(5)*100
    second=_select_policy(changed)
    assert first.loc[first.held_model.eq(MODELS[0]),"selected_shift_days"].item()==-7
    assert second.loc[second.held_model.eq(MODELS[0]),"selected_shift_days"].item()==-7


def run(output: Path, *, root: Path = ROOT):
    """Run only the declared 64-draw, 30-year, three-GCM sowing panel."""
    started=time.monotonic();root=Path(root).resolve();manifest=read_manifest(root)
    if tuple(manifest["models"]) != MODELS:raise ValueError("Frozen three-GCM membership changed")
    output=prepare_destination(output,root)
    inputs=_registered_inputs(root,manifest)
    registration_path=output/"registration.json"
    write_json(registration_path,_registration(inputs))
    _self_check()
    draws=pd.read_csv(root/DATA/"spatial_draws.csv")
    cells=cell_registry(root)
    unique=cells.loc[cells.cell_id.isin(draws.cell_id)].reset_index(drop=True)
    ix=pd.Index(unique.cell_id).get_indexer(draws.cell_id)
    if len(draws)!=64 or len(unique)!=62 or (ix<0).any():raise ValueError("Changed spatial design")
    draw_cells=unique.iloc[ix].reset_index(drop=True)
    regions=pd.read_csv(root/DATA/"regions.csv").set_index("cell_id").environment_region
    draws["environment_region"]=draws.cell_id.map(regions)
    if draws.environment_region.isna().any():raise ValueError("Missing declared region")
    phenology=json.loads((root/DATA/"phenology_parameters.json").read_text())
    fit=json.loads((root/DATA/"disease_parameters.json").read_text())["fitted"]
    thresholds={int(k):v for k,v in json.loads((root/DATA/"stage_thresholds.json").read_text())["all_stage_thresholds"].items()}
    previous=pd.read_parquet(root/PRIOR/"attribution_pairs.parquet")
    previous=previous.loc[previous.setting.eq("baseline")].set_index(IDENTITY)
    rows=[];verification=[];checkpoints=[];max_mass_error=0.;batches=0
    for model in MODELS:
        model_start=time.monotonic()
        for ref_year in REFERENCE_YEARS:
            for period,year in (("reference",ref_year),("future",ref_year+80)):
                dates,weather=load_weather(model,"ssp585",year,unique,root)
                weather={key:value[ix] for key,value in weather.items()}
                baseline_sowing=np.array([sowing_date_from_calendar(year,r.planting_doy,r.maturity_doy)
                    .to_datetime64().astype("datetime64[D]") for r in draw_cells.itertuples()])
                maturity=np.array([pd.Timestamp(year=year,month=1,day=1).to_datetime64().astype("datetime64[D]")
                    +np.timedelta64(int(round(r.maturity_doy))-1,"D") for r in draw_cells.itertuples()])
                season=[];daily=[];starts=[];ends=[];valid=[]
                for shift in SHIFTS:
                    sowing=baseline_sowing+np.timedelta64(shift,"D")
                    bounded=(pd.DatetimeIndex(sowing).year==pd.DatetimeIndex(baseline_sowing).year)
                    bounded&=(sowing>=dates[0].to_datetime64().astype("datetime64[D]"))&(sowing<maturity)
                    bounded&=draw_cells.calendar_valid.to_numpy()
                    metric,detail=simulate_draw_seasons(dates,weather,sowing,draw_cells.latitude.to_numpy(),
                        phenology,fit,thresholds,return_daily=True)
                    max_mass_error=max(max_mass_error,detail["mass_error"]);batches+=1
                    good=metric.valid_complete_season.to_numpy()&bounded
                    start,end=[detail["host"].stage_day_index[s] for s in (65,85)]
                    good&=(start>=0)&(end>=start)
                    part=draws[["spatial_draw_id","cell_id","area_mean_weight","environment_region"]].copy()
                    part["model"],part["reference_year"],part["period"]=model,ref_year,period
                    part["harvest_year"],part["shift_days"]=year,shift
                    part["baseline_sowing_date"],part["sowing_date"]=baseline_sowing,sowing
                    part["bounded_calendar"],part["valid_complete_season"]=bounded,good
                    part["host_status"]=np.where(~bounded,"outside_declared_calendar_bounds",metric.status)
                    for stage in (31,65,85):part[f"BBCH{stage}_date"]=metric[f"BBCH{stage}_date"]
                    for target,source in (("whole_proxy_days","GS65_85_lost_had3"),
                        ("whole_reference_days","GS65_85_reference_had3"),
                        ("grain_fill_days","grain_fill_days"),
                        ("grain_fill_mean_temperature_c","mean_temperature_grain_fill_c"),
                        ("grain_fill_rainfall_mm","rainfall_grain_fill_mm")):
                        part[target]=np.where(good,metric[source],np.nan)
                    interval=(np.arange(len(dates))[None,:]>=start[:,None])&(np.arange(len(dates))[None,:]<=end[:,None])
                    part["grain_fill_tmax_above30_days"]=np.where(good,((weather["tmax_c"]>30)&interval).sum(axis=1),np.nan)
                    reference=detail["reference_area"].sum(axis=2)
                    affected=(detail["reference_area"]*detail["functional_loss"]).sum(axis=2)
                    daily.append((reference,affected));starts.append(start);ends.append(end);valid.append(good)
                    season.append(part)
                    if shift==0:
                        corner="y00" if period=="reference" else "y11"
                        expected=previous.loc[pd.MultiIndex.from_frame(part[IDENTITY]),corner].to_numpy()
                        actual=part.whole_proxy_days.to_numpy()
                        np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-10,equal_nan=True)
                        diff=np.abs(actual-expected);finite=diff[np.isfinite(diff)]
                        verification.append(dict(model=model,harvest_year=year,period=period,
                            maximum_absolute_difference=float(finite.max()) if finite.size else 0.,
                            valid_draws=int(good.sum()),matches_retained_original=True))
                    del detail
                first,last,complete,nonempty=_shared_indices(starts,ends,valid)
                interval=(np.arange(len(dates))[None,:]>=first[:,None])&(np.arange(len(dates))[None,:]<=last[:,None])
                for part,(reference,affected) in zip(season,daily):
                    part["all_options_complete"]=complete
                    part["shared_interval_nonempty"]=nonempty
                    part["shared_status"]=np.where(~complete,"one_or_more_option_seasons_unavailable",
                        np.where(~nonempty,"zero_all_option_window_overlap","complete"))
                    part["shared_window_days"]=np.where(nonempty,last-first+1.,np.nan)
                    part["shared_proxy_days"]=np.where(nonempty,(affected*interval).sum(axis=1),np.nan)
                    part["shared_reference_days"]=np.where(nonempty,(reference*interval).sum(axis=1),np.nan)
                    if (part.shared_proxy_days>part.shared_reference_days+1e-10).any():
                        raise AssertionError("Shared affected tissue exceeds reference tissue")
                    rows.append(part)
            if ref_year in (1991,2000,2010,2020):
                checkpoint=dict(model=model,reference_year_completed=ref_year,completed_kernel_batches=batches,
                    elapsed_seconds=time.monotonic()-started,model_elapsed_seconds=time.monotonic()-model_start)
                checkpoints.append(checkpoint);write_json(output/"progress.json",checkpoint)
                print(json.dumps(checkpoint),flush=True)
        # A checkpoint preserves completed results without changing source inputs.
        pd.concat([r for r in rows if r.model.iloc[0]==model],ignore_index=True).to_parquet(output/f"raw_{model}.parquet",index=False)
    frame=pd.concat(rows,ignore_index=True)
    # One paired support flag per draw/year/model across all options and periods.
    within=frame.groupby(IDENTITY,sort=False).shared_interval_nonempty.agg(["all","size"])
    within["common_options_both_periods"]=within["all"]&within["size"].eq(2*len(SHIFTS))
    frame=frame.merge(within[["common_options_both_periods"]],left_on=IDENTITY,right_index=True,validate="many_to_one")
    panel=frame.groupby(PANEL_IDENTITY,sort=False).common_options_both_periods.agg(["all","size"])
    panel["common_three_gcm_panel"]=panel["all"]&panel["size"].eq(2*len(SHIFTS)*len(MODELS))
    frame=frame.merge(panel[["common_three_gcm_panel"]],left_on=PANEL_IDENTITY,right_index=True,validate="many_to_one")
    contrast_id=[*IDENTITY,"period"]
    baseline=frame.loc[frame.shift_days.eq(0)].set_index(contrast_id)
    for q in QUANTITIES:
        base=baseline[q].reindex(pd.MultiIndex.from_frame(frame[contrast_id])).to_numpy()
        frame["delta_"+q]=frame[q]-base
    if not np.allclose(frame.loc[frame.shift_days.eq(0),["delta_"+q for q in QUANTITIES]],0,equal_nan=False):
        # Unavailable seasons are NaN; finite baseline contrasts must be exactly zero.
        v=frame.loc[frame.shift_days.eq(0),["delta_"+q for q in QUANTITIES]].to_numpy()
        assert np.all(v[np.isfinite(v)]==0)
    summary=_summarize(frame,["period","shift_days","model"])
    regional=_summarize(frame,["environment_region","period","shift_days","model"])
    policy=_select_policy(regional)
    frame.to_parquet(output/"adaptation_pairs.parquet",index=False)
    frame.to_csv(output/"adaptation_pairs.csv.gz",index=False,compression="gzip")
    summary.to_csv(output/"fixed_shift_summary.csv",index=False)
    regional.to_csv(output/"fixed_shift_regions.csv",index=False)
    policy.to_csv(output/"leave_gcm_out_region_policy.csv",index=False)
    pd.DataFrame(verification).to_csv(output/"baseline_replay.csv",index=False)
    statuses=frame.groupby(["model","period","shift_days","environment_region","host_status","shared_status"],sort=False).agg(
        draw_years=("cell_id","size"),unique_cells=("cell_id","nunique"),area_time_weight=("area_mean_weight","sum")).reset_index()
    statuses.to_csv(output/"completion_and_missingness.csv",index=False)
    support=frame.loc[frame.shift_days.eq(0)].groupby(["environment_region","model","period"],sort=False).agg(
        requested_draw_years=("cell_id","size"),requested_unique_cells=("cell_id","nunique"),
        requested_weight=("area_mean_weight","sum"),all_options_complete=("all_options_complete","sum"),
        nonempty_shared_intervals=("shared_interval_nonempty","sum"),
        common_options_both_periods=("common_options_both_periods","sum"),
        common_three_gcm_draw_years=("common_three_gcm_panel","sum")).reset_index()
    support["zero_overlap_complete_seasons"]=support.all_options_complete-support.nonempty_shared_intervals
    support.to_csv(output/"support.csv",index=False)
    region_design=draws.groupby("environment_region",sort=False).agg(
        draws=("cell_id","size"),unique_cells=("cell_id","nunique"),sample_area_weight=("area_mean_weight","sum")).reset_index()
    region_design.to_csv(output/"regional_sample_design.csv",index=False)
    # Equal-GCM mean of the held-GCM policy results; no second optimization.
    policy_summary=policy.groupby("environment_region",sort=False).agg(
        eligible_region=("eligible_region","all"),held_unique_cells=("held_unique_cells","max"),
        selected_shifts=("selected_shift_days",lambda x:"|".join(str(int(v)) for v in x)),
        retained_weight_fraction=("retained_weight_fraction","mean"),
        **{q:(q,"mean") for q in policy.columns if q.startswith("held_delta_")}).reset_index()
    for q in [q for q in policy.columns if q.startswith("held_delta_")]:
        ranges=policy.groupby("environment_region")[q].agg(["min","max"])
        policy_summary[q+"_gcm_min"]=policy_summary.environment_region.map(ranges["min"])
        policy_summary[q+"_gcm_max"]=policy_summary.environment_region.map(ranges["max"])
    policy_summary.to_csv(output/"policy_tradeoffs.csv",index=False)
    # A whole-sampled-design policy changes only adequately represented regions.
    regional_weights=frame.loc[frame.common_three_gcm_panel&frame.period.eq("future")&frame.shift_days.eq(0)].groupby(
        ["model","environment_region"]).area_mean_weight.sum()
    policy=policy.copy();policy["retained_area_time_weight"]=[regional_weights.get((r.held_model,r.environment_region),0.) for r in policy.itertuples()]
    domain_rows=[]
    for model,part in policy.groupby("held_model",sort=False):
        row=dict(model=model,regions=len(part),unique_cells=int(part.held_unique_cells.sum()))
        for q in [q for q in policy.columns if q.startswith("held_delta_")]:
            selected=part.loc[part.retained_area_time_weight.gt(0)]
            row[q]=float(np.average(selected[q],weights=selected.retained_area_time_weight))
        domain_rows.append(row)
    domain=pd.DataFrame(domain_rows)
    ensemble=dict(model="ensemble",regions=int(domain.regions.max()),unique_cells=int(domain.unique_cells.max()))
    for q in [q for q in domain.columns if q.startswith("held_delta_")]:
        ensemble[q]=float(domain[q].mean());ensemble[q+"_gcm_min"]=float(domain[q].min());ensemble[q+"_gcm_max"]=float(domain[q].max())
    domain=pd.concat([domain,pd.DataFrame([ensemble])],ignore_index=True)
    domain.to_csv(output/"policy_sampled_design.csv",index=False)
    write_json(output/"quantity_definitions.json",dict(
        whole_proxy_days="own GS65-85 normalized post-latent affected-tissue integral; days",
        shared_proxy_days="same integral over common calendar-date intersection across all options; days",
        whole_reference_days="reference normalized top-three-leaf tissue area integrated over own GS65-85; days",
        shared_reference_days="reference normalized tissue integrated on the all-option shared calendar interval; days",
        delta="option minus unchanged sowing under identical GCM/year/draw weather; negative proxy delta denotes a smaller modeled symptom-tissue integral",
        grain_fill_days="modeled inclusive GS65-85 window duration; days, not a measured grain-fill observation",
        grain_fill_mean_temperature_c="mean daily mean air temperature during own GS65-85; degrees Celsius",
        grain_fill_tmax_above30_days="count of days with Tmax >30 C during own GS65-85; descriptive exposure, not a calibrated heat stress or yield loss",
        gcm_ranges="minimum-maximum among three specified climate models, not confidence intervals",
        repeated_numerical_results="numerical CSVs are reproducibility/plot sources, not duplicate manuscript display tables"))
    result=dict(status="complete",elapsed_seconds=time.monotonic()-started,kernel_batches=batches,
        scenario="ssp585",spatial_draws=64,unique_cells=62,season_option_draw_records=len(frame),
        baseline_checks=len(verification),maximum_baseline_absolute_difference=max(r["maximum_absolute_difference"] for r in verification),
        maximum_mass_error=max_mass_error,zero_overlap_complete_seasons=int(support.zero_overlap_complete_seasons.sum()),
        requested_draw_years_per_gcm=64*30,common_three_gcm_draw_years=int(panel.common_three_gcm_panel.sum()),
        sampled_design_sensitivity=True,agronomic_recommendation=False,physiological_or_yield_validation=False,
        registration_sha256=digest(registration_path),code_sha256=digest(Path(__file__)),input_hashes=inputs,
        policy_sampled_design=domain.to_dict("records"),
        output_hashes={p.name:digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    write_json(output/"receipt.json",result)
    print(json.dumps({k:result[k] for k in ("status","elapsed_seconds","kernel_batches","common_three_gcm_draw_years","maximum_baseline_absolute_difference")}),flush=True)
    return result


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=ROOT)
    parser.add_argument("--output",type=Path,default=ROOT/"data/final_analyses/adaptation")
    args=parser.parse_args()
    run(args.output,root=args.root)
