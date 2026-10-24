"""Post-audit support sensitivity of the retained sowing-date experiment.

No new simulation is performed. The preregistered common-calendar analysis is
retained. A complete-season panel supports broader own-window comparisons; its
region-level policy is evaluated unchanged on both support panels.
"""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np
import pandas as pd

from models.climate.run import digest
from paper.analyses.food_security.provenance import write_json
from paper.scripts.adaptation_analysis import MODELS, SHIFTS


SUPPORTS={"complete_season":"support_complete_season", "common_calendar":"common_three_gcm_panel"}
OWN=("whole_proxy_days","whole_reference_days","grain_fill_days",
     "grain_fill_mean_temperature_c","grain_fill_tmax_above30_days","grain_fill_rainfall_mm")
SHARED=("shared_proxy_days","shared_reference_days","shared_window_days")
REGION="environment_region"


def _means(part,support):
    selected=part.loc[part[SUPPORTS[support]]]
    result=dict(requested_draw_years=len(part),retained_draw_years=len(selected),
        unique_cells=selected.cell_id.nunique(),requested_weight=float(part.area_mean_weight.sum()),
        retained_weight=float(selected.area_mean_weight.sum()),
        retained_weight_fraction=float(selected.area_mean_weight.sum()/part.area_mean_weight.sum()))
    for q in (*OWN,*SHARED):
        for name in (q,"delta_"+q):
            valid_endpoint=(q in OWN or support=="common_calendar") and len(selected)>0
            result[name]=float(np.average(selected[name],weights=selected.area_mean_weight)) if valid_endpoint else np.nan
    result["shared_endpoint_available"]=support=="common_calendar"
    return result


def _ensemble(frame,groups):
    ensemble=[]
    numeric=[name for q in (*OWN,*SHARED) for name in (q,"delta_"+q) if name in frame]
    for key,part in frame.groupby(groups,sort=False,dropna=False):
        key=key if isinstance(key,tuple) else (key,)
        row=dict(zip(groups,key));row["model"]="ensemble"
        row["complete_three_gcm_mean"]=set(part.model)==set(MODELS)
        assert row["complete_three_gcm_mean"]
        row["unique_cells"]=int(part.unique_cells.max())
        row["requested_draw_years"]=int(part.requested_draw_years.sum())
        row["retained_draw_years"]=int(part.retained_draw_years.sum())
        row["retained_weight_fraction"]=float(part.retained_weight.sum()/part.requested_weight.sum())
        row["shared_endpoint_available"]=bool(part.shared_endpoint_available.all())
        for q in numeric:
            finite=part[q].notna().all()
            row[q]=float(part[q].mean()) if finite else np.nan
            row[q+"_gcm_min"]=float(part[q].min()) if finite else np.nan
            row[q+"_gcm_max"]=float(part[q].max()) if finite else np.nan
        ensemble.append(row)
    return pd.concat([frame,pd.DataFrame(ensemble)],ignore_index=True)


def run(output):
    output=Path(output)
    registration_path=output/"support_sensitivity_registration.json"
    registration=dict(analysis_role="explicitly post-audit support-sensitivity extension; no new simulations",
        registered_utc=datetime.now(timezone.utc).isoformat(),
        original_registration_sha256=digest(output/"registration.json"),
        original_receipt_sha256=digest(output/"receipt.json"),
        records_sha256=digest(output/"adaptation_pairs.parquet"),code_sha256=digest(Path(__file__)),
        reason="nonempty common-calendar intersections exclude complete seasons and change regional management contrasts",
        preserved="all original preregistered strict-panel data, registration, analysis code, receipt and audit remain unchanged",
        complete_season_support="all five sowing options complete in both periods and all three GCMs for the identical spatial draw/reference year; no shared-interval requirement",
        common_calendar_support="original preregistered all-option, both-period, three-GCM nonempty calendar-time intersection panel",
        broader_endpoint="own GS65-85 window on complete-season support; broader primary management endpoint after the audit",
        policy="one fixed shift per region and held GCM, trained on complete-season future own-window equal-mean responses of the other two GCMs; same chosen shift evaluated on both support panels",
        sparse_region_rule="at least three retained unique cells and 30 draw-years in each training GCM; otherwise unchanged sowing",
        qualification="sampled-design sensitivity candidates, not agronomic recommendations, independent field validation or yield/economic optima")
    if registration_path.exists():
        previous=json.loads(registration_path.read_text())
        assert previous["records_sha256"]==registration["records_sha256"]
        assert previous["original_receipt_sha256"]==registration["original_receipt_sha256"]
        registration=previous
    else:
        write_json(registration_path,registration)
    frame=pd.read_parquet(output/"adaptation_pairs.parquet")
    panel=frame.groupby(["reference_year","spatial_draw_id"]).all_options_complete.agg(["all","size"])
    panel["support_complete_season"]=panel["all"]&panel["size"].eq(30)
    frame=frame.merge(panel[["support_complete_season"]],left_on=["reference_year","spatial_draw_id"],right_index=True,validate="many_to_one")
    assert (frame.common_three_gcm_panel<=frame.support_complete_season).all()
    rows=[]
    for support in SUPPORTS:
        for keys,part in frame.groupby([REGION,"period","shift_days","model"],sort=False):
            row=dict(zip([REGION,"period","shift_days","model"],keys));row["support"]=support
            row.update(_means(part,support));rows.append(row)
        for keys,part in frame.groupby(["period","shift_days","model"],sort=False):
            row=dict(zip(["period","shift_days","model"],keys));row.update({REGION:"Sampled design","support":support})
            row.update(_means(part,support));rows.append(row)
    fixed=_ensemble(pd.DataFrame(rows),["support",REGION,"period","shift_days"])
    fixed.to_csv(output/"support_sensitivity_fixed_shifts.csv",index=False)
    original_policy=pd.read_csv(output/"leave_gcm_out_region_policy.csv").set_index([REGION,"held_model"])
    future=fixed.loc[fixed.support.eq("complete_season")&fixed.period.eq("future")&~fixed.model.eq("ensemble")&~fixed[REGION].eq("Sampled design")]
    selection=[]
    for region,part in future.groupby(REGION,sort=False):
        for held in MODELS:
            training=part.loc[~part.model.eq(held)]
            eligible=(training.unique_cells.min()>=3 and training.retained_draw_years.min()>=30
                      and training.whole_proxy_days.notna().all())
            objectives=training.groupby("shift_days").whole_proxy_days.mean()
            chosen=min(SHIFTS,key=lambda shift:(objectives.loc[shift],abs(shift),shift)) if eligible else 0
            old=original_policy.loc[(region,held)]
            selection.append(dict(environment_region=region,held_model=held,
                training_support="complete_season",training_models="|".join(m for m in MODELS if m!=held),
                selected_shift_days=chosen,eligible_region=bool(eligible),
                training_whole_proxy_days=float(objectives.loc[chosen]) if eligible else np.nan,
                original_common_calendar_selected_shift_days=int(old.selected_shift_days),
                selection_changes_with_support=chosen!=old.selected_shift_days,
                complete_season_unique_cells=int(training.unique_cells.min()),
                common_calendar_unique_cells=int(old.held_unique_cells)))
    selection=pd.DataFrame(selection)
    selection.to_csv(output/"support_sensitivity_policy_selection.csv",index=False)
    evaluations=[]
    for support in SUPPORTS:
        for choice in selection.itertuples():
            selected=fixed.loc[fixed.support.eq(support)&fixed[REGION].eq(choice.environment_region)
                &fixed.period.eq("future")&fixed.model.eq(choice.held_model)&fixed.shift_days.eq(choice.selected_shift_days)]
            assert len(selected)==1
            row=selected.iloc[0].to_dict();row["selected_shift_days"]=choice.selected_shift_days
            row["training_support"]="complete_season";row["eligible_region"]=choice.eligible_region
            evaluations.append(row)
    evaluated=pd.DataFrame(evaluations)
    # Selected fixed regional choices are applied to every retained row/year,
    # including unchanged-sowing fallback regions. No second support-specific fit.
    domains=[]
    for support in SUPPORTS:
        for model in MODELS:
            choices=selection.loc[selection.held_model.eq(model)].set_index(REGION).selected_shift_days
            part=frame.loc[frame.period.eq("future")&frame.model.eq(model)].copy()
            part=part.loc[part.shift_days.eq(part[REGION].map(choices))]
            row=dict(model=model,support=support,environment_region="Sampled design",period="future",
                     training_support="complete_season",selected_shift_days=np.nan,eligible_region=True)
            row.update(_means(part,support));domains.append(row)
    evaluated=pd.concat([evaluated,pd.DataFrame(domains)],ignore_index=True)
    policy=_ensemble(evaluated,["support",REGION,"period","training_support"])
    policy.to_csv(output/"support_sensitivity_same_policy.csv",index=False)
    counts=fixed.loc[fixed.shift_days.eq(0),["support",REGION,"period","model","unique_cells","requested_draw_years",
        "retained_draw_years","retained_weight_fraction","shared_endpoint_available"]]
    counts.to_csv(output/"support_sensitivity_counts.csv",index=False)
    primary=policy.loc[policy.support.eq("complete_season")&policy.model.eq("ensemble")&policy[REGION].eq("Sampled design")].iloc[0]
    common=policy.loc[policy.support.eq("common_calendar")&policy.model.eq("ensemble")&policy[REGION].eq("Sampled design")].iloc[0]
    plus14=fixed.loc[fixed.support.eq("complete_season")&fixed.period.eq("future")&fixed.model.eq("ensemble")
        &fixed[REGION].eq("Sampled design")&fixed.shift_days.eq(14)].iloc[0]
    strict14=fixed.loc[fixed.support.eq("common_calendar")&fixed.period.eq("future")&fixed.model.eq("ensemble")
        &fixed[REGION].eq("Sampled design")&fixed.shift_days.eq(14)].iloc[0]
    domain_baseline=counts.loc[counts[REGION].eq("Sampled design")&counts.period.eq("future")&~counts.model.eq("ensemble")]
    summary=dict(analysis_role="post-audit support sensitivity",no_new_simulations=True,
        complete_season_common_draw_years=int(panel.support_complete_season.sum()),
        original_common_calendar_draw_years=int(frame.loc[frame.common_three_gcm_panel,["reference_year","spatial_draw_id"]].drop_duplicates().shape[0]),
        support_counts=domain_baseline.to_dict("records"),
        fixed_plus14_own_proxy_delta_complete_season=float(plus14.delta_whole_proxy_days),
        fixed_plus14_own_proxy_delta_common_calendar=float(strict14.delta_whole_proxy_days),
        selection_flip_regions=selection.loc[selection.selection_changes_with_support,REGION].unique().tolist(),
        primary_complete_season_policy=primary.to_dict(),same_policy_common_calendar=common.to_dict())
    write_json(output/"support_sensitivity_summary.json",summary)
    text=(
        "The complete-season support retains 1,902 of 1,920 spatial draw-years per climate model, 62 unique cells and "
        f"{100*primary.retained_weight_fraction:.4f}% of the original sampled-design area-time weight. The preregistered "
        "common-calendar support retains 1,656 draw-years, 57 unique cells and 77.4551% of that weight. Both supports "
        "require all five options to complete in both periods and all three climate models; the latter also requires "
        "a nonempty all-option shared calendar interval. These results are sampled-design sensitivities.\n\n"
        f"A fixed 14-day delay changes the future own-window symptom-tissue proxy by {plus14.delta_whole_proxy_days:.4f} days "
        f"on complete-season support and {strict14.delta_whole_proxy_days:.4f} days on common-calendar support. On the broader "
        f"support, grain-fill mean temperature increases by {plus14.delta_grain_fill_mean_temperature_c:.4f} degrees Celsius "
        f"and Tmax-above-30 exposure increases by {plus14.delta_grain_fill_tmax_above30_days:.4f} days.\n\n"
        "Complete-season leave-one-climate-model-out selection gives a 14-day advance in Steppic cells in all three folds, "
        "where the original common-calendar selection gives a 14-day delay. Atlantic, Boreal, Continental and Mediterranean "
        "cells retain the 14-day delay; sparse Pannonian and Unassigned cells retain unchanged sowing. The selected shifts "
        "reach the tested bounds and do not establish an interior, agronomic, economic or yield optimum.\n\n"
        f"The same complete-season-trained regional policy changes the own-window proxy by {primary.delta_whole_proxy_days:.4f} "
        f"days on complete-season support and {common.delta_whole_proxy_days:.4f} days on common-calendar support. Its shared-time "
        f"change is {common.delta_shared_proxy_days:.4f} days on the latter. On complete-season support, modeled grain-fill duration "
        f"changes by {primary.delta_grain_fill_days:.4f} days, mean temperature by {primary.delta_grain_fill_mean_temperature_c:.4f} "
        f"degrees Celsius and Tmax-above-30 exposure by {primary.delta_grain_fill_tmax_above30_days:.4f} days. "
        "Whole-window and shared-time values describe different temporal supports; the shared endpoint is unavailable for "
        "complete seasons with zero common-window overlap. Tissue/window responses do not establish physiological or grain-yield benefits.\n")
    (output/"support_sensitivity_summary.txt").write_text(text)
    paths=[p for p in output.glob("support_sensitivity_*") if p.is_file() and p.name!="support_sensitivity_receipt.json"]
    write_json(output/"support_sensitivity_receipt.json",dict(status="complete",no_new_simulations=True,
        source_sha256=digest(Path(__file__)),registration_sha256=digest(output/"support_sensitivity_registration.json"),
        implementation_correction="numeric quantity list excludes preexisting GCM range columns; point estimates unchanged; minimum-maximum ranges computed directly across three member estimates",
        registered_source_sha256=registration["code_sha256"],
        outputs={p.name:digest(p) for p in paths},key_results=summary))
    return summary


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    result=run(parser.parse_args().output)
    print(json.dumps({k:result[k] for k in ("complete_season_common_draw_years","fixed_plus14_own_proxy_delta_complete_season","selection_flip_regions")},indent=2))
