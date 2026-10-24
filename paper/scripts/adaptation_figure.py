"""Publication-size plot of the registered sampled-design sowing experiment."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from models.climate.run import digest


def build_adaptation_figure(output):
    """Return a 19-cm figure and its plot-source specification without exporting."""
    output=Path(output)
    fixed=pd.read_csv(output/"support_sensitivity_fixed_shifts.csv")
    policies=pd.read_csv(output/"support_sensitivity_same_policy.csv")
    choices=pd.read_csv(output/"support_sensitivity_policy_selection.csv")
    regions=sorted(choices.loc[choices.eligible_region,"environment_region"].unique())
    with plt.rc_context({"font.family":"DejaVu Sans","font.size":12,
                         "axes.labelsize":12,"xtick.labelsize":12,"ytick.labelsize":12,
                         "legend.fontsize":12,"axes.linewidth":.8}):
        fig=plt.figure(figsize=(19/2.54,18.8/2.54),facecolor="white")
        a=fig.add_axes([.21,.745,.76,.18])
        b=fig.add_axes([.21,.415,.76,.20])
        c=fig.add_axes([.21,.095,.76,.18])
        colors={"common_calendar":"#34769A","complete_season":"#B87342"}
        labels={"common_calendar":"Common calendar support","complete_season":"Complete-season support"}
        for support,color in colors.items():
            selected=fixed.loc[fixed.support.eq(support)&fixed.period.eq("future")&fixed.model.eq("ensemble")
                               &fixed.environment_region.eq("Sampled design")].sort_values("shift_days")
            q="delta_whole_proxy_days"
            a.fill_between(selected.shift_days,selected[q+"_gcm_min"],selected[q+"_gcm_max"],
                           color=color,alpha=.13,linewidth=0)
            a.plot(selected.shift_days,selected[q],color=color,marker="o" if support=="common_calendar" else "s",
                   markersize=4,linewidth=1.2,label=labels[support])
        a.axhline(0,color="#A9ADB0",linewidth=.8,zorder=0)
        a.set_xticks([-14,-7,0,7,14]);a.set_xlim(-15,15)
        a.set_xlabel("Sowing-date shift (days)")
        a.set_ylabel("Own-window proxy\nchange (days)")
        a.legend(loc="lower center",bbox_to_anchor=(.5,1.06),ncol=2,frameon=False,
                 columnspacing=1.,handlelength=1.6,handletextpad=.4,borderaxespad=0)
        ypos=np.arange(len(regions))
        region_choices=choices.groupby("environment_region").agg(
            strict=("original_common_calendar_selected_shift_days","mean"),
            complete=("selected_shift_days","mean"),strict_n=("common_calendar_unique_cells","max"),
            complete_n=("complete_season_unique_cells","max")).loc[regions]
        for region in regions:
            part=choices.loc[choices.environment_region.eq(region)]
            assert part.selected_shift_days.nunique()==1 and part.original_common_calendar_selected_shift_days.nunique()==1
        for i,(support,color) in enumerate(colors.items()):
            shift=region_choices["strict" if i==0 else "complete"].to_numpy()
            b.plot(shift,ypos+(-.12 if i==0 else .12),linestyle="none",marker="o" if i==0 else "s",
                   color=color,markersize=5,clip_on=False)
        b.axvline(0,color="#A9ADB0",linewidth=.8,zorder=0)
        b.set_yticks(ypos,regions);b.invert_yaxis()
        b.set_ylim(len(regions)-.5,-.5);b.set_xlim(-16,16);b.set_xticks([-14,-7,0,7,14])
        b.set_xlabel("Selected sowing-date shift (days)",labelpad=1.5)
        regional_colors={"Atlantic":"#34769A","Boreal":"#8E6C8A","Continental":"#B87342",
                         "Mediterranean":"#887D42","Steppic":"#438A83","Sampled design":"#353B40"}
        for region in [*regions,"Sampled design"]:
            selected=policies.loc[policies.support.eq("complete_season")&policies.model.eq("ensemble")
                                  &policies.environment_region.eq(region)].iloc[0]
            x,y=[float(selected[q]) for q in ("delta_grain_fill_tmax_above30_days","delta_whole_proxy_days")]
            qx,qy="delta_grain_fill_tmax_above30_days","delta_whole_proxy_days"
            c.errorbar(x,y,xerr=[[x-selected[qx+"_gcm_min"]],[selected[qx+"_gcm_max"]-x]],
                       yerr=[[y-selected[qy+"_gcm_min"]],[selected[qy+"_gcm_max"]-y]],
                       color=regional_colors[region],marker="D" if region=="Sampled design" else "o",
                       markersize=4,linewidth=.8,capsize=2,label=region)
        c.axvline(0,color="#A9ADB0",linewidth=.8,zorder=0)
        c.axhline(0,color="#A9ADB0",linewidth=.8,zorder=0)
        c.set_xlabel("Change in GS65–85 days with Tmax >30°C")
        c.set_ylabel("Own-window proxy\nchange (days)")
        c.legend(loc="lower center",bbox_to_anchor=(.5,1.00),ncol=3,frameon=False,
                 columnspacing=1.,handlelength=1.2,handletextpad=.3,borderaxespad=0)
        for ax in (a,b,c):
            ax.spines[["top","right"]].set_visible(False)
            ax.tick_params(width=.8,length=3)
        fig.text(.025,.975,"(a)",fontsize=12,ha="left",va="top")
        fig.text(.025,.68,"(b)",fontsize=12,ha="left",va="top")
        fig.text(.025,.34,"(c)",fontsize=12,ha="left",va="top")
    spec=dict(name="sampled_sowing_date_adaptation_support_sensitivity",width_cm=19.,height_cm=18.8,font_pt=12,
        scenario="ssp585",period="2071-2100",model_count=3,
        source_files={name:digest(output/name) for name in ("support_sensitivity_fixed_shifts.csv","support_sensitivity_same_policy.csv","support_sensitivity_policy_selection.csv","support_sensitivity_counts.csv")},
        panels={"a":"Future fixed sowing shifts under unchanged weather; own-window endpoint on both common-calendar and complete-season support; equal-GCM means and model min-max bands",
                "b":"Fixed region-level leave-GCM-out choices trained separately on each support; each displayed choice agrees in all three held-GCM folds; caption gives retained cell counts",
                "c":"Same complete-season-trained policy evaluated on complete-season support: affected-tissue proxy versus hot-day exposure tradeoff; equal-GCM means and min-max ranges"},
        region_common_calendar_cells={r:int(region_choices.loc[r,"strict_n"]) for r in regions},
        region_complete_season_cells={r:int(region_choices.loc[r,"complete_n"]) for r in regions},
        limitations=["explicitly post-audit support-sensitivity extension; original preregistered strict analysis retained",
                     "shared-time endpoints remain available only on common-calendar support; same complete-trained policy compared across endpoints in source data",
                     "unbiased continental management inference not established",
                     "sparse regions retain unchanged sowing in the policy","model ranges are not confidence intervals",
                     "symptom-tissue proxy, not yield or a recommended agronomic practice"])
    spec["caption"]=(
        "(a) 2071–2100 SSP585 fixed sowing shifts under unchanged weather, expressed as changes in normalized "
        "affected-tissue days over each option’s own GS65–85 window. Lines show equal means of three GCMs; bands "
        "show model minima–maxima. (b) Regional leave-one-GCM-out choices trained separately on each support; each "
        "choice agrees across three folds. (c) The complete-season-trained policy evaluated on complete-season support, "
        "showing proxy changes and changes in GS65–85 days with Tmax >30°C; bars show model minima–maxima. "
        "Complete-season support requires all five options to complete in both periods and all three GCMs, retaining "
        "1,902/1,920 draw-years, 62 cells and 99.6407% of design weight. Nonempty common-calendar support retains "
        "1,656 draw-years, 57 cells and 77.4551%. Retained cells (common-calendar/complete-season) are Atlantic 8/8, "
        "Boreal 13/13, Continental 18/18, Mediterranean 7/9 and Steppic 9/12. Sparse Pannonian and Unassigned regions "
        "retain unchanged sowing. Complete-season support sensitivity is post-audit; common-calendar support was preregistered. The same "
        "complete-trained policy changes the proxy by −2.109 own-window and −1.882 shared-time days on strict support, "
        "and −1.891 own-window days on broader support. Model ranges are not confidence intervals. Bounded "
        "sampled-design sensitivities and descriptive thermal exposures do not establish agronomic or yield optima."
    )
    fig.gcb_method_spec=spec
    return fig,spec


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output",type=Path)
    parser.add_argument("--name",default="fig9.png")
    args=parser.parse_args()
    fig,spec=build_adaptation_figure(args.output)
    fig.savefig(args.output/args.name,dpi=300,facecolor="white")
    spec["builder_sha256"]=digest(Path(__file__))
    (args.output/"figure_spec.json").write_text(json.dumps(spec,indent=2)+"\n")
