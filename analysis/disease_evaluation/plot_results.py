#!/usr/bin/env python3
"""Export standalone research figures from the held-out benchmark CSVs."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
COLORS = {"secondary_seir_20d": "#176b84", "first_observation_persistence": "#8b9297",
          "global_logit_trend": "#bc6a2f", "secondary_seir_20d_beta_cap5": "#67a1ae",
          "secondary_seir_20d_beta_cap10": "#b0ced3"}
LABELS = {"secondary_seir_20d": "SEIR: L=20 d, β≤1", "first_observation_persistence": "First-value persistence",
          "global_logit_trend": "Global logit trend", "secondary_seir_20d_beta_cap5": "SEIR: L=20 d, β≤5",
          "secondary_seir_20d_beta_cap10": "SEIR: L=20 d, β≤10"}


def save(fig, stem):
    for extension in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{extension}", dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
                         "axes.labelsize": 10, "axes.spines.right": False,
                         "axes.spines.top": False, "pdf.fonttype": 42})
    predictions = pd.read_csv(OUT / "predictions.csv")
    metrics = pd.read_csv(OUT / "fold_metrics.csv")
    selected = predictions.loc[predictions.fold_type.eq("forward_year") & predictions.scored]
    primary = ["secondary_seir_20d", "first_observation_persistence", "global_logit_trend"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.8), sharex=True, sharey=True)
    for i, (ax, name) in enumerate(zip(axes, primary)):
        data = selected.loc[selected.model.eq(name)]
        ax.plot([0, 100], [0, 100], color="#50555a", lw=1, ls="--", zorder=0)
        for year, marker, color, fill in [(2018, "o", "#176b84", True), (2019, "^", "#bc6a2f", False)]:
            sub = data.loc[data.year.eq(year)]
            ax.scatter(sub.observed_percent, sub.predicted_percent, s=24, marker=marker,
                       facecolors=color if fill else "none", edgecolors=color, linewidths=.9,
                       alpha=.7, label=f"{year}: {len(sub)} future assessments")
        ax.set(xlim=(-2, 102), ylim=(-2, 102), xticks=np.arange(0, 101, 20), yticks=np.arange(0, 101, 20),
               xlabel="Observed infection proxy (%)", title=f"{chr(97+i)}  {LABELS[name]}")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(color="#e4e7e9", linewidth=.6, zorder=-1)
    axes[0].set_ylabel("Predicted infection proxy (%)")
    axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.25), ncols=2)
    fig.suptitle("Conditional forward-year predictions with held-out coordinates", y=.98, fontsize=14)
    fig.text(.02, .005, "First assessment conditions each series and is excluded. Points are correlated repeated leaf-rank assessments; no p-values or confidence intervals.", fontsize=9)
    fig.subplots_adjust(top=.83, bottom=.26, wspace=.18)
    save(fig, "observed_vs_heldout_predictions")

    names = primary + ["secondary_seir_20d_beta_cap5", "secondary_seir_20d_beta_cap10"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharex=True)
    for year, ax in zip((2018, 2019), axes):
        sub = metrics.loc[metrics.fold.eq(f"forward_{year}_coordinate_purged")].set_index("model")
        values = sub.loc[names, "rmse_pp"].to_numpy()
        bars = ax.barh(np.arange(len(names)), values, color=[COLORS[name] for name in names], height=.64,
                       edgecolor="#465057", linewidth=.6)
        for bar, value in zip(bars, values):
            ax.text(value + .45, bar.get_y() + bar.get_height()/2, f"{value:.2f}", va="center", fontsize=10)
        row = sub.iloc[0]
        ax.set(yticks=np.arange(len(names)), yticklabels=[LABELS[name] for name in names],
               xlim=(0, 44), xlabel="RMSE (percentage points)",
               title=f"{year}: {int(row.n_trials)} trials, {int(row.n_coordinate_years)} coordinate-years")
        ax.invert_yaxis()
        ax.grid(axis="x", color="#e4e7e9", linewidth=.6)
        ax.set_axisbelow(True)
    fig.suptitle("Forward-year performance and transmission-coefficient bound sensitivity", y=.98, fontsize=14)
    fig.text(.015, .025, "Equal coordinate-year weights; equal leaf-rank series within each coordinate-year; equal future dates within each series. β fitted to training disease only.", fontsize=9)
    fig.subplots_adjust(left=.19, right=.99, top=.81, bottom=.18, wspace=.72)
    save(fig, "forward_fold_performance")

    spatial = metrics.loc[metrics.fold_type.eq("leave_country_out") & metrics.model.isin(primary)]
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    country_folds = sorted(spatial.fold.unique())
    styles = [("secondary_seir_20d", "o"), ("first_observation_persistence", "s"), ("global_logit_trend", "^")]
    for name, marker in styles:
        sub = spatial.loc[spatial.model.eq(name)].set_index("fold").reindex(country_folds)
        ax.scatter(sub.rmse_pp, np.arange(len(country_folds)), marker=marker, s=42, color=COLORS[name], label=LABELS[name])
    labels = []
    for fold in country_folds:
        sub = spatial.loc[spatial.fold.eq(fold)].iloc[0]
        country_name = fold.removeprefix('leave_country_').replace('_', ' ').title()
        country_name = {"Slowakia": "Slovakia"}.get(country_name, country_name)
        labels.append(f"{country_name} (trials={int(sub.n_trials)})")
    ax.set(yticks=np.arange(len(country_folds)), yticklabels=labels, xlim=(0, 65),
           xlabel="Country-held-out RMSE (percentage points)",
           title="Conditional spatial transfer across country-held-out trials")
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e4e7e9", linewidth=.6)
    ax.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, -.22), ncols=3)
    fig.text(.025, .02, "Training uses other countries in 2017–2019; this spatial comparison is separate from forward-year evaluation. Small-country folds have few trials.", fontsize=9)
    fig.subplots_adjust(left=.30, right=.98, top=.90, bottom=.22)
    save(fig, "leave_country_performance")
    print("Exported 3 figures in PNG and PDF.")


if __name__ == "__main__":
    main()
