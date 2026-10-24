"""Reproducible descriptive analyses of the frozen European STB climate census.

No disease/phenology model is fitted or replayed. Cell means retain their exact
valid-year-pair counts; aggregation uses harvested area * valid pairs / 30,
followed by equal weighting of the three GCMs. All covariates are changes in the
modelled GS65--85 window, not fixed-calendar climate-change estimates. Symptom
timing is an internal frozen-model state, so its association with damage is not
an independent validation, mediation analysis, or causal attribution.

Run: .venv/bin/python -m analysis.gcb_preparation.climate_response_insights
Rerender: build_latitude_environment(data_directory, png_path), or
          build_climate_phenology(data_directory, png_path).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "data/final_analyses/climate_response"
SOURCE = ROOT / "data/final_study/evidence/full_grid_climate/results"
MODELS = ["ACCESS-CM2", "MPI-ESM1-2-HR", "MRI-ESM2-0"]
SCENARIOS = ["ssp126", "ssp245", "ssp585"]
PERIODS = ["2031-2060", "2071-2100"]
KEY = ["cell_id", "model", "scenario", "period"]
LOSS = "GS65_85_lost_had3"
TIMING = "F1_symptom_relative_anthesis_days"
TEMP = "mean_temperature_grain_fill_c"
RAIN = "rainfall_grain_fill_mm"
DURATION = "grain_fill_days"
FRACTION = "GS65_85_functional_lost_fraction"
METRICS = [LOSS, TIMING, TEMP, RAIN, DURATION, FRACTION]
REGIMES = ["Increase", "Mixed", "Decrease"]
SSP_COLORS = ["#0072B2", "#E69F00", "#D55E00"]
GCM_COLORS = ["#0072B2", "#D55E00", "#009E73"]


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def weighted_mean(values, weights):
    values, weights = np.asarray(values, float), np.asarray(weights, float)
    good = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[good], weights=weights[good])) if good.any() else np.nan


def weighted_correlation(x, y, weights):
    x, y, weights = [np.asarray(v, float) for v in (x, y, weights)]
    good = np.isfinite(x) & np.isfinite(y) & np.isfinite(weights) & (weights > 0)
    x, y, weights = x[good], y[good], weights[good]
    x = x - np.average(x, weights=weights)
    y = y - np.average(y, weights=weights)
    return float(np.sum(weights*x*y) / np.sqrt(np.sum(weights*x*x)*np.sum(weights*y*y)))


def summarize_groups(wide, grouping, metrics=METRICS):
    """Single-member summaries, always retaining the full fixed-area denominator."""
    rows = []
    for key, part in wide.groupby(grouping, observed=True, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        base = dict(zip(grouping, key))
        area = float(part.harvested_total_ha.sum())
        for metric in metrics:
            pair_column = "timing_valid_year_pairs" if metric == TIMING else "valid_year_pairs"
            weight = (part.harvested_total_ha*part[pair_column]/30).to_numpy(float)
            good = part[metric].notna().to_numpy() & (weight > 0)
            rows.append(dict(base, metric=metric, change=weighted_mean(part[metric], weight),
                fixed_landuse_area_ha=area, cells=len(part), valid_cells=int(good.sum()),
                pair_weight_ha=float(weight[good].sum()),
                valid_year_pairs=int(part.loc[good, pair_column].sum()),
                area_time_coverage=float(weight[good].sum()/area) if area else np.nan,
                minimum_valid_year_pairs=int(part[pair_column].min()),
                maximum_valid_year_pairs=int(part[pair_column].max())))
    return pd.DataFrame(rows)


def equal_gcm_summary(frame, grouping, value="change"):
    rows = []
    for key, part in frame.groupby(grouping, observed=True, sort=True):
        if not isinstance(key, tuple):
            key = (key,)
        if len(part) != 3 or set(part.model) != set(MODELS):
            raise AssertionError(f"Expected exactly three GCMs: {key}")
        complete = part[value].notna().all()
        rows.append(dict(zip(grouping, key), mean_change=float(part[value].mean()) if complete else np.nan,
            gcm_min=float(part[value].min()) if complete else np.nan,
            gcm_max=float(part[value].max()) if complete else np.nan,
            fixed_landuse_area_ha=float(part.fixed_landuse_area_ha.iloc[0]),
            minimum_area_time_coverage=float(part.area_time_coverage.min()),
            maximum_area_time_coverage=float(part.area_time_coverage.max()),
            minimum_pair_weight_ha=float(part.pair_weight_ha.min()),
            maximum_pair_weight_ha=float(part.pair_weight_ha.max()),
            available_gcms=int(part[value].count())))
    return pd.DataFrame(rows)


def load_sources():
    raw = pd.read_parquet(SOURCE / "full_grid_paired_changes_by_gcm.parquet")
    ensemble = pd.read_parquet(SOURCE / "full_grid_ensemble_paired_changes.parquet")
    registry = pd.read_parquet(ROOT / "data/final_climate/cells.parquet")
    metadata = ensemble[["cell_id", "latitude", "longitude", "harvested_total_ha",
                         "environment_region", "calendar_valid"]].drop_duplicates("cell_id")
    if len(metadata) != 14941 or metadata.cell_id.duplicated().any():
        raise AssertionError("Frozen fixed-landuse population changed")
    test = registry.merge(metadata, on="cell_id", suffixes=("_source", "_ensemble"), validate="one_to_one")
    for col in ["latitude", "longitude", "harvested_total_ha"]:
        np.testing.assert_array_equal(test[col+"_source"], test[col+"_ensemble"])
    if raw.duplicated(KEY + ["metric"]).any():
        raise AssertionError("Duplicate cell/member/metric records")
    # Timing is undefined in seasons without a detected F1 symptom. Its pair
    # support is a subset of complete seasons, whereas the other five metrics
    # share the complete-season cohort. Equal counts therefore identify the
    # same timing/complete-season pairs; unequal counts must not be mixed.
    selected = raw[raw.metric.isin(METRICS)]
    counts = selected.pivot(index=KEY, columns="metric", values="valid_year_pairs")
    complete_metrics = [m for m in METRICS if m != TIMING]
    if not counts[complete_metrics].eq(counts[LOSS], axis=0).all().all():
        raise AssertionError("Complete-season metric year-pair supports differ")
    if not counts[TIMING].le(counts[LOSS]).all():
        raise AssertionError("Symptom-pair support is not a subset of complete seasons")
    for col in ["change", "reference_on_common", "future_on_common"]:
        finite = selected.pivot(index=KEY, columns="metric", values=col).notna()
        if not finite.eq(finite[LOSS], axis=0).all().all():
            raise AssertionError("Metric missingness differs")
    if not np.array_equal(raw.change.notna(), raw.valid_year_pairs.gt(0)):
        raise AssertionError("Missing versus zero or pair accounting changed")
    good = raw.change.notna()
    np.testing.assert_allclose((raw.future_on_common-raw.reference_on_common)[good],
                               raw.change[good], rtol=1e-10, atol=1e-10)
    wide = selected.pivot(index=KEY, columns="metric", values="change").reset_index()
    core = selected[selected.metric.eq(LOSS)][KEY + ["valid_year_pairs", "reference_on_common", "future_on_common"]]
    wide = wide.merge(core, on=KEY, validate="one_to_one").merge(metadata, on="cell_id", validate="many_to_one")
    timing_count = counts[TIMING].rename("timing_valid_year_pairs").reset_index()
    wide = wide.merge(timing_count, on=KEY, validate="one_to_one")
    for metric, label in [(TEMP, "temperature_c"), (RAIN, "rainfall_mm")]:
        climate = selected[selected.metric.eq(metric)][KEY+["reference_on_common", "future_on_common"]].rename(
            columns={"reference_on_common": "reference_window_"+label, "future_on_common": "future_window_"+label})
        wide = wide.merge(climate, on=KEY, validate="one_to_one")
    wide["timing_matched_year_support"] = wide.timing_valid_year_pairs.eq(wide.valid_year_pairs) & wide.valid_year_pairs.gt(0)
    wide["pair_weight_ha"] = wide.harvested_total_ha*wide.valid_year_pairs/30
    wide["latitude_bin_lower"] = np.floor(wide.latitude).astype(int)
    wide["latitude_band_5deg_lower"] = (np.floor(wide.latitude/5)*5).astype(int)
    # Verify frozen equal-GCM arithmetic independently, without averaging member weights.
    check = selected.groupby(["cell_id", "scenario", "period", "metric"]).change.agg(["mean", "count"]).reset_index()
    check.loc[check["count"].ne(3), "mean"] = np.nan
    check = check.merge(ensemble[["cell_id", "scenario", "period", "metric", "mean_change"]],
                         on=["cell_id", "scenario", "period", "metric"], validate="one_to_one")
    np.testing.assert_allclose(check["mean"], check.mean_change, rtol=1e-12, atol=1e-12, equal_nan=True)
    return wide, metadata, dict(fixed_cells=len(metadata), calendar_valid_cells=int(metadata.calendar_valid.sum()),
        fixed_harvested_area_ha=float(metadata.harvested_total_ha.sum()),
        members=18, analysed_metrics=METRICS, identical_complete_season_metric_year_pair_support=True,
        timing_associations_restricted_to_identical_complete_season_pairs=True,
        timing_pair_support_differs_cell_members=int(wide.timing_valid_year_pairs.ne(wide.valid_year_pairs).sum()),
        frozen_equal_gcm_arithmetic_verified=True, finite_cell_member_pairs=int(wide[LOSS].notna().sum()),
        minimum_valid_pairs=int(wide.valid_year_pairs.min()), maximum_valid_pairs=int(wide.valid_year_pairs.max()))


def contribution_summaries(wide, dimension):
    rows = []
    for key, part in wide.groupby(["model", "scenario", "period"], sort=True):
        positive = part.pair_weight_ha * part[LOSS].clip(lower=0).fillna(0)
        negative = -part.pair_weight_ha * part[LOSS].clip(upper=0).fillna(0)
        denp, denn = float(positive.sum()), float(negative.sum())
        for group, g in part.groupby(dimension, observed=True, sort=True):
            index = g.index
            rows.append(dict(zip(["model", "scenario", "period"], key), **{dimension: group},
                fixed_landuse_area_ha=float(g.harvested_total_ha.sum()),
                fixed_area_share_percent=100*float(g.harvested_total_ha.sum()/part.harvested_total_ha.sum()),
                pair_weight_ha=float(g.pair_weight_ha.sum()),
                pair_area_share_percent=100*float(g.pair_weight_ha.sum()/part.pair_weight_ha.sum()),
                area_time_coverage=float(g.pair_weight_ha.sum()/g.harvested_total_ha.sum()),
                positive_contribution_had3_ha=float(positive.loc[index].sum()),
                negative_magnitude_contribution_had3_ha=float(negative.loc[index].sum()),
                all_domain_positive_contribution_had3_ha=denp,
                all_domain_negative_magnitude_contribution_had3_ha=denn,
                positive_contribution_share_percent=100*float(positive.loc[index].sum()/denp) if denp else np.nan,
                negative_contribution_share_percent=100*float(negative.loc[index].sum()/denn) if denn else np.nan))
    return pd.DataFrame(rows)


def response_regimes(wide):
    key = ["cell_id", "scenario", "period"]
    signs = wide.pivot(index=key, columns="model", values=LOSS)
    label = np.where(signs.notna().all(axis=1), np.where(signs.gt(0).all(axis=1), "Increase",
                       np.where(signs.lt(0).all(axis=1), "Decrease", "Mixed")), "Unavailable")
    regimes = signs.reset_index()[key].assign(response_regime=label)
    joined = wide.merge(regimes, on=key, validate="many_to_one")
    profiles = summarize_groups(joined, ["scenario", "period", "response_regime", "model"])
    return regimes, joined, profiles


def block_ids(part, block_size):
    lat = np.floor((part.latitude.to_numpy()+90)/block_size).astype(int)
    lon = np.floor((part.longitude.to_numpy()+180)/block_size).astype(int)
    return lat, lon


def matched_timing_support(wide, dimensions):
    """Quantify spatial and climate selection induced by identical-pair timing support."""
    rows = []
    grouping = ["model", "scenario", "period", *dimensions]
    for key, p in wide.groupby(grouping, observed=True, sort=True):
        base = dict(zip(grouping, key))
        for label, selected in [("full", p), ("matched_timing", p[p.timing_matched_year_support]),
                                ("excluded_timing", p[~p.timing_matched_year_support])]:
            rows.append(dict(base, support=label, cells=len(selected),
                fixed_landuse_area_ha=float(selected.harvested_total_ha.sum()),
                fixed_area_fraction=float(selected.harvested_total_ha.sum()/p.harvested_total_ha.sum()),
                valid_year_pairs=int(selected.valid_year_pairs.sum()),
                pair_weight_ha=float(selected.pair_weight_ha.sum()),
                pair_area_fraction=float(selected.pair_weight_ha.sum()/p.pair_weight_ha.sum()) if p.pair_weight_ha.sum() else np.nan,
                reference_window_temperature_c=weighted_mean(selected.reference_window_temperature_c, selected.pair_weight_ha),
                future_window_temperature_c=weighted_mean(selected.future_window_temperature_c, selected.pair_weight_ha),
                reference_window_rainfall_mm=weighted_mean(selected.reference_window_rainfall_mm, selected.pair_weight_ha),
                future_window_rainfall_mm=weighted_mean(selected.future_window_rainfall_mm, selected.pair_weight_ha)))
    return pd.DataFrame(rows)


def association_diagnostics(wide, bootstrap_replicates=500):
    """Descriptive WLS; 5/10-degree spatial CV; 5-degree block resampling.

    No cellwise p-values. Bootstrap intervals quantify sensitivity to spatial
    composition, not observation/model/forcing uncertainty. Response timing is
    a model state entering the response calculation, not independent evidence.
    """
    features = [TEMP, RAIN, DURATION, TIMING]
    specs = {"window_climate": features[:2], "window_climate_duration": features[:3],
             "window_climate_duration_timing": features}
    results, coefficients, correlations = [], [], []
    rng = np.random.default_rng(20261023)
    for key, original in wide.groupby(["model", "scenario", "period"], sort=True):
        part = original[original.timing_matched_year_support].dropna(subset=[LOSS, *features])
        part = part[part.pair_weight_ha.gt(0)]
        base = dict(zip(["model", "scenario", "period"], key),
            fixed_domain_landuse_area_ha=float(original.harvested_total_ha.sum()),
            selected_matched_landuse_area_ha=float(part.harvested_total_ha.sum()),
            selected_matched_pair_weight_ha=float(part.pair_weight_ha.sum()),
            matched_fixed_area_fraction=float(part.harvested_total_ha.sum()/original.harvested_total_ha.sum()),
            matched_pair_area_fraction=float(part.pair_weight_ha.sum()/original.pair_weight_ha.sum()))
        y, w = part[LOSS].to_numpy(), part.pair_weight_ha.to_numpy()
        for feature in features:
            correlations.append(dict(base, predictor=feature, weighted_r=weighted_correlation(part[feature], y, w),
                                     cells=len(part), pair_weight_ha=float(w.sum())))
        total_variance = np.sum(w*(y-np.average(y, weights=w))**2)
        for name, selected in specs.items():
            x = np.column_stack([np.ones(len(part)), part[selected].to_numpy()])
            sw = np.sqrt(w)
            beta = np.linalg.lstsq(x*sw[:, None], y*sw, rcond=None)[0]
            for size in [5, 10]:
                latblock, lonblock = block_ids(part, size)
                blocks = np.column_stack([latblock, lonblock])
                for assignment in [0, 1, 2]:
                    # Entire geographical blocks are withheld together. Different
                    # assignments test that one arbitrary block split does not drive the result.
                    lon_multiplier, lat_multiplier = [(23, 13), (17, 29), (11, 13)][assignment]
                    fold = (lonblock*lon_multiplier+latblock*lat_multiplier) % 5
                    prediction = np.full(len(part), np.nan)
                    for k in range(5):
                        train, test = fold != k, fold == k
                        if not train.any() or not test.any():
                            raise AssertionError("Empty spatial train/test fold")
                        b = np.linalg.lstsq(x[train]*sw[train, None], y[train]*sw[train], rcond=None)[0]
                        prediction[test] = x[test]@b
                    results.append(dict(base, association_model=name, spatial_block_degrees=size,
                        fold_assignment=assignment, fold_longitude_multiplier=lon_multiplier,
                        fold_latitude_multiplier=lat_multiplier, folds=5, cells=len(part),
                        spatial_blocks=len(np.unique(blocks, axis=0)), pair_weight_ha=float(w.sum()),
                        weighted_out_of_block_r2=1-float(np.sum(w*(y-prediction)**2)/total_variance),
                        weighted_out_of_block_rmse_had3=float(np.sqrt(np.average((y-prediction)**2, weights=w)))))
            if name != "window_climate_duration_timing":
                continue
            latblock, lonblock = block_ids(part, 5)
            _, cluster = np.unique(np.column_stack([latblock, lonblock]), axis=0, return_inverse=True)
            clusters = int(cluster.max()+1)
            gram = np.zeros((clusters, x.shape[1], x.shape[1]))
            rhs = np.zeros((clusters, x.shape[1]))
            for k in range(clusters):
                mask = cluster == k
                gram[k] = (x[mask]*w[mask, None]).T@x[mask]
                rhs[k] = x[mask].T@(w[mask]*y[mask])
            multiplicity = rng.multinomial(clusters, np.full(clusters, 1/clusters), size=bootstrap_replicates)
            resampled_gram = np.einsum("bk,kij->bij", multiplicity, gram)
            resampled_rhs = multiplicity@rhs
            boot = np.linalg.solve(resampled_gram, resampled_rhs[..., None])[..., 0]
            for i, feature in enumerate(["intercept", *features]):
                coefficients.append(dict(base, association_model=name, predictor=feature,
                    coefficient=float(beta[i]), block_bootstrap_lower_025=float(np.quantile(boot[:, i], .025)),
                    block_bootstrap_upper_975=float(np.quantile(boot[:, i], .975)),
                    bootstrap_replicates=bootstrap_replicates, bootstrap_block_degrees=5,
                    spatial_blocks=clusters, interval_scope="Sensitivity to spatial composition; not total uncertainty"))
    return pd.DataFrame(results), pd.DataFrame(coefficients), pd.DataFrame(correlations)


def timing_curves(wide):
    all_members = wide[wide.scenario.eq("ssp585") & wide.period.eq("2071-2100")].copy()
    part = all_members[all_members.timing_matched_year_support].copy()
    # Common physical bins, with unsupported tails retained in data and omitted
    # from the display when they represent <0.5% of domain pair-area support.
    part["timing_bin_lower_days"] = np.floor(part[TIMING]/2)*2
    part = part[part[TIMING].notna()]
    rows = []
    for (model, lower), p in part.groupby(["model", "timing_bin_lower_days"], sort=True):
        domain = all_members[all_members.model.eq(model)]
        rows.append(dict(model=model, scenario="ssp585", period="2071-2100", timing_bin_lower_days=float(lower),
            timing_bin_upper_days=float(lower+2),
            timing_mean_days=weighted_mean(p[TIMING], p.pair_weight_ha),
            loss_change_mean_had3=weighted_mean(p[LOSS], p.pair_weight_ha),
            fixed_landuse_area_ha=float(p.harvested_total_ha.sum()), pair_weight_ha=float(p.pair_weight_ha.sum()),
            domain_pair_area_share_percent=100*float(p.pair_weight_ha.sum()/domain.pair_weight_ha.sum()),
            matched_domain_pair_area_fraction=float(part.loc[part.model.eq(model), "pair_weight_ha"].sum()/domain.pair_weight_ha.sum()),
            valid_year_pairs=int(p.valid_year_pairs.sum()), cells=len(p)))
    return pd.DataFrame(rows)


def configure_plotting():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12, "axes.labelsize": 12,
        "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12, "axes.titlesize": 12,
        "axes.linewidth": .8, "figure.dpi": 150, "savefig.dpi": 300,
        "axes.spines.top": True, "axes.spines.right": True, "xtick.direction": "out", "ytick.direction": "out"})
    return plt


def panel(ax, tag):
    ax.text(.015, .98, f"({tag})", transform=ax.transAxes, ha="left", va="top", fontsize=12,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
    for spine in ax.spines.values():
        spine.set_visible(True)


def build_latitude_environment(data_directory, output_path=None):
    """Render from published CSVs, at exactly 19 cm width and 300 dpi."""
    plt = configure_plotting()
    from matplotlib.lines import Line2D
    data_directory = Path(data_directory)
    latitude = pd.read_csv(data_directory/"latitude_ensemble.csv")
    contribution = pd.read_csv(data_directory/"environment_contributions_by_gcm.csv")
    fig = plt.figure(figsize=(19/2.54, 18.2/2.54), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1.25])
    axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
    for ax, period, tag in zip(axes, PERIODS, ["a", "b"]):
        for scenario, color in zip(SCENARIOS, SSP_COLORS):
            p = latitude[latitude.period.eq(period) & latitude.scenario.eq(scenario) & latitude.metric.eq(LOSS)].sort_values("latitude_bin_lower")
            # Every plotted bin has at least 0.1% of the fixed European wheat
            # area and at least 90% paired-area-time coverage in every GCM.
            # Keep the entire 1-degree grid: deleting an unsupported interior
            # bin would falsely connect the line/ribbon across the data gap.
            p = p.set_index("latitude_bin_lower").reindex(range(
                int(latitude.latitude_bin_lower.min()), int(latitude.latitude_bin_lower.max())+1))
            supported = p.display_supported.fillna(False).astype(bool)
            p.loc[~supported, ["mean_change", "gcm_min", "gcm_max"]] = np.nan
            x = p.index.to_numpy()+.5
            ax.fill_between(x, p.gcm_min, p.gcm_max, color=color, alpha=.12, linewidth=0)
            ax.plot(x, p.mean_change, color=color, lw=1.7, marker="o", ms=2.5,
                    markeredgewidth=0)
        ax.axhline(0, color=".3", lw=.7, ls="--", zorder=0)
        ax.set(xlim=(35, 65), ylim=(-5, 15.5), yticks=[-4, 0, 4, 8, 12],
               xticks=[35, 45, 55, 65], xlabel="Latitude (°N)")
        if tag == "a":
            ax.set_ylabel("Affected-tissue change\n(days)")
        panel(ax, tag)
    axes[1].tick_params(labelleft=False)
    axes[0].legend(handles=[Line2D([], [], color=c, lw=2, label=l) for c, l in zip(SSP_COLORS,
                       ["SSP1–2.6", "SSP2–4.5", "SSP5–8.5"])], loc="upper left",
                       bbox_to_anchor=(.10, 1.01), frameon=False, handlelength=1.0, labelspacing=.2)
    ax = fig.add_subplot(grid[1, :])
    p = contribution[contribution.scenario.eq("ssp585") & contribution.period.eq("2071-2100")]
    names = p.groupby("environment_region").fixed_area_share_percent.mean().sort_values().index.tolist()
    y = np.arange(len(names))
    specs = [("fixed_area_share_percent", "Wheat area", "#777777", "s", -.21),
             ("positive_contribution_share_percent", "Increase contribution", "#D55E00", "o", 0),
             ("negative_contribution_share_percent", "Decrease contribution", "#0072B2", "D", .21)]
    for field, label, color, marker, offset in specs:
        stats = p.groupby("environment_region")[field].agg(["mean", "min", "max"]).reindex(names)
        ax.errorbar(stats["mean"], y+offset,
                    xerr=np.maximum(0, np.vstack([stats["mean"]-stats["min"], stats["max"]-stats["mean"]])),
                    color=color, marker=marker, linestyle="none", ms=4, lw=1, capsize=2, label=label)
    labels = ["Outside EEA" if n == "Outside EEA regions" else n for n in names]
    ax.set(yticks=y, yticklabels=labels, xlabel="Share of area or directional affected-tissue contribution (%)",
           xlim=(-1, 61), xticks=[0, 10, 20, 30, 40, 50, 60])
    ax.grid(axis="x", color=".9", lw=.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", handletextpad=.3, labelspacing=.2)
    panel(ax, "c")
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white")
    return fig


def build_climate_phenology(data_directory, output_path=None):
    """Display within-model co-changes; no fitted attribution graph is implied."""
    plt = configure_plotting()
    from matplotlib.lines import Line2D
    data_directory = Path(data_directory)
    curves = pd.read_csv(data_directory/"timing_cochange_bins_by_gcm.csv")
    profiles = pd.read_csv(data_directory/"response_regime_profiles_by_gcm.csv")
    profiles = profiles[profiles.scenario.eq("ssp585") & profiles.period.eq("2071-2100")]
    fig = plt.figure(figsize=(19/2.54, 15.5/2.54), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    ax = fig.add_subplot(grid[0, :])
    for model, color in zip(MODELS, GCM_COLORS):
        p = curves[curves.model.eq(model) & curves.domain_pair_area_share_percent.ge(.5)].sort_values("timing_mean_days")
        ax.plot(p.timing_mean_days, p.loss_change_mean_had3, color=color, lw=1.6, marker="o", ms=3)
    ax.axhline(0, color=".4", lw=.7, ls="--")
    ax.axvline(0, color=".4", lw=.7, ls="--")
    ax.set(xlabel="F1 symptom-timing shift relative to modelled GS65 (days)",
           ylabel="Affected-tissue change\n(days)", xticks=[-15, -10, -5, 0, 5],
           yticks=[-2.5, 0, 2.5, 5, 7.5, 10])
    ax.legend(handles=[Line2D([], [], color=c, marker="o", ms=3, lw=1.5, label=m)
                       for m, c in zip(MODELS, GCM_COLORS)], frameon=False, ncol=3,
              loc="upper center", bbox_to_anchor=(.51, 1.22), columnspacing=.8, handlelength=1.3)
    panel(ax, "a")
    for j, (metric, xlabel, tag) in enumerate([(TEMP, "Temperature\nchange (°C)", "b"),
                                               (RAIN, "Rainfall\nchange (mm)", "c"),
                                               (DURATION, "Duration\nchange (days)", "d")]):
        ax = fig.add_subplot(grid[1, j])
        for mi, (model, color) in enumerate(zip(MODELS, GCM_COLORS)):
            p = profiles[profiles.metric.eq(metric) & profiles.model.eq(model)].set_index("response_regime").reindex(REGIMES)
            ax.scatter(p.change, np.arange(3)+(mi-1)*.17, color=color, s=24, zorder=3)
        ax.axvline(0, color=".4", lw=.7, ls="--")
        ax.set(xlabel=xlabel, ylim=(2.6, -.6), yticks=np.arange(3),
               yticklabels=REGIMES if j == 0 else ["", "", ""],
               xticks=[[-2, 0, 2], [-10, 0, 10], [0, 5]][j])
        if j == 0:
            ax.set_ylabel("Affected-tissue\nresponse")
        ax.grid(axis="y", color=".9", lw=.6)
        ax.set_axisbelow(True)
        panel(ax, tag)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=300, facecolor="white")
    return fig


def main(output_directory=DEFAULT_OUTPUT, bootstrap_replicates=500):
    output = Path(output_directory)
    data = output/"data"
    data.mkdir(parents=True, exist_ok=True)
    wide, metadata, audit = load_sources()
    wide.to_parquet(data/"member_cell_cochanges.parquet", index=False, compression="zstd")
    for dimensions, name in [([], "overall"), (["environment_region"], "environment"),
                             (["latitude_band_5deg_lower"], "latitude")]:
        matched_timing_support(wide, dimensions).to_csv(data/f"timing_matched_support_{name}.csv", index=False)
    support = metadata.groupby(metadata.latitude.apply(np.floor).astype(int)).agg(
        fixed_landuse_area_ha=("harvested_total_ha", "sum"), cells=("cell_id", "size")).reset_index()
    support.to_csv(data/"fixed_latitude_area_support.csv", index=False)
    latitude = summarize_groups(wide, ["scenario", "period", "latitude_bin_lower", "model"])
    latitude.to_csv(data/"latitude_by_gcm.csv", index=False)
    latensemble = equal_gcm_summary(latitude, ["scenario", "period", "latitude_bin_lower", "metric"])
    latensemble["fixed_domain_area_share_percent"] = 100*latensemble.fixed_landuse_area_ha/audit["fixed_harvested_area_ha"]
    latensemble["display_supported"] = latensemble.fixed_domain_area_share_percent.ge(.1) & latensemble.minimum_area_time_coverage.ge(.9)
    latensemble.to_csv(data/"latitude_ensemble.csv", index=False)
    lat5 = summarize_groups(wide, ["scenario", "period", "latitude_band_5deg_lower", "model"])
    lat5.to_csv(data/"latitude_5degree_by_gcm.csv", index=False)
    equal_gcm_summary(lat5, ["scenario", "period", "latitude_band_5deg_lower", "metric"]).to_csv(data/"latitude_5degree_ensemble.csv", index=False)
    env = summarize_groups(wide, ["scenario", "period", "environment_region", "model"])
    env.to_csv(data/"environment_by_gcm.csv", index=False)
    equal_gcm_summary(env, ["scenario", "period", "environment_region", "metric"]).to_csv(data/"environment_ensemble.csv", index=False)
    contribution = contribution_summaries(wide, "environment_region")
    contribution.to_csv(data/"environment_contributions_by_gcm.csv", index=False)
    latcontribution = contribution_summaries(wide, "latitude_band_5deg_lower")
    latcontribution.to_csv(data/"latitude_contributions_by_gcm.csv", index=False)
    regimes, joined, profiles = response_regimes(wide)
    regimes.to_csv(data/"response_regime_cells.csv", index=False)
    profiles.to_csv(data/"response_regime_profiles_by_gcm.csv", index=False)
    curves = timing_curves(wide)
    curves.to_csv(data/"timing_cochange_bins_by_gcm.csv", index=False)
    cv, coefficients, correlations = association_diagnostics(wide, bootstrap_replicates)
    cv.to_csv(data/"association_spatial_cv.csv", index=False)
    coefficients.to_csv(data/"association_coefficients_spatial_bootstrap.csv", index=False)
    correlations.to_csv(data/"association_weighted_correlations.csv", index=False)
    # Descriptive sign co-occurrence with full landuse and valid area denominators.
    rows = []
    for key, p in wide.groupby(["model", "scenario", "period"], sort=True):
        masks = {"increased_loss_and_decreased_window_rainfall": p[LOSS].gt(0)&p[RAIN].lt(0),
                 "increased_loss_and_earlier_relative_symptoms": p[LOSS].gt(0)&p[TIMING].lt(0)&p.timing_matched_year_support,
                 "increased_loss_and_decreased_window_temperature": p[LOSS].gt(0)&p[TEMP].lt(0),
                 "decreased_loss_and_increased_window_temperature": p[LOSS].lt(0)&p[TEMP].gt(0)}
        for label, mask in masks.items():
            rows.append(dict(zip(["model", "scenario", "period"], key), cochange=label,
                cells=int(mask.sum()), selected_pair_weight_ha=float(p.loc[mask, "pair_weight_ha"].sum()),
                domain_pair_weight_ha=float(p.pair_weight_ha.sum()),
                domain_fixed_landuse_area_ha=float(p.harvested_total_ha.sum()),
                pair_area_share_percent=100*float(p.loc[mask, "pair_weight_ha"].sum()/p.pair_weight_ha.sum()),
                fixed_area_share_percent=100*float(p.loc[mask, "harvested_total_ha"].sum()/p.harvested_total_ha.sum())))
    signshares = pd.DataFrame(rows)
    signshares.to_csv(data/"climate_canopy_sign_cooccurrences.csv", index=False)
    # Check new region arithmetic against the frozen published regional results.
    frozen = pd.read_csv(SOURCE/"environmental_region_changes_by_gcm.csv")
    match = env.merge(frozen, on=["scenario", "period", "environment_region", "model", "metric"], suffixes=("_new", "_frozen"), validate="one_to_one")
    np.testing.assert_allclose(match.change_new, match.change_frozen, rtol=1e-11, atol=1e-11, equal_nan=True)
    audit["regional_arithmetic_matches_frozen_outputs"] = True
    # Display exports are self-contained and retain units and support metadata.
    for figure in ["fig3", "fig4"]:
        (data/figure).mkdir(exist_ok=True)
    for name in ["latitude_by_gcm.csv", "latitude_ensemble.csv", "environment_contributions_by_gcm.csv"]:
        (data/"fig3"/name).write_bytes((data/name).read_bytes())
    for name in ["timing_cochange_bins_by_gcm.csv", "response_regime_profiles_by_gcm.csv", "response_regime_cells.csv"]:
        (data/"fig4"/name).write_bytes((data/name).read_bytes())
    build_latitude_environment(data/"fig3", output/"fig3.png")
    build_climate_phenology(data/"fig4", output/"fig4.png")
    metadata_doc = dict(
        study_scope="Frozen European STB model census; fixed SPAM2020 all-wheat area and imposed winter/rainfed calendar",
        baseline="1991-2020", future_periods=PERIODS,
        metric_units={LOSS:"days; normalized symptom-tissue proxy, calculated as symptom-gated model damage weighted by reference upper-three-leaf area and divided by nominal three-leaf area (legacy source field GS65_85_lost_had3)",
            TIMING:"days; positive = later F1 symptoms relative to modelled GS65 gate",
            TEMP:"degrees Celsius; mean over each member's modelled GS65-85 window",
            RAIN:"mm; accumulated over each member's modelled GS65-85 window",
            DURATION:"days from modelled GS65 through GS85 inclusive",
            FRACTION:"unitless modelled fraction of GS65-85 reference healthy-area days lost"},
        aggregation="Per-member sum(area * valid_year_pairs/30 * cell change) / sum(area * valid_year_pairs/30); then arithmetic mean of exactly three GCMs",
        pairing="Frozen year-index pairing:1991 with2031/2071 through2020 with2060/2100; only common finite years within each cell-member. Climate/duration/fraction/loss share complete-season pairs. Timing aggregates use their own pair counts. Timing associations and curves retain only cell-members whose timing counts equal complete-season counts, guaranteeing identical years because symptom validity is a subset of complete-season validity.",
        denominator="All14941 fixed-landuse cells retained; missing metric values are not zero; each table reports valid paired-area-time support",
        latitude_plot_support="1-degree bins; >=0.1% of fixed domain wheat area and >=90% area-time coverage in every GCM; tails retained in data",
        response_regimes="Increase/decrease require strictly positive/negative cell mean change in all three GCMs; Mixed includes zero or disagreement; Unavailable retains missing GCMs",
        directional_contributions="Within each GCM, positive area*pair_fraction*change contributions and absolute negative contributions each have their own domain denominator; equal-GCM mean shares",
        timing_plot_support="2-day bins; displayed only if >=0.5% of domain paired-area weight in the respective GCM; remaining bins retained in data",
        association_models="Descriptive weighted linear regressions fitted separately in all18 model/scenario/period members; climate-only, plusduration, plustiming compared on identical cell/pair support after excluding cell-members with unequal symptom-timing and canopy year-pair counts",
        spatial_validation="5-fold spatial CV using5-degree and10-degree latitude-longitude blocks; three deterministic fold assignments; cellwise p-values prohibited",
        bootstrap="500 resamples of complete5-degree spatial blocks; 2.5/97.5 percentiles measure sensitivity to spatial composition, not total uncertainty",
        cautions=["Window climate incorporates climate change and shifts in modelled phenology; it is not a fixed-calendar climate-change estimate.",
            "Accumulated rainfall also depends on window duration; neither drying nor warming is causally isolated.",
            "F1 timing is a frozen-model state entering damage integration; its association is not independent process validation or causal mediation.",
            "Modelled GS65 gate was not independently validated as anatomical anthesis; no observed anthesis claim is made.",
            "GCM ranges are ensemble spread, not confidence intervals; block resampling excludes structural and forcing uncertainty.",
            "Cold-edge coverage is incomplete in some bins; unsupported bins are not displayed but remain in source exports.",
            "No model refit, forcing modification, climate download, calendar adaptation or absolute-yield prediction occurs."])
    (data/"data_dictionary.json").write_text(json.dumps(metadata_doc, indent=2)+"\n")
    for figure, mapping in [("fig3", {"a":"2031-2060 latitude changes", "b":"2071-2100 latitude changes",
                                      "c":"SSP5-8.5 late-century directional environmental contributions"}),
                            ("fig4", {"a":"SSP5-8.5 late-century matched-pair symptom-timing co-changes",
                                      "b":"Regime temperature changes", "c":"Regime rainfall changes", "d":"Regime duration changes"})]:
        (data/figure/"data_dictionary.json").write_text(json.dumps(dict(metadata_doc, panel_mapping=mapping), indent=2)+"\n")
    figures = {
      "fig3": "(a, b) Changes in the modelled GS65–85 affected-tissue integral by latitude for 2031–2060 (a) and 2071–2100 (b), relative to 1991–2020. Lines are equal-GCM means of harvested-area and valid-year-pair weighted 1° cell-bin means; ribbons span the three GCMs. Each displayed bin represents at least 0.1% of fixed European wheat area and has at least 90% paired-area-time coverage in every GCM. Unsupported bins are masked on the complete latitude grid, leaving breaks in lines and ribbons; excluded cold-edge/tail bins remain in the source data. (c) Fixed wheat-area shares and shares of the positive and negative-magnitude affected-tissue contributions by environmental zone for SSP5–8.5, 2071–2100. Dots are equal-GCM mean shares and whiskers span GCMs. Each directional denominator is the corresponding domain-wide sum of harvested area × valid-pair fraction × positive or absolute-negative cell mean change; directional shares each sum to 100%. The affected-tissue integral (days) is a normalized symptom-tissue proxy: symptom-gated model damage weighted by reference upper-three-leaf area, divided by nominal three-leaf area, and integrated over the modelled GS65–85 window. Environmental zones follow the retained spatial assignment; Outside EEA and Unassigned cells remain in all denominators.",
      "fig4": "Co-changes in the affected-tissue integral, modelled symptom timing and grain-fill conditions under SSP5–8.5 for 2071–2100 relative to 1991–2020. (a) Harvested-area and paired-year weighted means of affected-tissue change in common 2-day bins of the shift in F1 symptom timing relative to the modelled GS65 gate; negative timing shifts indicate earlier symptoms relative to that gate. Only cell-members with exactly equal symptom-timing and complete-season valid-year-pair counts are retained, ensuring that timing and affected-tissue changes use identical years. Displayed bins each contain at least 0.5% of the relevant GCM’s full-domain paired-area weight; all bins and matched-support fractions remain in source data. (b–d) GCM-specific area/pair-weighted mean changes in mean temperature (b), accumulated rainfall (c) and duration (d) within the respective modelled GS65–85 windows, for cells where the affected-tissue integral increases in all three GCMs (Increase), disagrees or is exactly zero (Mixed), or decreases in all three GCMs (Decrease). Response membership is shared across GCMs; unavailable cells are retained separately in source data. Window climate reflects both changed weather and shifted modelled phenology. Symptom timing is an internal state contributing to the modelled affected-tissue calculation; these descriptive associations do not establish independent validation, causal mediation or attribution. The modelled GS65 gate is not an independently validated observation of anatomical anthesis. The affected-tissue integral (days) is a normalized symptom-tissue proxy: symptom-gated model damage weighted by reference upper-three-leaf area, divided by nominal three-leaf area, and integrated over the modelled GS65–85 window."}
    (output/"figure_captions.json").write_text(json.dumps(figures, indent=2)+"\n")
    # Exact numerical handoff, without duplicating figure values as manuscript tables.
    chosen = latcontribution[latcontribution.scenario.eq("ssp585") & latcontribution.period.eq("2071-2100") & latcontribution.latitude_band_5deg_lower.isin([50,55])]
    concentration = chosen.groupby("model")[["fixed_area_share_percent", "positive_contribution_share_percent"]].sum()
    timing_coefs = coefficients[coefficients.predictor.eq(TIMING)]
    primary_cv = cv[cv.spatial_block_degrees.eq(5) & cv.fold_assignment.eq(0)]
    cp = primary_cv.pivot(index=["model", "scenario", "period"], columns="association_model", values="weighted_out_of_block_r2")
    strong = dict(
        late585_latitude50_60_fixed_area_share_percent=float(concentration.fixed_area_share_percent.mean()),
        late585_latitude50_60_increased_loss_contribution_share_percent_mean=float(concentration.positive_contribution_share_percent.mean()),
        late585_latitude50_60_increased_loss_contribution_share_percent_gcm_range=[float(concentration.positive_contribution_share_percent.min()),float(concentration.positive_contribution_share_percent.max())],
        relative_timing_correlations_all18_range=[float(correlations[correlations.predictor.eq(TIMING)].weighted_r.min()),float(correlations[correlations.predictor.eq(TIMING)].weighted_r.max())],
        relative_timing_adjusted_slopes_all18_range=[float(timing_coefs.coefficient.min()),float(timing_coefs.coefficient.max())],
        timing_slopes_negative_members=int(timing_coefs.coefficient.lt(0).sum()),
        timing_slopes_block_bootstrap_upper_below_zero_members=int(timing_coefs.block_bootstrap_upper_975.lt(0).sum()),
        spatial5degree_primary_cv_climate_r2_range=[float(cp.window_climate.min()),float(cp.window_climate.max())],
        spatial5degree_primary_cv_full_r2_range=[float(cp.window_climate_duration_timing.min()),float(cp.window_climate_duration_timing.max())],
        spatial5degree_primary_cv_full_vs_climate_improved_members=int(cp.window_climate_duration_timing.gt(cp.window_climate).sum()),
        late585_less_window_rain_and_increased_loss_pair_area_shares=signshares[signshares.scenario.eq("ssp585") & signshares.period.eq("2071-2100") & signshares.cochange.eq("increased_loss_and_decreased_window_rainfall")][["model", "pair_area_share_percent"]].to_dict("records"))
    # Entire CV sensitivity grid must show whether the conclusion survives.
    sens = cv.pivot(index=["model", "scenario", "period", "spatial_block_degrees", "fold_assignment"], columns="association_model", values="weighted_out_of_block_r2")
    strong["all_spatial_cv_designs_full_vs_climate_improved"] = int(sens.window_climate_duration_timing.gt(sens.window_climate).sum())
    strong["all_spatial_cv_designs_comparisons"] = len(sens)
    support_table = pd.read_csv(data/"timing_matched_support_overall.csv")
    selected_support = support_table[support_table.support.eq("matched_timing")]
    strong["timing_matched_fixed_area_fraction_all18_range"] = [float(selected_support.fixed_area_fraction.min()), float(selected_support.fixed_area_fraction.max())]
    strong["timing_matched_cells_all18_range"] = [int(selected_support.cells.min()), int(selected_support.cells.max())]
    strong["spatial_cv_all_designs_median_r2"] = {name:float(sens[name].median()) for name in sens.columns}
    strong["spatial_cv_all_designs_timing_addition_improved"] = int(sens.window_climate_duration_timing.gt(sens.window_climate_duration).sum())
    (output/"numeric_findings.json").write_text(json.dumps(strong, indent=2)+"\n")
    audit["timing_subset_proof"] = dict(
        kernel="models/climate/kernel.py:simulate_draw_seasons",
        logic="F1 symptom date is NaT whenever the season is invalid or the symptom index is negative. Relative timing is NaN if that date is NaT or the modelled GS65 index is negative. Canopy loss is finite whenever the season is valid and its GS65 start index is nonnegative. Thus finite relative timing implies finite canopy loss within a year; joint reference/future timing-pair validity is a subset of joint canopy-pair validity. The summary uses the same year-index pairing for both metrics. Equal per-cell/member pair counts imply the exact same paired years.",
        scope="Verified against retained frozen kernel and paired-summary source; not inferred from equal counts alone")
    audit["source_files"] = [{"path": str(p.relative_to(ROOT)), "sha256": sha256(p), "bytes": p.stat().st_size}
        for p in [SOURCE/"full_grid_paired_changes_by_gcm.parquet", SOURCE/"full_grid_ensemble_paired_changes.parquet",
                  SOURCE/"environmental_region_changes_by_gcm.csv", ROOT/"data/final_climate/cells.parquet",
                  ROOT/"data/final_climate/scientific_manifest.json", ROOT/"models/climate/kernel.py",
                  ROOT/"models/climate/summary.py", Path(__file__)]]
    audit["outputs"] = [{"path": str(p.relative_to(output)), "sha256": sha256(p), "bytes": p.stat().st_size}
                        for p in sorted(output.rglob("*")) if p.is_file() and p.name != "analysis_audit.json"]
    (output/"analysis_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({"output_directory": str(output), "audit": {k:v for k,v in audit.items() if k not in ["outputs", "source_files"]}, "findings": strong}, indent=2))
    return strong


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=500)
    args = parser.parse_args()
    main(args.output_directory, args.bootstrap_replicates)
