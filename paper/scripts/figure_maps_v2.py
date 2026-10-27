"""Native-cell climate and trial-location error maps at final publication size.

The builders return figures and geometry metadata without writing publication
files. The climate extent is identical to the previous EPSG:3035 cartography;
all source cell footprints and all trial sites are retained without interpolation.
"""
from functools import lru_cache
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap, Normalize, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("ssp126", "ssp245", "ssp585")
EVALUATIONS = ("temporal", "spatial_50km")
FONT_SIZE = 12
FRAME = "#353b40"
OCEAN = "#eaf2f8"
LAND = "#cccccc"
COUNTRY_LINE = "#82878a"
MISSING = "#737373"
LAND_LABEL = "No mapped wheat"
WATER_LABEL = "Water"
AGREEMENT_CATEGORIES = (
    ("all_3_increase", "All 3 increase", "#b2182b"),
    ("all_3_decrease", "All 3 decrease", "#2166ac"),
    ("mixed_or_zero", "Mixed or zero", "#dfbf64"),
    ("unavailable", "Unavailable", MISSING),
)
STYLE = {
    "font.family": "DejaVu Sans", "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE, "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE, "svg.fonttype": "none", "pdf.fonttype": 42,
}


@lru_cache(maxsize=1)
def _countries():
    path = (ROOT / "data/gcb_preparation_20261023/natural_earth/"
            "ne_110m_admin_0_countries.shp")
    return gpd.read_file(path).to_crs(3035)


def _require_columns(frame, required):
    missing = set(required).difference(frame.columns)
    if missing:
        raise ValueError(f"Missing map data columns: {sorted(missing)}")


def _map_axes(fig, bounds_cm, size_cm, extent, panel):
    width_cm, height_cm = size_cm
    ax = fig.add_axes([bounds_cm[0] / width_cm, bounds_cm[1] / height_cm,
                       bounds_cm[2] / width_cm, bounds_cm[3] / height_cm])
    ax.set_facecolor(OCEAN)
    _countries().plot(ax=ax, color=LAND, edgecolor="none", zorder=0)
    ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="", ylabel="")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(which="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(FRAME)
        spine.set_linewidth(0.8)
        spine.set_zorder(10)
    ax.text(0, 1.018, f"({panel})", transform=ax.transAxes, ha="left",
            va="bottom", fontsize=FONT_SIZE, color=FRAME, clip_on=False)
    return ax


def _country_boundaries(ax):
    _countries().boundary.plot(ax=ax, color=COUNTRY_LINE, linewidth=0.28, zorder=3)
    # GeoPandas attaches CRS axis labels even when geographic ticks are hidden.
    ax.set(xlabel="", ylabel="")


def _style_colourbar(bar):
    bar.outline.set_edgecolor(FRAME)
    bar.outline.set_linewidth(0.8)
    bar.ax.tick_params(which="both", width=0.8, length=3, pad=3,
                       labelsize=FONT_SIZE, colors=FRAME)


def _climate_support(grid):
    _require_columns(grid, (
        "cell_id", "scenario", "longitude", "latitude", "west", "east",
        "south", "north", "mean_change", "available_gcms",
        "positive_gcm_count", "negative_gcm_count",
    ))
    if set(grid.scenario) != set(SCENARIOS):
        raise ValueError("Climate map requires exactly SSP1–2.6, SSP2–4.5 and SSP5–8.5")
    support = grid.loc[grid.scenario.eq(SCENARIOS[0])].sort_values("cell_id").copy()
    if support.empty or not support.cell_id.is_unique:
        raise ValueError("Climate map requires one row per cell and scenario")
    bounds = ["longitude", "latitude", "west", "east", "south", "north"]
    for scenario in SCENARIOS:
        part = grid.loc[grid.scenario.eq(scenario)]
        if not part.cell_id.is_unique or set(part.cell_id) != set(support.cell_id):
            raise ValueError("The three scenarios must have identical unique cell support")
        part = part.set_index("cell_id").loc[support.cell_id]
        if not np.array_equal(part[bounds].to_numpy(float), support[bounds].to_numpy(float)):
            raise ValueError("Cell coordinates and native bounds differ among scenarios")
    if "period" in grid and set(grid.period) != {"2071-2100"}:
        raise ValueError("Climate map requires the 2071–2100 period")
    if not np.isfinite(support[bounds].to_numpy(float)).all():
        raise ValueError("Climate map coordinates and native bounds must be finite")
    if not ((support.east > support.west) & (support.north > support.south)).all():
        raise ValueError("Climate map contains invalid native cell bounds")
    counts = grid[["available_gcms", "positive_gcm_count", "negative_gcm_count"]].to_numpy(float)
    if not np.isfinite(counts).all() or (counts < 0).any() or (counts > 3).any():
        raise ValueError("Climate member counts must be finite integers from zero to three")
    if not np.equal(counts, np.floor(counts)).all() or (counts[:, 1:].sum(axis=1) > counts[:, 0]).any():
        raise ValueError("Positive and negative member counts exceed available members")
    return support.reset_index(drop=True)


def _agreement_frame(grid, support):
    agreement = (grid.loc[grid.scenario.eq("ssp585")].set_index("cell_id")
                 .loc[support.cell_id].reset_index())
    category = np.full(len(agreement), "unavailable", dtype=object)
    complete = agreement.available_gcms.eq(3) & np.isfinite(agreement.mean_change)
    category[complete] = "mixed_or_zero"
    category[complete & agreement.positive_gcm_count.eq(3)] = "all_3_increase"
    category[complete & agreement.negative_gcm_count.eq(3)] = "all_3_decrease"
    agreement["zero_gcm_count"] = (agreement.available_gcms -
                                   agreement.positive_gcm_count - agreement.negative_gcm_count)
    agreement["agreement_category"] = category
    code = {entry[0]: i for i, entry in enumerate(AGREEMENT_CATEGORIES)}
    label = {entry[0]: entry[1] for entry in AGREEMENT_CATEGORIES}
    agreement["agreement_code"] = agreement.agreement_category.map(code)
    agreement["agreement_label"] = agreement.agreement_category.map(label)
    agreement["unit"] = "categorical three-member direction agreement"
    agreement["mean_change_unit"] = "days"
    agreement["count_unit"] = "climate models"
    return agreement


def build_climate_map(grid):
    """Return ``(figure, SSP5–8.5 cell agreement, geometry)`` for four maps.

    Panels (a–c) retain native scenario means, with one shared ±12-day colour
    scale. Panel (d) reports full three-member direction agreement rather than
    a probability. Fewer than three available members, or an unavailable mean,
    produces the unavailable category; any zero or opposing signs among all
    three available members produces mixed-or-zero. No geographic scale is drawn.
    """
    support = _climate_support(grid)
    transformer = Transformer.from_crs(4326, 3035, always_xy=True)
    centres_x, centres_y = transformer.transform(
        support.longitude.to_numpy(float), support.latitude.to_numpy(float))
    # Preserve the previous cartographic extent exactly, including its southern
    # context, while using the full native footprint of every retained cell.
    extent = np.array([min(centres_x) - 150000, max(centres_x) + 150000,
                       min(centres_y) - 350000, max(centres_y) + 150000], float)
    lon = support[["west", "east", "east", "west"]].to_numpy(float)
    lat = support[["south", "south", "north", "north"]].to_numpy(float)
    xx, yy = transformer.transform(lon, lat)
    polygons = np.stack([xx, yy], axis=-1)
    centres_inside = ((centres_x > extent[0]) & (centres_x < extent[1]) &
                      (centres_y > extent[2]) & (centres_y < extent[3]))
    footprints_inside = ((xx >= extent[0]) & (xx <= extent[1]) &
                         (yy >= extent[2]) & (yy <= extent[3])).all(axis=1)
    if not centres_inside.all() or not footprints_inside.all():
        raise AssertionError("A native climate cell falls outside the preserved extent")
    agreement = _agreement_frame(grid, support)

    width_cm, height_cm = 19.0, 18.0
    margin_cm, gap_cm, row_gap_cm = 0.28, 0.46, 0.72
    map_width_cm = (width_cm - 2 * margin_cm - gap_cm) / 2
    map_height_cm = map_width_cm * (extent[3] - extent[2]) / (extent[1] - extent[0])
    bottom_cm = 2.25
    map_positions = [
        (margin_cm, bottom_cm + map_height_cm + row_gap_cm),
        (margin_cm + map_width_cm + gap_cm, bottom_cm + map_height_cm + row_gap_cm),
        (margin_cm, bottom_cm),
        (margin_cm + map_width_cm + gap_cm, bottom_cm),
    ]
    norm = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(MISSING)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), facecolor="white")
        axes = []
        for panel, scenario, position in zip("abc", SCENARIOS, map_positions):
            ax = _map_axes(fig, (*position, map_width_cm, map_height_cm),
                           (width_cm, height_cm), extent, panel)
            part = (grid.loc[grid.scenario.eq(scenario)].set_index("cell_id")
                    .loc[support.cell_id])
            values = np.ma.masked_invalid(part.mean_change.to_numpy(float))
            collection = PolyCollection(
                polygons, array=values, cmap=cmap, norm=norm, edgecolors="none",
                linewidths=0, antialiaseds=False, rasterized=True, zorder=2)
            ax.add_collection(collection)
            _country_boundaries(ax)
            axes.append(ax)

        ax = _map_axes(fig, (*map_positions[3], map_width_cm, map_height_cm),
                       (width_cm, height_cm), extent, "d")
        category_cmap = ListedColormap([entry[2] for entry in AGREEMENT_CATEGORIES])
        category_norm = BoundaryNorm(np.arange(5) - 0.5, 4)
        ax.add_collection(PolyCollection(
            polygons, array=agreement.agreement_code.to_numpy(float),
            cmap=category_cmap, norm=category_norm, edgecolors="none", linewidths=0,
            antialiaseds=False, rasterized=True, zorder=2))
        _country_boundaries(ax)
        axes.append(ax)

        # Parallel footers keep quantitative and categorical keys outside the
        # four maps, without reducing the map panels or the 12-point type.
        colourbar_ax = fig.add_axes([0.52 / width_cm, 1.81 / height_cm,
                                     8.38 / width_cm, 0.26 / height_cm])
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colourbar_ax,
                           orientation="horizontal", extend="both",
                           extendfrac=0.035, ticks=[-12, -6, 0, 6, 12])
        _style_colourbar(bar)
        bar.set_label("Change in normalized affected-tissue\nduration (days)",
                      fontsize=FONT_SIZE, color=FRAME, labelpad=4)
        bar.ax.xaxis.label.set_linespacing(1.0)
        # Matplotlib fills legend columns first. This ordering presents the
        # increasing/decreasing categories together on the first visual row.
        def category_key(index):
            return Patch(facecolor=AGREEMENT_CATEGORIES[index][2],edgecolor=FRAME,
                linewidth=.45,label=AGREEMENT_CATEGORIES[index][1])
        # Fill the two columns to read by rows: increase/decrease,
        # mixed/unavailable, then the land-use and water backgrounds.
        handles = [category_key(0),category_key(2),
            Patch(facecolor=LAND,edgecolor=FRAME,linewidth=.45,label=LAND_LABEL),
            category_key(1),category_key(3),
            Patch(facecolor=OCEAN,edgecolor=FRAME,linewidth=.45,label=WATER_LABEL)]
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.754, 0.065),
                   ncol=2, frameon=False, fontsize=FONT_SIZE, columnspacing=0.8,
                   handlelength=0.9, handleheight=0.85, handletextpad=0.4,
                   labelspacing=0.65, borderaxespad=0)

    geometry = {
        "width_cm": width_cm, "height_cm": height_cm, "projection": "EPSG:3035",
        "extent": extent.tolist(), "extent_preserved_from_previous_map": True,
        "cells_per_scenario": len(support), "native_cell_polygons": len(polygons),
        "cell_centres_inside_extent": int(centres_inside.sum()),
        "cell_footprints_inside_extent": int(footprints_inside.sum()),
        "map_width_cm": float(map_width_cm), "map_height_cm": float(map_height_cm),
        "map_axes": len(axes), "shared_geographical_scales": 0,
        "shared_colourbars": 1, "categorical_legends": 1, "font_pt": FONT_SIZE,
        "frame_linewidth_pt": 0.8, "colour_limits_days": [-12, 12],
        "panel_roles": {"a": "ssp126_mean_change", "b": "ssp245_mean_change",
                        "c": "ssp585_mean_change", "d": "ssp585_direction_agreement"},
        "agreement_scenario": "ssp585", "agreement_required_members": 3,
        "agreement_category_counts": {entry[0]: int(agreement.agreement_category.eq(entry[0]).sum())
                                      for entry in AGREEMENT_CATEGORIES},
        "agreement_is_probability": False, "interpolation": "none",
        "background_legend": {LAND_LABEL:LAND,WATER_LABEL:OCEAN,"Unavailable":MISSING},
        "background_scope": "Land outside mapped wheat support; no assertion about other crops",
        "zero_change_colour": "white midpoint within wheat cells",
    }
    fig.gcb_map_geometry = geometry
    return fig, agreement, geometry


def _site_support(site_metrics):
    _require_columns(site_metrics, (
        "evaluation", "site_id", "latitude", "longitude", "n", "rmse", "bias", "weight_sum",
    ))
    if set(site_metrics.evaluation) != set(EVALUATIONS):
        raise ValueError("Error map requires temporal and spatial_50km designs")
    if site_metrics.duplicated(["evaluation", "site_id"]).any():
        raise ValueError("Error map requires one row per site and evaluation design")
    if site_metrics.site_id.isna().any():
        raise ValueError("Error map requires nonmissing site identifiers")
    numeric = site_metrics[["latitude", "longitude", "n", "rmse", "bias", "weight_sum"]].to_numpy(float)
    if not np.isfinite(numeric).all():
        raise ValueError("Trial coordinates, counts and error metrics must be finite; no sites are dropped")
    if not site_metrics.latitude.between(-90, 90).all() or not site_metrics.longitude.between(-180, 180).all():
        raise ValueError("Trial coordinates must be valid geographic latitude and longitude")
    if ((site_metrics.n < 1) | (site_metrics.rmse < 0) | (site_metrics.weight_sum <= 0)).any():
        raise ValueError("Site counts and hierarchy mass must be positive and RMSE nonnegative")
    if not np.equal(site_metrics.n, np.floor(site_metrics.n)).all():
        raise ValueError("Site assessment counts must be integers")
    temporal = site_metrics.loc[site_metrics.evaluation.eq("temporal")].set_index("site_id")
    spatial = site_metrics.loc[site_metrics.evaluation.eq("spatial_50km")].set_index("site_id")
    if set(temporal.index) != set(spatial.index):
        raise ValueError("The two error-map designs must retain identical trial-site support")
    if not np.allclose(temporal[["latitude", "longitude"]],
                       spatial.loc[temporal.index, ["latitude", "longitude"]],
                       atol=1e-8, rtol=0):
        raise ValueError("Coordinates for a trial site differ between designs")
    return site_metrics.copy()


def build_error_map(site_metrics, *, colour_limit=None):
    """Return ``(figure, geometry)`` for site RMSE in the two held-out designs.

    Real trial coordinates are projected into EPSG:3035. Filled circles all
    have the same physical size and share a single viridis RMSE scale in
    percentage points. Sites with fewer than five assessments receive an
    additional hollow black ring. No sites are filtered and no values are
    interpolated; bias, counts and hierarchy mass remain in the source frame.
    """
    sites = _site_support(site_metrics)
    transformer = Transformer.from_crs(4326, 3035, always_xy=True)
    xx, yy = transformer.transform(sites.longitude.to_numpy(float), sites.latitude.to_numpy(float))
    sites["map_x_m"], sites["map_y_m"] = xx, yy
    width_cm, height_cm = 19.0, 9.5
    margin_cm, gap_cm = 0.28, 0.46
    map_width_cm = (width_cm - 2 * margin_cm - gap_cm) / 2
    map_height_cm = 6.68
    # One common aspect-matched extent preserves the full physical map size.
    # Padding and aspect adjustment enlarge only the geographic context.
    dx, dy = float(np.ptp(xx)), float(np.ptp(yy))
    span_x = dx + 2 * max(120000.0, 0.055 * dx)
    span_y = dy + 2 * max(120000.0, 0.075 * dy)
    aspect = map_height_cm / map_width_cm
    span_x = max(span_x, span_y / aspect)
    span_y = span_x * aspect
    middle_x, middle_y = (float(min(xx) + max(xx)) / 2,
                          float(min(yy) + max(yy)) / 2)
    extent = np.array([middle_x - span_x / 2, middle_x + span_x / 2,
                       middle_y - span_y / 2, middle_y + span_y / 2])
    inside = ((xx > extent[0]) & (xx < extent[1]) & (yy > extent[2]) & (yy < extent[3]))
    if not inside.all():
        raise AssertionError("A trial location falls outside the shared error-map extent")
    maximum = float(sites.rmse.max())
    # Round upward to a readable common limit while retaining every RMSE.
    colour_max = max(10.0, float(np.ceil(maximum / 10) * 10)) if colour_limit is None else float(colour_limit)
    if not np.isfinite(colour_max) or colour_max <= 0:
        raise ValueError('Colour limit must be finite and positive')
    overflow = maximum > colour_max
    norm = Normalize(vmin=0, vmax=colour_max, clip=False)
    cmap = plt.get_cmap('viridis').copy()
    over_colour = '#B2182B'
    cmap.set_over(over_colour)
    marker_size_pt2, small_n_ring_pt2 = 21.0, 36.0
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), facecolor="white")
        axes = []
        for i, (panel, evaluation) in enumerate(zip("ab", EVALUATIONS)):
            ax = _map_axes(fig, (margin_cm + i * (map_width_cm + gap_cm), 2.12,
                                map_width_cm, map_height_cm),
                           (width_cm, height_cm), extent, panel)
            part = sites.loc[sites.evaluation.eq(evaluation)].sort_values(
                ["rmse", "site_id"], kind="stable")
            _country_boundaries(ax)
            ax.scatter(part.map_x_m, part.map_y_m, c=part.rmse, cmap=cmap, norm=norm,
                       s=marker_size_pt2, marker="o", linewidths=0.28,
                       edgecolors="#ffffff", alpha=1, zorder=5, rasterized=True)
            small = part.loc[part.n.lt(5)]
            ax.scatter(small.map_x_m, small.map_y_m, s=small_n_ring_pt2,
                       marker="o", facecolors="none", edgecolors="#14191c",
                       linewidths=0.65, zorder=6, rasterized=True)
            axes.append(ax)
        colourbar_ax = fig.add_axes([0.78 / width_cm, 1.51 / height_cm,
                                     10.8 / width_cm, 0.29 / height_cm])
        ticks = np.array([0,5,10,15]) if colour_max==15 else np.linspace(0, colour_max, 5)
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=colourbar_ax,
                           orientation="horizontal", ticks=ticks, format="%g",
                           extend='max' if overflow else 'neither', extendfrac=.045)
        _style_colourbar(bar)
        bar.set_label("Site RMSE (percentage points)", fontsize=FONT_SIZE,
                      color=FRAME, labelpad=4)
        handle = Line2D([], [], color="#14191c", marker="o", linestyle="none",
                        markerfacecolor="none", markeredgewidth=0.65,
                        markersize=np.sqrt(small_n_ring_pt2), label="Fewer than 5\nassessments")
        handles = ([Patch(facecolor=over_colour, edgecolor=FRAME,
                    label=f'RMSE > {colour_max:g} pp')] if overflow else []) + [handle]
        fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.80, 0.104 if overflow else 0.136),
                   frameon=False, fontsize=FONT_SIZE, handlelength=0.8,
                   handletextpad=0.5, borderaxespad=0)

    geometry = {
        "width_cm": width_cm, "height_cm": height_cm, "projection": "EPSG:3035",
        "extent": extent.tolist(), "map_width_cm": map_width_cm,
        "map_height_cm": map_height_cm, "map_axes": len(axes),
        "site_centres_inside_extent": int(inside.sum()),
        "sites_per_design": {evaluation: int(sites.evaluation.eq(evaluation).sum())
                             for evaluation in EVALUATIONS},
        "small_n_sites_per_design": {
            evaluation: int((sites.evaluation.eq(evaluation) & sites.n.lt(5)).sum())
            for evaluation in EVALUATIONS},
        "small_n_threshold": 5, "small_n_definition": "assessment count n < 5",
        "marker_area_pt2": marker_size_pt2, "small_n_ring_area_pt2": small_n_ring_pt2,
        "all_sites_retained": True, "interpolation": "none",
        "shared_geographical_scales": 0, "shared_colourbars": 1,
        "font_pt": FONT_SIZE, "frame_linewidth_pt": 0.8,
        "colour_metric": "rmse",
        "point_draw_order": "ascending RMSE, then site_id; high errors remain visible where sites overlap", "colour_unit": "percentage points",
        "colour_limits": [0.0, colour_max],
        "above_colour_limit_counts": {
            evaluation: int((sites.evaluation.eq(evaluation) & sites.rmse.gt(colour_max)).sum())
            for evaluation in EVALUATIONS},
        "maximum_rmse": maximum, "over_colour": over_colour if overflow else None,
        "colourbar_extend": 'max' if overflow else 'neither',
        "values_clipped_or_removed": False,
        "panel_roles": {"a": "temporal", "b": "spatial_50km"},
    }
    fig.gcb_map_geometry = geometry
    return fig, geometry
