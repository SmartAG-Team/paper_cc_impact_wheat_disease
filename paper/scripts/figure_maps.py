"""Framed, common-support cartography for the GCB Figure 3 export.

The native quarter-degree cell footprints, projection and extent are unchanged.
All dimensions below are physical centimetres at the final 19 cm figure width.
"""
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import numpy as np
from pyproj import Transformer


ROOT = Path(__file__).resolve().parents[2]
SCENARIOS = ("ssp126", "ssp245", "ssp585")
FONT_SIZE = 12
FRAME = "#353b40"
OCEAN = "#edf3f6"
LAND = "#f0efeb"


def build_map_figure(grid):
    """Return Figure 3 from its archived, cell-level plotting data.

    ``grid`` is the published data/fig3/grid.csv DataFrame, including native
    west/east/south/north bounds. No filtering, resampling or land clipping is
    applied. The shared scale is in the EPSG:3035 metre coordinates; its rendered
    length exactly matches a 1,000,000 m horizontal span in every map.
    """
    required = {"cell_id", "scenario", "longitude", "latitude", "west", "east",
                "south", "north", "mean_change"}
    missing = required.difference(grid.columns)
    if missing:
        raise ValueError(f"Missing map data columns: {sorted(missing)}")
    support = grid.loc[grid.scenario.eq(SCENARIOS[0])].copy()
    if support.empty or not support.cell_id.is_unique:
        raise ValueError("Figure 3 requires one row per cell and scenario")
    support = support.sort_values("cell_id").reset_index(drop=True)
    for scenario in SCENARIOS:
        part = grid.loc[grid.scenario.eq(scenario)]
        if not part.cell_id.is_unique or set(part.cell_id) != set(support.cell_id):
            raise ValueError("All three scenarios must use identical cell support")
    if "period" in grid and set(grid.period) != {"2071-2100"}:
        raise ValueError("Figure 3 requires the 2071–2100 period")

    transformer = Transformer.from_crs(4326, 3035, always_xy=True)
    centres_x, centres_y = transformer.transform(
        support.longitude.to_numpy(), support.latitude.to_numpy())
    # Preserve the exact extent used by the archived Figure 3 builder.
    extent = np.array([min(centres_x) - 150000, max(centres_x) + 150000,
                       min(centres_y) - 350000, max(centres_y) + 150000], float)
    inside = ((centres_x > extent[0]) & (centres_x < extent[1]) &
              (centres_y > extent[2]) & (centres_y < extent[3]))
    if not inside.all():
        raise AssertionError("Mapped cell centres fall outside the shared extent")

    # Vectorised projection keeps every native cell polygon and value together.
    lon = support[["west", "east", "east", "west"]].to_numpy(float)
    lat = support[["south", "south", "north", "north"]].to_numpy(float)
    xx, yy = transformer.transform(lon, lat)
    polygons = np.stack([xx, yy], axis=-1)
    country_path = (ROOT / "data/gcb_preparation_20261023/natural_earth/"
                    "ne_110m_admin_0_countries.shp")
    countries = gpd.read_file(country_path).to_crs(3035)

    width_cm, height_cm = 19.0, 8.1
    margin_cm, gap_cm = 0.17, 0.27
    map_width_cm = (width_cm - 2 * margin_cm - 2 * gap_cm) / 3
    map_height_cm = map_width_cm * (extent[3] - extent[2]) / (extent[1] - extent[0])
    map_top_cm = height_cm - 0.72
    map_bottom_cm = map_top_cm - map_height_cm
    style = {
        "font.family": "DejaVu Sans", "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE, "xtick.labelsize": FONT_SIZE,
        "ytick.labelsize": FONT_SIZE, "axes.titlesize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE, "svg.fonttype": "none", "pdf.fonttype": 42,
    }
    with plt.rc_context(style):
        fig = plt.figure(figsize=(width_cm / 2.54, height_cm / 2.54), facecolor="white")
        axes = []
        norm = TwoSlopeNorm(vmin=-12, vcenter=0, vmax=12)
        for i, (panel, scenario) in enumerate(zip("abc", SCENARIOS)):
            left_cm = margin_cm + i * (map_width_cm + gap_cm)
            ax = fig.add_axes([left_cm / width_cm, map_bottom_cm / height_cm,
                               map_width_cm / width_cm, map_height_cm / height_cm])
            axes.append(ax)
            ax.set_facecolor(OCEAN)
            countries.plot(ax=ax, color=LAND, edgecolor="none", zorder=0)
            part = (grid.loc[grid.scenario.eq(scenario)].set_index("cell_id")
                    .loc[support.cell_id])
            collection = PolyCollection(
                polygons, array=np.ma.masked_invalid(part.mean_change.to_numpy(float)),
                cmap="RdBu_r", norm=norm, edgecolors="none", linewidths=0,
                antialiaseds=False, rasterized=True, zorder=2)
            ax.add_collection(collection)
            countries.boundary.plot(ax=ax, color="#747a7e", linewidth=0.28, zorder=3)
            ax.set(xlim=extent[:2], ylim=extent[2:], aspect="equal", xlabel="", ylabel="")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(which="both", length=0)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(FRAME)
                spine.set_linewidth(0.8)
                spine.set_zorder(10)
            ax.text(0.0, 1.035, f"({panel})", transform=ax.transAxes, ha="left",
                    va="bottom", fontsize=FONT_SIZE, color=FRAME, clip_on=False)

        # A dedicated white footer keeps the one scale away from all data.
        # Its axis has the SAME physical width and horizontal limits as a map.
        scale_ax = fig.add_axes([margin_cm / width_cm, 1.10 / height_cm,
                                 map_width_cm / width_cm, 1.1 / height_cm])
        scale_ax.set(xlim=extent[:2], ylim=(0, 1))
        scale_ax.set_axis_off()
        scale_start = extent[0] + 0.18 * (extent[1] - extent[0])
        scale_y, scale_h = 0.52, 0.11
        for segment, color in enumerate((FRAME, "white")):
            scale_ax.add_patch(Rectangle(
                (scale_start + segment * 500000, scale_y), 500000, scale_h,
                facecolor=color, edgecolor=FRAME, linewidth=0.8, clip_on=False))
        for distance in (0, 500000, 1000000):
            scale_ax.plot([scale_start + distance] * 2,
                          [scale_y - 0.035, scale_y + scale_h + 0.035],
                          color=FRAME, linewidth=0.8, clip_on=False)
        scale_ax.annotate("0", (scale_start, scale_y - 0.035),
                          xytext=(-2, -4), textcoords="offset points", ha="right",
                          va="top", fontsize=FONT_SIZE, color=FRAME)
        scale_ax.annotate("1000 km", (scale_start + 1000000, scale_y - 0.035),
                          xytext=(2, -4), textcoords="offset points", ha="left",
                          va="top", fontsize=FONT_SIZE, color=FRAME)

        colorbar_ax = fig.add_axes([0.335, 2.02 / height_cm, 0.605, 0.31 / height_cm])
        bar = fig.colorbar(collection, cax=colorbar_ax, orientation="horizontal",
                           extend="both", extendfrac=0.035, ticks=[-12, -6, 0, 6, 12])
        bar.outline.set_edgecolor(FRAME)
        bar.outline.set_linewidth(0.8)
        bar.ax.tick_params(axis="x", which="both", width=0.8, length=3,
                           pad=3, labelsize=FONT_SIZE, colors=FRAME)
        bar.set_label("Change in normalized affected-tissue\nduration (days)",
                      fontsize=FONT_SIZE, color=FRAME, labelpad=4)

    fig.gcb_map_geometry = {
        "width_cm": width_cm, "height_cm": height_cm, "projection": "EPSG:3035",
        "extent": extent.tolist(), "cells_per_scenario": len(support),
        "cell_centres_inside_extent": int(inside.sum()), "scale_length_m": 1000000,
        "scale_rendered_width_cm": 1000000 / (extent[1] - extent[0]) * map_width_cm,
        "map_width_cm": map_width_cm, "map_height_cm": map_height_cm,
        "map_axes": len(axes), "shared_geographical_scales": 1,
        "shared_colourbars": 1, "font_pt": FONT_SIZE,
        "native_cell_polygons": len(polygons), "colour_limits_days": [-12, 12],
    }
    return fig
