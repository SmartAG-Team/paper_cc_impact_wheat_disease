"""Source-backed Methods schematics at their final publication dimensions.

The two public builders return a Matplotlib figure and a JSON-serializable
diagram specification.  They perform no fitting, input mutation, or file I/O.
Positions in the support schematic are illustrative, not observed dates.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle, Ellipse, PathPatch
from matplotlib.path import Path as MplPath
import numpy as np


FONT = "DejaVu Sans"
FONT_SIZE = 12
INK = "#353B40"
BLUE = "#34769A"
ORANGE = "#B87342"
BLUE_PALE = "#EAF2F6"
ORANGE_PALE = "#F7EEE7"
NEUTRAL_PALE = "#F5F6F6"
GUIDE = "#A9ADB0"
LINE_WIDTH = 0.8

SOURCES = [
    "paper/Manuscript.md",
    "paper/Mathematical_Model_Specification.md",
    "paper/Supplementary_Information.md",
]


def _figure(height_cm):
    """Use physical figure coordinates; never shrink text to fit a node."""
    with plt.rc_context({
        "font.family": FONT,
        "font.size": FONT_SIZE,
        "mathtext.fontset": "dejavusans",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }):
        fig = plt.figure(figsize=(19 / 2.54, height_cm / 2.54), facecolor="white")
    return fig


def _panel(fig, bounds, label):
    ax = fig.add_axes(bounds)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()
    ax.add_patch(Rectangle(
        (0, 0), 1, 1, transform=ax.transAxes, fill=False,
        edgecolor=INK, linewidth=LINE_WIDTH, clip_on=False,
    ))
    _text(ax, 0.013, 0.982, label, ha="left", va="top")
    return ax


def _text(ax, x, y, text, *, color=INK, **kwargs):
    return ax.text(
        x, y, text, fontsize=FONT_SIZE, fontfamily=FONT,
        color=color, linespacing=1.12, zorder=5, **kwargs,
    )


def _node(ax, spec, key, bounds, text, *, fill=NEUTRAL_PALE, edge=INK):
    x, y, w, h = bounds
    ax.add_patch(Rectangle(
        (x, y), w, h, facecolor=fill, edgecolor=edge,
        linewidth=LINE_WIDTH, zorder=2,
    ))
    _text(ax, x + w / 2, y + h / 2, text, ha="center", va="center")
    spec["nodes"].append({
        "id": key, "bounds_axes": list(bounds), "label": text,
        "fill": fill, "edge": edge,
    })


def _arrow(ax, spec, source, target, points, *, color=INK, dashed=False, zorder=1):
    """Draw an orthogonal route with one terminal arrowhead."""
    for start, end in zip(points[:-2], points[1:-1]):
        ax.plot(
            [start[0], end[0]], [start[1], end[1]], color=color,
            linewidth=LINE_WIDTH, linestyle=(0, (3, 2)) if dashed else "-",
            solid_capstyle="butt", zorder=zorder,
        )
    ax.add_patch(FancyArrowPatch(
        points[-2], points[-1], arrowstyle="-|>", mutation_scale=8,
        linewidth=LINE_WIDTH, color=color, shrinkA=0, shrinkB=1.6,
        linestyle=(0, (3, 2)) if dashed else "-", zorder=zorder,
    ))
    spec["edges"].append({
        "source": source, "target": target,
        "route_axes": [list(point) for point in points],
        "style": "dashed check" if dashed else "solid dependency",
    })


def _crossing_gap(ax, x, y, *, height=0.024):
    """A white underpass keeps crossing dependencies visibly unconnected."""
    ax.add_patch(Rectangle(
        (x - 0.006, y - height / 2), 0.012, height,
        facecolor="white", edgecolor="none", zorder=2.5,
    ))


def _spec(name, height):
    return {
        "name": name,
        "kind": "methods schematic",
        "width_cm": 19.0,
        "height_cm": float(height),
        "font_family": FONT,
        "font_pt": FONT_SIZE,
        "line_width_pt": LINE_WIDTH,
        "source_paths": SOURCES.copy(),
        "nodes": [],
        "edges": [],
        "qualitative_positions": True,
    }


def _workflow_icon(ax, centre, kind, colour, halo):
    """Small conceptual glyphs in physical coordinates; no plotted measurements."""
    width,height=ax.figure.get_size_inches()*2.54*ax.get_position().size
    cx,cy=centre;sx=.20/width;sy=.20/height
    def xy(u,v):return cx+u*sx,cy+v*sy
    def line(points,lw=.8):
        q=np.asarray([xy(*v) for v in points]);ax.plot(q[:,0],q[:,1],color=colour,lw=lw,zorder=6,solid_capstyle='round')
    def oval(u,v,rx,ry,fill='none'):
        ax.add_patch(Ellipse(xy(u,v),2*rx*sx,2*ry*sy,facecolor=fill,edgecolor=colour,lw=.75,zorder=6))
    ax.add_patch(Ellipse((cx,cy),.53/width,.53/height,facecolor=halo,edgecolor='none',zorder=4))
    if kind in ['records','calendar','forecast']:
        line([(-.65,-.75),(.65,-.75),(.65,.75),(-.65,.75),(-.65,-.75)])
        if kind=='records':
            for y in [.35,-.05,-.45]:line([(-.35,y),(.35,y)])
        elif kind=='calendar':
            line([(-.65,.3),(.65,.3)]);line([(-.35,.55),(-.35,.95)]);line([(.35,.55),(.35,.95)])
            for x in [-.3,.3]:oval(x,-.2,.06,.06,colour)
        else:
            line([(-.35,-.35),(.35,.35)]);line([(-.1,.35),(.35,.35),(.35,-.1)])
    elif kind=='weather':
        oval(-.15,.25,.45,.45);line([(.1,-.25),(.65,-.25),(.65,.05),(.4,.3)])
        for x in [-.35,.2,.65]:line([(x,-.5),(x-.12,-.85)])
    elif kind=='fit':
        for y,x in [(.5,-.2),(0,.3),(-.5,-.35)]:
            line([(-.7,y),(.7,y)]);oval(x,y,.13,.13,halo)
    elif kind=='query':
        oval(0,0,.7,.7);line([(0,.5),(0,0),(.4,-.25)])
    elif kind=='host':
        line([(0,-.85),(0,.85)])
        for y in [-.4,.05,.5]:
            line([(0,y),(-.5,y+.3),(-.6,y+.6),(0,y+.15)])
            line([(0,y),(.5,y+.3),(.6,y+.6),(0,y+.15)])
    elif kind=='disease':
        vertices=[xy(-.6,-.65),xy(-.85,.55),xy(.5,.95),xy(.7,.7),xy(.8,-.3),xy(.1,-.9),xy(-.6,-.65)]
        ax.add_patch(PathPatch(MplPath(vertices,[MplPath.MOVETO]+[MplPath.CURVE4]*6),fill=False,edgecolor=colour,lw=.8,zorder=6))
        line([(-.5,-.55),(.35,.4)])
        for x,y in [(-.2,.25),(.3,-.2),(.15,.5)]:oval(x,y,.065,.065,colour)
    elif kind=='integral':
        _text(ax,cx,cy,'∫',ha='center',va='center',color=colour)
    elif kind=='outputs':
        for x in [-.7,.1]:
            for y in [-.7,.1]:line([(x,y),(x+.55,y),(x+.55,y+.55),(x,y+.55),(x,y)])
    elif kind=='replay':
        theta=np.linspace(.2,np.pi*1.75,32);line(list(zip(.65*np.cos(theta),.65*np.sin(theta))))
        line([(.3,-.65),(.65,-.45),(.3,-.3)])


def _workflow_card(ax,spec,key,bounds,title,body,icon,role='input'):
    """Quiet rounded cards, one accent and a consistent12pt typographic hierarchy."""
    x,y,w,h=bounds;width,height=ax.figure.get_size_inches()*2.54*ax.get_position().size
    palettes={
        'input':('white','#CADBE5',BLUE,BLUE_PALE,INK),
        'host':('#FCF7F1','#DCC3AA',ORANGE,'#F4E7D7',INK),
        'process':('white','#CFD6DA',INK,'#EEF1F2',INK),
        'output':('#254F66','#254F66','white','#39677F','white'),
        'check':('white','#BBC1C5','#737D83','#F1F3F4',INK),
    }
    fill,edge,accent,halo,body_colour=palettes[role]
    patch=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={.14/width}',
        mutation_aspect=width/height,facecolor=fill,edgecolor=edge,lw=.8,zorder=2)
    ax.add_patch(patch)
    if role!='output':
        stripe=Rectangle((x,y),.065/width,h,facecolor=accent,edgecolor='none',zorder=3)
        stripe.set_clip_path(patch);ax.add_patch(stripe)
    _workflow_icon(ax,(x+.48/width,y+h-.39/height),icon,accent,halo)
    title_colour='#945B2C' if role=='host' else '#5F6A70' if role=='check' else accent
    title_text=_text(ax,x+.88/width,y+h-.39/height,title,color=title_colour,ha='left',va='center',fontweight='bold')
    body_text=_text(ax,x+.32/width,y+h-.80/height,body,color=body_colour,ha='left',va='top')
    spec['nodes'].append(dict(id=key,panel=ax.get_label()[-1],bounds_axes=list(bounds),label=title+'\n'+body,
        title=title,body=body,icon=icon,fill=fill,edge=edge,role=role,title_colour=title_colour))
    if not hasattr(ax,'workflow_card_text'):ax.workflow_card_text=[]
    ax.workflow_card_text.append((bounds,[title_text,body_text]))


def build_workflow():
    """Two visually separated paths with the original scientific dependencies."""
    height=16.5;fig=_figure(height);spec=_spec('study_workflow',height)
    panel_a=[.02,.60,.96,.38];panel_b=[.02,.02,.96,.56]
    a=_panel(fig,panel_a,'(a)');b=_panel(fig,panel_b,'(b)')
    a.set_label('workflow_a');b.set_label('workflow_b')
    for ax,fill in [(a,'#F7FAFC'),(b,'#FAFAF8')]:
        ax.add_patch(Rectangle((0,0),1,1,facecolor=fill,edgecolor='none',zorder=-2))
    spec['panels']={'a':panel_a,'b':panel_b}
    _workflow_card(a,spec,'monitoring_records',(.04,.58,.28,.30),'Dated disease','BASF · Corteva\nUntreated fields','records')
    _workflow_card(a,spec,'weather_routing',(.04,.12,.28,.30),'Weather inputs','Station / reanalysis','weather')
    _workflow_card(a,spec,'historical_statistical_fit',(.39,.58,.31,.30),'Historical fit','Earlier seasons only','fit','process')
    _workflow_card(a,spec,'query_context',(.39,.12,.31,.30),'Forecast day q','Own-field disease < q\nWeather through q − 1','query','process')
    _workflow_card(a,spec,'monitoring_evaluation',(.78,.28,.19,.44),'Forecasts','Severity\nTemporal\nSpatial','forecast','output')
    _arrow(a,spec,'monitoring_records','historical_statistical_fit',[(.32,.73),(.39,.73)],color=BLUE)
    _arrow(a,spec,'weather_routing','query_context',[(.32,.27),(.39,.27)],color=BLUE)
    _arrow(a,spec,'monitoring_records','query_context',[(.32,.73),(.355,.73),(.355,.27),(.39,.27)],color=BLUE)
    _arrow(a,spec,'weather_routing','historical_statistical_fit',[(.32,.27),(.355,.27),(.355,.73),(.39,.73)],color=BLUE)
    a.plot([.355,.355],[.27,.73],ls='none',marker='o',markersize=2.2,color=BLUE,zorder=3)
    _arrow(a,spec,'historical_statistical_fit','monitoring_evaluation',[(.70,.73),(.74,.73),(.74,.56),(.78,.56)])
    _arrow(a,spec,'query_context','monitoring_evaluation',[(.70,.27),(.74,.27),(.74,.44),(.78,.44)])
    _workflow_card(b,spec,'calendar_and_site',(.04,.62,.25,.28),'GGCMI','Sowing date\nSite coordinates','calendar')
    _workflow_card(b,spec,'climate_weather',(.04,.30,.25,.22),'Daily climate','Tmean · Tmax\nRH · precipitation','weather')
    _workflow_card(b,spec,'host_trajectory',(.36,.62,.32,.28),'Host development','Tmean · Tmax\nDaylength · vernalization\nArea · renewal · window','host','host')
    _workflow_card(b,spec,'stb_progression',(.36,.30,.32,.22),'STB progression','Fixed parameters','disease','process')
    _workflow_card(b,spec,'symptom_integral',(.75,.62,.22,.28),'Integral','Symptom tissue\nNormalized days','integral','host')
    _workflow_card(b,spec,'climate_outputs',(.75,.30,.22,.22),'Outputs','Attribution\nExposure','outputs','output')
    _workflow_card(b,spec,'original_f1_replay',(.36,.05,.32,.15),'F1 replay','Retrospective check','replay','check')
    _arrow(b,spec,'climate_weather','host_trajectory',[(.29,.485),(.325,.485),(.325,.70),(.36,.70)],color=BLUE)
    _arrow(b,spec,'calendar_and_site','host_trajectory',[(.29,.77),(.36,.77)],color=BLUE)
    _arrow(b,spec,'climate_weather','stb_progression',[(.29,.41),(.36,.41)],color=BLUE)
    _arrow(b,spec,'host_trajectory','stb_progression',[(.52,.62),(.52,.52)],color=ORANGE)
    _arrow(b,spec,'stb_progression','symptom_integral',[(.68,.41),(.715,.41),(.715,.70),(.75,.70)])
    _arrow(b,spec,'host_trajectory','symptom_integral',[(.68,.76),(.75,.76)],color=ORANGE)
    spec['edges'][-1]['label']='area / window'
    _arrow(b,spec,'symptom_integral','climate_outputs',[(.86,.62),(.86,.52)],color=ORANGE)
    _arrow(b,spec,'stb_progression','original_f1_replay',[(.52,.30),(.52,.20)],color=GUIDE,dashed=True)
    spec['routing_notes']={
        'monitoring_shared_input_rail':{'x_axes':.355,'junctions_axes':[[.355,.27],[.355,.73]],
            'meaning':'both inputs feed historical fitting and query context; no source-to-source dependency'},
        'climate_lanes':'upper host path; lower disease path; direct host-to-integral dependency',
        'icons':'conceptual symbols only; no measured or simulated values',
        'style':'consistent12pt; bold headings; restrained role accents; deep-blue forecast/output cards',
    }

    spec["provenance"] = {
        "monitoring": {
            "provider_versions": ["BASF Zenodo 6521175", "Corteva Zenodo 6390211"],
            "input_population": "dated untreated disease assessments",
            "query_rule": "strictly earlier same-field disease; weather through q-1",
            "withheld": ["all query-day disease", "future disease", "other-field current-season disease"],
            "weather": "retained qualified station/reanalysis routing",
            "statistical_fitting": "earlier seasons; inherited crop-feature ancestry retained",
            "statistical_components": ["process/history prediction", "logit discrepancy"],
            "ancestry_qualification": "the entire upstream crop-feature pipeline was not reconstructed independently in every earlier year",
            "folds": {"temporal": 5, "chronology_plus_50km": 15},
            "evaluation_population": {"records": 2017, "field_seasons": 192, "sites": 121},
            "evidence_role": "retrospective adaptive observation-conditioned monitoring",
        },
        "climate": {
            "daily_weather_variables": ["mean temperature", "maximum temperature", "relative humidity", "precipitation"],
            "management": "GGCMI winter-wheat rainfed sowing calendar and site coordinates",
            "host_weather_inputs": ["mean temperature", "maximum temperature"],
            "host_calendar_site_inputs": ["sowing date", "latitude for daylength"],
            "host_development_controls": ["temperature", "daylength", "vernalization"],
            "host_to_integral_dependency": "host trajectory directly determines tissue area, renewal and integration window",
            "disease_weather_inputs": ["mean temperature", "maximum temperature", "relative humidity", "precipitation"],
            "disease_parameters": "retained fixed climate formulation; unaffected by monitoring scores",
            "monitoring_to_climate_parameter_transfer": False,
            "disease_to_host_feedback": False,
            "integral": "normalized post-latent symptomatic reference tissue, including nonsporulating, infectious and dead states",
            "calibration_or_replay_population": {"original_F1_records": 115, "original_2017_2018_calibration": 45, "original_2019_evaluation": 70},
            "replay_edge": "retrospective endpoint check only; not a newly independent validation or monitoring-to-climate fitting dependency",
            "interpretation": "symptom proxy; neither measured physiological healthy-area loss nor grain-yield loss",
            "no_inferred_inputs": ["cultivar-specific resistance", "satellite measurements"],
        },
    }
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    from matplotlib.transforms import Bbox
    for ax in [a,b]:
        for bounds,texts in ax.workflow_card_text:
            box=ax.transData.transform_bbox(Bbox.from_bounds(*bounds))
            for text in texts:
                q=text.get_window_extent(renderer)
                assert box.contains(q.x0,q.y0) and box.contains(q.x1,q.y1), ('Workflow text outside card',text.get_text())
    spec['design']={'card_corner_radius_cm':.14,'icon_diameter_cm':.53,'icon_data':'conceptual only',
        'header_font_pt':12,'body_font_pt':12,'header_weight':'bold','output_fill':'#254F66',
        'all_card_text_inside_bounds':True,'model_dependencies_unchanged':True}
    fig.gcb_method_spec = spec
    return fig, spec


def build_attribution_design():
    """Return ``(figure, specification)`` for factorial and interval support."""
    height = 13.5
    fig = _figure(height)
    spec = _spec("attribution_design", height)
    panel_a = [0.02, 0.45, 0.96, 0.53]
    panel_b = [0.02, 0.02, 0.96, 0.40]
    a = _panel(fig, panel_a, "(a)")
    b = _panel(fig, panel_b, "(b)")
    spec["panels"] = {"a": panel_a, "b": panel_b}

    # The first loss index is weather (columns), the second host (rows).
    _text(a, 0.4375, 0.862, "Historical weather (0)", ha="center", va="center", color=BLUE)
    _text(a, 0.8025, 0.862, "Future weather (1)", ha="center", va="center", color=BLUE)
    _text(a, 0.13, 0.65, "Historical\nhost (0)", ha="center", va="center", color=ORANGE)
    _text(a, 0.13, 0.37, "Future\nhost (1)", ha="center", va="center", color=ORANGE)
    cells = [
        ("L00", (0.28, 0.53, 0.315, 0.24), "$L_{00}$\nHistorical weather\nHistorical host", 0, 0),
        ("L10", (0.645, 0.53, 0.315, 0.24), "$L_{10}$\nFuture weather\nHistorical host", 1, 0),
        ("L01", (0.28, 0.25, 0.315, 0.24), "$L_{01}$\nHistorical weather\nFuture host", 0, 1),
        ("L11", (0.645, 0.25, 0.315, 0.24), "$L_{11}$\nFuture weather\nFuture host", 1, 1),
    ]
    for key, bounds, text, weather, host in cells:
        _node(a, spec, key, bounds, text,
              fill=BLUE_PALE if host == 0 else ORANGE_PALE)
        spec["nodes"][-1].update(weather_index=weather, host_index=host)
    _text(a, 0.025, 0.175,
          "Host replacement: tissue availability, area, renewal and window timing",
          ha="left", va="center")
    _text(a, 0.025, 0.078,
          "Frozen disease parameters; first index weather, second index host",
          ha="left", va="center")

    # A single elapsed-sowing-time coordinate system, with no pseudo-date ticks.
    historical = [0.36, 0.91]
    future = [0.27, 0.76]
    common = [max(historical[0], future[0]), min(historical[1], future[1])]
    b.add_patch(Rectangle(
        (common[0], 0.425), common[1] - common[0], 0.42,
        facecolor="#E6E8E9", edgecolor="none", zorder=0,
    ))
    _text(b, sum(common) / 2, 0.9, "$C = I_0 \\cap I_1$", ha="center", va="center")
    for y, span, color, label in [
        (0.735, historical, BLUE, "Historical $I_0$"),
        (0.515, future, ORANGE, "Future $I_1$"),
    ]:
        _text(b, 0.025, y, label, ha="left", va="center")
        b.plot(span, [y, y], color=color, linewidth=7, solid_capstyle="butt", zorder=2)
        for x in span:
            b.plot([x, x], [y - 0.065, y + 0.065], color=color,
                   linewidth=LINE_WIDTH, zorder=3)
    _arrow(b, spec, "elapsed_time_origin", "elapsed_time",
           [(0.27, 0.315), (0.94, 0.315)])
    _text(b, 0.605, 0.203, "Elapsed days since sowing", ha="center", va="center")
    _text(b, 0.025, 0.073,
          "Own intervals: $I_0$, $I_1$     Shared time: the same $C$ in all four runs",
          ha="left", va="center")
    spec["factorial"] = {
        "column_variable": "disease weather w", "row_variable": "host trajectory h",
        "historical_index": 0, "future_index": 1,
        "first_index": "weather", "second_index": "host",
        "corners": {"L00": [0, 0], "L10": [1, 0], "L01": [0, 1], "L11": [1, 1]},
        "host_replacement": ["tissue availability", "area", "renewal", "window timing"],
        "disease_parameters": "unchanged across all corners",
        "disease_to_host_feedback": False,
    }
    spec["integration_support"] = {
        "coordinate": "elapsed days since sowing",
        "schematic_units": "unitless display position; no observed or simulated dates",
        "historical_interval_position": historical,
        "future_interval_position": future,
        "common_interval_position": common,
        "whole_window_rule": "each Lwh uses its host-specific interval Ih",
        "shared_time_rule": "all four corners use C = I0 intersect I1 in elapsed time",
        "trajectory_rule": "neither host trajectory is extended beyond its reached window",
        "interpretation": "different integration estimands; shared time does not identify a uniquely correct biological support",
        "numeric_effect_results": False,
    }
    fig.gcb_method_spec = spec
    return fig, spec
