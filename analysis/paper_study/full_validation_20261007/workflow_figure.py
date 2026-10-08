"""Editable vector research workflow for the wheat/STB manuscript."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
from matplotlib.path import Path as MplPath


def render_workflow(destination, export):
    fig,ax=plt.subplots(figsize=(9.4,6.15))
    fig.subplots_adjust(left=.012,right=.995,bottom=.018,top=.985)
    ax.set(xlim=(0,14.1),ylim=(0,8.8));ax.axis('off')
    ink='#243039';grey='#63717b';green='#426d5b';gold='#a17937';blue='#275b7d'
    cards=[]

    def card(key,x,y,w,h,title,lines,color,fill,text_x=None):
        box=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.015,rounding_size=.065',
            linewidth=.7,edgecolor='#cfd7db',facecolor=fill,zorder=2)
        ax.add_patch(box)
        ax.plot([x+.03,x+.03],[y+.16,y+h-.16],color=color,linewidth=2.,solid_capstyle='round',zorder=3)
        tx=x+.20 if text_x is None else text_x
        header=ax.text(tx,y+h-.25,title,fontsize=10.1,color=ink,weight='bold',ha='left',va='top',zorder=4)
        body=ax.text(tx,y+h-.68,'\n'.join(lines),fontsize=9.2,color='#49545b',ha='left',va='top',
            linespacing=1.55,zorder=4)
        cards.append((key,box,header,body))

    def arrow(start,end,color=grey,dashed=False,via=None):
        if via:
            points=[start,*via,end]
            ax.plot([p[0] for p in points[:-1]],[p[1] for p in points[:-1]],color=color,
                linewidth=.95,zorder=1)
            start=points[-2]
        ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=9.,
            linewidth=.95,color=color,linestyle='--' if dashed else '-',zorder=1,
            shrinkA=0,shrinkB=0))

    for letter,x,title in [('a',.25,'Forcing and assumptions'),('b',4.25,'Coupled seasonal processes'),
            ('c',10.25,'Evaluation and impact')]:
        ax.text(x,8.5,letter,fontsize=12.,weight='bold',color=ink,va='top')
        ax.text(x+.33,8.48,title,fontsize=10.5,weight='bold',color=ink,va='top')
        ax.plot([x,x+(3.10 if letter!='b' else 5.25)],[8.06,8.06],color='#d9dfe2',linewidth=.8)

    card('weather',.25,6.37,3.10,1.32,'Weather and sowing',
        ['Temperature and humidity','Rain, latitude and sowing'],grey,'#f5f7f8')
    card('scenarios',.25,4.02,3.10,1.81,'Canopy and sources',
        ['Local overwintering source','Imported background','Nominal leaf-area capacity'],grey,'#f5f7f8')
    card('literature',.25,1.48,3.10,1.69,'Yield-response data',
        ['STB and wheat-rust evidence','Leaf area, duration and yield'],grey,'#f5f7f8')
    card('phenology',4.25,6.37,5.25,1.32,'Crop and leaf development',
        ['TPV clock → BBCH stage events','Leaf appearance and unfolding'],green,'#f0f6f2')
    card('disease',4.25,4.02,5.25,1.81,'Seasonal disease development',
        ['Overwintering sources and airborne input','Latent development and symptom appearance',
         'Secondary cycles on available leaves'],gold,'#faf6ed')
    card('upper_leaf',4.25,1.48,5.25,1.69,'Upper-leaf exposure',
        ['F1–F3 during grain filling','Healthy-area-duration deficit','Conditional yield transfer'],green,'#f0f6f2',text_x=5.64)
    card('validation',10.25,5.92,3.55,1.77,'Observational evaluation',
        ['German BBCH station holdouts','BASF/Corteva stage and disease','Published canopy–yield pairs'],blue,'#f0f5f9')
    card('climate',10.25,3.47,3.55,1.81,'European climate scenarios',
        ['3 climate models × 3 SSPs','Baseline and future periods','51,840 spatial draw-seasons'],blue,'#f0f5f9')
    card('outputs',10.25,1.48,3.55,1.36,'Crop-protection endpoints',
        ['Occurrence and timing on F1–F3','Grain fill and yield transfer'],blue,'#f0f5f9')

    # Connectors occupy the gutters; no arrow runs through a text card.
    arrow((3.38,7.10),(4.22,7.10),green)
    arrow((3.75,7.10),(4.22,5.45),gold,via=[(3.75,5.45)])
    arrow((3.38,4.72),(4.22,4.72),gold)
    arrow((6.87,6.34),(6.87,5.86),green)
    arrow((6.87,3.99),(6.87,3.20),green)
    arrow((3.38,2.37),(4.22,2.37),grey)
    arrow((9.53,7.10),(10.22,7.10),blue)
    arrow((9.53,5.25),(10.22,6.22),blue,via=[(9.86,5.25),(9.86,6.22)])
    arrow((12.02,5.89),(12.02,5.31),blue)
    ax.text(12.19,5.59,'Frozen model',fontsize=8.3,color=blue,va='center')
    arrow((9.53,4.49),(10.22,4.49),blue)
    arrow((12.02,3.44),(12.02,2.87),blue)
    arrow((9.53,2.37),(10.22,2.37),blue)

    # A schematic wheat stem identifies the three target leaves without
    # implying an observed total leaf number or measured canopy area.
    leaf_patches=[];leaf_labels=[]
    stem_x=4.87
    ax.plot([stem_x,stem_x],[1.70,2.89],color=green,linewidth=1.05,zorder=4)
    for rank,base,direction in [(1,2.69,1),(2,2.31,-1),(3,1.97,1)]:
        tip_x=stem_x+direction*.49;tip_y=base+.27
        verts=[(stem_x,base),(stem_x+direction*.15,base+.36),(tip_x,tip_y),
            (stem_x+direction*.21,base-.04),(stem_x,base),(stem_x,base)]
        path=MplPath(verts,[MplPath.MOVETO,MplPath.CURVE3,MplPath.CURVE3,
            MplPath.CURVE3,MplPath.CURVE3,MplPath.CLOSEPOLY])
        patch=PathPatch(path,facecolor='#b9d2c1',edgecolor=green,linewidth=.8,zorder=4)
        ax.add_patch(patch);leaf_patches.append(patch)
        label_x,label_y = {1:(4.97,base+.30),2:(4.39,base+.40),3:(5.07,base-.24)}[rank]
        leaf_labels.append(ax.text(label_x,label_y,f'F{rank}',fontsize=7.3,color=green,ha='left',va='bottom',zorder=4))

    ax.text(.25,.65,'Stage observations constrain development; disease signs constrain visible expression.',
        fontsize=9.,color=grey,va='center')
    ax.text(.25,.29,'Yield transfer is conditional on healthy-area and tolerance assumptions.',
        fontsize=9.,color=grey,va='center')
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    import numpy as np
    for label in leaf_labels:
        bb=label.get_window_extent(renderer)
        xx,yy=np.meshgrid(np.linspace(bb.x0,bb.x1,12),np.linspace(bb.y0,bb.y1,12))
        points=np.column_stack([xx.ravel(),yy.ravel()])
        for patch in leaf_patches:
            outline=patch.get_path().transformed(patch.get_transform())
            assert not outline.contains_points(points).any(),label.get_text()
    bounds=[]
    for key,box,header,body in cards:
        area=box.get_window_extent(renderer)
        for role,text in [('title',header),('body',body)]:
            bb=text.get_window_extent(renderer)
            assert area.x0<=bb.x0 and bb.x1<=area.x1 and area.y0<=bb.y0 and bb.y1<=area.y1,(key,role)
        bounds.append(dict(node=key,text_within_card=True))
    destination=Path(destination)
    (destination/'fig1_workflow_layout_check.json').write_text(json.dumps(dict(status='passed',nodes=bounds,
        leaf_labels_do_not_overlap_leaf_shapes=True,vector_exports=True,field_sources_and_assumptions_labelled=True),indent=2)+'\n')
    export(fig,destination,'fig1_framework')
    return ('Figure 1 | Evidence-to-impact workflow for seasonal wheat Septoria simulation. '
        '(a) Weather, management, source/canopy assumptions and published STB, yellow-rust and leaf-rust yield-response evidence. '
        '(b) Coupled crop development, continuity of local residue and living-canopy sources, imported pressure, '
        'leaf disease development and top-three-leaf exposure. The leaf schematic identifies F1–F3; it does not represent '
        'measured canopy area or an observed total leaf count. '
        '(c) Outcome-specific evaluation and frozen-model climate scenarios. German station holdouts concern retained phenology; '
        'BASF/Corteva stages and symptom records constrain added development and disease timing. French results retain their '
        'archived model and conditioned-transfer identity. Published rust evidence concerns crop response; epidemic simulation remains STB-specific. Calibration is external to the pure simulator. '
        'Three climate models, three SSPs, three 30-year periods and64 spatial draws produce51,840 draw-season simulations. '
        'Yield transfer remains conditional on functional green-area and tolerance assumptions.')
