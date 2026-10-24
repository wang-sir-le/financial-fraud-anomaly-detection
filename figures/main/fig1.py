"""Draw the adopted Figure 1 with named feature, metric and saved-output icons.

Usage: python fig1.py [--out-dir PATH] [--font-family FAMILY]
Requires Python 3.10+, Matplotlib and adjacent data/fig1_protocol.json.
The protocol is authenticated before drawing. No manuscript, author workspace,
model fitting, prediction, resampling, network service or private dependency is
used. Physical output width is exactly 170 mm, with minimum 8 pt lettering.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ft2font import FT2Font
from matplotlib.patches import Arc, Circle, Ellipse, FancyArrowPatch, FancyBboxPatch, PathPatch, Polygon, Rectangle
from matplotlib.path import Path as MPath

PROTOCOL_HASH = "3e5863cd422e6a6263d616e9e3b7df6d5f4ce10278023691619f41a8293c8c7a"

def resolve_font(requested: str | None) -> tuple[str, str]:
    """Prefer installed Arial, then the metric-compatible Liberation Sans."""
    candidates = [requested] if requested else ["Arial", "Liberation Sans"]
    for family in candidates:
        try:
            filename = font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        return str(family), filename
    raise RuntimeError("No suitable installed font found. Install Arial or Liberation Sans, or use --font-family; glyph and geometry checks still apply.")

def draw(P: dict, DEST: Path, font_family: str, font_path: str) -> dict:
    DEST.mkdir(parents=True, exist_ok=True)
    W, H = 174.0, 118.0
    INK, MUTED, EDGE = '#152126', '#37484E', '#57666A'
    PAPER, PANEL = '#FFFFFF', '#FCFCFA'
    TRAIN, VAL, TEST = '#B7CEE0', '#C9DDC6', '#F2B8AB'
    BLUE, CORAL, SLATE, GREEN, LILAC = '#DFEDF0', '#F9DFD6', '#E2E9EE', '#E1EDDE', '#EDE1EF'
    plt.rcParams.update({
        'font.family': font_family, 'font.size': 8.0, 'axes.unicode_minus': True,
        'svg.hashsalt': 'kais-fig1-meaningful-data-icons-20261024',
        'svg.fonttype': 'path', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'savefig.facecolor': PAPER, 'hatch.linewidth': 0.4,
    })
    physical_w = 170.0
    physical_h = physical_w * H / W
    coordinate_to_mm = physical_w / W
    fig = plt.figure(figsize=(physical_w / 25.4, physical_h / 25.4), facecolor=PAPER)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, W), ylim=(H, 0), aspect='equal')
    ax.set_axis_off()
    labels, regions, strokes = [], [], []

    def text(x, y, value, size=8, bold=False, align='left', color=INK, region=None):
        t = ax.text(x, y, value, fontsize=size, fontweight='bold' if bold else 'normal',
                    color=color, ha=align, va='center', zorder=10, linespacing=1.16)
        labels.append((t, region))
        return t

    def rect(x, y, w, h, fill='white', edge=EDGE, lw=.5, r=0, z=1, dash=None):
        if r:
            p = FancyBboxPatch((x,y), w,h, boxstyle=f'round,pad=0,rounding_size={r}',
                              facecolor=fill, edgecolor=edge if lw else 'none', linewidth=lw, zorder=z)
        else:
            p = Rectangle((x,y), w,h, facecolor=fill, edgecolor=edge if lw else 'none', linewidth=lw, zorder=z)
        if dash: p.set_linestyle((0,dash))
        ax.add_patch(p)
        if lw: strokes.append(lw)
        return p

    def line(x1,y1,x2,y2, color=EDGE, lw=.5, dash=None, z=3):
        ln, = ax.plot([x1,x2], [y1,y2], color=color, lw=lw, zorder=z, solid_capstyle='round')
        if dash: ln.set_linestyle((0,dash))
        strokes.append(lw)

    def arrow(x1,y1,x2,y2, color=INK, lw=.65, head=6):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2), arrowstyle='->', color=color,
                                   lw=lw, mutation_scale=head, shrinkA=0, shrinkB=0, zorder=7))
        strokes.append(lw)

    def circle(x,y,r,fill='white',edge=EDGE,lw=.5,z=4):
        ax.add_patch(Circle((x,y),r,facecolor=fill,edgecolor=edge,lw=lw,zorder=z))
        strokes.append(lw)

    def ellipse(x,y,w,h,fill='white',edge=EDGE,lw=.5,z=4):
        ax.add_patch(Ellipse((x,y),w,h,facecolor=fill,edgecolor=edge,lw=lw,zorder=z))
        strokes.append(lw)

    def outline_path(vertices, codes, fill='none', color=EDGE, lw=.55, z=4):
        ax.add_patch(PathPatch(MPath(vertices,codes),facecolor=fill,edgecolor=color,linewidth=lw,
                              joinstyle='round',capstyle='round',zorder=z))
        strokes.append(lw)

    def band(letter, title, y, height):
        rect(2.2,y,169.6,height,fill=PANEL,edge=INK,lw=.6,r=6.5,dash=(3.1,2.4))
        text(1.5,y,letter,size=11,bold=True)
        t = text(6.5,y,title,size=10,bold=True)
        t.set_bbox(dict(facecolor=PAPER,edgecolor='none',pad=2.0))

    def database(x,y,w=7.2,h=9):
        rect(x-w/2,y-h/2,w,h-1.25,fill='#F7F8F5',edge=INK,lw=.55,z=4)
        for yy in (y+h/2-1.25,y+.5,y-2):
            ax.add_patch(Arc((x,yy),w,2.5,theta1=0,theta2=180,color=INK,lw=.55,zorder=5))
        ellipse(x,y-h/2,w,2.5,fill='#F7F8F5',edge=INK,lw=.55,z=5)

    def chevron(x,y,w,h,color):
        ax.add_patch(Polygon([(x,y-h/2),(x+w-1,y-h/2),(x+w,y),(x+w-1,y+h/2),
                              (x,y+h/2),(x+1,y)],closed=True,facecolor=color,edgecolor=EDGE,lw=.5,zorder=3))
        strokes.append(.5)

    def trees(cx,cy,s=1):
        # Branching trees indicate the fitted boosting models, avoiding NN imagery.
        for dx,dy in ((-3.7,-.8),(3.7,.8)):
            x,y=cx+dx*s,cy+dy*s
            line(x,y-2.8*s,x-2.0*s,y,lw=.5)
            line(x,y-2.8*s,x+2.0*s,y,lw=.5)
            line(x-2.0*s,y,x-3.0*s,y+2.4*s,lw=.5)
            line(x-2.0*s,y,x-1.0*s,y+2.4*s,lw=.5)
            for xx,yy in ((x,y-2.8*s),(x-2*s,y),(x+2*s,y),(x-3*s,y+2.4*s),(x-1*s,y+2.4*s)):
                circle(xx,yy,.66*s,fill='#C9CBBE',lw=.45)

    def folder(cx,cy):
        x,y=cx-6.3,cy-3.7
        outline_path([(x,y+1),(x+.6,y),(x+3.5,y),(x+4.3,y+1),(x+9.1,y+1),
                      (x+9.8,y+2),(x+9.3,y+6.8),(x,y+6.8),(x,y+1)],
                     [MPath.MOVETO]+[MPath.LINETO]*7+[MPath.CLOSEPOLY],fill='#D3E0E2')
        outline_path([(x,y+6.8),(x+1.1,y+2.4),(x+10.4,y+2.4),(x+9.3,y+6.8),(x,y+6.8)],
                     [MPath.MOVETO]+[MPath.LINETO]*3+[MPath.CLOSEPOLY],fill='#F9FCFB')
        circle(cx+3.8,cy+2.1,1.8,fill=BLUE)
        line(cx+2.8,cy+2.1,cx+4.8,cy+2.1,lw=.6,z=6)
        line(cx+3.8,cy+1.1,cx+3.8,cy+3.1,lw=.6,z=6)
        # Small event groups, deliberately without measured axes or curves.
        for j in range(3):
            rect(cx+6.8+j*.8,cy-2.6,.35,4.8,fill=('#B4C7C4','#9A86AF','#BEC9CF')[j],lw=0,z=4)

    def features_icon(cx,cy):
        # Named history quantities and symbols; no invented feature measurements.
        x,y,w,h=cx-12.5,61.5,25,9.6
        rect(x,y,w,h,fill='#FFFFFF',edge=EDGE,lw=.45,r=.4,z=4)
        rect(x+19.2,y+.1,5.7,h-.2,fill='#F8E8DF',lw=0,z=5)
        line(x+19.2,y,x+19.2,y+h,lw=.4,z=6)
        for yy in (y+3.2,y+6.4):line(x,yy,x+w,yy,lw=.4,z=6)
        for row,(name,symbol) in enumerate((('Prior count','n'),('Mean amount','μ'),('Recency','Δt'))):
            yy=y+1.6+row*3.2
            text(x+1.0,yy,name,region=(x+.7,y+.05,x+18.9,y+h-.05))
            text(x+22.1,yy,symbol,align='center',region=(x+19.4,y+.05,x+w-.1,y+h-.05))

    def scales(cx,cy):
        line(cx,cy-2.3,cx,cy+2.5,lw=.45)
        line(cx-2.5,cy-1.5,cx+2.5,cy-2.1,lw=.45)
        line(cx-1.6,cy+2.6,cx+1.6,cy+2.6,lw=.45)
        for dx,dy in ((-2.1,-1.55),(2.1,-2.0)):
            line(cx+dx,cy+dy,cx+dx-1.1,cy+dy+2.5,lw=.4)
            line(cx+dx,cy+dy,cx+dx+1.1,cy+dy+2.5,lw=.4)
            ax.add_patch(Arc((cx+dx,cy+dy+2.35),2.2,1.1,theta1=0,theta2=180,edgecolor=EDGE,lw=.45,zorder=5))

    def people(cx,cy):
        for dx,dy,fill in ((-1.4,0,'#F0C2B7'),(1.4,-1.4,'#BDCFCB'),(1.4,1.5,'#CDBDD9')):
            circle(cx+dx,cy+dy-.65,.55,fill=fill,lw=.4)
            ellipse(cx+dx,cy+dy+.65,1.9,1.05,fill=fill,lw=.4)

    def readout(cx,cy):
        # A report schema with explicit metric names, rather than an unlabeled map.
        x,y,w,h=146.0,61.5,18.0,9.6
        outline_path([(x,y),(x+w-1.6,y),(x+w,y+1.6),(x+w,y+h),(x,y+h),(x,y)],
                     [MPath.MOVETO]+[MPath.LINETO]*4+[MPath.CLOSEPOLY],fill='#FCFAFD',lw=.5,z=4)
        line(x+w-1.6,y,x+w-1.6,y+1.6,lw=.4,z=6)
        line(x+w-1.6,y+1.6,x+w,y+1.6,lw=.4,z=6)
        for row,name in enumerate(('AP','Recall','Cost')):
            yy=y+1.6+row*3.2
            text(x+1.4,yy,name,region=(x+1,y+.05,x+11.4,y+h-.05))
            line(x+11.8,yy+.35,x+w-1.3,yy+.35,color='#89949C',lw=.45,z=6)

    def saved_output(x,y,w,h,name):
        # Existing saved artifact, not a new numerical result.
        outline_path([(x,y),(x+w-1.3,y),(x+w,y+1.3),(x+w,y+h),(x,y+h),(x,y)],
                     [MPath.MOVETO]+[MPath.LINETO]*4+[MPath.CLOSEPOLY],fill='#FAFBFC',lw=.45,z=4)
        line(x+w-1.3,y,x+w-1.3,y+1.3,lw=.4,z=6)
        line(x+w-1.3,y+1.3,x+w,y+1.3,lw=.4,z=6)
        text(x+w/2-.2,y+h/2,name,align='center',region=(x+.3,y+.3,x+w-.5,y+h-.3))

    # a: separate temporal designs. Every displayed number is taken from protocol.
    band('a','Separate temporal designs',4.2,39.2)
    line(74,8.1,74,40.7,color='#B3BDBF',lw=.45,dash=(2,2))
    line(119,8.1,119,40.7,color='#B3BDBF',lw=.45,dash=(2,2))
    region_pay=(4.8,7.7,72.2,41.8)
    region_ieee=(76.5,7.7,117,41.8)
    region_bank=(121,7.7,169.5,41.8)
    regions.extend([region_pay,region_ieee,region_bank])
    database(9.8,16,7.3,9.2)
    text(17.1,9.8,'PaySim',9.5,True,region=region_pay)
    text(17.1,13.4,'Mobile-money simulation',region=region_pay)
    text(17.1,16.8,f"n = {P['pay_sim']['total_rows']:,} rows",bold=True,region=region_pay)
    for xx,lab in zip((23,41,59),('Train','Val','Test')): text(xx,20.6,lab,align='center',region=region_pay)
    for row,f in enumerate(P['pay_sim']['folds']):
        y=24.3+row*3.9
        text(6.2,y,f"F{f['fold']}",bold=True,region=region_pay)
        for col,(key,colr) in enumerate(zip(('train','validation','test'),(TRAIN,VAL,TEST))):
            x=14+col*18
            width=18 if not (row==2 and col==2) else 21
            chevron(x,y,width,3.65,colr)
            text(x+width/2,y,f"{f[key][0]}–{f[key][1]}",align='center',region=(x+.8,y-1.7,x+width-.5,y+1.7))
    text(6.2,36.5,'LightGBM (5 seeds)',region=region_pay)
    text(45.1,36.5,'Post-hoc logistic',region=region_pay)
    text(6.2,40,'Simulation steps',color=MUTED,region=region_pay)

    text(77,9.8,'IEEE-CIS',9.5,True,region=region_ieee)
    text(77,13.4,'Anonymised transactions',region=region_ieee)
    text(77,16.8,f"n = {P['ieee_cis']['total_labelled_rows']:,} rows",bold=True,region=region_ieee)
    for xx,lab in zip((88,98.8,109.6),('Train','Val','Test')): text(xx,20.6,lab,align='center',region=region_ieee)
    for row,f in enumerate(P['ieee_cis']['folds']):
        y=24.3+row*3.9
        text(77,y,f"F{f['fold']}",bold=True,region=region_ieee)
        for col,(key,colr) in enumerate(zip(('train','validation','test'),(TRAIN,VAL,TEST))):
            x=82.6+col*10.8
            rect(x,y-1.825,10.8,3.65,fill=colr,lw=.45,z=3)
            text(x+5.4,y,f"{f[key+'_percent_of_full_labelled_rows']:.0f}%",align='center',region=(x+.2,y-1.7,x+10.6,y+1.7))
    text(77,36.5,'Test: 59,054 rows/fold',region=region_ieee)
    text(77,40,'LightGBM (5 seeds)',region=region_ieee)

    text(122,9.8,'BankSim',9.5,True,region=region_bank)
    text(122,13.4,'Simulated bank payments',region=region_bank)
    text(122,16.8,f"n = {P['bank_sim']['total_rows']:,} rows",bold=True,region=region_bank)
    database(165.2,13.7,5.8,6.4)
    for xx,lab in zip((129.5,145,160.5),('Train','Val','Test')): text(xx,20.6,lab,align='center',region=region_bank)
    for col,(key,colr) in enumerate(zip(('train','validation','test'),(TRAIN,VAL,TEST))):
        x=121.8+col*15.5
        chevron(x,24.3,15.5,3.65,colr)
        a,b=P['bank_sim']['split'][key]['steps']
        text(x+7.75,24.3,f'{a}–{b}',align='center',region=(x+.5,22.6,x+15,26.0))
    for y,lab in zip((28.8,32.5,36.2,39.9),(
            'LightGBM / XGBoost (5 seeds)',
            'Local protocol frozen pre-fit',
            'B1 / seed 42: Val-only selection',
            'Settings locked before Test')):
        text(122,y,lab,region=region_bank)
    text(W/2,46.3,'Steps: PaySim / BankSim; IEEE-CIS: % of all labelled rows.',align='center',color=MUTED)
    arrow(W/2,48.2,W/2,50.1,lw=.65,head=5)

    # b: original five-card workflow, with dataset-specific decisions retained.
    band('b','Study-specific workflows',51.4,41.8)
    cards=[(5,27,'Inputs',BLUE),(35,29,'Features',CORAL),(67,29,'Models',SLATE),
           (99,39,'Decisions',GREEN),(141,28,'Readouts',LILAC)]
    for x,w,title,fill in cards:
        rect(x,55.2,w,33.2,fill=fill,lw=0,r=5.2)
        text(x+w/2,58.9,title,9.2,True,align='center',region=(x+.8,55.5,x+w-.8,62))
        if x!=141: arrow(x+w+.35,73.6,x+w+2.65,73.6,lw=.65,head=5)

    folder(18.2,66)
    for y,label in zip((72.0,75.6,79.2,82.8),('Attributes','Time group','Entity key','Prior events')):
        text(18.5,y,label,align='center',region=(6,70,31,88))
    text(18.5,86.4,'Schematic fields',align='center',color=MUTED,region=(6,70,31,88))
    features_icon(49.5,66)
    for y,label in zip((73.1,77.3,81.5,85.7),('Current context','Strict prior history','Entity aggregates','No label history')):
        text(49.5,y,label,align='center',region=(35.8,70,63.2,88))
    trees(81.5,64.0,.64)
    for y,label,bold in ((68.4,'Original: 5 seeds',True),(72.0,'LightGBM: all three',False),
                         (75.6,'XGBoost: BankSim',False),(79.2,'Post-hoc logistic',True),
                         (82.8,'PaySim / IEEE-CIS',False),(86.4,'1 model / condition',False)):
        text(81.5,y,label,bold=bold,align='center',region=(67.7,66.7,95.3,88.2))
    line(71,77.4,92,77.4,color='#BEBBAE',lw=.45)
    scales(103.3,58.7)
    people(133.7,58.9)
    for y,label,bold in (
        (63.6,'PaySim',True),(66.8,'Earlier Val: Platt',False),
        (70.0,'Later Val: threshold',False),(73.2,'Fixed → Test; Val cap ≤3%',False),
        (76.4,'Test workload varies',False),(80.1,'IEEE-CIS / BankSim',True),
        (83.3,'Completed Test ranking',False),(86.5,'k = ceil(0.03N)',False)):
        text(118.5,y,label,bold=bold,align='center',region=(100.2,62.0,136.8,88.2))
    line(104,78.25,133,78.25,color='#BFC4AA',lw=.45)
    readout(154.7,65.7)
    for y,label in zip((72.9,76.3,79.7,83.1,86.5),('AP; TP / recall','Precision / alert rate','Score-stage audit*','*PaySim only','Analytical cost')):
        text(155,y,label,align='center',region=(142,71.2,168,88.2))
    text(W/2,91,'PaySim: later exact-3% contrasts are separate.',align='center',color=MUTED)
    arrow(W/2,94.0,W/2,96.3,lw=.65,head=5)

    # c: separate subsequent analyses; S22 explicitly has new historical fits.
    band('c','Later analyses and scope',98.4,15.1)
    rect(5,101.6,80.5,10.1,fill='#E8EBED',lw=0,r=2.8)
    rect(88.5,101.6,80.5,10.1,fill='#E8EBED',lw=0,r=2.8)
    text(7.5,103.8,'S21 | Existing outputs',8.5,True,region=(6.8,102,70,112))
    text(7.5,107.1,'Headroom / exchanges / time structure',region=(6.8,102,70,112))
    text(7.5,110.3,'No new fits / scores / resampling',region=(6.8,102,70,112))
    saved_output(71.5,102.2,11.5,4.2,'Scores')
    saved_output(71.5,107.0,11.5,4.2,'Draws')
    text(91,103.8,'S22 | Retrospective development',8.5,True,region=(90.2,102,155.8,112))
    text(91,107.1,'Exposed BankSim; 6 LightGBM C07 fits',region=(90.2,102,155.8,112))
    text(91,110.3,'Seed 42; no CIs or independent validation',region=(90.2,102,155.8,112))
    trees(161.4,106.5,.8)
    text(W/2,116,'Named fields and files are schematic; datasets and fits remain separate.',align='center',color=MUTED)

    # Verify used glyphs in the actual regular/bold font faces before export.
    font_faces = []
    for weight in ("normal", "bold"):
        face = font_manager.findfont(font_manager.FontProperties(family=font_family, weight=weight), fallback_to_default=False)
        available = FT2Font(face).get_charmap()
        used = {character for obj,_ in labels if obj.get_fontweight() == weight for character in obj.get_text() if not character.isspace()}
        missing = sorted(character for character in used if ord(character) not in available)
        if missing:
            raise RuntimeError(f"Font {font_family!r} ({weight}) lacks required glyphs: {missing}")
        font_faces.append({"weight":weight,"font_file_name":Path(face).name,"all_used_glyphs_available":True})
    # Check real rendered lettering, assigned cells, canvas and overlaps.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inverse = ax.transData.inverted()
    issues, records = [], []
    for obj, region in labels:
        bbox = obj.get_window_extent(renderer=renderer).transformed(inverse)
        x0,x1 = sorted((float(bbox.x0), float(bbox.x1)))
        y0,y1 = sorted((float(bbox.y0), float(bbox.y1)))
        record = {"text":obj.get_text(), "font_pt":obj.get_fontsize(),
                  "bounds_coordinate_units":[x0,y0,x1,y1],
                  "bounds_mm":[v*coordinate_to_mm for v in (x0,y0,x1,y1)]}
        records.append(record)
        if obj.get_fontsize() < 8:
            issues.append({"kind":"font_below_8_pt", "label":record})
        if x0 < 0 or y0 < 0 or x1 > W or y1 > H:
            issues.append({"kind":"canvas", "label":record})
        if region:
            a,b,c,d = region
            if x0<a-.03 or y0<b-.03 or x1>c+.03 or y1>d+.03:
                issues.append({"kind":"cell", "region":region, "label":record})
    for index,a in enumerate(records):
        for b in records[index+1:]:
            x0,y0,x1,y1 = a["bounds_coordinate_units"]
            a0,b0,a1,b1 = b["bounds_coordinate_units"]
            if min(x1,a1)-max(x0,a0)>.03 and min(y1,b1)-max(y0,b0)>.03:
                issues.append({"kind":"text_overlap", "a":a["text"], "b":b["text"]})
    report = {
        "figure":"fig1", "adopted_revision":"semantic_data_icons_20261024",
        "canvas_mm":[physical_w,physical_h], "geometry_coordinate_units":[W,H],
        "minimum_font_pt":min(t.get_fontsize() for t,_ in labels),
        "maximum_font_pt":max(t.get_fontsize() for t,_ in labels),
        "minimum_stroke_pt":min(strokes), "text_count":len(labels),
        "font_family":font_family, "font_file_name":Path(font_path).name, "font_faces":font_faces,
        "protocol_sha256":PROTOCOL_HASH,
        "all_text_inside_canvas_and_assigned_cells":not issues,
        "draw_only_no_models_predictions_or_resampling":True,
        "cross_platform_byte_identity_claimed":False,
        "semantic_icon_definitions":{
            "Features":"Prior count n, prior mean amount μ, recency Δt: named field definitions, without measured values.",
            "Readouts":"Report fields AP, Recall and Cost: no fabricated performance values.",
            "S21":"Scores and Draws: existing saved score vectors and bootstrap draws; the draw-distribution diagnostic is specific to PaySim."
        },
        "issues":issues, "text":records
    }
    (DEST/"fig1_validation.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    if issues:
        raise AssertionError("Figure geometry check failed: "+json.dumps(issues,ensure_ascii=False))
    metadata = {
        "Title":"Separate temporal designs, study-specific workflows and subsequent analyses",
        "Description":"Named feature fields, metric reports and saved-output icons replace decorative matrices. Study-specific protocols and S21/S22 analysis boundaries remain distinct.",
        "Creator":"Python / Matplotlib; native vector primitives", "Date":None
    }
    fig.savefig(DEST/"fig1.svg", metadata=metadata)
    with matplotlib.rc_context({"svg.fonttype":"none"}):
        fig.savefig(DEST/"fig1_editable.svg", metadata=metadata)
    fig.savefig(DEST/"fig1.pdf",metadata={"Title":metadata["Title"],"Creator":metadata["Creator"],"CreationDate":None,"ModDate":None})
    os.environ.setdefault("SOURCE_DATE_EPOCH","0")
    fig.savefig(DEST/"fig1.eps", metadata={"Creator":metadata["Creator"]})
    fig.savefig(DEST/"fig1.png",dpi=600)
    fig.savefig(DEST/"fig1_preview.png",dpi=300)
    namespace={"s":"http://www.w3.org/2000/svg"}
    for filename in ("fig1.svg","fig1_editable.svg"):
        path=DEST/filename
        svg=path.read_text(encoding="utf-8")
        svg=re.sub(r'(<svg\b[^>]*?)width="[^"]+" height="[^"]+"',
                   rf'\1width="{physical_w:g}mm" height="{physical_h:.12f}mm"',svg,count=1)
        svg=svg.replace('<svg ','<svg role="img" aria-labelledby="fig1-title fig1-desc" ',1)
        accessibility=(
            '<title id="fig1-title">Separate temporal designs and study-specific workflows</title>'
            '<desc id="fig1-desc">Named prior-count, amount-mean and recency fields; AP/Recall/Cost report fields; saved Scores and Draws files. No invented measurements. Dataset designs, model scopes and S21/S22 analysis boundaries remain separate.</desc>'
        )
        svg=re.sub(r'(<svg\b.*?>)',lambda match:match.group(1)+"\n "+accessibility,svg,count=1,flags=re.S)
        path.write_text(svg,encoding="utf-8")
        root=ET.parse(path).getroot()
        report.setdefault("exports",{})[filename]={
            "text_elements":len(root.findall('.//s:text',namespace)),
            "image_elements":len(root.findall('.//s:image',namespace)),
            "width":root.get("width"), "height":root.get("height")
        }
    plt.close(fig)
    report["publication_sha256"]={
        filename:hashlib.sha256((DEST/filename).read_bytes()).hexdigest()
        for filename in ("fig1.svg","fig1_editable.svg","fig1.pdf","fig1.eps","fig1.png","fig1_preview.png")
    }
    (DEST/"fig1_validation.json").write_text(json.dumps(report,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return {key:value for key,value in report.items() if key not in ("text","semantic_icon_definitions")}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir",type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument("--font-family",help="Installed font family; default Arial, then metric-compatible Liberation Sans")
    args=parser.parse_args()
    source=Path(__file__).resolve().parent/"data"/"fig1_protocol.json"
    raw=source.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PROTOCOL_HASH:
        raise RuntimeError("Frozen figure protocol SHA-256 mismatch")
    protocol=json.loads(raw.decode("utf-8-sig"))
    family,path=resolve_font(args.font_family)
    print(json.dumps(draw(protocol,args.out_dir,family,path),indent=2,ensure_ascii=False))

if __name__=="__main__":
    main()
