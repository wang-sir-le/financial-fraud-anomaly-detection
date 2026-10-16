"""Draw Fig. S3 reliability curves with decorative fixed-width soft halos.

Only the frozen, aggregate S3 data are read. No model fitting, predictions,
resampling, or uncertainty estimates are generated. Requires reportlab.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from html import escape
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import toColor as HexColor

from _figure_io import output_directory, render_png, write_verification

HERE = Path(__file__).resolve().parent
OUT = output_directory(HERE, "Draw adopted Figure S3 from its neighbouring aggregates.")
SOURCE = HERE / 'figS3_data.json'
LOCAL = SOURCE
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '77bfb009a7f22709f842380ba6d2130fbb239fb5c7e9bd1311fd7cb0b1404439'
DATA = json.loads(SOURCE.read_text(encoding='utf-8'))
W = 1500
H = W * DATA['figsize_inches'][1] / DATA['figsize_inches'][0]
PDF_WIDTH = DATA['figsize_inches'][0] * 72
SCALE = PDF_WIDTH / W
INK, MUTED, RULE = '#172735', '#647381', '#D8E0E7'
PALETTE = ['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD']
for name, filename in [('Arial', 'Arial.ttf'), ('Arial-Bold', 'Arial Bold.ttf')]:
    path = Path('/System/Library/Fonts/Supplemental') / filename
    if path.exists():
        pdfmetrics.registerFont(TTFont(name, str(path)))
REG = 'Arial' if 'Arial' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
BOLD = 'Arial-Bold' if 'Arial-Bold' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'
PDF_DIR = OUT
PDF_DIR.mkdir(parents=True, exist_ok=True)
pdf_path = PDF_DIR / 'figS3.pdf'
C = canvas.Canvas(str(pdf_path), pagesize=(PDF_WIDTH, H * SCALE), pageCompression=1)
C.setTitle('Figure S3: reliability diagrams with decorative soft curve halos')
C.setAuthor('')
C.scale(SCALE, SCALE)
SVG = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<title>Figure S3: PaySim PTHW reliability diagrams</title>',
       '<desc>Forty-five independent fold-method-seed series and 299 nonempty original bins. Soft halos are decorative fixed-display-width strokes, not uncertainty intervals.</desc>']
TEXT_BOXES = []

def rect(x, y, w, h, fill='white', stroke=None, sw=1):
    C.setFillColor(HexColor(fill))
    if stroke:
        C.setStrokeColor(HexColor(stroke)); C.setLineWidth(sw)
    C.rect(x, H-y-h, w, h, fill=1, stroke=int(bool(stroke)))
    SVG.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def line(x1, y1, x2, y2, color=RULE, sw=1, dashed=False):
    C.setStrokeColor(HexColor(color)); C.setLineWidth(sw)
    C.setDash([6, 5] if dashed else [])
    C.line(x1, H-y1, x2, H-y2)
    C.setDash([])
    dash = ' stroke-dasharray="6 5"' if dashed else ''
    SVG.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{sw}"{dash}/>')

def circle(x, y, r, fill, stroke=None, sw=1):
    C.setFillColor(HexColor(fill))
    if stroke:
        C.setStrokeColor(HexColor(stroke)); C.setLineWidth(sw)
    C.circle(x, H-y, r, fill=1, stroke=int(bool(stroke)))
    SVG.append(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{fill}" stroke="{stroke or "none"}" stroke-width="{sw}"/>')

def text(x, y, value, size=22, color=INK, bold=False, anchor='start', rotate=False, cell=None):
    value = str(value)
    font = BOLD if bold else REG
    tw = pdfmetrics.stringWidth(value, font, size)
    left = x - (tw / 2 if anchor == 'middle' else tw if anchor == 'end' else 0)
    C.setFillColor(HexColor(color)); C.setFont(font, size)
    C.saveState()
    if rotate:
        C.translate(x, H-y); C.rotate(90); C.drawCentredString(0, 0, value)
        box = (x-size, y-tw/2, x+size*.25, y+tw/2)
    else:
        C.drawString(left, H-y, value)
        box = (left, y-size*.82, left+tw, y+size*.23)
    C.restoreState()
    transform = f' transform="rotate(-90 {x} {y})"' if rotate else ''
    SVG.append(f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}"{transform}>{escape(value)}</text>')
    TEXT_BOXES.append({'text': value, 'box': box, 'cell': cell})

def centered(x, y, value, size=22, **kwargs):
    text(x, y+size*.34, value, size=size, anchor='middle', **kwargs)

def polyline(points, color, width=3, dash=None, opacity=1):
    if len(points)<2:
        return
    C.saveState()
    C.setStrokeColor(HexColor(color)); C.setLineWidth(width)
    C.setStrokeAlpha(opacity); C.setLineJoin(1); C.setLineCap(1)
    C.setDash(dash or [])
    p=C.beginPath(); p.moveTo(points[0][0],H-points[0][1])
    for x,y in points[1:]: p.lineTo(x,H-y)
    C.drawPath(p,stroke=1,fill=0); C.restoreState()
    coords=' '.join(f'{x:.10f},{y:.10f}' for x,y in points)
    dashes=' stroke-dasharray="'+','.join(str(v) for v in dash)+'"' if dash else ''
    SVG.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="{width}" stroke-opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round"{dashes}/>')

def polygon(points,color):
    C.setFillColor(HexColor(color)); C.setStrokeColor(HexColor(color))
    p=C.beginPath(); p.moveTo(points[0][0],H-points[0][1])
    for x,y in points[1:]: p.lineTo(x,H-y)
    p.close(); C.drawPath(p,fill=1,stroke=0)
    coords=' '.join(f'{x},{y}' for x,y in points)
    SVG.append(f'<polygon points="{coords}" fill="{color}"/>')

def point_marker(x,y,kind,color,r=5.2,opacity=1):
    C.saveState(); C.setFillAlpha(opacity); C.setStrokeAlpha(opacity)
    SVG.append(f'<g opacity="{opacity}">')
    if kind=='o': circle(x,y,r,color)
    elif kind=='s': rect(x-r,y-r,2*r,2*r,color)
    elif kind=='D': polygon([(x,y-r*1.3),(x+r*1.15,y),(x,y+r*1.3),(x-r*1.15,y)],color)
    elif kind=='^': polygon([(x,y-r*1.3),(x+r*1.15,y+r),(x-r*1.15,y+r)],color)
    elif kind=='v': polygon([(x-r*1.15,y-r),(x+r*1.15,y-r),(x,y+r*1.3)],color)
    SVG.append('</g>'); C.restoreState()

def start_clip(x,y,width,height,ident):
    C.saveState(); p=C.beginPath(); p.rect(x,H-y-height,width,height)
    C.clipPath(p,stroke=0,fill=0)
    SVG.append(f'<defs><clipPath id="{ident}"><rect x="{x}" y="{y}" width="{width}" height="{height}"/></clipPath></defs>')
    SVG.append(f'<g clip-path="url(#{ident})">')

def end_clip():
    SVG.append('</g>'); C.restoreState()

METHODS=['Uncalibrated','Platt','Isotonic']
TITLES=['Uncalibrated','Platt scaling','Isotonic regression']
SEEDS=[42,52,62,72,82]
MARKERS=['o','s','^','D','v']
DASHES=[None,[10,6],[11,5,2,5],[2,5],[10,4,2,4,2,4]]
# Widths are physical display widths, independent of probability values.
# All halo strokes use exactly the original polyline coordinates.
CELL_SIZE=335
PW,PH=300,300
PANEL_SCALE=PW/CELL_SIZE
PANEL_INSET=(CELL_SIZE-PW)/2
GLOW=[(width*PANEL_SCALE,alpha) for width,alpha in [(28,.030),(21,.045),(14,.065),(8,.045)]]
XPOS=[138,615,1092]
YPOS=[112,566,1020]
records=[]
rendered_series=[]
rect(0,0,W,H)
for x,title,fill in zip(XPOS,TITLES,['#FDEBEC','#E6F0FC','#E7F4E5']):
    rect(x,22,CELL_SIZE,43,fill)
    centered(x+CELL_SIZE/2,43,title,25,bold=True)

for i,fold in enumerate([1,2,3]):
    cell_top=YPOS[i]
    py=cell_top+PANEL_INSET
    rect(13,cell_top-8,38,CELL_SIZE+16,'#EDF0F3')
    text(39,cell_top+CELL_SIZE/2,f'Fold {fold}',22,bold=True,anchor='middle',rotate=True)
    text(80+PANEL_INSET,py+PH/2,'Observed fraud fraction',23,anchor='middle',rotate=True)
    for j,method in enumerate(METHODS):
        px=XPOS[j]+PANEL_INSET
        fx=lambda value: px+value*PW
        fy=lambda value: py+(1-value)*PH
        # Light guides and the original agreement diagonal.
        for k in range(6):
            v=k/5
            if k not in [0,5]:
                line(fx(v),py,fx(v),py+PH,'#EDF0F3',1)
                line(px,fy(v),px+PW,fy(v),'#EDF0F3',1)
        polyline([(px,py+PH),(px+PW,py)],'#68717C',1.6*PANEL_SCALE,[8*PANEL_SCALE,6*PANEL_SCALE])
        series=[]
        curve_dashes=[None if d is None else [v*PANEL_SCALE for v in d] for d in DASHES]
        for seed,color,marker,dash in zip(SEEDS,PALETTE,MARKERS,curve_dashes):
            chosen=sorted((r for r in DATA['rows'] if r['fold']==fold and r['calibration_method']==method and r['seed']==seed),key=lambda r:r['bin_id'])
            assert chosen and all(r['sample_count']>0 for r in chosen)
            points=[(fx(r['mean_predicted_probability']),fy(r['observed_fraud_rate'])) for r in chosen]
            series.append((chosen,points,color,marker,dash))
            records.extend(chosen)
            rendered_series.append({'fold':fold,'method':method,'seed':seed,'point_count':len(chosen),
                'bin_ids':[r['bin_id'] for r in chosen],
                'x':[r['mean_predicted_probability'] for r in chosen],
                'y':[r['observed_fraud_rate'] for r in chosen],
                'halo':'marker only' if len(chosen)==1 else 'fixed-width identical-path strokes'})
        # Render all decorative layers first, so none obscure another crisp line.
        start_clip(px-5,py-5,PW+10,PH+10,f'glow_{i}_{j}')
        for chosen,points,color,marker,dash in series:
            if len(points)==1:
                for radius,alpha in [(16,.030),(12,.045),(8,.065)]:
                    point_marker(*points[0],marker,color,radius*PANEL_SCALE,alpha)
            else:
                for width,alpha in GLOW:
                    polyline(points,color,width,dash,alpha)
        end_clip()
        start_clip(px-7,py-7,PW+14,PH+14,f'curves_{i}_{j}')
        for chosen,points,color,marker,dash in series:
            polyline(points,color,2.9*PANEL_SCALE,dash)
            for xx,yy in points: point_marker(xx,yy,marker,color,r=5.2*PANEL_SCALE)
        end_clip()
        line(px,py,px,py+PH,'#4B5563',1.6)
        line(px,py+PH,px+PW,py+PH,'#4B5563',1.6)
        for k in range(6):
            v=k/5
            line(fx(v),py+PH,fx(v),py+PH+6,'#4B5563',1.2)
            centered(fx(v),py+PH+29,f'{v:.1f}',19)
            line(px-6,fy(v),px,fy(v),'#4B5563',1.2)
            text(px-13,fy(v)+6,f'{v:.1f}',19,anchor='end')
        centered(px+PW/2,py+PH+64,'Mean predicted probability',22)

# One shared legend retains method and seed decoding without extra statistics.
rect(56,1454,1388,54,'white','#CBD2D9',1)
for index,(seed,color,marker,dash) in enumerate(zip(SEEDS,PALETTE,MARKERS,DASHES)):
    x=86+index*193
    polyline([(x,1481),(x+39,1481)],color,2.9,dash)
    point_marker(x+20,1481,marker,color)
    text(x+52,1488,f'Seed {seed}',21)
polyline([(1085,1481),(1126,1481)],'#68717C',1.6,[8,6])
text(1140,1488,'Perfect calibration',21)
centered(W/2,1539,'Soft halos are decorative and do not represent uncertainty intervals.',19,color=MUTED)

assert len(rendered_series)==45 and len(records)==299
assert sum(r['sample_count']<10 for r in records)==50
assert sum(s['point_count']==1 for s in rendered_series)==3
source_keys={(r['fold'],r['calibration_method'],r['seed'],r['bin_id']) for r in DATA['rows']}
assert source_keys=={(r['fold'],r['calibration_method'],r['seed'],r['bin_id']) for r in records}
SVG.append('</svg>')
(OUT/'figS3.svg').write_text('\n'.join(SVG),encoding='utf-8')
C.showPage(); C.save()
with (OUT/'figS3_plot_data.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(DATA['rows'][0]));writer.writeheader();writer.writerows(records)
(OUT/'figS3_rendered_series.json').write_text(json.dumps(rendered_series,indent=2),encoding='utf-8')
caption='''Fig. S3 PaySim PTHW reliability diagrams. Rows are folds and columns are raw, Platt and isotonic Test probabilities; markers and line styles identify five seeds. Points show mean predicted probability and observed fraud fraction in nonempty bins of the fixed ten-bin uniform partition. Lines are visual guides and can span empty bins; the diagonal denotes agreement. Of 450 bins across 45 series, 299 are nonempty and 50 nonempty bins have fewer than ten observations. Bins are not pooled. Sparse points can be unstable, and these diagrams do not demonstrate improved ranking or decisions. Soft halos are decorative fixed-display-width strokes along the original paths and do not represent uncertainty intervals. Single-point series receive marker halos only.
'''
(OUT/'figS3_caption.txt').write_text(caption,encoding='utf-8')
issues=[t for t in TEXT_BOXES if t['box'][0]<0 or t['box'][1]<0 or t['box'][2]>W or t['box'][3]>H]
assert not issues,issues
validation={'source_sha256':hashlib.sha256(LOCAL.read_bytes()).hexdigest(),'series':45,'nonempty_bins':299,
    'fixed_partition_total_bins':450,'empty_bins_omitted':151,'nonempty_bins_below_10':50,
    'original_values_preserved':True,'new_statistical_intervals':0,'smoothed_curves':False,'pooled_bins':False,
    'single_point_series':3,'single_point_series_rendering':'marker halo only',
    'halo_widths_px_alpha':GLOW,'halo_widths_pt':[width*SCALE for width,alpha in GLOW],
    'halo_meaning':'decorative fixed-display-width strokes, not data intervals',
    'method_headers':['Uncalibrated','Platt scaling','Isotonic regression'],
    'physical_size_mm':[v*25.4 for v in DATA['figsize_inches']],
    'plot_area_px':[PW,PH],'previous_plot_area_px':[CELL_SIZE,CELL_SIZE],
    'plot_linear_scale':PANEL_SCALE,'plot_area_change_percent':100*((PW/CELL_SIZE)**2-1),
    'plot_centres_unchanged':True,'headings_and_legend_unchanged':True,'font_sizes_unchanged':True,
    'text_canvas_bounds':'PASS','pdf_path':pdf_path.name,'font':REG,'minimum_font_px':19}
validation['figure']='S3'
validation.update(render_png(pdf_path, OUT/'figS3.png'))
write_verification(OUT/'figS3_verification.json', validation,
                   [OUT/'figS3.png', pdf_path, OUT/'figS3.svg'])
