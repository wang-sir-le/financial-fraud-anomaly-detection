"""Draw six whole-window BankSim S9 condition tiles and four normalized contrasts.

Only the frozen, aggregate S9 data are read. No model fitting, predictions,
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
OUT = output_directory(HERE, "Draw adopted Figure S9 from its neighbouring aggregates.")
SOURCE = HERE / 'figS9_data.json'
LOCAL = SOURCE
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == 'e86eb2bb6c432ff3e9ef5656a90aaf97433c7f538fca3d98fe0cd917c17aad92'
DATA = json.loads(SOURCE.read_text(encoding='utf-8'))
W,H = 1600,1100
PDF_WIDTH = 190 / 25.4 * 72
SCALE = PDF_WIDTH / W
INK, MUTED, RULE = '#252A2B', '#626D73', '#D8DEE2'
BLUE,ORANGE,PURPLE = '#318DCF','#EF8130','#9960A9'
for name, filename in [('TimesNewRoman', 'Times New Roman.ttf'), ('TimesNewRoman-Bold', 'Times New Roman Bold.ttf')]:
    path = Path('/System/Library/Fonts/Supplemental') / filename
    if path.exists():
        pdfmetrics.registerFont(TTFont(name, str(path)))
REG = 'TimesNewRoman' if 'TimesNewRoman' in pdfmetrics.getRegisteredFontNames() else 'Times-Roman'
BOLD = 'TimesNewRoman-Bold' if 'TimesNewRoman-Bold' in pdfmetrics.getRegisteredFontNames() else 'Times-Bold'
PDF_DIR = OUT
PDF_DIR.mkdir(parents=True, exist_ok=True)
pdf_path = PDF_DIR / 'figS9.pdf'
C = canvas.Canvas(str(pdf_path), pagesize=(PDF_WIDTH, H * SCALE), pageCompression=1)
C.setTitle('Figure S9: Condition performance and normalized history-increment contrasts')
C.setAuthor('')
C.scale(SCALE, SCALE)
SVG = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<title>Figure S9: Six whole-window condition tiles and four normalized contrasts</title>',
       '<desc>Six whole-window fixed C07 seed 42 conditions and the original four history-increment contrasts. Uniform row colors identify direct merchant input. Bar heights encode I in percentage points and labels give J in captured-fraud counts, with each fraud denominator shown. No inferential intervals or significance symbols.</desc>']
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
    size = max(size, 24)
    if rotate:
        anchor = 'middle'
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
    SVG.append(f'<text x="{x}" y="{y}" font-family="Times New Roman, Times, serif" font-size="{size}" font-weight="{700 if bold else 400}" fill="{color}" text-anchor="{anchor}"{transform}>{escape(value)}</text>')
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

LOOKUP={(r['history'],r['merchant']):r for r in DATA['whole_window']}
assert len(LOOKUP)==6
for row in LOOKUP.values():
    assert row['P']==2160 and row['K']==5950
    assert abs(row['recall']-row['TP']/row['P'])<1e-12
    assert row['remaining_space']==row['P']-row['TP']
source_paths=[HERE.parents[1]/item['path'] for item in DATA['sources']]
source_check='metadata only; source CSVs not present in this isolated drawing copy'
if all(path.is_file() for path in source_paths):
    for item,path in zip(DATA['sources'],source_paths):
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
    source_check='PASS'

rect(0,0,W,H)
text(44,49,'Input × history conditions in BankSim',40,bold=True)
text(44,91,'Fixed C07, seed 42  |  Retrospective development',27,color=MUTED)
text(W-44,91,'Whole window: P = 2,160; K = 5,950',27,color=MUTED,anchor='end')

text(44,157,'a',39,bold=True)
text(132,157,'Detection performance by input and history condition',32,bold=True)
text(44,202,'Direct merchant',26,color=MUTED)
text(44,231,'input',26,color=MUTED)
for history,x in [('Hnone',295),('H7',710),('Horig',1125)]:
    centered(x+195,205,history,32,bold=True)

CELLS=[]
for merchant,y,fill,tint,description in [
    ('M-off',238,'#E8F1E4','#527A4E','(absent)'),
    ('M-on',403,'#EFE8F4','#7A5A8A','(present)')]:
    text(44,y+60,merchant,34,bold=True)
    text(44,y+96,description,27,color=MUTED)
    for history,x in [('Hnone',295),('H7',710),('Horig',1125)]:
        row=LOOKUP[(history,merchant)]
        rect(x,y,390,146,fill)
        line(x,y+146,x+390,y+146,tint,2)
        centered(x+195,y+40,f"TP = {row['TP']:,}",37,bold=True)
        centered(x+195,y+83,f"Recall {100*row['recall']:.4f}%",30)
        centered(x+195,y+120,f"Uncaptured frauds: {row['remaining_space']}",28,color=MUTED)
        CELLS.append({**row,'display_TP':f"TP = {row['TP']:,}",
                      'display_recall':f"{100*row['recall']:.4f}%",'row_fill':fill})

text(44,611,'b',39,bold=True)
text(132,611,'Difference of history increments',32,bold=True)
text(132,651,'J = (TP[Horig] - TP[H7])M-on - (TP[Horig] - TP[H7])M-off;   I = 100J/P (pp)',26)

PX,PY,PW,PH=150,690,1365,230
YMIN,YMAX=-1.7,.4
fy=lambda value:PY+(YMAX-value)/(YMAX-YMIN)*PH
for value in [-1.5,-1.0,-.5,0.0]:
    yy=fy(value)
    if value!=0:line(PX,yy,PX+PW,yy,'#E3E7EA',1)
    line(PX-7,yy,PX,yy,'#515D64',1.2)
    text(PX-19,yy+8,f'{value:.1f}',26,anchor='end')
line(PX,fy(0),PX+PW,fy(0),'#596770',1.7,True)
line(PX,PY,PX,PY+PH,'#515D64',1.6)
line(PX,PY+PH,PX+PW,PY+PH,'#515D64',1.6)
text(47,PY+PH/2,'Contrast I (pp)',29,rotate=True)

# Whole-window and local-segment results have different denominators and budgets.
line(548,PY+9,548,PY+PH-9,'#BBC5CB',1.2,True)
BARS=[]
for x,row in zip([365,710,1055,1400],DATA['contrasts']):
    assert row['J']==row['delta1_TP']-row['delta0_TP']
    assert abs(row['I_pp']-100*row['J']/row['common_P'])<1e-12
    positive=row['I_pp']>0
    color='#2D5D9D' if positive else '#A82D3C'
    fill='#81A4D6' if positive else '#D9979D'
    yy=fy(row['I_pp']);zz=fy(0)
    rect(x-62,min(yy,zz),124,abs(yy-zz),fill)
    if positive:
        value_y=yy-48
        count_y=yy-17
    else:
        value_y=yy+28
        count_y=yy+58
    text(x,value_y,f"I = {row['I_pp']:+.4f} pp",28,color=color,bold=True,anchor='middle')
    text(x,count_y,f"J = {row['J']:+d} TP",27,color=color,anchor='middle')
    line(x,PY+PH,x,PY+PH+7,'#515D64',1.2)
    centered(x,960,row['label'],28)
    centered(x,995,f"P = {row['common_P']:,}",25,color=MUTED)
    BARS.append({**row,'x':x,'y_value':yy,'y_zero':zz,'bar_width':124,
                 'bar_height':abs(yy-zz),'display_I':f"{row['I_pp']:+.4f} pp",
                 'display_J':f"{row['J']:+d} TP"})

text(44,1041,'Single fits; no inferential intervals. M-off retains merchant as the history key.',25,color=MUTED)
text(44,1078,'Segment budgets are local and do not decompose the whole-window selection.',25,color=MUTED)

assert len(CELLS)==6 and len(BARS)==4
assert [r['J'] for r in BARS]==[-6,-8,-3,1]
assert {r['row_fill'] for r in CELLS if r['merchant']=='M-off'}=={'#E8F1E4'}
assert {r['row_fill'] for r in CELLS if r['merchant']=='M-on'}=={'#EFE8F4'}
issues=[t for t in TEXT_BOXES if t['box'][0]<0 or t['box'][1]<0 or t['box'][2]>W or t['box'][3]>H]
assert not issues,issues
SVG.append('</svg>')
(OUT/'figS9.svg').write_text('\n'.join(SVG),encoding='utf-8')
C.showPage();C.save()
with (OUT/'figS9_whole_window_cells.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(CELLS[0]));writer.writeheader();writer.writerows(CELLS)
with (OUT/'figS9_contrast_values.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=list(BARS[0]));writer.writeheader();writer.writerows(BARS)
(OUT/'figS9_render_audit.json').write_text(json.dumps({'cells':CELLS,'bars':BARS},indent=2))
caption='''Fig. S9 Direct merchant input and accessible history in six fixed retrospective BankSim conditions. (a) Whole-window tiles report TP, recall and uncaptured frauds (P-TP) at K=5,950 among P=2,160 frauds (N=198,311). Uniform row colors identify the direct merchant input condition and do not encode TP magnitude. (b) Bars show I=100J/P in percentage points using unrounded values; labels report I, J and the fraud denominator P for each window. J=(TP[Horig]-TP[H7])M-on-(TP[Horig]-TP[H7])M-off. The whole-window denominator is 2,160 and each segment denominator is 720. The normalized contrasts facilitate comparison of descriptive magnitudes across these denominators; the four results are not independent replications. M-off removes only the direct merchant predictor; merchant remains the history key. H7 uses seven strictly earlier complete simulation steps. Horig retains the original 1/7/30-step aggregates and strictly earlier recency/availability, whose source may extend beyond 30 steps. Segment budgets are local, so their selections do not decompose the whole-window list and the segment contrasts must not be summed to reproduce the whole-window contrast. These single fits under C07 and seed 42 use the already-exposed development environment, without inferential intervals or significance symbols; they do not establish substitution, sufficiency or equivalence.
'''
(OUT/'figS9_caption.txt').write_text(caption,encoding='utf-8')
validation={'source_data_sha256':hashlib.sha256(LOCAL.read_bytes()).hexdigest(),'source_files_hash_check':source_check,
    'whole_window_cells':6,'whole_window_P':2160,'whole_window_K':5950,'segment_P':720,
    'bars':4,'bar_metric':'I (pp), unrounded','contrasts_J':[-6,-8,-3,1],
    'contrasts_I_pp':[r['I_pp'] for r in BARS],'all_TP_recall_headroom_checked':True,
    'all_J_I_arithmetic_checked':True,'continuous_TP_color_scale':False,
    'uniform_categorical_row_colors':True,'duplicated_TP_line_panel':False,
    'significance_symbols':0,'uncertainty_intervals':0,'new_predictions':0,'new_fits':0,'new_resampling':0,
    'source_values_changed':False,'font':REG,'minimum_font_px':24,'minimum_font_pt':24*SCALE,
    'text_canvas_bounds':'PASS','pdf_width_mm':190,'pdf_height_mm':190*H/W,
    'project_assets_replaced':False}
validation['figure']='S9'
validation.update(render_png(pdf_path, OUT/'figS9.png'))
write_verification(OUT/'figS9_verification.json', validation,
                   [OUT/'figS9.png', pdf_path, OUT/'figS9.svg'])
