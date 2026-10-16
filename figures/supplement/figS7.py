"""Draw the adopted three-row BankSim Fig. S7 from frozen aggregates.

Only the frozen, aggregate S7 data are read. No model fitting, predictions,
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
OUT = output_directory(HERE, "Draw adopted Figure S7 from its neighbouring aggregates.")
SOURCE = HERE / 'figS7.data.json'
LOCAL = SOURCE
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == '38b255fd70e19919bdb3f7a38f7fdba591cc3663ffef49740caaa18c14d1d63a'
DATA = json.loads(SOURCE.read_text(encoding='utf-8'))
W,H = 1600,1185
PDF_WIDTH = 190 / 25.4 * 72
SCALE = PDF_WIDTH / W
INK, MUTED, RULE = '#172735', '#647381', '#D8E0E7'
BLUE,ORANGE = '#1675B6','#EC7B2C'
for name, filename in [('Arial', 'Arial.ttf'), ('Arial-Bold', 'Arial Bold.ttf')]:
    path = Path('/System/Library/Fonts/Supplemental') / filename
    if path.exists():
        pdfmetrics.registerFont(TTFont(name, str(path)))
REG = 'Arial' if 'Arial' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
BOLD = 'Arial-Bold' if 'Arial-Bold' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'
PDF_DIR = OUT
PDF_DIR.mkdir(parents=True, exist_ok=True)
pdf_path = PDF_DIR / 'figS7.pdf'
C = canvas.Canvas(str(pdf_path), pagesize=(PDF_WIDTH, H * SCALE), pageCompression=1)
C.setTitle('Figure S7 preview: BankSim temporal structure and whole-window capture contributions')
C.setAuthor('')
C.scale(SCALE, SCALE)
SVG = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
       '<title>Figure S7 preview: BankSim descriptive time structure</title>',
       '<desc>Fifty-four simulation steps. Transaction and fraud counts and paired five-seed mean capture differences from frozen whole-window top-3% selections. No step-wise reranking or new intervals.</desc>']
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

GROUPS={name:sorted([r for r in DATA['rows'] if r['family']==name],key=lambda r:r['step'])
        for name in ['LightGBM','XGBoost']}
assert len(DATA['rows'])==108
assert all(len(rows)==54 for rows in GROUPS.values())
assert [r['step'] for r in GROUPS['LightGBM']]==list(range(126,180))
assert [(r['step'],r['N'],r['F']) for r in GROUPS['LightGBM']]==[(r['step'],r['N'],r['F']) for r in GROUPS['XGBoost']]
assert {r['F'] for r in DATA['rows']}=={40}
assert sum(r['N'] for r in GROUPS['LightGBM'])==198311
assert sum(r['F'] for r in GROUPS['LightGBM'])==2160

PX,PW,PH=138,1280,210
XTICKS=[126,130,140,150,160,170,179]
XDOMAIN=(125,180)
PLOT_RECORDS=[]
PATH_AUDIT=[]
rect(0,0,W,H)

def legend_item(x,y,label,color,dash=None):
    polyline([(x,y),(x+43,y)],color,3.2,dash)
    text(x+56,y+7,label,23)

def setup_panel(letter,title,heading_y,plot_y,limits,ticks,ylabel,left_color=INK):
    text(39,heading_y,letter,37,bold=True)
    text(PX,heading_y,title,30,bold=True)
    fx=lambda value: PX+(value-XDOMAIN[0])/(XDOMAIN[1]-XDOMAIN[0])*PW
    fy=lambda value: plot_y+(limits[1]-value)/(limits[1]-limits[0])*PH
    for v in XTICKS:
        xx=fx(v)
        line(xx,plot_y,xx,plot_y+PH,'#EDF1F4',1)
        line(xx,plot_y+PH,xx,plot_y+PH+7,'#697786',1.2)
        centered(xx,plot_y+PH+28,str(v),22)
    for v in ticks:
        yy=fy(v)
        line(PX,yy,PX+PW,yy,'#E5EBF0',1)
        line(PX-7,yy,PX,yy,'#697786',1.2)
        label=f'{v:,.0f}' if abs(v)>=1000 else f'{v:g}'
        text(PX-17,yy+7,label,22,color=left_color,anchor='end')
    line(PX,plot_y,PX,plot_y+PH,'#697786',1.6)
    line(PX,plot_y+PH,PX+PW,plot_y+PH,'#697786',1.6)
    text(43,plot_y+PH/2,ylabel,25,color=left_color,anchor='middle',rotate=True)
    return fx,fy

def draw_series(panel,name,metric,fx,fy,color,dash=None):
    chosen=GROUPS['LightGBM'] if metric in ['N','F'] else GROUPS[name]
    values=[r[metric] for r in chosen]
    points=[(fx(r['step']),fy(r[metric])) for r in chosen]
    polyline(points,color,3.4,dash)
    PLOT_RECORDS.extend({'panel':panel,'series':name,'metric':metric,'step':r['step'],'value':r[metric]} for r in chosen)
    PATH_AUDIT.append({'panel':panel,'series':name,'metric':metric,'steps':[r['step'] for r in chosen],'values':values})

fx,fy=setup_panel('a','BankSim temporal structure',45,115,(3500,3820),[3500,3600,3700,3800],'Transactions',BLUE)
legend_item(PX+8,82,'Transactions (left axis)',BLUE)
legend_item(PX+386,82,'Frauds (right axis)',ORANGE,[10,7])
text(PX+PW,44,'Test steps 126-179',22,color=MUTED,anchor='end')
text(PX+PW,89,'40 frauds at every step',22,color=ORANGE,anchor='end')
draw_series('a','Transactions','N',fx,fy,BLUE)
fy_fraud=lambda value:115+(50-value)/50*PH
draw_series('a','Frauds','F',fx,fy_fraud,ORANGE,[10,7])
line(PX+PW,115,PX+PW,115+PH,ORANGE,1.4)
for value in [0,10,20,30,40,50]:
    yy=fy_fraud(value)
    line(PX+PW,yy,PX+PW+7,yy,ORANGE,1.2)
    text(PX+PW+17,yy+7,str(value),22,color=ORANGE)
text(1510,115+PH/2,'Frauds',25,color=ORANGE,anchor='middle',rotate=True)

for letter,title,heading_y,plot_y,metric,limits,ticks in [
    ('b','H0 contribution (whole-window top-3%)',407,477,'H0',(-.5,10),[0,2,4,6,8,10]),
    ('c','H1 contribution (whole-window top-3%)',769,839,'H1',(-2.5,3.5),[-2,-1,0,1,2,3])]:
    fx,fy=setup_panel(letter,title,heading_y,plot_y,limits,ticks,'Δ captured frauds')
    legend_item(PX+8,heading_y+37,'LightGBM',BLUE)
    legend_item(PX+260,heading_y+37,'XGBoost',ORANGE,[10,6])
    line(PX,fy(0),PX+PW,fy(0),'#8E98A3',1.6)
    draw_series(letter,'LightGBM',metric,fx,fy,BLUE)
    draw_series(letter,'XGBoost',metric,fx,fy,ORANGE,[10,6])

centered(PX+PW/2,1102,'Simulation step',25)
centered(W/2,1140,'Whole-window top-3% selections attributed to steps; no step-wise reranking.',21,color=MUTED)
centered(W/2,1170,'H0 = B0H - B0; H1 = B1H - B1. Contribution curves show five-seed mean differences.',20,color=MUTED)

assert len(PLOT_RECORDS)==324 and len(PATH_AUDIT)==6
sums={family:{metric:sum(r[metric] for r in rows) for metric in ['H0','H1']} for family,rows in GROUPS.items()}
for family,expected in [('LightGBM',{'H0':208.4,'H1':1.0}),('XGBoost',{'H0':211.4,'H1':9.6})]:
    for metric,value in expected.items(): assert abs(sums[family][metric]-value)<1e-9
issues=[t for t in TEXT_BOXES if t['box'][0]<0 or t['box'][1]<0 or t['box'][2]>W or t['box'][3]>H]
assert not issues,issues
SVG.append('</svg>')
(OUT/'figS7.svg').write_text('\n'.join(SVG),encoding='utf-8')
C.showPage();C.save()
with (OUT/'figS7_plot_data.csv').open('w',newline='',encoding='utf-8') as f:
    writer=csv.DictWriter(f,fieldnames=['panel','series','metric','step','value']);writer.writeheader();writer.writerows(PLOT_RECORDS)
(OUT/'figS7_path_audit.json').write_text(json.dumps(PATH_AUDIT,indent=2),encoding='utf-8')
caption='''Fig. S7 Descriptive BankSim temporal structure and step contributions to whole-window capture effects at 3% capacity. (a) Transaction counts (left axis, blue) and fraud counts (right axis, orange dashed line) for all 54 simulation steps, 126-179. Each step contains 40 frauds. (b) H0=B0H-B0 and (c) H1=B1H-B1, shown for LightGBM and XGBoost as five-seed means of paired captured-fraud count differences. Each condition and seed uses its original completed-Test-window top-3% selection (K=5,950); captured frauds are attributed to their simulation steps before differences and seed means are formed. No per-step selection, capacity allocation or reranking is performed. Step contributions sum to the original whole-window H0/H1 estimates. Lines are visual guides to the realised sequence; the plot adds no uncertainty intervals and does not establish stationarity or a temporal mechanism. Simulation steps are not independent business days.
'''
(OUT/'figS7_caption.txt').write_text(caption,encoding='utf-8')
validation={'source_sha256':hashlib.sha256(LOCAL.read_bytes()).hexdigest(),'original_source_csv_sha256':DATA['source_sha256'],
    'source_rows':108,'steps_per_family':54,'step_range':[126,179],'transaction_total':198311,'fraud_total':2160,
    'frauds_per_step':40,'plotted_points':324,'series':6,'sum_H0_H1':sums,
    'new_fits':0,'new_predictions':0,'reranking':False,'new_intervals':0,'smoothing':False,
    'data_values_changed':False,'text_canvas_bounds':'PASS','pdf_width_mm':190,'pdf_height_mm':190*H/W,
    'font':REG,'minimum_font_px':20}
validation['figure']='S7'
validation.update(render_png(pdf_path, OUT/'figS7.png'))
write_verification(OUT/'figS7_verification.json', validation,
                   [OUT/'figS7.png', pdf_path, OUT/'figS7.svg'])
