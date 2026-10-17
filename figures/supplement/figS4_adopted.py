"""Draw the manuscript-adopted score-stage contrast view from frozen means.

Unlike the historical figS4.py view, the third contrast is PTH-P0 (P_BUNDLE),
not PTHW-PTH. No fitting, scoring, calibration, or resampling is performed.
Requires reportlab and pymupdf; run from any working directory.
"""
from pathlib import Path
import csv
import hashlib
import json
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.colors import HexColor
import pymupdf

HERE = Path(__file__).resolve().parent
SOURCE = HERE / 'figS4_adopted_data.csv'
ROWS = list(csv.DictReader(SOURCE.open(encoding='utf-8-sig', newline='')))
W, H = 481.8, 345.72
for name, file in [('Arial','Arial.ttf'),('Arial-Bold','Arial Bold.ttf')]:
    path = Path('/System/Library/Fonts/Supplemental') / file
    if path.exists(): pdfmetrics.registerFont(TTFont(name,str(path)))
REG = 'Arial' if 'Arial' in pdfmetrics.getRegisteredFontNames() else 'Helvetica'
BOLD = 'Arial-Bold' if 'Arial-Bold' in pdfmetrics.getRegisteredFontNames() else 'Helvetica-Bold'
c = canvas.Canvas(str(HERE / 'figS4_adopted.pdf'), pagesize=(W,H))
c.setTitle('PaySim score-processing contrasts in the adopted manuscript')
c.setAuthor('')

def text(x,y,s,size=9,color='#172735',bold=False):
    c.setFillColor(HexColor(color));c.setFont(BOLD if bold else REG,size);c.drawString(x,H-y,s)
def line(x1,y1,x2,y2,color='#9AA4A8',width=.7):
    c.setStrokeColor(HexColor(color));c.setLineWidth(width);c.line(x1,H-y1,x2,H-y2)

text(14,19,'Score-processing contrasts',13,bold=True)
text(14,38,'PaySim · LightGBM · Raw versus clipped / Platt AP contrasts',9,color='#59676E')
colors={1:'#AD6345',2:'#8A7195',3:'#667D56'}
left,right,top,bottom=142,461,90,284
lo,hi=-.24,.22
def xx(v):return left+(v-lo)/(hi-lo)*(right-left)
line(xx(0),top-12,xx(0),bottom,color='#7F898D',width=.8)
point_rows=[]
for g,(contrast,label) in enumerate([('P_H_BASE','History above P0'),('P_H_TIME','History above PT'),('P_BUNDLE','Bundle above P0')]):
    for f in [1,2,3]:
        y=top+(g*3+f-1)*22.5
        sel={r['score_stage']:float(r['delta_ap']) for r in ROWS if int(r['fold'])==f and r['contrast']==contrast}
        assert set(sel)=={'raw','clipped','platt'}
        assert abs(sel['clipped']-sel['platt'])<1e-14
        text(44,y+3,label+f' · F{f}',9)
        color=colors[f];line(xx(sel['raw']),y,xx(sel['platt']),y,color,1.2)
        c.setStrokeColor(HexColor(color));c.setFillColor(HexColor('#FFFFFF'));c.setLineWidth(1.1)
        c.circle(xx(sel['raw']),H-y,3.1,stroke=1,fill=1)
        c.setFillColor(HexColor(color));c.rect(xx(sel['platt'])-2.7,H-y-2.7,5.4,5.4,fill=1,stroke=0)
        point_rows.append({'contrast':contrast,'fold':f,**sel})
    if g<2:line(left,top+(g*3+2)*22.5+11,right,top+(g*3+2)*22.5+11,'#D9E0E3',.6)
line(left,bottom,right,bottom)
for val in [-.2,-.1,0,.1,.2]:
    x=xx(val);line(x,bottom,x,bottom+3);text(x-8,bottom+14,f'{val:.1f}',8)
text(205,320,'AP difference (original AP units)',10)
c.setStrokeColor(HexColor('#465357'));c.setFillColor(HexColor('#FFFFFF'));c.circle(192,H-60,3,stroke=1,fill=1)
text(202,63,'Raw',9)
c.setFillColor(HexColor('#465357'));c.rect(264,H-63,6,6,stroke=0,fill=1);text(279,63,'Clipped / Platt',9)
text(14,339,'Five-seed means; connectors denote processing, not uncertainty or time.',8,color='#59676E')
text(xx(.06),top+7*22.5-7,'+0.0390 → -0.0041',9,color=colors[2])
c.save()
with pymupdf.open(HERE/'figS4_adopted.pdf') as d:
    pix=d[0].get_pixmap(matrix=pymupdf.Matrix(600/72,600/72),alpha=False)
    pix.set_dpi(600,600);pix.save(HERE/'figS4_adopted.png')
(HERE/'figS4_adopted_validation.json').write_text(json.dumps({
    'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'contrasts':['P_H_BASE','P_H_TIME','P_BUNDLE'], 'points':point_rows,
    'clipped_platt_equal':True,'no_new_fitting_scoring_or_resampling':True,
    'historical_view':'figS4.py and figS4_data.json retain the earlier P_WEIGHT selection',
    'png_dpi':600},indent=2))
print('Adopted S4 exported: 9 raw-versus-clipped/Platt contrasts; P_BUNDLE retained.')
