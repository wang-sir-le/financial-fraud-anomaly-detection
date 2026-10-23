"""Native vector reconstruction of the user's second reference image.

The matrices, icons, score strips and split bars are conceptual glyphs. No
experimental scatterplots, heatmaps, score samples or performance curves are
invented. All visible lettering remains editable SVG text. Pillow and lxml
are used for source validation; no network access is used.
"""
from pathlib import Path
from html import escape
import math
import json

from PIL import ImageFont
from lxml import etree as ET

OUT=Path(__file__).resolve().parent
STEM='fig02_protocol_labels_20261020'
W,H=2400,1400
REG=Path('/System/Library/Fonts/Supplemental/Arial.ttf')
BOLD=Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')
C=dict(ink='#102846',muted='#536d89',blue='#3477b7',line='#c4d9eb',
       pale='#eff7fd',green='#6b9849',orange='#ed8127',purple='#7953b1',
       arrow='#4b637d',cream='#fbf6eb')
parts=[]
audit=[]

def emit(tag,attrs,body=''):
    a=' '.join(f'{k.replace("_","-")}="{escape(str(v),quote=True)}"' for k,v in attrs.items())
    parts.append(f'<{tag} {a}>{body}</{tag}>')

def rect(x,y,w,h,fill='white',stroke=None,r=12,sw=1.3,dash=None):
    a=dict(x=x,y=y,width=w,height=h,rx=r,fill=fill,stroke=stroke or C['line'],stroke_width=sw)
    if dash:a['stroke_dasharray']=dash
    emit('rect',a)

def line(x1,y1,x2,y2,color=None,sw=2,dash=None):
    a=dict(x1=x1,y1=y1,x2=x2,y2=y2,stroke=color or C['blue'],stroke_width=sw,stroke_linecap='round')
    if dash:a['stroke_dasharray']=dash
    emit('line',a)

def circle(x,y,r,fill,stroke='none',sw=2):
    emit('circle',dict(cx=x,cy=y,r=r,fill=fill,stroke=stroke,stroke_width=sw))

def ellipse(x,y,rx,ry,fill,stroke=None,sw=2):
    emit('ellipse',dict(cx=x,cy=y,rx=rx,ry=ry,fill=fill,stroke=stroke or C['blue'],stroke_width=sw))

def path(d,fill='none',stroke=None,sw=2):
    emit('path',dict(d=d,fill=fill,stroke=stroke or C['blue'],stroke_width=sw,
                     stroke_linecap='round',stroke_linejoin='round'))

def poly(points,color=None,sw=3,fill='none',closed=False):
    emit('polygon' if closed else 'polyline',dict(points=' '.join(f'{x},{y}' for x,y in points),
         fill=fill,stroke=color or C['blue'],stroke_width=sw,stroke_linecap='round',stroke_linejoin='round'))

def arrow(points,color=None,sw=4,size=12):
    color=color or C['arrow'];poly(points,color,sw)
    px,py=points[-2];x,y=points[-1];a=math.atan2(y-py,x-px)
    p=(x-size*math.cos(a)+size*.55*math.sin(a),y-size*math.sin(a)-size*.55*math.cos(a))
    q=(x-size*math.cos(a)-size*.55*math.sin(a),y-size*math.sin(a)+size*.55*math.cos(a))
    poly([(x,y),p,q],color,1,color,True)

def text(x,y,s,size=31,bold=False,color=None,anchor='start',maxw=None):
    size=max(size,41)
    f=ImageFont.truetype(str(BOLD if bold else REG),size)
    width=f.getlength(s)
    left=x-width/2 if anchor=='middle' else x-width if anchor=='end' else x
    b=f.getbbox(s,anchor='ls');bounds=[left+b[0],y+b[1],left+b[2],y+b[3]]
    emit('text',dict(x=x,y=y,font_family='Arial, Helvetica, sans-serif',font_size=size,
         font_weight=700 if bold else 400,fill=color or C['ink'],text_anchor=anchor),escape(s))
    audit.append(dict(text=s,size=size,width=width,max_width=maxw,bounds=bounds))

def lines(x,y,ss,size=31,gap=37,bold=False,color=None,maxw=None):
    for i,s in enumerate(ss):text(x,y+i*gap,s,size,bold,color,maxw=maxw)

def card(x,y,w,h,fill='white',stroke=None,r=14):
    rect(x,y,w,h,fill,stroke,r)

def database(x,y,w=55,h=76,color=None):
    color=color or C['blue']
    path(f'M{x},{y+12} L{x},{y+h-11} C{x},{y+h+3} {x+w},{y+h+3} {x+w},{y+h-11} L{x+w},{y+12}',
         'white',color,3)
    for yy in [y+12,y+32,y+52]:
        path(f'M{x},{yy} C{x},{yy+14} {x+w},{yy+14} {x+w},{yy}',stroke=color,sw=3)
    ellipse(x+w/2,y+12,w/2,11,'#edf6fc',color,3)

def document(x,y,w=70,h=100,checks=False,dots=False):
    rect(x,y,w,h,'#f8fcff','#78a2c5',7,2)
    for i in range(3):
        yy=y+23+i*(h-31)/3
        if checks:
            rect(x+10,yy-9,16,16,C['blue'],C['blue'],2,1)
            poly([(x+13,yy-1),(x+17,yy+3),(x+23,yy-5)],'white',2)
            line(x+34,yy,x+w-12,yy,'#b2ccdf',2)
        elif dots:
            circle(x+18,yy,8,['#d96d59','#e6a25e','#71a5cb'][i])
            line(x+34,yy,x+w-12,yy,'#b2ccdf',2)
        else:line(x+14,yy,x+w-14,yy,'#86abc9',3)

def clock(x,y,r=22,color=None):
    col=color or C['blue'];circle(x,y,r,'white',col,3)
    poly([(x,y-r*.57),(x,y),(x+r*.46,y+3)],col,3)

def phone(x,y,w=47,h=79):
    rect(x,y,w,h,'#f5fbff',C['blue'],6,3)
    line(x+10,y+12,x+w-10,y+12,C['blue'],2)
    circle(x+w/2,y+h-11,3,C['blue'])
    text(x+w/2,y+51,'$',29,True,C['blue'],'middle')

def bankcard(x,y,w=110,h=65):
    rect(x,y,w,h,'#f3f8fe',C['blue'],7,3)
    rect(x,y+14,w,14,C['blue'],C['blue'],0,0)
    rect(x+12,y+40,19,12,'#cee1ef','#84a9c7',2,1)
    line(x+46,y+50,x+w-12,y+50,'#8aabc7',3)

def shop(x,y,w=125,h=90):
    rect(x+11,y+24,w-22,h-24,'#f3f8fc',C['blue'],3,3)
    poly([(x,y+24),(x+13,y),(x+w-13,y),(x+w,y+24)],C['blue'],3,'#d8e9f7',True)
    for j in range(5):rect(x+j*w/5,y+24,w/5,14,'#7f9fbd' if j%2 else '#d2e4f3',C['blue'],1,1)
    rect(x+22,y+53,30,37,'white',C['blue'],2,2)
    rect(x+66,y+53,35,22,'white',C['blue'],2,2)

def cart(x,y):
    poly([(x,y),(x+8,y),(x+15,y+30),(x+47,y+30),(x+56,y+7),(x+12,y+7)],C['muted'],3,'#8397a9',True)
    for xx in [x+20,x+43]:circle(xx,y+40,4,C['muted'])

def pin(x,y):
    path(f'M{x},{y+46} C{x-40},{y+2} {x-20},{y-20} {x},{y-20} C{x+20},{y-20} {x+40},{y+2} {x},{y+46} Z',
         '#7790a7','#7790a7',1)
    circle(x,y,7,'white')

def model_tree(x,y):
    nodes=[(x,y),(x-42,y+52),(x+42,y+52),(x-68,y+102),(x-24,y+102),(x+22,y+102),(x+67,y+102)]
    for i,j in [(0,1),(0,2),(1,3),(1,4),(2,5),(2,6)]:line(*nodes[i],*nodes[j],C['muted'],4)
    for i,(xx,yy) in enumerate(nodes):circle(xx,yy,21 if i==0 else 14,'url(#node)','#70889f',1)

def gear(x,y,r):
    pts=[]
    for i in range(48):
        a=i*2*math.pi/48
        rr=r*(1 if i%4 in [0,1] else .78)
        pts.append((x+rr*math.cos(a),y+rr*math.sin(a)))
    poly(pts,'#718aa3',1,'url(#gear)',True)
    circle(x,y,r*.39,'#f5f8fb','#6e87a0',2)

def grid(x,y,w,h,cols=5,rows=7,kind='raw',opacity=1):
    # Constant column accents indicate matrix structure, never observed values.
    parts.append(f'<g opacity="{opacity}">')
    for r in range(rows):
        for c in range(cols):
            fill='white'
            if kind=='raw' and c in [0,cols-1]:fill='#d6eaf8'
            if kind=='past' and c==cols-2:fill='#f5e7c8'
            if kind=='scores' and c==0:fill='#bdddf3'
            rect(x+c*w/cols,y+r*h/rows,w/cols,h/rows,fill,'#89a7c0',0,.9)
    parts.append('</g>')

def draw():
    parts.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="167.64mm" height="97.79mm" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">')
    parts.append('<title id="title">Study-specific experimental framework</title>')
    parts.append('<desc id="desc">Current-input baselines and optional temporal-history increments define separate component configurations. W denotes positive-class weight two in PTHW. Models fit Train labels independently by dataset. PaySim uses raw, clipped and Platt scores, with earlier-Validation calibration and later-Validation cost-minimising threshold selection under a 3 percent alert cap. Full-Test-score AP is separate from decision-based evaluation. PaySim also has separate exact-3-percent component contrasts; IEEE-CIS and BankSim use a primary 3 percent exact top-k budget on completed Test windows. Test labels are used only for metrics, never review selection. History construction excludes fraud labels and current groups. All icons and matrices are schematic.</desc>')
    parts.append('''<defs>
      <linearGradient id="outerBlue" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#e8f3fb"/><stop offset="1" stop-color="#f5faff"/></linearGradient>
      <linearGradient id="outerCream" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#faf4e8"/><stop offset="1" stop-color="#fffdf8"/></linearGradient>
      <linearGradient id="modelFill" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f5fbff"/><stop offset="1" stop-color="#e5f1fb"/></linearGradient>
      <linearGradient id="node" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#526d88"/><stop offset="1" stop-color="#8499ad"/></linearGradient>
      <linearGradient id="gear" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#4d6c8b"/><stop offset="1" stop-color="#a0b5c7"/></linearGradient>
      <linearGradient id="scoreBand" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#d8ebf9"/><stop offset=".55" stop-color="#91badd"/><stop offset="1" stop-color="#3a77ae"/></linearGradient>
    </defs>''')
    rect(0,0,W,H,'white','white',0,0)
    rect(24,70,710,1280,'url(#outerBlue)','#c7dfef',26,1.2)
    rect(795,70,700,1280,'url(#outerCream)','#ecdfc7',26,1.2)
    rect(1550,70,825,1280,'url(#outerBlue)','#c7dfef',26,1.2)
    text(379,126,'Features & supervised learning',44,True,anchor='middle',maxw=662)
    text(1145,126,'Datasets & model training',44,True,anchor='middle',maxw=650)
    text(1962,126,'Scores & review decisions',44,True,anchor='middle',maxw=770)

    # Features: icons and stacked structural matrices closely match the template.
    rect(45,165,302,360,'#fbfdff','#8fb7d7',12,1.4,'5 4')
    rect(405,165,302,360,'#fbfdff','#8fb7d7',12,1.4,'5 4')
    text(196,214,'Raw / current',41,True,anchor='middle',maxw=282)
    text(196,262,'attributes',41,True,anchor='middle',maxw=282)
    text(556,214,'Past-only',41,True,anchor='middle',maxw=282)
    text(556,262,'history',41,True,anchor='middle',maxw=282)
    database(81,276,35,51)
    cart(73,364);clock(98,454,21,C['muted'])
    grid(150,300,152,188,4,8,'raw')
    text(320,500,'…',41,anchor='middle')
    grid(429,300,159,115,5,5,'past')
    grid(453,325,159,115,5,5,'past')
    grid(477,350,159,115,5,5,'past')
    arrow([(445,487),(667,487)],C['blue'],2.4,9)
    text(196,575,'Baseline',41,anchor='middle',maxw=282)
    text(556,575,'Optional',41,anchor='middle',maxw=282)
    arrow([(196,590),(196,615)],C['arrow'],4,10)
    arrow([(556,590),(556,615)],C['arrow'],4,10)
    # Configuration choices replace the mandatory Raw-plus-history join.
    card(70,630,640,370,'#ffffff','#bcd8ed',16)
    text(95,675,'Contrast configurations',41,True,maxw=590)
    text(95,725,'PaySim: P0 = Raw',41,maxw=590)
    text(95,767,'PT · PH · PTH · PTHW',41,maxw=590)
    text(95,809,'T: time transforms · H: history',41,color=C['muted'],maxw=590)
    text(95,851,'W: positive-class weight 2',41,color=C['muted'],maxw=590)
    text(95,893,'IEEE-CIS: M0–M3',41,maxw=590)
    text(95,935,'BankSim: B0 / B0H',41,maxw=590)
    text(95,977,'B1 / B1H',41,maxw=590)
    # Model family specification is routed to fitting, not to a different dataset.
    card(70,1030,640,285,'url(#modelFill)',C['blue'],16)
    model_tree(166,1135)
    text(270,1073,'Supervised models',41,True,maxw=414)
    # All model methods share the same type style at the manuscript's 8.118 pt.
    model_font_size=41
    text(270,1118,'PaySim / IEEE-CIS',41,color=C['muted'],maxw=430)
    text(270,1166,'LGBM · Logistic',model_font_size,maxw=430)
    text(270,1210,'BankSim',41,color=C['muted'],maxw=430)
    text(270,1258,'LGBM · XGB',model_font_size,maxw=430)

    # Dataset cards keep the verified source-release counts, replacing fake plots.
    for y,name,color,ss,kind in [
        (165,'PaySim',C['green'],['6,362,620 synthetic','mobile-money transactions','Recipient history'],'phone'),
        (365,'IEEE-CIS',C['orange'],['590,540 labelled','transactions','Card-related history'],'card'),
        (565,'BankSim',C['purple'],['594,643 simulated','transactions','Merchant history'],'shop')]:
        card(815,y,660,190,'#fffefa','#dfd8cc',14)
        rect(842,y+16,245,55,color,color,7,0)
        text(964,y+56,name,38,True,'white','middle',222)
        text(844,y+117,'Source: '+ss[0].split()[0],41,maxw=378)
        text(844,y+169,ss[2],41,color=C['muted'],maxw=382)
        if kind=='phone':
            database(1254,y+66,64,78,C['green']);phone(1353,y+91,44,72)
            arrow([(1329,y+114),(1345,y+114)],C['green'],2.5,7)
        elif kind=='card':
            bankcard(1252,y+73,110,65);bankcard(1293,y+105,110,65)
        else:shop(1256,y+73,134,91)
    poly([(1010,773),(1010,791),(1260,791),(1260,773)],C['arrow'],6)
    arrow([(1135,791),(1135,852)],C['arrow'],6,15)
    card(845,870,610,210,'url(#modelFill)',C['blue'],17)
    gear(926,954,44);gear(979,909,28)
    text(1035,921,'Training & scoring',35,True,maxw=390)
    text(1035,964,'Fit on Train labels',41,maxw=390)
    text(1035,1005,'Per-workflow fits',41,maxw=390)
    text(1150,1053,'Weights: PaySim / IEEE-CIS',41,color=C['muted'],anchor='middle',maxw=570)
    arrow([(1135,1094),(1135,1127)],C['arrow'],5,12)
    card(850,1145,600,195,'#ffffff','#ded9cf',15)
    text(1150,1187,'Model scores',35,True,anchor='middle',maxw=550)
    lines(875,1235,['PaySim: raw / clipped / Platt','IEEE-CIS: raw scores',
                    'BankSim: raw margins'],41,45,maxw=550)
    # Feature representations and family choices feed independent dataset fitting.
    arrow([(720,815),(758,815),(758,978),(832,978)],C['arrow'],6,14)
    arrow([(720,1172),(777,1172),(777,978),(832,978)],C['arrow'],5,13)

    # Scores branch directly to AP and to dataset-specific review rules.
    poly([(1459,1248),(1516,1248),(1516,225)],C['arrow'],6)
    for yy in [225,488]:arrow([(1516,yy),(1556,yy)],C['arrow'],6,14)

    # Abstract score-list and checklist glyphs do not depict observations.
    card(1570,165,785,120,'#ffffff','#bcd5e8',16)
    rect(1595,189,66,70,'#f8fcff','#78a2c5',7,2)
    for yy,ww in [(207,35),(225,26),(243,18)]:
        circle(1607,yy,3,C['blue'])
        line(1617,yy,1617+ww,yy,C['blue'],3)
    text(1694,211,'Ranking',41,True,maxw=540)
    text(1694,264,'All Test scores + labels',41,maxw=540)
    rect(2250,192,85,65,'#e6f1fb','#c4d9eb',10,1)
    text(2292,240,'AP',52,True,C['blue'],'middle',75)

    card(1570,305,785,150,'#ffffff','#bcd5e8',16)
    document(1595,339,66,87,checks=True)
    text(1694,350,'Review',41,True,maxw=330)
    text(2330,350,'Test labels: metrics only',41,color=C['muted'],anchor='end',maxw=460)
    text(1694,398,'TP / FP / FN · Precision · Recall',41,maxw=640)
    text(1694,441,'Alert rate: A/N',41,maxw=640)

    # One shared heading replaces the former nested outer decision card.
    text(1595,506,'Review rules',41,True,maxw=345)
    card(1570,530,375,340,'#ffffff','#bad3e7',14)
    card(1960,530,395,340,'#ffffff','#bad3e7',14)
    text(1595,574,'PaySim transfer',41,True,maxw=325)
    # Threshold-control glyph, clear of the title and protocol lettering.
    for yy,xx in [(552,1922),(573,1933)]:
        line(1915,yy,1940,yy,C['blue'],2.5)
        circle(xx,yy,5,'white',C['blue'],2)
    lines(1595,630,['Early Val → Platt','Late Val → τ*','min cost; A/N≤3%',
                    'Test: same τ*','Variable alerts'],41,48,maxw=325)
    text(2000,574,'Exact top-k',41,True,maxw=260)
    rect(2288,550,46,45,'#f8fcff','#78a2c5',5,2)
    for yy in [560,572,584]:
        poly([(2294,yy),(2297,yy+3),(2302,yy-3)],C['blue'],1.8)
        line(2308,yy,2326,yy,'#86abc9',2)
    # Dataset-specific budgets remain separate from threshold transfer.
    text(1971,630,'IEEE-CIS / BankSim',41,maxw=373)
    text(1971,670,'Primary 3%',41,maxw=373)
    text(1971,722,'PaySim: component',41,maxw=373)
    text(1971,762,'contrasts at 3%',41,maxw=373)
    text(1971,814,'Completed Test',41,maxw=373)
    text(1971,854,'windows',41,maxw=373)
    # Both decision rules feed the review branch; AP bypasses these rules.
    poly([(1767,530),(1767,520),(2167,520),(2167,530)],C['arrow'],3)
    arrow([(1962,520),(1962,459)],C['arrow'],3,7)

    # Temporal split bars are schematic, not proportional to actual row counts.
    card(1570,895,785,190,'#ffffff','#bad3e7',14)
    text(1595,940,'Time splits',41,True,maxw=360)
    text(2330,940,'Train / Val / Test',41,color=C['muted'],anchor='end',maxw=330)
    text(1595,990,'PaySim / IEEE-CIS',41,maxw=460)
    text(1595,1045,'3 folds',41,maxw=170)
    for i,a in enumerate([110,135,160]):
        xx,yy=1785,1012+17*i
        rect(xx,yy,a,8,'#c4deef','#9fc2db',0,.8)
        rect(xx+a,yy,27,8,'#efdfbe','#d2bd92',0,.8)
        rect(xx+a+27,yy,45,8,'#7facd3','#608fba',0,.8)
    text(2100,990,'BankSim',41,maxw=230)
    text(2100,1045,'1 Test',41,maxw=230)
    xx,yy=2100,1063
    rect(xx,yy,145,8,'#c4deef','#9fc2db',0,.8)
    rect(xx+145,yy,30,8,'#efdfbe','#d2bd92',0,.8)
    rect(xx+175,yy,45,8,'#7facd3','#608fba',0,.8)

    card(1570,1110,785,65,'#ffffff','#bad3e7',12)
    clock(1602,1142,18,C['muted'])
    text(1638,1155,'History: past-only · no fraud labels',41,maxw=690)

    card(1570,1200,395,140,'#ffffff','#bad3e7',12)
    card(1980,1200,375,140,'#ffffff','#bad3e7',12)
    text(1590,1238,'Analytical cost',41,True,maxw=355)
    # Calculator and CI glyphs convey metric types without numerical results.
    rect(1595,1253,55,40,'#f8fcff','#78a2c5',5,2)
    rect(1603,1259,39,9,'#d5e8f6','#78a2c5',1,1)
    for xx in [1606,1620,1634]:
        for yy in [1276,1286]:rect(xx,yy,6,4,C['blue'],C['blue'],1,0)
    text(1680,1284,'FP + c × FN',41,maxw=265)
    text(1590,1326,'PaySim, IEEE-CIS',41,maxw=355)
    text(2000,1238,'Conditional CIs',41,True,maxw=334)
    text(2000,1284,'Paired bootstrap',41,maxw=334)
    text(2000,1326,'Fixed models',41,maxw=250)
    line(2263,1321,2333,1321,C['blue'],2.5)
    for xx in [2263,2333]:line(xx,1313,xx,1329,C['blue'],2.5)
    circle(2298,1321,5,C['blue'])
    parts.append('</svg>')

if __name__=='__main__':
    draw()
    failures=[a for a in audit if a['max_width'] is not None and a['width']>a['max_width']]
    if failures:raise ValueError(json.dumps(failures,ensure_ascii=False,indent=2))
    for a in audit:
        x1,y1,x2,y2=a['bounds']
        assert x1>=0 and y1>=0 and x2<=W and y2<=H,a
    xml='\n'.join(parts)
    root=ET.fromstring(xml.encode('utf-8'),parser=ET.XMLParser(resolve_entities=False,no_network=True))
    (OUT/(STEM+'.svg')).write_text(xml,encoding='utf-8')
    report=dict(canvas=[W,H],native_size_mm=[167.64,97.79],editable_text_elements=len(audit),
                min_native_font_pt=min(a['size'] for a in audit)*167.64/25.4*72/W,
                font_at_original_manuscript_167_64mm=min(a['size'] for a in audit)*167.64/25.4*72/W,
                raster_embeds=len(root.findall('.//{http://www.w3.org/2000/svg}image')),
                experimental_curves_or_observations=False,text_width_checks='PASS',text=audit)
    (OUT/'checks.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='text'},ensure_ascii=False,indent=2))
