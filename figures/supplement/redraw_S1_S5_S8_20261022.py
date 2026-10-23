"""Publication exports from byte-preserved aggregate figure inputs only.

No transaction records, labels, model scores, fitted models, or resampling
arrays are accessed.  Values and interval endpoints are read verbatim from
the three neighbouring JSON inputs.  Run in Python with matplotlib/numpy.
"""
from pathlib import Path
import os,sys,json,hashlib,csv
HERE=Path(__file__).resolve().parent
TASK=HERE.parent
if (TASK/'deps').exists():sys.path.insert(0,str(TASK/'deps'))
os.environ.setdefault('MPLCONFIGDIR',str(TASK/'mplcache'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.colors import TwoSlopeNorm
from pypdf import PdfReader
from PIL import Image

DPI=650
BLUE='#0072B2';ORANGE='#D55E00';GREEN='#009E73';INK='#111111';GRAY='#8A8A8A';GRID='#EFEFEF'
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':8.5,'axes.labelsize':8.5,'axes.titlesize':9.5,'xtick.labelsize':8,
 'ytick.labelsize':8,'legend.fontsize':8.5,'text.color':INK,'axes.labelcolor':INK,
 'axes.edgecolor':'#333333','xtick.color':INK,'ytick.color':INK,'axes.linewidth':.65,
 'xtick.major.width':.6,'ytick.major.width':.6,'xtick.major.size':2.7,'ytick.major.size':2.7,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path','savefig.facecolor':'white',
 'figure.facecolor':'white','axes.unicode_minus':True})
DATA={n:json.loads((HERE/f'fig{n}_data.json').read_text()) for n in ['S1','S5','S8']}
assert DATA['S1']['rows']==DATA['S8']['rows']

def clean(ax,grid=None):
 ax.spines[['top','right']].set_visible(False)
 if grid:ax.set_axisbelow(True);ax.grid(axis=grid,color=GRID,linewidth=.6)
def panel(ax,label):ax.set_title(label,loc='left',pad=7,fontweight='normal')
def font_info(pdf):
 page=PdfReader(pdf).pages[0];records=[]
 for _,ref in page['/Resources']['/Font'].items():
  font=ref.get_object();base=str(font.get('/BaseFont'))
  descendants=font.get('/DescendantFonts',[font])
  for desc in descendants:
   d=desc.get_object();fd=d.get('/FontDescriptor')
   if fd:
    fdo=fd.get_object();embedded=any(k in fdo for k in ['/FontFile','/FontFile2','/FontFile3'])
    records.append({'font':base,'subtype':str(font.get('/Subtype')),'embedded':embedded})
 assert records and all(x['embedded'] for x in records)
 return records
def finish(fig,name,scientific,annotations=None):
 fig.canvas.draw();renderer=fig.canvas.get_renderer();outside=[];text_records=[]
 for art in fig.findobj(match=Text):
  if not art.get_visible() or not art.get_text().strip():continue
  bb=art.get_window_extent(renderer)
  if not bb.width or not bb.height:continue
  text_records.append({'text':art.get_text(),'size_pt':art.get_fontsize()})
  if bb.x0<-.5 or bb.y0<-.5 or bb.x1>fig.bbox.width+.5 or bb.y1>fig.bbox.height+.5:outside.append(art.get_text())
 assert not outside,(name,outside)
 assert min(x['size_pt'] for x in text_records)>=8
 if annotations:
  boxes=[(a.get_text(),a.get_window_extent(renderer)) for a in annotations]
  for i,(txt,bb) in enumerate(boxes):
   # get_window_extent on Annotation includes leaders.  Actual text extents
   # are checked separately with the Text superclass.
   bb=Text.get_window_extent(annotations[i],renderer)
   for j in range(i):
    other=Text.get_window_extent(annotations[j],renderer)
    assert not bb.overlaps(other),(name,txt,boxes[j][0])
 fig.savefig(HERE/f'fig{name}.png',dpi=DPI)
 fig.savefig(HERE/f'fig{name}.pdf')
 fig.savefig(HERE/f'fig{name}.svg')
 with Image.open(HERE/f'fig{name}.png') as im:pixel=list(im.size)
 inches=list(fig.get_size_inches());effective=[pixel[i]/inches[i] for i in range(2)]
 assert min(effective)>=600
 info={'figure':name,'source_json_sha256':hashlib.sha256((HERE/f'fig{name}_data.json').read_bytes()).hexdigest(),
       'inches':inches,'pixels':pixel,'effective_dpi_xy':effective,'minimum_visible_font_pt':min(x['size_pt'] for x in text_records),
       'fonts_pdf':font_info(HERE/f'fig{name}.pdf'),'svg_text_outlined':True,'all_text_within_canvas':True,
       'scientific':scientific,'new_fits_scores_resampling':0,'text_records':text_records,
       'files':{ext:hashlib.sha256((HERE/f'fig{name}.{ext}').read_bytes()).hexdigest() for ext in ['png','pdf','svg']}}
 (HERE/f'fig{name}_verification_20261022.json').write_text(json.dumps(info,ensure_ascii=False,indent=2))
 plt.close(fig)

def draw_s1():
 d=DATA['S1'];fig,axs=plt.subplots(1,2,figsize=d['figsize_inches'])
 fig.subplots_adjust(left=.095,right=.985,bottom=.19,top=.87,wspace=.35)
 style={'History':(BLUE,'o'),'Weighting':(ORANGE,'s')}
 offsets={('paysim','History',1):(5,5),('paysim','History',2):(5,5),('paysim','History',3):(5,5),
          ('paysim','Weighting',1):(5,9),('paysim','Weighting',2):(-12,9),('paysim','Weighting',3):(7,-12),
          ('ieee_cis','History',1):(16,-7),('ieee_cis','History',2):(-28,4),('ieee_cis','History',3):(5,5),
          ('ieee_cis','Weighting',1):(-5,-19),('ieee_cis','Weighting',2):(-5,27),('ieee_cis','Weighting',3):(5,5)}
 plotted=[];near=[]
 for ax,ds,title in zip(axs,['paysim','ieee_cis'],['(a) PaySim','(b) IEEE-CIS']):
  clean(ax);ax.axhline(0,color=GRAY,ls='--',lw=.7);ax.axvline(0,color=GRAY,ls='--',lw=.7)
  for row in d['rows']:
   if row['dataset']!=ds:continue
   color,marker=style[row['component']];x,y=row['delta_ap'],row['delta_recall_pp']
   ax.scatter(x,y,c=color,marker=marker,s=25,zorder=3)
   leader=ds=='ieee_cis' and row['fold']<3
   a=ax.annotate(f"F{row['fold']}",(x,y),xytext=offsets[ds,row['component'],row['fold']],textcoords='offset points',fontsize=8,
       color=color if leader else INK,arrowprops={'arrowstyle':'-','color':color,'linewidth':.5,'shrinkA':1.5,'shrinkB':3.5} if leader else None)
   if leader:near.append(a)
   plotted.append({'dataset':ds,'fold':row['fold'],'component':row['component'],'x':x,'y':y})
  panel(ax,title);ax.set_xlabel('AP difference');ax.set_ylabel('Recall difference (pp)')
 axs[0].set(xlim=(-.17,.19),ylim=(-9,68),xticks=[-.1,0,.1],yticks=[0,20,40,60])
 axs[1].set(xlim=(-.025,.045),ylim=(-5,6),xticks=[-.02,0,.02,.04],yticks=[-4,-2,0,2,4,6])
 axs[0].legend(handles=[Line2D([],[],color=c,marker=m,linestyle='None',markersize=4.5,label=k) for k,(c,m) in style.items()],loc='upper left',frameon=False,handletextpad=.35,borderpad=.1)
 assert len(plotted)==12
 finish(fig,'S1',{'points':plotted,'intervals':0,'independent_dataset_axes':True,'all_four_IEEE_origin_labels_separated':True},near)

def draw_s5():
 d=DATA['S5'];fig=plt.figure(figsize=d['figsize_inches'])
 gs=fig.add_gridspec(2,2,height_ratios=[9,6],left=.17,right=.985,bottom=.115,top=.94,wspace=.82,hspace=.36)
 axs=[fig.add_subplot(gs[i,j]) for i in range(2) for j in range(2)]
 stage_styles={'raw':(BLUE,'o',.20,'Raw'),'clipped':(ORANGE,'s',0,'Clipped'),'platt':(GREEN,'^',-.20,'Platt')}
 labs={'P_H_BASE':'PH − P0','P_H_TIME':'PTH − PT','P_WEIGHT':'PTHW − PTH','I_HISTORY':'M1 − M0','I_WEIGHT':'M2 − M1'}
 plotted=[]
 for ax,ds,metric,letter in zip(axs,['paysim','paysim','ieee_cis','ieee_cis'],['delta_ap','delta_recall','delta_ap','delta_recall'],['(a)','(b)','(c)','(d)']):
  clean(ax,'x');ax.axvline(0,color=GRAY,lw=.7,ls='--')
  contrasts=['P_H_BASE','P_H_TIME','P_WEIGHT'] if ds=='paysim' else ['I_HISTORY','I_WEIGHT']
  keys=[(c,f) for c in contrasts for f in [1,2,3]]
  for index,(contrast,fold) in enumerate(keys):
   y=len(keys)-1-index
   rr=[r for r in d['rows'] if r['dataset']==ds and r['metric']==metric and r['contrast']==contrast and r['fold']==fold]
   assert len(rr)==(3 if ds=='paysim' else 1)
   for r in rr:
    stage=r['score_stage'] if ds=='paysim' else 'raw';color,marker,offset,_=stage_styles[stage]
    if ds=='ieee_cis':offset=0
    factor=100 if metric=='delta_recall' else 1
    est,low,high=[r[k]*factor for k in ['estimate','ci_lower','ci_upper']]
    ax.errorbar(est,y+offset,xerr=[[est-low],[high-est]],fmt=marker,ms=3.4,color=color,ecolor=color,elinewidth=.85,capsize=0,zorder=3)
    plotted.append({**r,'display_factor':factor,'display_estimate':est,'display_lower':low,'display_upper':high})
   if index in [2,5] and index<len(keys)-1:ax.axhline(y-.5,color=GRID,lw=.7,zorder=0)
  ax.set_yticks(range(len(keys)));ax.set_yticklabels([labs[c]+f'  F{f}' for c,f in reversed(keys)],fontsize=8)
  ax.set_ylim(-.65,len(keys)-.35);ax.set_xlabel('AP difference' if metric=='delta_ap' else 'Recall difference (pp)')
  panel(ax,letter+' '+('PaySim' if ds=='paysim' else 'IEEE-CIS'))
 axs[0].set(xlim=(-.17,.14),xticks=[-.1,0,.1]);axs[1].set(xlim=(-5,44),xticks=[0,20,40])
 axs[2].set(xlim=(-.022,.044),xticks=[-.02,0,.02,.04]);axs[3].set(xlim=(-5,5.5),xticks=[-5,0,5])
 fig.legend(handles=[Line2D([],[],color=c,marker=m,ls='-',ms=3.5,lw=.85,label=l) for c,m,_,l in stage_styles.values()],loc='lower center',bbox_to_anchor=(.5,.018),ncol=3,frameon=False,handletextpad=.4,columnspacing=2)
 assert len(plotted)==66
 finish(fig,'S5',{'intervals':plotted,'interval_count':66,'AP_intervals':33,'recall_intervals':33,'pointwise_interval_level':.95,'independent_axes':True})

def draw_s8():
 d=DATA['S8'];fig,axs=plt.subplots(2,2,figsize=d['figsize_inches'])
 fig.subplots_adjust(left=.125,right=.955,top=.91,bottom=.09,wspace=.65,hspace=.44)
 panels=[]
 for ax,ds,metric,letter in zip(axs.flat,['paysim','paysim','ieee_cis','ieee_cis'],['delta_ap','delta_recall_pp','delta_ap','delta_recall_pp'],['(a)','(b)','(c)','(d)']):
  arr=np.array([[next(r[metric] for r in d['rows'] if r['dataset']==ds and r['component']==c and r['fold']==f) for f in [1,2,3]] for c in ['History','Weighting']])
  vmax=float(abs(arr).max());im=ax.pcolormesh(np.arange(-.5,3,1),np.arange(-.5,2,1),arr,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-vmax,vcenter=0,vmax=vmax),shading='flat',rasterized=False)
  ax.set(xticks=range(3),xticklabels=['F1','F2','F3'],yticks=range(2),yticklabels=['History','Weighting'],xlim=(-.5,2.5),ylim=(1.5,-.5))
  ax.tick_params(length=0,pad=5)
  for sp in ax.spines.values():sp.set_visible(False)
  ax.set_xticks(np.arange(-.5,3,1),minor=True);ax.set_yticks(np.arange(-.5,2,1),minor=True);ax.grid(which='minor',color='white',lw=.7);ax.tick_params(which='minor',bottom=False,left=False)
  texts=[]
  for (i,j),v in np.ndenumerate(arr):
   dec=4 if metric=='delta_ap' else 2;color='white' if abs(v)/vmax>.55 else INK;txt=f'{v:+.{dec}f}'
   ax.text(j,i,txt,ha='center',va='center',fontsize=8.5,color=color);texts.append(txt)
  panel(ax,letter+' '+('PaySim' if ds=='paysim' else 'IEEE-CIS')+(': AP' if metric=='delta_ap' else ': Recall (pp)'))
  bar=fig.colorbar(im,ax=ax,fraction=.047,pad=.065,aspect=18,ticks=[-vmax,0,vmax]);bar.solids.set_rasterized(False);bar.solids.set_edgecolor('face');bar.outline.set_visible(False);bar.ax.tick_params(labelsize=8,length=2,pad=2)
  dec=3 if metric=='delta_ap' else 1;bar.set_ticklabels([f'{-vmax:.{dec}f}','0',f'{vmax:.{dec}f}'])
  panels.append({'dataset':ds,'metric':metric,'matrix':arr.tolist(),'display_strings':texts,'color_range':[-vmax,vmax]})
 finish(fig,'S8',{'panels':panels,'numeric_cells':24,'intervals':0,'independent_symmetric_scales':True,'paysim_history_is_PTH_minus_P0':True})

if __name__=='__main__':
 draw_s1();draw_s5();draw_s8()
 summary={'date':'2026-10-22','figures':{n:json.loads((HERE/f'fig{n}_verification_20261022.json').read_text()) for n in ['S1','S5','S8']},'original_input_jsons_preserved':True,'new_fits_scores_resampling':0}
 (HERE/'figures_s1_s5_s8_exports.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
 print(json.dumps({n:{k:v for k,v in x.items() if k in ['inches','pixels','effective_dpi_xy','minimum_visible_font_pt','fonts_pdf']} for n,x in summary['figures'].items()},indent=2))
