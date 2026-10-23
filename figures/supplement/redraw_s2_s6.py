"""Production redraws from unchanged S2 aggregates and retained S6 draw values.

Requires numpy, matplotlib, Pillow and PyMuPDF. No model, raw transaction,
new inference or new bootstrap code is called. The optional local deps folder
is used for this task's isolated plotting environment only.
"""
from pathlib import Path
import csv, hashlib, json, os, sys
HERE=Path(__file__).resolve().parent
DEPS=HERE.parent/'deps'
if DEPS.is_dir():sys.path.insert(0,str(DEPS))
os.environ.setdefault('MPLCONFIGDIR',str(HERE/'.mplconfig_s2_s6'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.text import Text
from PIL import Image
import pymupdf

INK='#25343D';BLUE='#234E70';TEAL='#008C95';GRAY='#7C8B93';GRID='#E8EEF1'
DPI=650
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.5,
 'text.color':INK,'axes.labelcolor':INK,'axes.edgecolor':GRAY,
 'xtick.color':INK,'ytick.color':INK,'axes.linewidth':.6,
 'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'path',
 'savefig.facecolor':'white','figure.facecolor':'white',
 'axes.unicode_minus':True,'path.simplify':False})
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def export(fig,stem,size):
    fig.canvas.draw();ren=fig.canvas.get_renderer()
    labels=[];outside=[]
    for t in fig.findobj(match=Text):
        if not t.get_visible() or not t.get_text().strip():continue
        b=t.get_window_extent(ren)
        if b.width==0 or b.height==0:continue
        labels.append({'text':t.get_text(),'point_size':float(t.get_fontsize()),'bounds_pixels':list(map(float,b.extents))})
        if b.x0<-.8 or b.y0<-.8 or b.x1>fig.bbox.width+.8 or b.y1>fig.bbox.height+.8:outside.append(t.get_text())
    assert not outside,outside
    assert min(x['point_size'] for x in labels)>=8
    outputs={}
    for ext in ['png','pdf','svg']:
        p=HERE/f'{stem}.{ext}'
        fig.savefig(p,dpi=DPI if ext=='png' else None,bbox_inches=None)
        outputs[ext]={'filename':p.name,'sha256':sha(p),'bytes':p.stat().st_size}
    plt.close(fig)
    with Image.open(HERE/f'{stem}.png') as im:
        pixels=list(im.size);meta=im.info.get('dpi')
    actual=[pixels[0]/size[0],pixels[1]/size[1]]
    assert min(actual)>=600
    pdf=pymupdf.open(HERE/f'{stem}.pdf');page=pdf[0]
    fonts=[]
    for row in page.get_fonts(full=True):
        xref,ext,kind,name,*_=row
        found=pdf.extract_font(xref)
        fonts.append({'xref':xref,'type':kind,'name':name,'embedded_bytes':len(found[3]),'embedded':bool(found[3])})
    assert fonts and all(x['embedded'] for x in fonts)
    assert not page.get_images()
    dimensions=[float(page.rect.width),float(page.rect.height)]
    assert np.allclose(dimensions,np.array(size)*72,atol=.001)
    # Render the vector PDF independently for final visual inspection.
    page.get_pixmap(matrix=pymupdf.Matrix(2,2),alpha=False).save(HERE/f'{stem}_pdf_preview.png')
    pdf.close()
    svg=(HERE/f'{stem}.svg').read_text()
    assert '<text' not in svg and '<image' not in svg
    return {'outputs':outputs,'physical_inches':list(size),'png_pixels':pixels,'png_dpi_metadata':meta,
            'effective_dpi_at_unchanged_docx_extent':actual,'pdf_points':dimensions,
            'pdf_fonts':fonts,'pdf_contains_raster_images':False,'svg_glyphs_outlined':True,
            'minimum_visible_font_pt':min(x['point_size'] for x in labels),'visible_text':labels,
            'text_outside_canvas':outside}

def s2():
    source=HERE/'figS2_data.json';d=json.loads(source.read_text());rows=d['rows']
    assert [r['fold'] for r in rows]==[1,2,3]
    size=(6.6,1.77375);w,h=np.array(size)*72
    fig=plt.figure(figsize=size,dpi=180)
    def tx(x,y,t,**kw):return fig.text(x/w,y/h,t,va='center',**kw)
    for y in [124,110.5]:
        fig.add_artist(Line2D([7/w,313/w],[y/h,y/h],transform=fig.transFigure,color=INK,lw=.5))
    tx(7,117.1,'Contrast / fold',fontsize=8.5)
    tx(113,117.1,'Estimate',fontsize=8.5,ha='center')
    tx(230,117.1,'95% CI',fontsize=8.5,ha='center')
    tx(7,98.7,'IEEE-CIS · C02 (M2 − M0) · Exact 3% review capacity',fontsize=8.5,fontweight='bold')
    row_y=[83,68,53]
    num=lambda v:f'{v:.3f}'.replace('-','−')
    for r,y in zip(rows,row_y):
        tx(7,y,f"Fold {r['fold']}",fontsize=8.5)
        tx(113,y,num(r['estimate_pp']),ha='center',fontsize=8.5)
        tx(230,y,f"({num(r['lower_pp'])}, {num(r['upper_pp'])})",ha='center',fontsize=8.5)
    bottom,top=45,124
    ax=fig.add_axes([335/w,bottom/h,(w-342)/w,(top-bottom)/h])
    ax.set_ylim(bottom,top);ax.set_xlim(-2,2);ax.set_xticks([-2,-1,0,1,2]);ax.set_yticks([])
    ax.spines[['left','top','right']].set_visible(False)
    ax.tick_params(axis='x',labelsize=8,length=2.7,width=.5,pad=2.5)
    for x in [-2,-1,1,2]:ax.axvline(x,color=GRID,lw=.65,linestyle=(0,(3.5,3.5)),zorder=0)
    ax.axvline(0,color=GRAY,lw=.75,linestyle=(0,(3.5,3.5)),zorder=1)
    plotted=[]
    for r,y in zip(rows,row_y):
        est,lo,hi=[r[k] for k in ['estimate_pp','lower_pp','upper_pp']]
        a=ax.errorbar(est,y,xerr=[[est-lo],[hi-est]],fmt='s',markersize=4.3,
                      color=BLUE,ecolor='#929A9F',elinewidth=.85,capsize=1.8,capthick=.65,zorder=3)
        seg=a.lines[2][0].get_segments()[0]
        got={'fold':r['fold'],'estimate_pp':float(a.lines[0].get_xdata()[0]),'lower_pp':float(seg[0][0]),'upper_pp':float(seg[1][0])}
        assert all(abs(got[k]-r[k])<1e-12 for k in ['estimate_pp','lower_pp','upper_pp']);plotted.append(got)
    ax.set_xlabel('ΔRecall@3% (pp)',fontsize=8.5,labelpad=4)
    tx(7,24,'Five-seed means; pointwise conditional 95% CIs.',fontsize=8)
    tx(7,10,'Paired moving-block percentiles; crossing zero does not establish equivalence.',fontsize=8)
    result=export(fig,'figS2',size)
    result.update({'figure':'S2','source':source.name,'source_sha256':sha(source),'plotted_values_checked':plotted,
       'points':3,'intervals':3,'preserved':'All estimates, original pointwise conditional 95% paired moving-block percentile intervals, exact3% decision and five-seed means',
       'new_inference':False,'new_model_scores':0,'new_resampling':0})
    return result

def s6():
    source=HERE/'figS6_existing_5000.csv'
    with source.open(encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    rows=sorted(rows,key=lambda r:int(r['replicate_id']))
    assert len(rows)==5000 and [int(r['replicate_id']) for r in rows]==list(range(1,5001))
    values=np.array([float(r['value']) for r in rows]);assert np.isfinite(values).all()
    x=np.sort(values);unique,counts=np.unique(x,return_counts=True);y=np.cumsum(counts)/len(x)
    # Draw the exact right-continuous empirical CDF. A left baseline and right
    # plateau are display extensions only, outside the first/last observed x.
    xx=np.r_[unique[0],unique,unique[-1]];yy=np.r_[0,y,1]
    assert np.all(np.diff(xx)>=0) and np.all(np.diff(yy)>=0) and yy[-1]==1
    ci=np.quantile(values,[.025,.975],method='linear');assert np.allclose(ci,[-166.2,746.8],rtol=0,atol=1e-10)
    size=(6.6,2.834916885)
    fig=plt.figure(figsize=size,dpi=180)
    ax=fig.add_axes([.105,.20,.875,.64])
    ax.step(xx,yy,where='post',color=BLUE,lw=1.15,zorder=3)
    ax.axvline(39.4,color=INK,lw=.95,linestyle=(0,(3,2.5)),label='Observed: +39.4',zorder=2)
    ax.axvline(-166.2,color=TEAL,lw=.95,linestyle=(0,(5,3)),label='Original 2.5%: −166.2',zorder=2)
    ax.axvline(746.8,color=TEAL,lw=.95,linestyle=(0,(5,3)),label='Original 97.5%: 746.8',zorder=2)
    ax.set_xlim(-280,1170);ax.set_ylim(-.05,1.05)
    ax.set_xticks([-200,0,200,400,600,800,1000]);ax.set_yticks([0,.2,.4,.6,.8,1])
    ax.tick_params(labelsize=8,length=3,width=.6)
    ax.set_xlabel('Paired ΔTP at 3% capacity',fontsize=9,labelpad=4)
    ax.set_ylabel('Empirical cumulative proportion',fontsize=9,labelpad=4)
    ax.spines[['top','right']].set_visible(False);ax.grid(color=GRID,lw=.55,zorder=0)
    ax.legend(loc='lower right',fontsize=8,frameon=True,facecolor='white',edgecolor='#DCE5EA',framealpha=.97,
              borderpad=.5,labelspacing=.4,handlelength=2.2)
    fig.text(.54,.94,'PaySim · Fold 3 · PTHW − PTH · LightGBM · Platt scores',ha='center',va='center',fontsize=9)
    result=export(fig,'figS6',size)
    result.update({'figure':'S6','source':source.name,'source_sha256':sha(source),'original_draws_read':5000,
      'observed_effect':39.4,'original_ci':[-166.2,746.8],'recomputed_ci_from_retained_draws':ci.tolist(),
      'draw_min':float(values.min()),'draw_max':float(values.max()),'unique_draw_values':len(unique),
      'ecdf_definition':'Right-continuous empirical CDF: cumulative number of retained draws <=x /5000; ties combined exactly; no smoothing',
      'curve_data_sha256':hashlib.sha256(np.column_stack([unique,y]).astype('<f8').tobytes()).hexdigest(),
      'preserved':'Retained5000draws; all three original marker values; 3% capacity, Fold3, fixed LightGBM models and Platt score stage',
      'new_model_scores':0,'new_resampling':0,'new_inference':False})
    with (HERE/'figS6_ecdf_plot_data.csv').open('w',newline='') as f:
        wr=csv.writer(f);wr.writerow(['delta_tp','cumulative_proportion','multiplicity'])
        wr.writerows(zip(unique,y,counts))
    return result

if __name__=='__main__':
    result={'date':'2026-10-22','python':sys.version,'matplotlib':matplotlib.__version__,'numpy':np.__version__,
      'task':'S2/S6 production outputs at unchanged Word figure extents','figures':[s2(),s6()],
      'raster_upsampling_used':False,'requires_final_visual_review':True}
    target=HERE/'figures_s2_s6_validation.json'
    target.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps({f['figure']:{k:f[k] for k in ['png_pixels','effective_dpi_at_unchanged_docx_extent','minimum_visible_font_pt','pdf_fonts']} for f in result['figures']},ensure_ascii=False,indent=2))
