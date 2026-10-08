"""Restyle a completed ECDF display path, without reading or inferring draws.

The verbatim PDF path comes from figS6.sourcevectors.json.  The same uniform
page transformation applies to its path, axis rectangle, ticks and marker
positions.  Scientific coordinates are not recovered from that path.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pymupdf

HERE = Path(__file__).resolve().parent
BLUE = '#234E70'
TEAL = '#008C95'
INK = '#263746'


def rgb_pdf(hex_color: str) -> str:
    return ' '.join(f'{int(hex_color[i:i + 2], 16) / 255:.10f}' for i in (1, 3, 5))


def main() -> None:
    source_path = HERE / 'figS6.sourcevectors.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    curve = source['curve_path_verbatim_pdf_operators']
    assert hashlib.sha256(curve.encode('latin1')).hexdigest() == source['curve_path_sha256']
    assert source['docx_s6_image_matches_source_png'] is True
    _, _, width, height = source['source_page_points']
    x0, y0, aw, ah = map(float, source['axis_box_pdf_points'])
    physical_width = 6.6
    scale = physical_width * 72 / width
    target_height = height * scale
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9.0,
        'axes.labelsize': 9.0, 'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5,
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK, 'ytick.color': INK,
        'axes.edgecolor': '#8D969D', 'axes.linewidth': 0.6, 'axes.spines.top': False,
        'axes.spines.right': False, 'pdf.fonttype': 42,
    })
    fig = plt.figure(figsize=(physical_width, target_height / 72))
    ax = fig.add_axes((x0 / width, y0 / height, aw / width, ah / height))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([(float(p) - x0) / aw for p, _ in source['x_ticks_from_completed_pdf']],
                  [label for _, label in source['x_ticks_from_completed_pdf']])
    ax.set_yticks([(float(p) - y0) / ah for p, _ in source['y_ticks_from_completed_pdf']],
                  [label for _, label in source['y_ticks_from_completed_pdf']])
    ax.set_xlabel('Paired ΔTP at 3% capacity', labelpad=4)
    ax.set_ylabel('Empirical cumulative proportion', labelpad=5)
    ax.tick_params(length=3.0, width=0.6)
    ax.grid(axis='y', color='#DCE5EA', linewidth=0.55)
    fig.text(0.54, 0.962, 'PaySim · Fold 3 · PTHW − PTH · LightGBM · Platt scores',
             ha='center', va='center', fontsize=9.0, fontweight='medium')
    base = io.BytesIO()
    fig.savefig(base, format='pdf', bbox_inches=None, facecolor='white')
    plt.close(fig)
    base.seek(0)
    pdf = pymupdf.open(stream=base.getvalue(), filetype='pdf')
    page = pdf[0]
    assert abs(page.rect.width - width * scale) < 1e-4
    assert abs(page.rect.height - height * scale) < 1e-4
    clip = ' '.join(source['axis_box_pdf_points']) + ' re W n'
    operations = [f'q {scale:.16f} 0 0 {scale:.16f} 0 0 cm', clip,
                  '2 J 1 j 1.5 w [] 0 d', f'{rgb_pdf(BLUE)} RG', curve, 'S']
    for marker in source['reference_lines']:
        color = INK if marker['role'] == 'observed' else TEAL
        dash = '[3.0 2.5]' if marker['role'] == 'observed' else '[6.0 3.0]'
        xp = marker['x_pdf_points']
        operations.extend([f'{rgb_pdf(color)} RG 1.1 w {dash} 0 d',
                           f'{xp} {y0} m {xp} {y0 + ah} l S'])
    operations.append('Q')
    appended = pdf.get_new_xref()
    pdf.update_object(appended, '<<>>')
    pdf.update_stream(appended, ('\n'.join(operations) + '\n').encode('latin1'))
    contents = page.get_contents() + [appended]
    pdf.xref_set_key(page.xref, 'Contents', '[' + ' '.join(f'{xref} 0 R' for xref in contents) + ']')

    # Put a compact legend on the empty lower-right region.  Its white backing
    # masks only the marker lines, as in the original finished figure.
    p = pdf[0]
    legend_x = width * scale * 0.735
    legend_y = target_height * 0.67
    p.draw_rect(pymupdf.Rect(legend_x - 3, legend_y - 9,
                            width * scale * 0.982, legend_y + 25),
                color=None, fill=(1, 1, 1), overlay=True)
    for row, (label, color, dash) in enumerate([
        ('Observed: +39.4', INK, '[3 2] 0'),
        ('Original 2.5%: -166.2', TEAL, '[5 2.5] 0'),
        ('Original 97.5%: 746.8', TEAL, '[5 2.5] 0'),
    ]):
        y = legend_y + row * 11
        rgb = tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))
        p.draw_line(pymupdf.Point(legend_x, y - 2.5),
                    pymupdf.Point(legend_x + 15, y - 2.5),
                    color=rgb, width=0.8, dashes=dash, overlay=True)
        p.insert_text(pymupdf.Point(legend_x + 19, y), label,
                      fontname='helv', fontsize=7.5, color=tuple(int(INK[i:i + 2], 16) / 255 for i in (1, 3, 5)))
    pdf.save(HERE / 'figS6.pdf', garbage=0, deflate=True)
    pdf.close()
    pdf = pymupdf.open(HERE / 'figS6.pdf')
    page = pdf[0]
    final_streams = b'\n'.join(pdf.xref_stream(xref) for xref in page.get_contents())
    assert curve.encode('latin1') in final_streams
    assert not page.get_images()
    zoom = 2526 / page.rect.width
    probe = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), alpha=False)
    assert probe.width == 2526 and probe.height == 1086
    assert min(probe.samples[-probe.stride:]) == 255  # one wholly white rounding row
    clip_rect = pymupdf.Rect(0, 0, page.rect.width, (1085 - 0.0001) / zoom)
    image = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom), clip=clip_rect, alpha=False)
    assert image.width == 2526 and image.height == 1085
    image.set_dpi(400, 400)
    image.save(HERE / 'figS6.png')
    pdf.close()
    source['redraw'] = {
        'output_curve_path_sha256': hashlib.sha256(curve.encode('latin1')).hexdigest(),
        'path_verbatim_present_in_final_pdf': True,
        'uniform_affine_pdf_transform': [scale, 0, 0, scale, 0, 0],
        'physical_width_inches': physical_width,
        'raster_pixels': [image.width, image.height],
        'raster_effective_dpi': 2526 / physical_width,
        'png_dpi_metadata': [400, 400],
        'raster_rounding_margin': 'One verified wholly white bottom raster row omitted; PDF and plotted paths unchanged.',
        'new_vector_pdf_raster_images': 0,
        'changed': ['stroke colors', 'font and label layout', 'grid', 'legend layout'],
        'unchanged': ['curve path commands', 'axis rectangle relative geometry',
                      'axis tick positions and numbers', 'three marker positions and numbers'],
        'model_fit': 0, 'preprocessing_fit': 0, 'model_load': 0,
        'prediction': 0, 'draw_read': 0, 'resampling': 0,
    }
    source_path.write_text(json.dumps(source, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(source['redraw'], ensure_ascii=False))


if __name__ == '__main__':
    main()
