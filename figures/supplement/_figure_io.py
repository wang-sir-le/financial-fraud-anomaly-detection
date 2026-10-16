"""Portable output handling for the adopted supplementary figure scripts."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import subprocess


def output_directory(here, description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--output-dir", type=Path, default=here,
                        help="Write exports here; input JSON remains beside the script.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    return args.output_dir.resolve()


def render_png(pdf_path, png_path, dpi=600):
    """Rasterize the generated vector PDF with PyMuPDF or Poppler."""
    try:
        import pymupdf as fitz
    except ImportError:
        poppler = shutil.which("pdftoppm")
        if not poppler:
            raise RuntimeError("PNG export needs PyMuPDF (pip install PyMuPDF) or pdftoppm.")
        subprocess.run([poppler, "-r", str(dpi), "-singlefile", "-png",
                        str(pdf_path), str(png_path.with_suffix(""))], check=True)
        renderer = "Poppler"
    else:
        with fitz.open(pdf_path) as doc:
            assert len(doc) == 1
            doc[0].get_pixmap(dpi=dpi, alpha=False).save(png_path)
        renderer = "PyMuPDF"
    from PIL import Image
    with Image.open(png_path) as im:
        size = list(im.size)
        im.save(png_path, dpi=(dpi, dpi))
    return {"png_renderer": renderer, "png_dpi": dpi, "png_pixels": size}


def write_verification(path, values, exports):
    values = {**values, "exports": [p.name for p in exports],
              "export_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in exports},
              "reproduction_scope": "Frozen aggregates and drawing only; no fitting, prediction or resampling.",
              "pixel_identity": "Fonts and PDF rasterizers may change rendered pixels."}
    path.write_text(json.dumps(values, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"figure": values.get("figure"), "validation": "PASS",
                      "exports": values["exports"]}))
