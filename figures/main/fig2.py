"""Regenerate the adopted October 20 Figure 2 SVG; PNG export is separate."""
from pathlib import Path
import runpy
import shutil

if __name__ == "__main__":
    here = Path(__file__).resolve().parent
    source = here / "fig2_source_20261020"
    runpy.run_path(str(source / "draw_figure.py"), run_name="__main__")
    shutil.copyfile(source / "fig02_protocol_labels_20261020.svg", here / "fig2.svg")
    print("Updated fig2.svg. The adopted fig2.png is retained; PNG export is separate.")
