# Current standalone figures

`main/` contains Figures 1-6. Figures 1-2 are protocol diagrams; Figures 3-5 are the latest October 8, 2026 revisions; Figure 6 retains the October 7 data redraw. `supplement/` contains Figures S1-S8 from the completed October 7 redraw.

Every figure has its original standalone Python script and published PNG/PDF exports. Main Figures 1-5 additionally have editable SVG exports. Data plots read adjacent JSON files containing existing aggregate estimates, intervals, bins, counts or saved curve geometry. No row-level transactions, scores, labels, model states or Bootstrap samples are read. Run a script in a copy of the figure folder to preserve the original published exports.

The recorded environment is Python 3.11.9, matplotlib 3.10.8 and NumPy 2.4.4. Figure S6 additionally uses PyMuPDF 1.28.2. Fonts can affect appearance on another machine. The file manifest verifies delivered bytes, not cross-platform rendering identity. Original revision histories record the scientific interval conventions and the distinction between observation-point arithmetic and confidence limits.
