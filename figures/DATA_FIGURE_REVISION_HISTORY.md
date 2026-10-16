> Historical entries below describe their recorded revision dates. The final adopted October 16 presentation is documented in [UPDATES_20261016.md](../docs/UPDATES_20261016.md).

# 学术数据图重绘包

本包包含主稿 Fig.3–6 和补充 Fig.S1–S9。概念图 Fig.1–2 保持原样。

每幅图提供高分辨率 PNG、单页矢量 PDF、独立 Python 绘图脚本和相邻紧凑输入。输入采用 JSON 或 CSV，保存既有汇总指标、既有区间或已经完成的成图矢量路径；不含逐笔交易、逐笔标签、逐笔预测分数、特征矩阵、模型或 bootstrap 逐次样本。

运行某幅图的脚本即可重绘该图。脚本只读取包内相邻紧凑输入，不依赖原实验目录，也不会调用训练、预测、排序、AP 计算或统计回放入口。执行会更新相应图文件；先复制图包可保留交付版本。

本轮绘图使用本机已有 Python 3.11.9、matplotlib 3.10.8、numpy 2.4.4；S6 另使用已有 PyMuPDF 1.28.2。未安装或升级依赖。Word 文档另用内置文档依赖读取与核验。

## 保留的科学含义

- Fig.3：原 PTHW−P0 效应、95%区间和次要 equal-fold 汇总。
- Fig.4：H0 使用原95%区间，H1/D 使用原98.75% Bonferroni marginal 区间；面板尺度不同。观察点算术边界不作为置信限。
- Fig.5/S2：原 corrected C02 三折区间；跨零不等于等效。
- Fig.6：既有五seed情境均值；离散点之间的连线不代表连续最优。负成本结果保留。
- S1/S8：原描述性均值；PaySim History 指 PTH−P0 的时间与历史组合；热图分别使用对称零中心色阶。
- S3：保留45个seed序列、299个非空bin；不合并稀疏bin。线可跨空bin，只作视觉引导。
- S4/S5：原三score stages及全部区间；重合的 clipped/Platt 均保留。
- S6：逐字保留原成图的2058段曲线路径，统一变换曲线、轴及参考线。未读取、反推或重算逐次 draws。
- S7：全部54个simulation steps、两流程原whole-window TopK的分步归属；没有按step重新排名或分配容量。交易数与欺诈数改为独立面板，避免双纵轴量纲混读。

数值源、筛选口径及来源哈希保留在各JSON。统一的 MANIFEST.json 记录交付文件哈希与核查状态。


## 2026-10-15 Fig. S3 visual update

Adopted the soft-halo PNG, PDF and editable SVG for the S3 reliability curves and replaced only S3 in the current supplementary manuscript. All 45 fold-method-seed series and 299 original nonempty-bin points are unchanged. The halos use constant display widths along the original paths; single-point series use marker halos only. They are decorative, not uncertainty intervals. The figure and caption state this distinction.

The same-day layout refinement reduced each S3 plotting area from 335 x 335 to 300 x 300 display units, centred in its original cell. Fonts, header bands, fold bars, legend, overall figure size and document drawing extents are unchanged; curve strokes, dash spacing, markers and halos scale with the plot areas.


## 2026-10-15 Review fixes and source synchronization

- Figure 6 now matches the five-panel manuscript artwork adopted on October 14, including panel (e) with six saved pointwise conditional 95% reference recall intervals. The separate dataset scales and the distinction between threshold transfer and exact completed-window ranking are preserved. The default PNG is byte-identical to the current manuscript image. Compact data, source hashes, independent redraw receipts, the full caption and the pinned plotting environment are archived beside the entry point. All 111 points and twelve interval endpoints are unchanged; no new intervals or experiments were generated.
- Figure S1 was restored to its original two-scatter-panel image, original drawing extents and short caption at the author's request. The standalone original PNG/PDF/script were restored and combined-layout companion files archived. Figure S8 retains the separate heatmap. The S3 soft halos and reduced plot areas, Figure 6 source synchronization and Figure S9 reconstruction are preserved.
- Figure S9 now has a standalone script, compact aggregate JSON, caption, PNG and editable vector SVG/PDF. It reconstructs the six-condition tile figure from the retained S22 result CSVs, preserving all 26 numeric labels. The original drawing script was not located; this is a content- and layout-matched reconstruction, not a pixel-identical copy of the manuscript raster. The single-seed retrospective scope, local segment budgets and absence of confidence intervals remain explicit.
- The figure index, main captions and current hash manifest were refreshed. The original seven-section `docs/SUPPLEMENT.md` now has a visible archive notice distinguishing it from the current S1–S23 manuscript supplement. No statistical pipeline, fitted model or experimental outcome was changed.


## 2026-10-16 Final supplementary presentation

Figure S2 adopts the compact table-and-forest style, retaining the three original effects, intervals and zero guide. Figure S7 adopts the three-row layout selected after the landscape comparison. Figure S9 replaces the intermediate paired-point design with six whole-window condition tiles and four I (pp) bars, retaining J and denominator labels. Segment budgets remain local and the whole-window and segment contrasts are not independent replications. The single-seed retrospective evidence has no inferential intervals.

The S3 drawing script now reproduces the recorded soft halos and reduced 300-by-300 plot areas; its 45 series and 299 original bin points are unchanged. Halos remain decorative. S1 retains the original two-scatter layout after withdrawal of the heatmap proposal; S8 remains separate. The S2/S3/S7/S9 scripts updated in this release use adjacent inputs and explicit `--output-dir` directories. Unchanged supplementary entry points retain their historical output behavior. Only the final adopted exports and compact reconstruction assets are published.
