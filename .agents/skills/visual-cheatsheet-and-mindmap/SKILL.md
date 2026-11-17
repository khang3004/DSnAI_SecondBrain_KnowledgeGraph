---
name: visual-cheatsheet-and-mindmap
description: >-
  Authors and geometrically audits 2-page Landscape A4 visual mindmaps and
  cheatsheets (*_cheatsheet.tex) and builds atlas.pdf. Enforces zero text
  overflow, zero PGFPlots axis/curve crossings, and zero mini-box spills via
  audit_cheatsheets.py.
---

# 2-Page Visual Cheatsheet & Mindmap Authoring

## Prerequisites
Read `monographs/AGENTS.md` and `latex_core/cheatsheet.tex`.

## Layout Architecture
- **Page 1 (`MINDMAP`):** `\csbegin{domain_idx}{Parent}{Title}{Hook}` $\to$ `\csroot{Short Title}{mini-TikZ}` + `\csbranch{1..6}{A..F}{Header}{Body}`.
  - Inside `\csroot`, `cheatsheet.tex` resets `shape=rectangle, line width=0.4pt, text width=, text=black` before the nested `tikzpicture` so child nodes do not inherit the outer circle.
  - Keep mini-TikZ diagrams compact (`scale=0.45..0.55`) and ensure labels stay well inside rectangles/ellipses.
- **Page 2 (`CHEATSHEET`):** `\csnextpage` + `\begin{cscolumns}...\end{cscolumns}` containing `cscard` blocks (`FORMULAS TO REMEMBER`, `RECIPE`, `DECISION GUIDE`, `CHEAT-LINE`, `PGFPlots` chart, `PITFALL CHECKLIST`, `10-SECOND VISUAL RECAP`).

## PGFPlots & TikZ Anti-Overlap Rules
1. **Never let curves exceed `[ymin, ymax]`:** Restrict `domain=` so log-scale or steep curves stop cleanly at `ymin`/`ymax` (PGFPlots draws unclipped paths across `xlabel` and neighboring cards before PDF clipping!).
2. **Legend placement:** Place legends in empty corners (`at={(0.03,0.92)}` or `at={(0.97,0.91)}`) with `fill=white, fill opacity=0.92, draw=gray!40`. Avoid `\frac` inside legend entries; raise `ymax` if the legend touches top ticks or curves.
3. **Standalone & Atlas Inclusion:** Add `\subfile{..._cheatsheet.tex}` to `atlas.tex` in exact taxonomy order.

## Mandatory Geometry Audit Gate
```bash
.venv/bin/python scripts/clean_aux.py
.venv/bin/python scripts/compile_qa.py --file <leaf_cheatsheet.tex>
.venv/bin/python scripts/compile_qa.py --file atlas.tex
.venv/bin/python scripts/audit_cheatsheets.py
.venv/bin/python scripts/clean_aux.py
```
Must achieve `errors=0 | overfull_hbox=0` AND `Audited N pages in atlas.pdf: 0 pages have geometric issues.`
