# Monographs & Visual Cheatsheets Guidelines (`monographs/`)

Every leaf topic in `data/taxonomy_tree.json` has **two** paired XeLaTeX files compiled via `subfiles` (`\documentclass[../../../../master.tex]{subfiles}`):
1. `<leaf_id>.tex` — Rigorous cross-book synthesis monograph.
2. `<leaf_id>_cheatsheet.tex` — 2-page Landscape A4 visual mindmap + cheatsheet.

## 1. Cross-Book Monograph Standards (`<leaf_id>.tex`)
- **Unified Macros (`latex_core/macros.tex`):** Always use `\vecb{x}`, `\mat{A}`, `\T`, `\inv`, `\pinv`, `\eye`, `\zeros`, `\ones`, `\grad`, `\hess`, `\jac`, `\E`, `\Prob`, `\Normal`, `\KL{p}{q}`, `\norm{x}`, `\abs{x}`, `\inner{x}{y}`, `\bookcite{book_slug}{Ch. X, pp. Y--Z}`.
- **Custom Callout Environments (`latex_core/preamble.tex`):**
  - `coremath` — Formal definitions, theorems, and derivations.
  - `intuition` — Geometric/physical intuition ("Why it works").
  - `crossbook` — Notation reconciliation tables & comparative synthesis across textbooks.
  - `pitfall` — Numerical stability warnings, edge cases, and failure modes.
- **LaTeX Safety Rules:**
  - Escape `&` as `\&` in prose, section titles, and `tcolorbox` titles (`\begin{crossbook}[Title A \& B]`).
  - Wrap primed transposes in parentheses `(\tilde{\vecb{x}}')\T` to prevent `Double superscript` errors.
  - Use defined palette colors only: `monographNavy`, `monographTeal`, `monographIndigo`, `monographCrimson`, `monographSlate`.
  - Wrap paragraphs containing long `\texttt{book_slug}` citations in `\begin{sloppypar}...\end{sloppypar}` if `Overfull \hbox` occurs.
- **Standalone & Master Compilation:** Every `.tex` file must compile standalone (`compile_qa.py --file <path>`) AND as part of `master.tex` (regenerated via `scripts/build_master.py`).

## 2. 2-Page Visual Cheatsheet & Mindmap Standards (`<leaf_id>_cheatsheet.tex`)
- **Page 1 (`MINDMAP`):** Uses `\csbegin`, `\csroot{Title}{mini-tikz}`, and `\csbranch{1..6}{A..F}{Header}{Body}`.
  - Inside `\csroot`, `cheatsheet.tex` resets `shape=rectangle, line width=0.4pt, text width=, text=black` so nested TikZ nodes never inherit the outer circle shape.
  - Keep mini-TikZ diagrams compact (`scale=0.45..0.55`) and verify node labels stay strictly inside their bounding boxes/ellipses (no `MINI_SPILL`).
- **Page 2 (`CHEATSHEET`):** Uses `\csnextpage` + `\begin{cscolumns}...\end{cscolumns}` with `cscard` blocks (`FORMULAS`, `RECIPE`, `DECISION GUIDE`, `CHEAT-LINE`, `PGFPlots` diagram, `PITFALL CHECKLIST`, `10-SECOND RECAP`).
  - **PGFPlots Rules:**
    - Always clip `domain=` so curves never evaluate below `ymin` or above `ymax` (unclipped curves in PGFPlots draw outside the axis and cross `xlabel` or neighboring cards!).
    - Keep legends away from curves and away from axis borders (`at={(0.03,0.92)}` or `at={(0.97,0.91)}` with `fill=white, fill opacity=0.92, draw=gray!40`).
    - Never use `\frac` inside legend entries (causes tall legends that hit top ticks/borders).
- **Mandatory Geometry QA:** After editing any cheatsheet or `atlas.tex`, run:
  ```bash
  .venv/bin/python scripts/clean_aux.py
  .venv/bin/python scripts/compile_qa.py --file atlas.tex
  .venv/bin/python scripts/audit_cheatsheets.py
  ```
  Target: `errors=0 | overfull_hbox=0` and `0 pages have geometric issues`.
