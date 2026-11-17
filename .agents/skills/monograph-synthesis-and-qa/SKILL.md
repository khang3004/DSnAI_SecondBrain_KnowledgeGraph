---
name: monograph-synthesis-and-qa
description: >-
  Synthesizes cross-book AI/ML LaTeX leaf monographs in monographs/, extracts
  300 DPI source figures if needed, reconciles notation across textbooks,
  regenerates master.tex, and runs automated XeLaTeX compilation + self-healing
  QA until exit=0, errors=0, and overfull_hbox=0.
---

# Cross-Book Monograph Synthesis & QA

## Prerequisites
Read `monographs/AGENTS.md` for macro, callout, and LaTeX safety conventions.

## Step-by-Step Procedure

### 1. Read Source Pages & Extract Figures (Optional)
- Inspect exact mapped pages from `books_monographs/` using `scripts/page_reader.py` or `pymupdf`.
- If iconic source figures are needed, run `scripts/figure_extractor.py` at 300 DPI to `assets/figures/<book_slug>_p<page>_fig<idx>.png` and embed with `\monographfig`.

### 2. Author the Leaf Monograph (`<leaf_id>.tex`)
1. Start with `\documentclass[../../../../master.tex]{subfiles}` and `\monographleaf{Domain}{Parent}{Sub-Topic}{Leaf Title}{\bookcite{...}{...}}`.
2. **Cross-Book Notation Table:** Include a `crossbook` environment reconciling divergent textbook symbols onto unified `latex_core/macros.tex` notation.
3. **Derivations & Intuition:** Use `coremath` for rigorous equations/theorems, `intuition` for geometric/physical intuition, `algorithm` (`algorithm2e`) for pseudocode, and `pitfall` for failure modes.
4. **Vector Visuals:** Include TikZ diagrams or PGFPlots curves using only defined palette colors (`monographNavy`, `monographTeal`, `monographIndigo`, `monographCrimson`, `monographSlate`).
5. **LaTeX Hygiene:** Escape `&` as `\&` in prose/titles, wrap primed transposes as `(\tilde{\vecb{x}}')\T`, and use `\begin{sloppypar}...\end{sloppypar}` around paragraphs with long `\texttt{book_slug}` citations.

### 3. Rebuild Master & Compile QA
```bash
.venv/bin/python scripts/build_master.py --taxonomy data/taxonomy_tree.json --output master.tex
.venv/bin/python scripts/clean_aux.py
.venv/bin/python scripts/compile_qa.py --file <path_to_leaf.tex>
.venv/bin/python scripts/compile_qa.py --file master.tex --json-out data/qa_final_master.json
.venv/bin/python scripts/clean_aux.py
```
All targets must pass with `exit=0 | errors=0 | overfull_hbox=0`.
