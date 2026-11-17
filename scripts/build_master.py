#!/usr/bin/env python3
r"""
Master Monograph Generator (`./master.tex`) from Taxonomy Tree.

Generates `./master.tex` referencing `latex_core/preamble.tex`,
`latex_core/macros.tex`, and all 47 leaf monographs ordered strictly by
Domain -> Parent Topic -> Sub-Topic -> Leaf from `./data/taxonomy_tree.json`.
Uses `\IfFileExists` guards so `./master.tex` compiles cleanly after any
partial domain batch as well as after complete synthesis.

Usage:
    .venv/bin/python scripts/build_master.py --taxonomy data/taxonomy_tree.json --output master.tex
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Optional


def _latex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain metadata strings."""
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in text)


def generate_master_tex(taxonomy_path: Path, output_path: Path, include_only_existing: bool = False) -> None:
    """Generate unified `master.tex` wrapping all hierarchical monograph domains and leaves."""
    if not taxonomy_path.is_file():
        raise FileNotFoundError(f"Taxonomy file not found: {taxonomy_path}")

    with taxonomy_path.open("r", encoding="utf-8") as f:
        tree: dict[str, Any] = json.load(f)

    lines: list[str] = [
        "% =============================================================================",
        "% Unified Cross-Book AI/ML Knowledge Monograph — Master Document",
        "% Compile with: latexmk -xelatex -interaction=nonstopmode -halt-on-error master.tex",
        "% =============================================================================",
        r"\documentclass[11pt,a4paper,oneside]{report}",
        "",
        r"\usepackage{subfiles}",
        r"\input{\subfix{latex_core/preamble.tex}}",
        r"\input{\subfix{latex_core/macros.tex}}",
        r"\input{\subfix{latex_core/cheatsheet.tex}}",
        "",
        r"\begin{document}",
        "",
        "% --- Title Page --------------------------------------------------------------",
        r"\begin{titlepage}",
        r"  \centering",
        r"  \vspace*{2.0cm}",
        r"  {\small\textsf{\color{monographTeal}\textbf{CROSS-BOOK SYNTHESIS ARCHITECTURE}}}\\[0.8em]",
        r"  \textcolor{monographNavy}{\rule{0.85\textwidth}{2pt}}\\[1.5em]",
        r"  {\Huge\bfseries\color{monographNavy} Foundations and Frontiers of\\Modern Machine Intelligence}\\[1.2em]",
        r"  {\Large\color{monographSlate} A Unified Mathematical, Algorithmic, and Agentic Systems Monograph}\\[1.5em]",
        r"  \textcolor{monographRule}{\rule{0.65\textwidth}{1pt}}\\[2.2em]",
        r"  \begin{tcolorbox}[",
        r"    enhanced, colback=monographNavyBg, colframe=monographNavy,",
        r"    boxrule=0.8pt, leftrule=3.5pt, width=0.88\textwidth, arc=3pt",
        r"  ]",
        r"    \small\color{monographSlate}",
        r"    \textbf{Editorial Scope:} Cross-book mathematical derivations, reconciled notation,",
        r"    geometric intuition, 300~DPI benchmark figures, and TikZ/PGFPlots analyses distilled",
        r"    across 34 foundational textbooks spanning Linear Algebra, Optimization, Probabilistic ML,",
        r"    Deep Learning, Generative Diffusion \& Optimal Transport, Transformers/LLMs,",
        r"    Reinforcement Learning, Alignment, and Production Agentic Systems.",
        r"  \end{tcolorbox}",
        r"  \vfill",
        r"  {\footnotesize\textsf{\color{monographSlate} Generated from 47 Hierarchical Leaf Monographs $\cdot$ Typeset with XeLaTeX}}",
        r"\end{titlepage}",
        "",
        r"\pagenumbering{roman}",
        r"\setcounter{tocdepth}{1}",
        r"{\sloppy\emergencystretch=3em\tableofcontents}",
        r"\cleardoublepage",
        r"\pagenumbering{arabic}",
        "",
    ]

    leaf_count = 0
    for dom in tree.get("domains", []):
        dom_title = _latex_escape(str(dom.get("title", "")))
        dom_leaves = [
            lf
            for pt in dom.get("parent_topics", [])
            for st in pt.get("sub_topics", [])
            for lf in st.get("leaves", [])
        ]
        if include_only_existing and not any(Path(lf["tex_path"]).is_file() for lf in dom_leaves):
            continue

        lines.append(f"% === DOMAIN: {dom.get('id')} ===")
        lines.append(rf"\part{{{dom_title}}}")
        lines.append("")
        for pt in dom.get("parent_topics", []):
            lines.append(f"% -- Parent Topic: {pt.get('title')}")
            for st in pt.get("sub_topics", []):
                lines.append(f"% ---- Sub-Topic: {st.get('title')}")
                for lf in st.get("leaves", []):
                    tex_path = str(lf["tex_path"])
                    if include_only_existing and not Path(tex_path).is_file():
                        continue
                    lines.append(rf"\IfFileExists{{{tex_path}}}{{\subfile{{{tex_path}}}}}{{}}")
                    leaf_count += 1
        lines.append("")

    atlas_paths = _collect_cheatsheet_paths(tree)
    if atlas_paths:
        lines.append("% === VISUAL ATLAS: mindmap + cheatsheet pairs (2 landscape pages per leaf) ===")
        lines.append(r"\part{Visual Atlas: Mindmaps \& Cheatsheets}")
        lines.extend(rf"\IfFileExists{{{p}}}{{\subfile{{{p}}}}}{{}}" for p in atlas_paths)
        lines.append("")

    lines.append(r"\end{document}")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(
        f"Generated {output_path} referencing {leaf_count} leaf monographs "
        f"and {len(atlas_paths)} cheatsheets."
    )


def _collect_cheatsheet_paths(tree: dict[str, Any]) -> list[str]:
    """Return existing `<leaf>_cheatsheet.tex` paths in taxonomy order.

    Each leaf `foo.tex` owns an optional companion `foo_cheatsheet.tex` located
    in the same directory (mindmap page + cheatsheet page).
    """
    paths: list[str] = []
    for dom in tree.get("domains", []):
        for pt in dom.get("parent_topics", []):
            for st in pt.get("sub_topics", []):
                for lf in st.get("leaves", []):
                    leaf = Path(str(lf["tex_path"]))
                    companion = leaf.with_name(f"{leaf.stem}_cheatsheet.tex")
                    if companion.is_file():
                        paths.append(companion.as_posix())
    return paths


def generate_atlas_tex(taxonomy_path: Path, output_path: Path) -> int:
    """Generate `./atlas.tex`: a pocket-size document made of cheatsheets only.

    Returns:
        Number of cheatsheets referenced.
    """
    with taxonomy_path.open("r", encoding="utf-8") as f:
        tree: dict[str, Any] = json.load(f)
    paths = _collect_cheatsheet_paths(tree)
    lines: list[str] = [
        "% Visual Atlas — mindmaps + cheatsheets only (generated by scripts/build_master.py)",
        r"\documentclass[11pt,a4paper,oneside]{report}",
        r"\usepackage{subfiles}",
        r"\input{\subfix{latex_core/preamble.tex}}",
        r"\input{\subfix{latex_core/macros.tex}}",
        r"\input{\subfix{latex_core/cheatsheet.tex}}",
        r"\begin{document}",
        r"\pagenumbering{gobble}",
        *[rf"\subfile{{{p}}}" for p in paths],
        r"\end{document}",
        "",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Generated {output_path} referencing {len(paths)} cheatsheets.")
    return len(paths)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate ./master.tex from taxonomy_tree.json")
    p.add_argument("--taxonomy", type=Path, default=Path("data/taxonomy_tree.json"))
    p.add_argument("--output", type=Path, default=Path("master.tex"))
    p.add_argument("--only-existing", action="store_true", help="Include only existing leaf .tex files")
    p.add_argument("--atlas", type=Path, default=None, help="Also write a cheatsheets-only atlas .tex here")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    generate_master_tex(args.taxonomy, args.output, include_only_existing=args.only_existing)
    if args.atlas is not None:
        generate_atlas_tex(args.taxonomy, args.atlas)
    return 0


if __name__ == "__main__":
    sys.exit(main())
