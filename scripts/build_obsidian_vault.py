"""Build a densely connected Obsidian Knowledge Graph vault from the AI/ML monographs.

Uses a balanced-brace LaTeX parser so nested macros, captions, tables, callouts,
and equations convert cleanly to Obsidian Markdown + MathJax with zero broken
delimiters or raw LaTeX leaks.
"""

from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

from concept_graph_spec import CONCEPT_SPECS, ConceptSpec

ROOT = Path(__file__).resolve().parent.parent
VAULT = ROOT / "obsidian_vault"


def slugify_note(name: str) -> str:
    """Sanitize a note title so it is safe as an Obsidian filename and wikilink target."""
    cleaned = name.replace("/", " - ").replace(":", " -").replace("\\", "").replace("#", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(" &")
    return cleaned


def read_balanced(text: str, pos: int, open_ch: str = "{", close_ch: str = "}") -> tuple[str, int] | None:
    """Read a balanced {...} or [...] group starting at pos; return (content, next_pos)."""
    if pos >= len(text) or text[pos] != open_ch:
        return None
    depth = 0
    i = pos
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            i += 2
            continue
        if ch == open_ch:
            depth += 1
        elif ch == close_ch:
            depth -= 1
            if depth == 0:
                return text[pos + 1 : i], i + 1
        i += 1
    return None


def replace_macro_1arg(text: str, macro: str, fmt: Any) -> str:
    """Replace \\macro[opt]{arg} (and optional * variant) with balanced-brace awareness."""
    out: list[str] = []
    i = 0
    n = len(text)
    pat = re.compile(rf"\\{re.escape(macro)}(?![A-Za-z])\*?\s*")
    while i < n:
        m = pat.match(text, i)
        if m:
            j = m.end()
            if j < n and text[j] == "[":
                opt_res = read_balanced(text, j, "[", "]")
                if opt_res is not None:
                    j = opt_res[1]
                    while j < n and text[j].isspace():
                        j += 1
            if j < n and text[j] == "{":
                res = read_balanced(text, j)
                if res is not None:
                    arg, nxt = res
                    out.append(fmt(arg))
                    i = nxt
                    continue
            tok_m = re.match(r"(\\[A-Za-z]+(?:\{[^{}]*\})?|[A-Za-z0-9])", text[j:])
            if tok_m:
                out.append(fmt(tok_m.group(1)))
                i = j + tok_m.end()
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def replace_macro_2arg(text: str, macro: str, fmt: Any, opt_arg: bool = False) -> str:
    """Replace \\macro{a}{b} (or \\macro[opt]{a}{b}) with balanced-brace awareness."""
    out: list[str] = []
    i = 0
    n = len(text)
    pat = re.compile(rf"\\{re.escape(macro)}(?![A-Za-z])\*?\s*")
    while i < n:
        m = pat.match(text, i)
        if m:
            j = m.end()
            opt = ""
            if opt_arg and j < n and text[j] == "[":
                opt_res = read_balanced(text, j, "[", "]")
                if opt_res is not None:
                    opt, j = opt_res
                    while j < n and text[j].isspace():
                        j += 1
            if j < n and text[j] == "{":
                r1 = read_balanced(text, j)
                if r1 is not None:
                    a1, j2 = r1
                    while j2 < n and text[j2].isspace():
                        j2 += 1
                    if j2 < n and text[j2] == "{":
                        r2 = read_balanced(text, j2)
                        if r2 is not None:
                            a2, nxt = r2
                            out.append(fmt(a1, a2, opt) if opt_arg else fmt(a1, a2))
                            i = nxt
                            continue
        out.append(text[i])
        i += 1
    return "".join(out)


def replace_macro_3arg(text: str, macro: str, fmt: Any, opt_arg: bool = False) -> str:
    """Replace \\macro[opt]{a}{b}{c} with balanced-brace awareness."""
    out: list[str] = []
    i = 0
    n = len(text)
    pat = re.compile(rf"\\{re.escape(macro)}(?![A-Za-z])\*?\s*")
    while i < n:
        m = pat.match(text, i)
        if m:
            j = m.end()
            opt = ""
            if opt_arg and j < n and text[j] == "[":
                opt_res = read_balanced(text, j, "[", "]")
                if opt_res is not None:
                    opt, j = opt_res
                    while j < n and text[j].isspace():
                        j += 1
            args: list[str] = []
            cur = j
            ok = True
            for _ in range(3):
                while cur < n and text[cur].isspace():
                    cur += 1
                res = read_balanced(text, cur) if cur < n and text[cur] == "{" else None
                if res is None:
                    ok = False
                    break
                args.append(res[0])
                cur = res[1]
            if ok:
                out.append(fmt(args[0], args[1], args[2], opt) if opt_arg else fmt(args[0], args[1], args[2]))
                i = cur
                continue
        out.append(text[i])
        i += 1
    return "".join(out)


def convert_latex_table_to_markdown(block: str) -> str:
    """Convert a LaTeX tabular/tabularx block into a clean Markdown table."""
    cap_m = re.search(r"\\caption(?:of\{table\})?\s*\{", block)
    caption = ""
    if cap_m:
        res = read_balanced(block, cap_m.end() - 1)
        if res:
            caption = clean_latex_text(res[0]).strip()

    tab_m = re.search(r"\\begin\{tabularx?\}(?:\{[^}]*\})?\s*", block)
    if not tab_m:
        return f"\n> [!note] Table\n> {caption}\n" if caption else ""
    col_res = read_balanced(block, tab_m.end())
    body_start = col_res[1] if col_res else tab_m.end()
    tab_end_m = re.search(r"\\end\{tabularx?\}", block[body_start:])
    if not tab_end_m:
        return f"\n> [!note] Table\n> {caption}\n" if caption else ""
    tbody = block[body_start : body_start + tab_end_m.start()]
    for tok in (r"\toprule", r"\midrule", r"\bottomrule", r"\hline", r"\centering", r"\addlinespace"):
        tbody = tbody.replace(tok, "")
    raw_rows = [r.strip() for r in re.split(r"\\\\(?:\[[^\]]*\])?", tbody) if r.strip()]
    rows: list[list[str]] = []
    for rr in raw_rows:
        cells = []
        for c in re.split(r"(?<!\\)&", rr):
            cell = clean_latex_text(c).replace("\n", " ").strip()
            cell = re.sub(r"\[\[([^\]|]+)\|[^\]]+\]\]", r"[[\1]]", cell)
            # Replace bare | inside inline math $...$ with \mid so Obsidian never splits table columns
            cell = re.sub(
                r"\$([^$]+)\$",
                lambda m: "$" + m.group(1).replace("\\|", "\\parallel ").replace("|", "\\mid ") + "$",
                cell,
            )
            cells.append(cell.replace("|", "\\|"))
        if any(cells):
            rows.append(cells)
    if not rows:
        return f"\n> [!note] Table\n> {caption}\n" if caption else ""
    ncols = max(len(r) for r in rows)
    norm_rows = [r + [""] * (ncols - len(r)) for r in rows]
    md_lines = []
    if caption:
        md_lines.append(f"\n**Table:** {caption}\n")
    md_lines.append("| " + " | ".join(norm_rows[0]) + " |")
    md_lines.append("| " + " | ".join("---" for _ in range(ncols)) + " |")
    for r in norm_rows[1:]:
        md_lines.append("| " + " | ".join(r) + " |")
    return "\n" + "\n".join(md_lines) + "\n"


def convert_algorithm_block(block: str) -> str:
    """Convert an algorithm2e environment into a clean Obsidian callout with pseudocode."""
    cap_m = re.search(r"\\caption\s*\{", block)
    title = "Algorithm"
    if cap_m:
        res = read_balanced(block, cap_m.end() - 1)
        if res:
            title = clean_latex_text(res[0]).strip()
            block = block[: cap_m.start()] + block[res[1] :]

    body = re.sub(r"^\\begin\{algorithm\*?\}(?:\[[^\]]*\])?", "", block.strip())
    body = re.sub(r"\\end\{algorithm\*?\}$", "", body).strip()
    for tok in (r"\DontPrintSemicolon", r"\SetAlgoLined", r"\SetKwInOut", r"\BlankLine", r"\;", r"\Indp", r"\Indm"):
        body = body.replace(tok, "")
    body = re.sub(r"\\label\{[^}]*\}", "", body)

    # Convert common algorithm2e keywords cleanly
    body = replace_macro_1arg(body, "KwIn", lambda a: f"**Input:** {a}")
    body = replace_macro_1arg(body, "KwOut", lambda a: f"**Output:** {a}")
    body = replace_macro_1arg(body, "KwData", lambda a: f"**Data:** {a}")
    body = replace_macro_1arg(body, "KwResult", lambda a: f"**Result:** {a}")
    # Strip optional inline comments like \For(\tcp*{...}) before parsing 2-arg control blocks
    body = replace_macro_1arg(body, "tcp", lambda a: f" *// {a}*")
    body = replace_macro_1arg(body, "Comment", lambda a: f" *// {a}*")
    body = re.sub(r"(\\(?:ForEach|For|While|If|lIf|uIf|ElseIf|lElseIf|uElseIf|Else|lElse|Repeat))\s*\([^)]*\)", r"\1", body)
    body = replace_macro_1arg(body, "textsc", lambda a: a)

    for _ in range(3):
        for kw, label in (
            ("ForEach", "for each"),
            ("For", "for"),
            ("While", "while"),
            ("If", "if"),
            ("lIf", "if"),
            ("uIf", "if"),
            ("ElseIf", "else if"),
            ("lElseIf", "else if"),
            ("uElseIf", "else if"),
            ("Repeat", "repeat until"),
        ):
            body = replace_macro_2arg(body, kw, lambda cond, blk, lb=label: f"**{lb}** {cond}:\n{blk}")
        body = replace_macro_3arg(body, "eIf", lambda cond, t_blk, e_blk: f"**if** {cond}:\n{t_blk}\n**else**:\n{e_blk}")
        body = replace_macro_1arg(body, "Else", lambda blk: f"**else**:\n{blk}")
        body = replace_macro_1arg(body, "lElse", lambda blk: f"**else**:\n{blk}")
    body = replace_macro_1arg(body, "Return", lambda a: f"**return** {a}")
    body = re.sub(r"\\Return(?![A-Za-z])", "**return**", body)
    body = body.replace(r"\KwTo", "to").replace(r"\KwDownTo", "downto")

    body = clean_latex_text(body)
    lines = [f"> {ln}" if ln.strip() else ">" for ln in body.splitlines()]
    return f"\n> [!example] Algorithm: {title}\n" + "\n".join(lines) + "\n"


def convert_figure_block(block: str) -> str:
    """Extract figure caption using balanced braces."""
    cap_m = re.search(r"\\caption(?:of\{figure\})?\s*\{", block)
    cap = "TikZ / PGFPlots Diagram"
    if cap_m:
        res = read_balanced(block, cap_m.end() - 1)
        if res:
            cap = clean_latex_text(res[0]).strip()
    return f"\n> [!example] Visual Diagram (LaTeX / PDF)\n> *{cap}* — see rendered vector figure in `master.pdf` / `atlas.pdf`.\n"


def clean_latex_text(text: str) -> str:
    """Expand custom monograph LaTeX macros into Obsidian-compatible MathJax & Markdown."""
    cleaned_lines: list[str] = []
    for ln in text.splitlines():
        if ln.lstrip().startswith("%"):
            continue
        ln = re.sub(r"(?<!\\)%.*$", "", ln)
        cleaned_lines.append(ln)
    s = "\n".join(cleaned_lines)

    s = re.sub(r"\\documentclass(?:\[[^\]]*\])?\{[^}]*\}", "", s)
    s = s.replace("\\begin{document}", "").replace("\\end{document}", "")
    leaf_m = re.search(r"\\monographleaf\s*", s)
    if leaf_m:
        cur = leaf_m.end()
        for _ in range(5):
            while cur < len(s) and s[cur].isspace():
                cur += 1
            res = read_balanced(s, cur) if cur < len(s) and s[cur] == "{" else None
            if res:
                cur = res[1]
        s = s[: leaf_m.start()] + s[cur:]

    s = re.sub(r"\\label\{[^}]*\}", "", s)
    s = re.sub(r"\\ref\{([^}]*)\}", r"`\1`", s)
    s = re.sub(r"\\eqref\{([^}]*)\}", r"`(\1)`", s)
    s = re.sub(r"\\captionsetup\{[^}]*\}", "", s)
    s = re.sub(r"\\setlength\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\renewcommand\{[^}]*\}\{[^}]*\}", "", s)
    s = re.sub(r"\\(?:footnotesize|scriptsize|small|tiny|normalsize|centering|noindent|sloppy|begingroup|endgroup|medskip|bigskip|smallskip|hfill|allowbreak|nonumber|qed|raggedright|clearpage)(?![A-Za-z])", "", s)
    s = re.sub(r"\\emergencystretch\s*=\s*\S+", "", s)
    s = re.sub(r"\\par(?![A-Za-z])", "\n", s)
    s = re.sub(r"\\newline(?![A-Za-z])", "\n", s)
    s = s.replace(r"\L{}", "Ł").replace(r"V\~{u}", "Vũ").replace(r"\textasciicircum", "^").replace(r"\eqqcolon", "=:")
    s = re.sub(r"\\verb\|([^|]*)\|", r"`\1`", s)
    s = re.sub(r"\.\\(?=\s)", ".", s)

    s = replace_macro_3arg(
        s,
        "monographfig",
        lambda path, cap, _lbl, _opt: f"\n> [!note] Source Figure\n> *Figure:* {clean_latex_text(cap)} (`{path}`)\n",
        opt_arg=True,
    )

    s = re.sub(r"\\begin\{algorithm\*?\}(?:\[[^\]]*\])?.*?\\end\{algorithm\*?\}", lambda m: convert_algorithm_block(m.group(0)), s, flags=re.DOTALL)
    s = re.sub(r"\\begin\{figure\}(?:\[[^\]]*\])?.*?\\end\{figure\}", lambda m: convert_figure_block(m.group(0)), s, flags=re.DOTALL)
    s = re.sub(r"\\begin\{table\}(?:\[[^\]]*\])?.*?\\end\{table\}", lambda m: convert_latex_table_to_markdown(m.group(0)), s, flags=re.DOTALL)
    s = re.sub(r"\\begin\{tabularx?\}.*?\\end\{tabularx?\}", lambda m: convert_latex_table_to_markdown(m.group(0)), s, flags=re.DOTALL)
    s = s.replace("\\begin{center}", "").replace("\\end{center}", "")
    s = replace_macro_2arg(s, "captionof", lambda _t, _c: "")
    s = replace_macro_2arg(s, "texorpdfstring", lambda a, _b: a)
    s = replace_macro_1arg(s, "paragraph", lambda a: f"\n**{a}.** ")
    s = replace_macro_1arg(s, "textit", lambda a: f"*{a}*")
    s = replace_macro_1arg(s, "textup", lambda a: a)
    s = replace_macro_1arg(s, "textsc", lambda a: a)
    s = replace_macro_1arg(s, "textsuperscript", lambda a: f"^{a}")

    for _ in range(3):
        s = replace_macro_2arg(s, "inner", lambda a, b: rf"\langle {a}, {b} \rangle")
        s = replace_macro_2arg(s, "KL", lambda a, b: rf"D_{{\mathrm{{KL}}}}({a} \parallel {b})")
        s = replace_macro_2arg(s, "JS", lambda a, b: rf"D_{{\mathrm{{JS}}}}({a} \parallel {b})")
        s = replace_macro_2arg(s, "pd", lambda a, b: rf"\frac{{\partial {a}}}{{\partial {b}}}")
        s = replace_macro_2arg(s, "dd", lambda a, b: rf"\frac{{\mathrm{{d}}{a}}}{{\mathrm{{d}}{b}}}")
        s = replace_macro_2arg(s, "notebadge", lambda a, b: f"**{a}**: ${b}$")
        s = replace_macro_2arg(s, "Wass", lambda a, b, opt: f"W_{{{opt or 'p'}}}({a}, {b})", opt_arg=True)

        s = replace_macro_1arg(s, "vecb", lambda a: rf"\boldsymbol{{{a}}}")
        s = replace_macro_1arg(s, "mat", lambda a: rf"\boldsymbol{{{a}}}")
        s = replace_macro_1arg(s, "bm", lambda a: rf"\boldsymbol{{{a}}}")
        s = replace_macro_1arg(s, "norm", lambda a: rf"\lVert {a} \rVert")
        s = replace_macro_1arg(s, "abs", lambda a: rf"\lvert {a} \rvert")
        s = replace_macro_1arg(s, "set", lambda a: rf"\{{{a}\}}")
        s = replace_macro_1arg(s, "ceil", lambda a: rf"\lceil {a} \rceil")
        s = replace_macro_1arg(s, "floor", lambda a: rf"\lfloor {a} \rfloor")
        s = replace_macro_1arg(s, "emph", lambda a: f"*{a}*")
        s = replace_macro_1arg(s, "textbf", lambda a: f"**{a}**")
        s = replace_macro_1arg(s, "texttt", lambda a: f"`{a}`")

    simple_macros = {
        r"\R": r"\mathbb{R}",
        r"\N": r"\mathbb{N}",
        r"\Z": r"\mathbb{Z}",
        r"\C": r"\mathbb{C}",
        r"\Hilbert": r"\mathcal{H}",
        r"\Simplex": r"\Delta",
        r"\T": r"^{\top}",
        r"\inv": r"^{-1}",
        r"\pinv": r"^{+}",
        r"\eye": r"\boldsymbol{I}",
        r"\zeros": r"\boldsymbol{0}",
        r"\ones": r"\boldsymbol{1}",
        r"\grad": r"\nabla",
        r"\hess": r"\nabla^2",
        r"\jac": r"\boldsymbol{J}",
        r"\Tr": r"\mathrm{Tr}",
        r"\diag": r"\mathrm{diag}",
        r"\rank": r"\mathrm{rank}",
        r"\spn": r"\mathrm{span}",
        r"\nullsp": r"\mathrm{null}",
        r"\colsp": r"\mathrm{col}",
        r"\sign": r"\mathrm{sign}",
        r"\softmax": r"\mathrm{softmax}",
        r"\softplus": r"\mathrm{softplus}",
        r"\sigmoid": r"\sigma",
        r"\ReLU": r"\mathrm{ReLU}",
        r"\argmin": r"\arg\min",
        r"\argmax": r"\arg\max",
        r"\minimize": r"\mathrm{minimize}",
        r"\maximize": r"\mathrm{maximize}",
        r"\Prob": r"\mathbb{P}",
        r"\E": r"\mathbb{E}",
        r"\Var": r"\mathrm{Var}",
        r"\Cov": r"\mathrm{Cov}",
        r"\Corr": r"\mathrm{Corr}",
        r"\Entropy": r"\mathrm{H}",
        r"\MI": r"\mathrm{I}",
        r"\Normal": r"\mathcal{N}",
        r"\Uniform": r"\mathcal{U}",
        r"\Bernoulli": r"\mathrm{Bern}",
        r"\Cat": r"\mathrm{Cat}",
        r"\given": r"\mid",
        r"\indep": r"\perp\!\!\!\perp",
        r"\simiid": r"\overset{\mathrm{i.i.d.}}{\sim}",
        r"\States": r"\mathcal{S}",
        r"\Actions": r"\mathcal{A}",
        r"\Rewards": r"\mathcal{R}",
        r"\Trans": r"\mathcal{P}",
        r"\Bellman": r"\mathcal{T}",
        r"\Qval": r" Q",
        r"\Vval": r" V",
        r"\Adv": r" A",
        r"\dom": r"\mathrm{dom}",
        r"\coloneqq": r":=",
    }
    for k, v in sorted(simple_macros.items(), key=lambda kv: -len(kv[0])):
        s = re.sub(re.escape(k) + r"(?![A-Za-z])", lambda _m, rep=v: rep, s)

    s = re.sub(r"\\section\*?\{([^}]*)\}", r"\n## \1\n", s)
    s = re.sub(r"\\subsection\*?\{([^}]*)\}", r"\n### \1\n", s)
    s = re.sub(r"\\subsubsection\*?\{([^}]*)\}", r"\n#### \1\n", s)

    for env in ("equation", "equation*", "align", "align*", "gather", "gather*"):
        s = re.sub(
            rf"\\begin\{{{re.escape(env)}\}}(.*?)\\end\{{{re.escape(env)}\}}",
            lambda m, is_al=("align" in env): (
                f"\n$$\n\\begin{{aligned}}\n{m.group(1).strip()}\n\\end{{aligned}}\n$$\n"
                if is_al
                else f"\n$$\n{m.group(1).strip()}\n$$\n"
            ),
            s,
            flags=re.DOTALL,
        )

    s = re.sub(r"\\begin\{(?:itemize|enumerate)\}(?:\[[^\]]*\])?", "", s)
    s = re.sub(r"\\end\{(?:itemize|enumerate)\}", "", s)
    s = re.sub(r"^\s*\\item\s+", "- ", s, flags=re.MULTILINE)

    callout_map = {
        "coremath": ("important", "Core Math"),
        "intuition": ("tip", "Intuition"),
        "crossbook": ("info", "Cross-Book Synthesis"),
        "pitfall": ("warning", "Pitfall"),
    }
    for env, (ctype, default_title) in callout_map.items():
        pattern = rf"\\begin\{{{env}\}}(?:\[([^\]]*)\])?(.*?)\\end\{{{env}\}}"

        def _repl(m: re.Match[str], ct: str = ctype, dt: str = default_title) -> str:
            title = m.group(1).strip() if m.group(1) else dt
            body = m.group(2).strip()
            quoted = "\n".join(f"> {ln}" if ln.strip() else ">" for ln in body.splitlines())
            return f"\n> [!{ct}] {title}\n{quoted}\n"

        s = re.sub(pattern, _repl, s, flags=re.DOTALL)

    s = s.replace("~", " ").replace("---", "—").replace("--", "–").replace("``", '"').replace("''", '"')
    s = s.replace(r"\&", "&").replace(r"\;", " ").replace(r"\,", " ")
    s = re.sub(r"\\vspace\*?\{[^}]*\}", "", s)
    s = re.sub(r"\\hspace\*?\{[^}]*\}", " ", s)
    # Ensure inline $...$ never spans newlines (outside $$...$$)
    disp_parts = re.split(r"(\$\$.*?\$\$)", s, flags=re.DOTALL)
    for di in range(0, len(disp_parts), 2):
        disp_parts[di] = re.sub(
            r"(?<!\$)\$(?!\$)([^$]+?)(?<!\$)\$(?!\$)",
            lambda m: "$" + re.sub(r"\s*\n\s*", " ", m.group(1)).strip() + "$",
            disp_parts[di],
            flags=re.DOTALL,
        )
    s = "".join(disp_parts)

    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def link_bookcites(text: str, slug_to_note: dict[str, str]) -> tuple[str, list[str]]:
    """Replace \\bookcite{slug}{loc} with Obsidian wikilinks to 03_Sources/."""
    cited: list[str] = []
    out = replace_macro_2arg(text, "bookcite", lambda s, l: _link_bookcite_helper(s, l, slug_to_note, cited))
    return out, cited


def _link_bookcite_helper(slug: str, loc: str, slug_to_note: dict[str, str], cited: list[str]) -> str:
    slug = slug.strip()
    loc = loc.strip().replace("~", " ")
    note = slug_to_note.get(slug, f"Source - {slug}")
    if note not in cited:
        cited.append(note)
    return f"[[{note}|{slug}]] ({loc})"


def autolink_concepts_in_markdown(
    md: str,
    alias_to_note: list[tuple[str, str]],
    exclude_note: str | None = None,
) -> tuple[str, list[str]]:
    """Link first prose occurrence of known concepts outside math, links, and headings."""
    linked: set[str] = set()
    token_pat = re.compile(
        r"(\$\$.*?\$\$|\$[^$\n]+\$|\[\[[^\]]+\]\]|`[^`]+`|^#{1,6}\s+.*$)",
        flags=re.DOTALL | re.MULTILINE,
    )
    parts = token_pat.split(md)
    for idx in range(0, len(parts), 2):
        prose = parts[idx]
        for alias, target in alias_to_note:
            if target == exclude_note or target in linked or len(alias) < 3:
                continue
            pat = re.compile(rf"(?<![A-Za-z0-9_\[])\b({re.escape(alias)})\b(?![A-Za-z0-9_\]])")
            m = pat.search(prose)
            if m:
                matched_text = m.group(1)
                replacement = f"[[{target}]]" if matched_text == target else f"[[{target}|{matched_text}]]"
                prose = prose[: m.start()] + replacement + prose[m.end() :]
                linked.add(target)
        parts[idx] = prose
    return "".join(parts), sorted(linked)


def extract_cheatsheet_summary(cs_path: Path) -> tuple[str, list[tuple[str, str]]]:
    """Extract hook tagline and branch titles + key formulas using balanced braces."""
    if not cs_path.exists():
        return "", []
    raw = cs_path.read_text(encoding="utf-8", errors="ignore")
    hook = ""
    csb_m = re.search(r"\\csbegin\{\d+\}", raw)
    if csb_m:
        cur = csb_m.end()
        args: list[str] = []
        for _ in range(3):
            while cur < len(raw) and raw[cur].isspace():
                cur += 1
            res = read_balanced(raw, cur) if cur < len(raw) and raw[cur] == "{" else None
            if res:
                args.append(res[0])
                cur = res[1]
        if len(args) == 3:
            hook = clean_latex_text(args[2])

    branches: list[tuple[str, str]] = []
    for m in re.finditer(r"\\csbranch\{\d+\}\{[A-G]\}", raw):
        cur = m.end()
        title_res = read_balanced(raw, cur) if cur < len(raw) and raw[cur] == "{" else None
        if not title_res:
            continue
        raw_title = title_res[0]
        btitle = clean_latex_text(raw_title).replace("$-$", "—").replace("$", "").strip()
        btitle = re.sub(r"^(\d+)\s+", r"\1. ", btitle)
        body_start = title_res[1]
        body_res = read_balanced(raw, body_start) if body_start < len(raw) and raw[body_start] == "{" else None
        fm = ""
        if body_res:
            fm_m = re.search(r"\\csfm\s*\{", body_res[0])
            if fm_m:
                fm_res = read_balanced(body_res[0], fm_m.end() - 1)
                if fm_res:
                    fm = clean_latex_text(fm_res[0])
        branches.append((btitle, fm))
    return hook, branches


def mermaid_safe_label(text: str) -> str:
    """Sanitize text for Mermaid flowchart labels."""
    s = re.sub(r"[^\w\s\-.,/+()=]", "", text)
    s = re.sub(r"\s+", " ", s).strip()
    return s[:48]


def yaml_list(items: list[str] | tuple[str, ...], as_links: bool = True) -> str:
    """Format a YAML list of strings or wikilinks."""
    if not items:
        return "[]"
    formatted = [f'"[[{x}]]"' if as_links else f'"{x}"' for x in items]
    return "\n" + "\n".join(f"  - {item}" for item in formatted)


def build_vault() -> None:
    """Build the complete Obsidian Knowledge Graph vault."""
    taxonomy = json.loads((ROOT / "data" / "taxonomy_tree.json").read_text(encoding="utf-8"))
    book_index = json.loads((ROOT / "data" / "book_index.json").read_text(encoding="utf-8"))

    if VAULT.exists():
        shutil.rmtree(VAULT)
    for sub in (".obsidian", "00_MOC", "01_Monographs", "02_Concepts", "03_Sources", "04_Cheatsheets"):
        (VAULT / sub).mkdir(parents=True, exist_ok=True)

    books_raw = book_index.get("books", [])
    slug_to_book: dict[str, dict[str, Any]] = {}
    slug_to_source_note: dict[str, str] = {}
    for b in books_raw:
        if b.get("is_duplicate"):
            continue
        slug = b["book_slug"]
        title = b.get("inferred_title") or b.get("metadata", {}).get("title") or slug
        title = re.sub(r"\s+", " ", str(title)).strip()[:80]
        note_name = slugify_note(f"Source - {title}")
        slug_to_book[slug] = b
        slug_to_source_note[slug] = note_name

    leaf_info: dict[str, dict[str, Any]] = {}
    domain_mocs: list[tuple[int, str, str]] = []
    pt_mocs: list[tuple[str, str, str]] = []
    st_mocs: list[tuple[str, str, str]] = []

    for d_idx, dom in enumerate(taxonomy["domains"], start=1):
        d_note = slugify_note(f"D{d_idx:02d} - {dom['title']}")
        domain_mocs.append((d_idx, dom["title"], d_note))
        for pt in dom["parent_topics"]:
            pt_note = slugify_note(f"PT {d_idx:02d} - {pt['title']}")
            pt_mocs.append((d_note, pt["title"], pt_note))
            for st in pt["sub_topics"]:
                st_note = slugify_note(f"ST {d_idx:02d} - {st['title']}")
                st_mocs.append((pt_note, st["title"], st_note))
                for lf in st["leaves"]:
                    m_note = slugify_note(lf["title"])
                    cs_note = slugify_note(f"Cheatsheet - {lf['title']}")
                    leaf_info[lf["id"]] = {
                        "leaf": lf,
                        "domain_idx": d_idx,
                        "domain_note": d_note,
                        "domain_title": dom["title"],
                        "pt_note": pt_note,
                        "pt_title": pt["title"],
                        "st_note": st_note,
                        "st_title": st["title"],
                        "monograph_note": m_note,
                        "cheatsheet_note": cs_note,
                    }

    spec_by_name: dict[str, ConceptSpec] = {c.name: c for c in CONCEPT_SPECS}
    prereqs: dict[str, set[str]] = defaultdict(set)
    enables: dict[str, set[str]] = defaultdict(set)
    generalizes: dict[str, set[str]] = defaultdict(set)
    special_of: dict[str, set[str]] = defaultdict(set)
    related: dict[str, set[str]] = defaultdict(set)
    contrasts: dict[str, set[str]] = defaultdict(set)
    duals: dict[str, set[str]] = defaultdict(set)

    for c in CONCEPT_SPECS:
        for p in c.prerequisites:
            assert p in spec_by_name, f"Missing prerequisite target: {p} (from {c.name})"
            prereqs[c.name].add(p)
            enables[p].add(c.name)
        for g in c.generalizes:
            assert g in spec_by_name, f"Missing generalizes target: {g} (from {c.name})"
            generalizes[c.name].add(g)
            special_of[g].add(c.name)
        for r in c.related:
            assert r in spec_by_name, f"Missing related target: {r} (from {c.name})"
            related[c.name].add(r)
            related[r].add(c.name)
        for ct in c.contrasts_with:
            assert ct in spec_by_name, f"Missing contrasts target: {ct} (from {c.name})"
            contrasts[c.name].add(ct)
            contrasts[ct].add(c.name)
        for dl in c.dual_to:
            assert dl in spec_by_name, f"Missing dual target: {dl} (from {c.name})"
            duals[c.name].add(dl)
            duals[dl].add(c.name)

    alias_pairs: list[tuple[str, str]] = []
    for c in CONCEPT_SPECS:
        alias_pairs.append((c.name, c.name))
        for al in c.aliases:
            alias_pairs.append((al, c.name))
    alias_pairs.sort(key=lambda x: -len(x[0]))

    source_cited_by_monograph: dict[str, set[str]] = defaultdict(set)
    source_cited_by_concept: dict[str, set[str]] = defaultdict(set)
    leaf_to_concepts: dict[str, list[str]] = defaultdict(list)
    concept_referenced_in_monographs: dict[str, set[str]] = defaultdict(set)

    for c in CONCEPT_SPECS:
        leaf_to_concepts[c.leaf_id].append(c.name)

    monograph_data: dict[str, dict[str, Any]] = {}
    for leaf_id, info in leaf_info.items():
        lf = info["leaf"]
        tex_path = ROOT / lf["tex_path"]
        raw_tex = tex_path.read_text(encoding="utf-8", errors="ignore") if tex_path.exists() else ""
        linked_tex, cited_sources = link_bookcites(raw_tex, slug_to_source_note)
        md_body = clean_latex_text(linked_tex)

        for src in lf.get("sources", []):
            s_note = slug_to_source_note.get(src["book_slug"])
            if s_note and s_note not in cited_sources:
                cited_sources.append(s_note)

        for s_note in cited_sources:
            source_cited_by_monograph[s_note].add(info["monograph_note"])

        md_linked, inline_concepts = autolink_concepts_in_markdown(md_body, alias_pairs)
        own_concepts = sorted(set(leaf_to_concepts[leaf_id]) | set(inline_concepts))
        for cname in own_concepts:
            concept_referenced_in_monographs[cname].add(info["monograph_note"])

        m_prereqs: set[str] = set()
        m_enables: set[str] = set()
        m_related: set[str] = set()
        for cname in leaf_to_concepts[leaf_id]:
            m_prereqs |= prereqs[cname]
            m_enables |= enables[cname]
            m_related |= related[cname]
            for s_note in cited_sources:
                source_cited_by_concept[s_note].add(cname)

        monograph_data[leaf_id] = {
            "cited_sources": sorted(cited_sources),
            "own_concepts": own_concepts,
            "m_prereqs": sorted(m_prereqs),
            "m_enables": sorted(m_enables),
            "m_related": sorted(m_related),
        }

        frontmatter = (
            "---\n"
            f'type: monograph\n'
            f'domain: "D{info["domain_idx"]:02d}"\n'
            f'parent_topic: "[[{info["pt_note"]}]]"\n'
            f'sub_topic: "[[{info["st_note"]}]]"\n'
            f'cheatsheet: "[[{info["cheatsheet_note"]}]]"\n'
            f'part_of: {yaml_list([info["st_note"], info["domain_note"]])}\n'
            f'concepts: {yaml_list(own_concepts)}\n'
            f'prerequisites: {yaml_list(sorted(m_prereqs))}\n'
            f'enables: {yaml_list(sorted(m_enables))}\n'
            f'related: {yaml_list(sorted(m_related))}\n'
            f'sources: {yaml_list(sorted(cited_sources))}\n'
            f'tags:\n  - monograph\n  - domain-{info["domain_idx"]:02d}\n'
            "---\n\n"
        )

        nav_header = (
            f"# {lf['title']}\n\n"
            f"> [!abstract] Monograph Navigation\n"
            f"> - **Domain:** [[{info['domain_note']}|{info['domain_title']}]]\n"
            f"> - **Parent / Sub-Topic:** [[{info['pt_note']}|{info['pt_title']}]] → [[{info['st_note']}|{info['st_title']}]]\n"
            f"> - **Visual Cheatsheet:** [[{info['cheatsheet_note']}]]\n"
            f"> - **Description:** {lf['description']}\n\n"
        )

        rel_section = (
            "\n\n---\n\n## Knowledge Graph Connections\n\n"
            f"- **Core Atomic Concepts ({len(leaf_to_concepts[leaf_id])}):** "
            + (", ".join(f"[[{c}]]" for c in sorted(leaf_to_concepts[leaf_id])) or "None")
            + "\n"
            f"- **Prerequisites (Directed $\\leftarrow$):** "
            + (", ".join(f"[[{c}]]" for c in sorted(m_prereqs)) or "None")
            + "\n"
            f"- **Enables / Unlocks (Directed $\\rightarrow$):** "
            + (", ".join(f"[[{c}]]" for c in sorted(m_enables)) or "None")
            + "\n"
            f"- **Related Concepts (Undirected $\\leftrightarrow$):** "
            + (", ".join(f"[[{c}]]" for c in sorted(m_related)) or "None")
            + "\n"
            f"- **Synthesized Sources:** "
            + (", ".join(f"[[{s}]]" for s in sorted(cited_sources)) or "None")
            + "\n"
        )

        out_path = VAULT / "01_Monographs" / f"{info['monograph_note']}.md"
        out_path.write_text(frontmatter + nav_header + md_linked + rel_section, encoding="utf-8")

    # 5. Generate 47 Cheatsheet Companion Notes (04_Cheatsheets/)
    for leaf_id, info in leaf_info.items():
        lf = info["leaf"]
        cs_tex = ROOT / lf["tex_path"].replace(".tex", "_cheatsheet.tex")
        hook, branches = extract_cheatsheet_summary(cs_tex)
        own_c = sorted(leaf_to_concepts[leaf_id])
        m_data = monograph_data[leaf_id]

        mermaid_lines = [
            "```mermaid",
            "flowchart LR",
            f'  ROOT["{mermaid_safe_label(lf["title"])}"]',
        ]
        for idx, (btitle, _fm) in enumerate(branches[:6], start=1):
            mermaid_lines.append(f'  B{idx}["{mermaid_safe_label(btitle)}"]')
            mermaid_lines.append(f"  ROOT --> B{idx}")
        for idx, cname in enumerate(own_c[:6], start=1):
            mermaid_lines.append(f'  C{idx}(("{mermaid_safe_label(cname)}"))')
            mermaid_lines.append(f"  ROOT -.-> C{idx}")
        mermaid_lines.append("```")

        fm_table = ""
        if branches:
            fm_table = "## Branch Summary & Key Equations\n\n"
            for btitle, fm in branches:
                fm_table += f"- **{btitle}**"
                if fm:
                    fm_table += f"\n  $$\n  {fm}\n  $$\n"
                else:
                    fm_table += "\n"

        cs_fm = (
            "---\n"
            "type: cheatsheet\n"
            f'domain: "D{info["domain_idx"]:02d}"\n'
            f'monograph: "[[{info["monograph_note"]}]]"\n'
            f'part_of: {yaml_list([info["monograph_note"], info["st_note"]])}\n'
            f'concepts: {yaml_list(own_c)}\n'
            "tags:\n  - cheatsheet\n  - mindmap\n"
            f"  - domain-{info['domain_idx']:02d}\n"
            "---\n\n"
        )
        cs_body = (
            f"# Cheatsheet — {lf['title']}\n\n"
            f"> [!summary] 10-Second Visual Hook\n> {hook or lf['description']}\n\n"
            f"- **Full Monograph:** [[{info['monograph_note']}]]\n"
            f"- **Sub-Topic MOC:** [[{info['st_note']}]]\n"
            f"- **Atomic Concepts:** {', '.join(f'[[{c}]]' for c in own_c) or 'None'}\n\n"
            "## Interactive Mindmap Overview\n\n"
            + "\n".join(mermaid_lines)
            + "\n\n"
            + fm_table
            + "\n## Related Sources\n\n"
            + (", ".join(f"[[{s}]]" for s in m_data["cited_sources"]) or "None")
            + "\n"
        )
        (VAULT / "04_Cheatsheets" / f"{info['cheatsheet_note']}.md").write_text(cs_fm + cs_body, encoding="utf-8")

    # 6. Generate Atomic Concept Notes (02_Concepts/)
    for c in CONCEPT_SPECS:
        info = leaf_info[c.leaf_id]
        m_note = info["monograph_note"]
        cs_note = info["cheatsheet_note"]
        sources = monograph_data[c.leaf_id]["cited_sources"]
        ref_monos = sorted(concept_referenced_in_monographs[c.name] | {m_note})

        c_pre = sorted(prereqs[c.name])
        c_en = sorted(enables[c.name])
        c_gen = sorted(generalizes[c.name])
        c_spc = sorted(special_of[c.name])
        c_rel = sorted(related[c.name])
        c_con = sorted(contrasts[c.name])
        c_dua = sorted(duals[c.name])

        frontmatter = (
            "---\n"
            f"type: {c.kind}\n"
            f'domain: "D{c.domain_idx:02d}"\n'
            f"aliases: {yaml_list(c.aliases, as_links=False)}\n"
            f'part_of: {yaml_list([m_note, cs_note, info["domain_note"]])}\n'
            f"prerequisites: {yaml_list(c_pre)}\n"
            f"enables: {yaml_list(c_en)}\n"
            f"generalizes: {yaml_list(c_gen)}\n"
            f"special_case_of: {yaml_list(c_spc)}\n"
            f"related: {yaml_list(c_rel)}\n"
            f"contrasts_with: {yaml_list(c_con)}\n"
            f"dual_to: {yaml_list(c_dua)}\n"
            f"sources: {yaml_list(sources)}\n"
            f"tags:\n  - {c.kind}\n  - domain-{c.domain_idx:02d}\n"
            "---\n\n"
        )

        formula_str = c.formula.replace(r"\argmax", r"\arg\max").replace(r"\argmin", r"\arg\min")
        formula_block = f"\n$$\n{formula_str}\n$$\n" if formula_str else ""
        body = (
            f"# {c.name}\n\n"
            f"> [!abstract] Definition ({c.kind.title()})\n> {c.summary}\n"
            f"{formula_block}\n"
            "## Semantic Graph Relations\n\n"
            "| Direction | Relation | Connected Nodes |\n"
            "| :--- | :--- | :--- |\n"
            f"| Directed ($\\leftarrow$) | **Prerequisites** (`requires`) | {', '.join(f'[[{x}]]' for x in c_pre) or '—'} |\n"
            f"| Directed ($\\rightarrow$) | **Enables / Unlocks** (`enables`) | {', '.join(f'[[{x}]]' for x in c_en) or '—'} |\n"
            f"| Directed ($\\supset$) | **Generalizes** | {', '.join(f'[[{x}]]' for x in c_gen) or '—'} |\n"
            f"| Directed ($\\subset$) | **Special Case Of** | {', '.join(f'[[{x}]]' for x in c_spc) or '—'} |\n"
            f"| Undirected ($\\leftrightarrow$) | **Related Concepts** | {', '.join(f'[[{x}]]' for x in c_rel) or '—'} |\n"
            f"| Undirected ($\\leftrightarrow$) | **Contrasts With** | {', '.join(f'[[{x}]]' for x in c_con) or '—'} |\n"
            f"| Undirected ($\\rightleftarrows$) | **Dual To** | {', '.join(f'[[{x}]]' for x in c_dua) or '—'} |\n\n"
            "## Monograph & Visual Anchors\n\n"
            f"- **Primary Monograph:** [[{m_note}]]\n"
            f"- **Visual Cheatsheet:** [[{cs_note}]]\n"
            f"- **Referenced Across Monographs:** {', '.join(f'[[{x}]]' for x in ref_monos)}\n"
            f"- **Primary Textbooks:** {', '.join(f'[[{s}]]' for s in sources) or '—'}\n"
        )
        (VAULT / "02_Concepts" / f"{c.name}.md").write_text(frontmatter + body, encoding="utf-8")

    # 7. Generate 34 Source Notes (03_Sources/)
    for slug, b in slug_to_book.items():
        s_note = slug_to_source_note[slug]
        monos = sorted(source_cited_by_monograph[s_note])
        concepts = sorted(source_cited_by_concept[s_note])
        kws = [kw["term"] for kw in b.get("top_keywords", [])[:12]] if isinstance(b.get("top_keywords"), list) else []
        frontmatter = (
            "---\n"
            "type: source\n"
            f'book_slug: "{slug}"\n'
            f'category: "{b.get("library_category", "unknown")}"\n'
            f'pages: {b.get("page_count", 0)}\n'
            f"cited_by_monographs: {yaml_list(monos)}\n"
            f"cited_by_concepts: {yaml_list(concepts)}\n"
            "tags:\n  - source\n  - textbook\n"
            "---\n\n"
        )
        body = (
            f"# {s_note}\n\n"
            f"- **Slug:** `{slug}`\n"
            f"- **Category:** `{b.get('library_category', 'unknown')}`\n"
            f"- **Pages:** {b.get('page_count', 0)}\n"
            + (f"- **Top Keywords:** {', '.join(kws)}\n" if kws else "")
            + "\n## Synthesized in Monographs\n\n"
            + ("\n".join(f"- [[{m}]]" for m in monos) if monos else "- *(Index reference)*\n")
            + "\n\n## Key Atomic Concepts\n\n"
            + (", ".join(f"[[{c}]]" for c in concepts) if concepts else "—")
            + "\n"
        )
        (VAULT / "03_Sources" / f"{s_note}.md").write_text(frontmatter + body, encoding="utf-8")

    # 8. Generate MOCs (00_MOC/)
    for pt_note, st_title, st_note in st_mocs:
        leaves_in_st = [v for v in leaf_info.values() if v["st_note"] == st_note]
        d_note = leaves_in_st[0]["domain_note"]
        c_list = sorted({c for v in leaves_in_st for c in leaf_to_concepts[v["leaf"]["id"]]})
        fm = (
            "---\n"
            "type: sub_topic_moc\n"
            f'part_of: {yaml_list([pt_note, d_note])}\n'
            f'monographs: {yaml_list([v["monograph_note"] for v in leaves_in_st])}\n'
            f"concepts: {yaml_list(c_list)}\n"
            "tags:\n  - moc\n  - sub-topic\n---\n\n"
        )
        body = (
            f"# {st_title}\n\n"
            f"- **Parent Topic:** [[{pt_note}]]\n"
            f"- **Domain:** [[{d_note}]]\n\n"
            "## Monographs & Cheatsheets\n\n"
            + "\n".join(
                f"- [[{v['monograph_note']}]] | [[{v['cheatsheet_note']}|Cheatsheet]]"
                for v in leaves_in_st
            )
            + "\n\n## Atomic Concepts\n\n"
            + (", ".join(f"[[{c}]]" for c in c_list) or "—")
            + "\n"
        )
        (VAULT / "00_MOC" / f"{st_note}.md").write_text(fm + body, encoding="utf-8")

    for d_note, pt_title, pt_note in pt_mocs:
        sts_in_pt = [(st_title, st_note) for p, st_title, st_note in st_mocs if p == pt_note]
        leaves_in_pt = [v for v in leaf_info.values() if v["pt_note"] == pt_note]
        fm = (
            "---\n"
            "type: parent_topic_moc\n"
            f'part_of: "[[{d_note}]]"\n'
            f"sub_topics: {yaml_list([s[1] for s in sts_in_pt])}\n"
            "tags:\n  - moc\n  - parent-topic\n---\n\n"
        )
        body = (
            f"# {pt_title}\n\n"
            f"- **Domain:** [[{d_note}]]\n\n"
            "## Sub-Topics\n\n"
            + "\n".join(f"- [[{s[1]}|{s[0]}]]" for s in sts_in_pt)
            + "\n\n## Monographs\n\n"
            + "\n".join(f"- [[{v['monograph_note']}]]" for v in leaves_in_pt)
            + "\n"
        )
        (VAULT / "00_MOC" / f"{pt_note}.md").write_text(fm + body, encoding="utf-8")

    for d_idx, d_title, d_note in domain_mocs:
        pts_in_dom = [(pt_title, pt_note) for dn, pt_title, pt_note in pt_mocs if dn == d_note]
        leaves_in_dom = [v for v in leaf_info.values() if v["domain_note"] == d_note]
        concepts_in_dom = sorted(c.name for c in CONCEPT_SPECS if c.domain_idx == d_idx)
        fm = (
            "---\n"
            "type: domain_moc\n"
            f'domain: "D{d_idx:02d}"\n'
            'part_of: "[[00_Home_MOC]]"\n'
            f"parent_topics: {yaml_list([p[1] for p in pts_in_dom])}\n"
            f"concepts: {yaml_list(concepts_in_dom)}\n"
            "tags:\n  - moc\n  - domain\n---\n\n"
        )
        body = (
            f"# Domain {d_idx:02d}: {d_title}\n\n"
            "- **Home:** [[00_Home_MOC]] | **Concept Index:** [[00_Concept_Graph_Index]]\n\n"
            "## Parent Topics\n\n"
            + "\n".join(f"- [[{p[1]}|{p[0]}]]" for p in pts_in_dom)
            + "\n\n## Leaf Monographs & Cheatsheets\n\n"
            + "\n".join(
                f"- [[{v['monograph_note']}]] · [[{v['cheatsheet_note']}|Cheatsheet]]"
                for v in leaves_in_dom
            )
            + "\n\n## Atomic Knowledge Graph Nodes\n\n"
            + (", ".join(f"[[{c}]]" for c in concepts_in_dom))
            + "\n"
        )
        (VAULT / "00_MOC" / f"{d_note}.md").write_text(fm + body, encoding="utf-8")

    by_kind: dict[str, list[ConceptSpec]] = defaultdict(list)
    for c in CONCEPT_SPECS:
        by_kind[c.kind].append(c)
    idx_lines = [
        "---\ntype: concept_index\npart_of: \"[[00_Home_MOC]]\"\ntags:\n  - moc\n  - index\n---\n",
        "# Concept Knowledge Graph Index\n",
        f"Total atomic nodes: **{len(CONCEPT_SPECS)}** across 7 domains.\n",
    ]
    for kind in ("theorem", "object", "algorithm", "architecture", "concept", "metric"):
        items = by_kind.get(kind, [])
        idx_lines.append(f"## {kind.title()}s ({len(items)})\n")
        for c in items:
            idx_lines.append(f"- [[{c.name}]] *(D{c.domain_idx:02d})* — {c.summary}")
        idx_lines.append("")
    (VAULT / "00_MOC" / "00_Concept_Graph_Index.md").write_text("\n".join(idx_lines), encoding="utf-8")

    home_md = (
        "---\n"
        "type: home_moc\n"
        "tags:\n  - moc\n  - home\n"
        "---\n\n"
        "# AI/ML SecondBrain Knowledge Graph\n\n"
        "> [!info] Graph Architecture (Directed + Undirected)\n"
        "> - **[[00_Concept_Graph_Index]]**: Atomic concepts, theorems, algorithms, objects, architectures, and metrics.\n"
        "> - **Directed Edges:** `prerequisites` ($\\leftarrow$), `enables` ($\\rightarrow$), `generalizes` ($\\supset$), `special_case_of` ($\\subset$), `part_of`.\n"
        "> - **Undirected Edges:** `related` ($\\leftrightarrow$), `contrasts_with` ($\\leftrightarrow$), `dual_to` ($\\rightleftarrows$).\n\n"
        "## 7 Knowledge Domains\n\n"
        + "\n".join(f"- **D{d_idx:02d}:** [[{d_note}|{d_title}]]" for d_idx, d_title, d_note in domain_mocs)
        + "\n\n## Core Cross-Domain Bridges\n\n"
        "- **Geometry & Optimization:** [[Singular Value Decomposition (SVD)]] ↔ [[Low-Rank Approximation (Eckart-Young)]] → [[LoRA & QLoRA]]\n"
        "- **Information & Alignment:** [[KL Divergence]] → [[Variational Inference & ELBO]] → [[Variational Autoencoder (VAE)]] / [[TRPO]] → [[RLHF (KL-Regularized)]] ↔ [[DPO (Direct Preference Optimization)]]\n"
        "- **Transport & Diffusion:** [[Wasserstein Distance]] ⇄ [[Kantorovich Duality]] → [[Wasserstein GAN (WGAN)]] / [[Score-Based SDE]] → [[Probability Flow ODE]] → [[Flow Matching & Rectified Flow]]\n"
        "- **Transformers & Agents:** [[Self-Attention & Multi-Head Attention]] → [[Transformer Architecture]] → [[Chain-of-Thought & Test-Time Compute]] → [[ReAct & Tool-Use Loop]] → [[Model Context Protocol (MCP)]] ↔ [[Agent-to-Agent Protocol (A2A)]]\n"
    )
    (VAULT / "00_MOC" / "00_Home_MOC.md").write_text(home_md, encoding="utf-8")

    (VAULT / ".obsidian" / "app.json").write_text(
        json.dumps(
            {
                "useMarkdownLinks": False,
                "strictLineBreaks": False,
                "readableLineLength": True,
                "showLineNumber": True,
                "newFileLocation": "folder",
                "newFileFolderPath": "02_Concepts",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (VAULT / ".obsidian" / "core-plugins.json").write_text(
        json.dumps(
            {
                "file-explorer": True,
                "global-search": True,
                "switcher": True,
                "graph": True,
                "backlink": True,
                "outgoing-link": True,
                "tag-pane": True,
                "page-preview": True,
                "outline": True,
                "properties": True,
                "bookmarks": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (VAULT / ".obsidian" / "graph.json").write_text(
        json.dumps(
            {
                "collapse-filter": False,
                "search": "",
                "showTags": False,
                "showAttachments": False,
                "hideUnresolved": False,
                "showOrphans": True,
                "collapse-color-groups": False,
                "colorGroups": [
                    {"query": "path:00_MOC", "color": {"a": 1, "rgb": 14725410}},
                    {"query": "path:01_Monographs", "color": {"a": 1, "rgb": 3763174}},
                    {"query": "path:02_Concepts", "color": {"a": 1, "rgb": 2274177}},
                    {"query": "path:04_Cheatsheets", "color": {"a": 1, "rgb": 584166}},
                    {"query": "path:03_Sources", "color": {"a": 1, "rgb": 9856202}},
                ],
                "collapse-display": False,
                "showArrow": True,
                "textFadeMultiplier": -0.5,
                "nodeSizeMultiplier": 1.15,
                "lineSizeMultiplier": 0.9,
                "collapse-forces": False,
                "centerStrength": 0.45,
                "repelStrength": 12.5,
                "linkStrength": 0.85,
                "linkDistance": 210,
                "scale": 0.85,
                "close": False,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"Built Obsidian vault at {VAULT}: "
        f"{len(leaf_info)} monographs, {len(leaf_info)} cheatsheets, "
        f"{len(CONCEPT_SPECS)} atomic concepts, {len(slug_to_book)} sources, "
        f"{2 + len(domain_mocs) + len(pt_mocs) + len(st_mocs)} MOCs."
    )


if __name__ == "__main__":
    build_vault()
