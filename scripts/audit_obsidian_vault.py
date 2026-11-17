"""Audit the Obsidian Knowledge Graph vault for link integrity, connectivity, and symmetry."""

from __future__ import annotations

import re
from collections import Counter, defaultdict, deque
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VAULT = ROOT / "obsidian_vault"


def extract_wikilinks(text: str) -> list[str]:
    """Extract target note names from [[target]] or [[target|alias]]."""
    raw = re.findall(r"\[\[([^\]|#]+)(?:#[^\]|]*)?(?:\|[^\]]+)?\]\]", text)
    return [r.strip() for r in raw if r.strip()]


def parse_yaml_list_field(text: str, field: str) -> set[str]:
    """Parse a YAML frontmatter list field of wikilinks."""
    fm_m = re.match(r"^---\n(.*?)\n---", text, flags=re.DOTALL)
    if not fm_m:
        return set()
    fm = fm_m.group(1)
    # Match single-line [] or multi-line - "[[...]]"
    m = re.search(rf"^{field}:\s*(\[\]|(?:\n\s+-\s+.*)+)", fm, flags=re.MULTILINE)
    if not m or m.group(1).strip() == "[]":
        return set()
    return set(extract_wikilinks(m.group(1)))


def main() -> int:
    md_files = sorted(VAULT.rglob("*.md"))
    name_to_path: dict[str, Path] = {p.stem: p for p in md_files}
    adj: dict[str, set[str]] = defaultdict(set)
    out_edges = 0
    broken: list[tuple[str, str]] = []

    for p in md_files:
        src = p.stem
        txt = p.read_text(encoding="utf-8")
        for tgt in extract_wikilinks(txt):
            if tgt not in name_to_path:
                broken.append((src, tgt))
            elif tgt != src:
                adj[src].add(tgt)
                adj[tgt].add(src)
                out_edges += 1

    # Orphans & degree
    orphans = [name for name in name_to_path if len(adj[name]) == 0]
    low_deg = [name for name in name_to_path if len(adj[name]) < 2]

    # Connected components
    visited: set[str] = set()
    components = 0
    for node in name_to_path:
        if node in visited:
            continue
        components += 1
        q = deque([node])
        visited.add(node)
        while q:
            cur = q.popleft()
            for nb in adj[cur]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

    # Symmetry & inverse checks on 02_Concepts
    concept_files = sorted((VAULT / "02_Concepts").glob("*.md"))
    c_fields: dict[str, dict[str, set[str]]] = {}
    for cp in concept_files:
        txt = cp.read_text(encoding="utf-8")
        c_fields[cp.stem] = {
            k: parse_yaml_list_field(txt, k)
            for k in (
                "prerequisites",
                "enables",
                "generalizes",
                "special_case_of",
                "related",
                "contrasts_with",
                "dual_to",
            )
        }

    asymmetry_errors: list[str] = []
    directed_count = 0
    undirected_count = 0
    for a, f in c_fields.items():
        directed_count += len(f["prerequisites"]) + len(f["generalizes"])
        undirected_count += len(f["related"]) + len(f["contrasts_with"]) + len(f["dual_to"])
        for b in f["prerequisites"]:
            if a not in c_fields.get(b, {}).get("enables", set()):
                asymmetry_errors.append(f"{a}.prerequisites -> {b} missing inverse {b}.enables")
        for b in f["enables"]:
            if a not in c_fields.get(b, {}).get("prerequisites", set()):
                asymmetry_errors.append(f"{a}.enables -> {b} missing inverse {b}.prerequisites")
        for b in f["generalizes"]:
            if a not in c_fields.get(b, {}).get("special_case_of", set()):
                asymmetry_errors.append(f"{a}.generalizes -> {b} missing inverse {b}.special_case_of")
        for sym in ("related", "contrasts_with", "dual_to"):
            for b in f[sym]:
                if a not in c_fields.get(b, {}).get(sym, set()):
                    asymmetry_errors.append(f"{a}.{sym} <-> {b} not symmetric")

    avg_deg = sum(len(adj[n]) for n in name_to_path) / max(1, len(name_to_path))
    top_hubs = Counter({n: len(adj[n]) for n in c_fields}).most_common(10)

    # Obsidian Markdown & MathJax Render Checks
    raw_latex_pat = re.compile(
        r"(\\vecb\b|\\mat\b|\\bm\b|\\abs\b|\\norm\b|\\KL\b|\\begin\{tabularx?\}|\\begin\{algorithm|\\toprule|\\midrule|\\bottomrule|\\captionof\b|\\monographfig\b|\\csfm\b|\\ForEach\b|\\While\b|\\KwIn\b|\\KwOut\b|\\Return\b|\\begingroup\b|\\endgroup\b|\\sloppy\b|\\texorpdfstring\b)"
    )
    render_errors: list[str] = []
    for p in md_files:
        txt = p.read_text(encoding="utf-8")
        body = re.sub(r"^---\n.*?\n---\n", "", txt, flags=re.DOTALL)
        # Strip mermaid/code blocks before math/latex checks
        body_no_code = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
        body_no_code = re.sub(r"`[^`]+`", "", body_no_code)
        for m in raw_latex_pat.finditer(body_no_code):
            render_errors.append(f"{p.name}: raw LaTeX leak '{m.group(1)}'")
        # Check balanced $$ blocks
        dd_count = len(re.findall(r"(?<!\\)\$\$", body_no_code))
        if dd_count % 2 != 0:
            render_errors.append(f"{p.name}: odd number of $$ delimiters ({dd_count})")
        # Check no raw \[ \] \( \) LaTeX math delimiters remain
        if re.search(r"(?<!\\)(?:\\\[|\\\]|\\\(|\\\))", body_no_code):
            render_errors.append(f"{p.name}: raw \\[ or \\( math delimiter")
        # Check every Markdown table header has a blank line (or bare '>') before it for Obsidian
        all_lines = body_no_code.splitlines()
        for lno in range(1, len(all_lines)):
            if re.match(r"^(?:>\s*)?\|[\s\-:|]+\|\s*$", all_lines[lno]):
                if lno >= 2 and all_lines[lno - 2].strip() not in ("", ">"):
                    render_errors.append(f"{p.name}:L{lno}: table missing blank line before header -> {all_lines[lno-2][:40]}")
        # Check inline $ per line (after stripping $$...$$)
        no_disp = re.sub(r"\$\$.*?\$\$", "", body_no_code, flags=re.DOTALL)
        for lno, ln in enumerate(no_disp.splitlines(), 1):
            sc = len(re.findall(r"(?<!\\)\$", ln))
            if sc % 2 != 0:
                render_errors.append(f"{p.name}:L{lno}: odd inline $ ({sc}) -> {ln[:60]}")

    print(f"Vault notes: {len(md_files)} | Total wikilink edges: {out_edges}")
    print(f"Connected components: {components} | Orphans: {len(orphans)} | Degree<2: {len(low_deg)}")
    print(f"Broken links: {len(broken)} | Asymmetry errors: {len(asymmetry_errors)} | Render errors: {len(render_errors)}")
    print(
        f"Concept graph: {len(concept_files)} atomic nodes | "
        f"Directed edges: {directed_count} | Undirected edges (halved): {undirected_count // 2} | "
        f"Vault avg degree: {avg_deg:.2f}"
    )
    print("Top 10 Concept Hubs:", ", ".join(f"{n} ({d})" for n, d in top_hubs))

    if broken:
        print("Sample broken:", broken[:10])
    if asymmetry_errors:
        print("Sample asymmetry:", asymmetry_errors[:10])
    if render_errors:
        print("Sample render errors:", render_errors[:15])

    return 0 if (not broken and not orphans and not asymmetry_errors and not render_errors and components == 1) else 1


if __name__ == "__main__":
    raise SystemExit(main())
