#!/usr/bin/env python3
"""
Targeted PDF & Index Reader for Cross-Book Knowledge Synthesis.

Helps synthesis subagents inspect TOC subheadings, search relevant pages inside
mapped source ranges, and read exact 1-based PDF page text cleanly without
loading full textbooks into context.

Usage examples:
    .venv/bin/python scripts/page_reader.py --leaf vector_spaces_norms_and_projections --toc
    .venv/bin/python scripts/page_reader.py --leaf vector_spaces_norms_and_projections --query "pseudoinverse,gram-schmidt"
    .venv/bin/python scripts/page_reader.py --book linear_algebra --pages 348-356
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Optional

import pymupdf


def _load_json(path: Path) -> Any:
    """Load JSON file with explicit error context."""
    if not path.is_file():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _parse_page_spec(spec: str, max_page: int) -> list[int]:
    """Parse 1-based page specification like '10,15-20' into sorted unique page indices."""
    pages: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo = max(1, int(lo_s.strip()))
            hi = min(max_page, int(hi_s.strip()))
            pages.update(range(lo, hi + 1))
        else:
            p = int(part)
            if 1 <= p <= max_page:
                pages.add(p)
    return sorted(pages)


def _find_leaf(tree: dict[str, Any], leaf_id: str) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Locate a leaf node and its ancestor Domain, Parent Topic, and Sub-Topic nodes."""
    for dom in tree.get("domains", []):
        for pt in dom.get("parent_topics", []):
            for st in pt.get("sub_topics", []):
                for lf in st.get("leaves", []):
                    if lf.get("id") == leaf_id:
                        return dom, pt, st, lf
    raise KeyError(f"Leaf ID '{leaf_id}' not found in taxonomy tree.")


def _normalize_page_text(raw: str) -> str:
    """Collapse excessive blank lines while preserving equation lines and lists."""
    lines = [re.sub(r"[ \t]+", " ", ln).rstrip() for ln in raw.splitlines()]
    out: list[str] = []
    blank_run = 0
    for ln in lines:
        if not ln:
            blank_run += 1
            if blank_run <= 1:
                out.append("")
        else:
            blank_run = 0
            out.append(ln)
    return "\n".join(out).strip()


def cmd_leaf_toc(
    book_index: dict[str, Any],
    taxonomy: dict[str, Any],
    leaf_id: str,
) -> int:
    """Print leaf metadata, source page ranges, printed_page_offset, and overlapping TOC headings."""
    books = {b["book_slug"]: b for b in book_index.get("books", [])}
    dom, pt, st, lf = _find_leaf(taxonomy, leaf_id)

    print(f"LEAF: {lf['title']} ({lf['id']})")
    print(f"PATH: {dom['title']} > {pt['title']} > {st['title']}")
    print(f"TEX : {lf['tex_path']}")
    print(f"DESC: {lf['description']}")
    print("VISUALS:")
    for kv in lf.get("key_visuals", []):
        print(f"  - {kv}")
    print("\nMAPPED SOURCES:")
    for src in lf.get("sources", []):
        slug = src["book_slug"]
        bk = books[slug]
        offset = bk.get("printed_page_offset")
        p0, p1 = int(src["page_start"]), int(src["page_end"])
        print(
            f"\n  [{slug}] PDF pp. {p0}-{p1} (book total={bk['page_count']}, printed_offset={offset}) | {src['chapter_or_section']}"
        )
        print(f"    Role: {src['synthesis_role']}")
        toc_hits = [
            t
            for t in bk.get("toc", [])
            if p0 - 2 <= int(t.get("pdf_page", t.get("page", 0))) <= p1
        ]
        for t in toc_hits[:35]:
            lvl = int(t.get("level", 1))
            pg = t.get("pdf_page", t.get("page"))
            indent = "      " + ("  " * max(0, lvl - 1))
            print(f"{indent}p.{pg}: {t.get('title')}")
        if len(toc_hits) > 35:
            print(f"      ... ({len(toc_hits) - 35} more TOC entries in range)")
    return 0


def _toc_section_for_page(toc: list[dict[str, Any]], pno: int) -> str:
    """Resolve the deepest TOC heading covering 1-based PDF page pno."""
    active: list[str] = []
    for entry in toc:
        start = int(entry.get("page", 0))
        end = int(entry.get("end_page", start))
        if start <= pno <= max(start, end):
            active.append(str(entry.get("title", "")))
    return " > ".join(active[-2:]) if active else ""


def cmd_leaf_query(
    book_index: dict[str, Any],
    taxonomy: dict[str, Any],
    leaf_id: str,
    query: str,
    top_k: int,
) -> int:
    """Rank mapped pages of a leaf monograph by keyword/term match."""
    books = {b["book_slug"]: b for b in book_index.get("books", [])}
    _, _, _, lf = _find_leaf(taxonomy, leaf_id)
    terms = [t.strip().lower() for t in re.split(r"[,;\s]+", query) if t.strip()]

    scored: list[tuple[float, str, int, list[str], str, list[str]]] = []
    seen: set[tuple[str, int]] = set()
    for src in lf.get("sources", []):
        slug = src["book_slug"]
        bk = books[slug]
        p0, p1 = int(src["page_start"]), int(src["page_end"])
        toc = bk.get("toc", [])
        for pg in bk.get("pages", []):
            pno = int(pg.get("page", 0))
            if not (p0 <= pno <= p1) or (slug, pno) in seen:
                continue
            seen.add((slug, pno))
            top_terms = [str(k) for k in pg.get("top_terms", [])]
            topic_emb = pg.get("topic_embedding", {})
            sec = _toc_section_for_page(toc, pno)
            hay = " ".join([sec] + top_terms + list(topic_emb.keys())).lower()
            hits = [tm for tm in terms if tm in hay]
            if hits:
                weight = sum(float(topic_emb.get(tm, 0.25)) for tm in hits) + 0.5 * len(hits)
                scored.append((weight, slug, pno, hits, sec, top_terms[:6]))

    scored.sort(key=lambda x: (-x[0], x[1], x[2]))
    print(f"Query hits for leaf '{leaf_id}' terms={terms} (showing top {top_k} of {len(scored)}):")
    for weight, slug, pno, hits, sec, tt in scored[:top_k]:
        print(f"  - [{slug} p.{pno}] score={weight:.2f} matches={hits} sec={sec!r} terms={tt}")
    return 0


def cmd_read_pages(
    book_index: dict[str, Any],
    book_slug: str,
    pages_spec: str,
    max_chars_per_page: int,
) -> int:
    """Extract cleaned text of specific 1-based PDF pages from a book."""
    books = {b["book_slug"]: b for b in book_index.get("books", [])}
    if book_slug not in books:
        raise KeyError(f"Unknown book_slug '{book_slug}'.")
    bk = books[book_slug]
    doc = pymupdf.open(bk["file_path"])
    try:
        pages = _parse_page_spec(pages_spec, len(doc))
        offset = bk.get("printed_page_offset")
        print(f"BOOK: {book_slug} | File: {bk['file_name']} | printed_offset={offset} | pages={pages_spec}")
        for pno in pages:
            raw = doc[pno - 1].get_text("text", sort=True)
            txt = _normalize_page_text(raw)
            if max_chars_per_page > 0 and len(txt) > max_chars_per_page:
                txt = txt[:max_chars_per_page] + "\n... [truncated]"
            printed_est = (pno - int(offset)) if isinstance(offset, int) else None
            p_tag = f"PDF p.{pno}" + (f" (printed ~p.{printed_est})" if printed_est and printed_est > 0 else "")
            print(f"\n--- {book_slug} | {p_tag} ---")
            print(txt)
    finally:
        doc.close()
    return 0


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inspect mapped textbook sections and read PDF pages.")
    p.add_argument("--index", type=Path, default=Path("data/book_index.json"))
    p.add_argument("--taxonomy", type=Path, default=Path("data/taxonomy_tree.json"))
    p.add_argument("--leaf", type=str, default=None, help="Leaf ID to inspect")
    p.add_argument("--toc", action="store_true", help="Show TOC headings in leaf source ranges")
    p.add_argument("--query", type=str, default=None, help="Keyword query within leaf mapped pages")
    p.add_argument("--top-k", type=int, default=20, help="Max query hits to display")
    p.add_argument("--book", type=str, default=None, help="Book slug to read")
    p.add_argument("--pages", type=str, default=None, help="1-based PDF page spec, e.g. '208-216'")
    p.add_argument("--max-chars", type=int, default=4500, help="Max chars per page when printing")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    book_index = _load_json(args.index)

    if args.book and args.pages:
        return cmd_read_pages(book_index, args.book, args.pages, args.max_chars)

    if args.leaf:
        taxonomy = _load_json(args.taxonomy)
        if args.query:
            return cmd_leaf_query(book_index, taxonomy, args.leaf, args.query, args.top_k)
        return cmd_leaf_toc(book_index, taxonomy, args.leaf)

    raise ValueError("Provide either (--book & --pages) or (--leaf with optional --toc/--query).")


if __name__ == "__main__":
    sys.exit(main())
