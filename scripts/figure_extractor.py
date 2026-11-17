#!/usr/bin/env python3
"""
300 DPI Figure Extractor for Cross-Book Knowledge Monograph Synthesis.

Detects both caption-anchored figures (raster, vector, and hybrid raster+vector
diagrams such as UDL/Murphy/Raschka/Goodfellow/Sutton/Tiep-Vu layouts) and
uncaptioned architectural diagrams inside mapped PDF page ranges, cropping cleanly
without prose bleed and rendering at 300 DPI to:
    ./assets/figures/<book_slug>_p<page>_fig<idx>.png

Usage examples:
    .venv/bin/python scripts/figure_extractor.py --book understandingdl_simonprince --pages 44-46
    .venv/bin/python scripts/figure_extractor.py --leaf vector_spaces_norms_and_projections
    .venv/bin/python scripts/figure_extractor.py --domain 01_mathematical_foundations --manifest-out data/figures_domain01.json
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import sys
from typing import Any, Optional

import numpy as np
import pymupdf

# Regex matching candidate figure captions at the start of a text block.
CAPTION_PREFIX_RE = re.compile(
    r"^(?P<label>(?:Figure|Fig\.?|H[ìi]nh)\s+\d+(?:[\.\-]\d+)*(?:[a-z])?)"
    r"(?P<delim>\s*[:\.\-—]|\s+)",
    re.IGNORECASE,
)

# Inline body-prose indicators: if a block begins with "Figure X.Y <verb>" in
# normal body font, it is almost always a textual cross-reference rather than a caption.
PROSE_VERB_AFTER_LABEL_RE = re.compile(
    r"^(?:Figure|Fig\.?|H[ìi]nh)\s+\d+(?:[\.\-]\d+)*(?:[a-z])?\s+"
    r"(?:shows?|illustrates?|depicts?|demonstrates?|compares?|summarizes?|"
    r"presents?|plots?|displays?|gives?|provides?|is|are|was|were|has|have|"
    r"can|may|will|would|and|or|also|again|below|above|on|in|"
    r"tương\s+ứng|cho\s+thấy|minh\s+họa|biểu\s+diễn|là|được|có|thể\s+hiện)\b",
    re.IGNORECASE,
)

TABLE_OR_ALGO_PREFIX_RE = re.compile(
    r"^(?:Table|Bảng|Algorithm|Thuật\s+toán|Box|Example|Exercise|Problem)\s+\d+",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class CaptionBlock:
    """Structured representation of a detected figure caption on a PDF page."""

    label: str
    full_text: str
    bbox: pymupdf.Rect
    is_bold_or_sans: bool
    font_size: float
    has_explicit_delim: bool


@dataclass(frozen=True)
class ExtractedFigure:
    """Metadata for a single detected or rendered 300 DPI figure asset."""

    book_slug: str
    page: int
    fig_idx: int
    kind: str  # "captioned" or "uncaptioned_visual"
    label: str
    caption: str
    bbox: list[float]
    pixel_width: int
    pixel_height: int
    dpi: int
    output_path: str
    source_mode: str  # "raster", "vector", "hybrid", or "scan_crop"


def _load_book_index(index_path: Path) -> dict[str, dict[str, Any]]:
    """Load book metadata keyed by book_slug from ./data/book_index.json."""
    if not index_path.is_file():
        raise FileNotFoundError(f"Book index not found: {index_path}")
    with index_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {b["book_slug"]: b for b in data.get("books", [])}


def _parse_page_spec(spec: str, max_page: int) -> list[int]:
    """Parse 1-based page specifications like '10,15-22,40' into sorted unique pages."""
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


def _rect_intersection_area(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Compute intersection area between two PyMuPDF rectangles."""
    inter = a & b
    return 0.0 if inter.is_empty else float(inter.width * inter.height)


def _rect_distance(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    """Compute shortest Euclidean distance between two axis-aligned rectangles."""
    dx = max(0.0, a.x0 - b.x1, b.x0 - a.x1)
    dy = max(0.0, a.y0 - b.y1, b.y0 - a.y1)
    return (dx * dx + dy * dy) ** 0.5


def _merge_close_rects(rects: list[pymupdf.Rect], x_tol: float, y_tol: float) -> list[pymupdf.Rect]:
    """Iteratively merge rectangles whose horizontal/vertical gaps fall within tolerances."""
    if not rects:
        return []
    groups = [pymupdf.Rect(r) for r in rects]
    changed = True
    while changed:
        changed = False
        next_groups: list[pymupdf.Rect] = []
        used = [False] * len(groups)
        for i in range(len(groups)):
            if used[i]:
                continue
            cur = pymupdf.Rect(groups[i])
            used[i] = True
            merged_in_pass = True
            while merged_in_pass:
                merged_in_pass = False
                for j in range(i + 1, len(groups)):
                    if used[j]:
                        continue
                    expanded = pymupdf.Rect(
                        cur.x0 - x_tol,
                        cur.y0 - y_tol,
                        cur.x1 + x_tol,
                        cur.y1 + y_tol,
                    )
                    if expanded.intersects(groups[j]):
                        cur.include_rect(groups[j])
                        used[j] = True
                        merged_in_pass = True
                        changed = True
            next_groups.append(cur)
        groups = next_groups
    return groups


def _extract_clean_visuals(page: pymupdf.Page) -> tuple[list[pymupdf.Rect], list[pymupdf.Rect]]:
    """
    Collect non-artifact raster image bboxes and vector drawing bboxes from a page.

    Filters out printer trim/crop marks near corners, full-page backgrounds,
    1px header/footer rules, and tiny decorative glyphs.
    """
    pw = float(page.rect.width)
    ph = float(page.rect.height)
    interior = pymupdf.Rect(22.0, 22.0, pw - 22.0, ph - 22.0)

    rasters: list[pymupdf.Rect] = []
    for im in page.get_image_info(xrefs=False):
        r = pymupdf.Rect(im["bbox"]) & page.rect
        if r.is_empty:
            continue
        # Skip 1px decorative rules, tiny icons, and corner brand logos.
        if r.width < 32.0 or r.height < 18.0:
            continue
        if r.width < 68.0 and r.height < 68.0 and (r.y1 < 0.14 * ph or r.y0 > 0.86 * ph):
            continue
        # Skip full-page slide background images when nearly blank/border.
        if r.width > 0.96 * pw and r.height > 0.96 * ph:
            continue
        rasters.append(r)

    drawings: list[pymupdf.Rect] = []
    for drw in page.get_drawings():
        r = pymupdf.Rect(drw["rect"])
        if r.is_empty or not r.intersects(interior):
            continue
        r = r & interior
        # Ignore hairline rules outside main content bands and full-page frames.
        if r.width > 0.85 * pw and r.height > 0.85 * ph:
            continue
        if r.width <= 1.2 and r.height <= 1.2:
            continue
        # Ignore full-width header/footer divider rules near page top/bottom.
        if r.height <= 3.5 and r.width > 0.55 * pw and (r.y1 < 0.17 * ph or r.y0 > 0.86 * ph):
            continue
        drawings.append(r)

    return rasters, drawings


def _analyze_text_blocks(
    page: pymupdf.Page,
) -> tuple[list[tuple[pymupdf.Rect, str, float, bool]], float]:
    """
    Extract text blocks with dominant font size and bold/sans style flag,
    plus the page's estimated body font size.
    """
    ph = float(page.rect.height)
    text_blocks: list[tuple[pymupdf.Rect, str, float, bool]] = []
    font_size_weights: dict[float, int] = {}

    text_dict = page.get_text("dict")
    for blk in text_dict.get("blocks", []):
        if blk.get("type") != 0:
            continue
        spans = [sp for ln in blk.get("lines", []) for sp in ln.get("spans", []) if sp.get("text", "").strip()]
        if not spans:
            continue
        rect = pymupdf.Rect(blk["bbox"])
        text = " ".join(" ".join(sp["text"] for sp in ln.get("spans", [])) for ln in blk.get("lines", []))
        clean_text = re.sub(r"\s+", " ", text).strip()
        if not clean_text:
            continue

        # Weight font sizes outside headers/footers to identify body font size.
        if 0.10 * ph < rect.y0 and rect.y1 < 0.90 * ph and len(clean_text) > 40:
            for sp in spans:
                sz = round(float(sp["size"]), 1)
                font_size_weights[sz] = font_size_weights.get(sz, 0) + len(sp["text"])

        first_sp = spans[0]
        first_size = float(first_sp["size"])
        first_font = str(first_sp.get("font", "")).lower()
        is_bold_or_sans = bool(int(first_sp.get("flags", 0)) & 16) or any(
            tok in first_font for tok in ("bold", "sans", "semibold", "demi", "cmbx", "lmsans")
        )
        text_blocks.append((rect, clean_text, first_size, is_bold_or_sans))

    body_font_size = max(font_size_weights, key=lambda k: font_size_weights[k]) if font_size_weights else 10.5
    return text_blocks, body_font_size


def _detect_captions(
    page: pymupdf.Page,
    text_blocks: list[tuple[pymupdf.Rect, str, float, bool]],
    body_font_size: float,
    rasters: list[pymupdf.Rect],
    drawings: list[pymupdf.Rect],
) -> list[CaptionBlock]:
    """Identify genuine figure captions and reject body prose mentions."""
    captions: list[CaptionBlock] = []
    ph = float(page.rect.height)
    pw = float(page.rect.width)

    for rect, clean_text, fsize, is_bold_or_sans in text_blocks:
        # Skip running headers/footers.
        if rect.y1 < 0.08 * ph or rect.y0 > 0.94 * ph:
            continue
        m = CAPTION_PREFIX_RE.match(clean_text)
        if not m:
            continue

        delim = m.group("delim").strip()
        has_explicit_delim = delim in {":", ".", "-", "—"}

        # Reject prose sentences like "Figure 3.7 shows that..." unless explicitly styled as caption.
        if not has_explicit_delim and PROSE_VERB_AFTER_LABEL_RE.match(clean_text):
            if not is_bold_or_sans and fsize >= body_font_size - 0.3:
                continue

        # Must have significant visual elements near this block (above, beside, or right below).
        search_band = pymupdf.Rect(
            0.0,
            max(0.0, rect.y0 - 0.72 * ph),
            pw,
            min(ph, rect.y1 + 0.35 * ph),
        )
        nearby_rasters = [r for r in rasters if r.intersects(search_band)]
        nearby_drawings = [d for d in drawings if d.intersects(search_band)]
        if not nearby_rasters and len(nearby_drawings) < 4:
            continue

        # Additional safeguard against long body paragraphs mentioning Figure X.Y at start.
        if not has_explicit_delim and not is_bold_or_sans and fsize >= body_font_size and len(clean_text) > 420:
            continue

        captions.append(
            CaptionBlock(
                label=re.sub(r"\s+", " ", m.group("label")).strip(),
                full_text=clean_text,
                bbox=rect,
                is_bold_or_sans=is_bold_or_sans,
                font_size=fsize,
                has_explicit_delim=has_explicit_delim,
            )
        )

    # Sort top-to-bottom for deterministic caption zone partitioning.
    captions.sort(key=lambda c: (c.bbox.y0, c.bbox.x0))
    return captions


def _is_diagram_label_block(
    blk_rect: pymupdf.Rect,
    blk_text: str,
    blk_size: float,
    body_font_size: float,
    visual_union: pymupdf.Rect,
    pw: float,
) -> bool:
    """Determine if a text block is an embedded plot label/axis tick belonging to a figure."""
    if TABLE_OR_ALGO_PREFIX_RE.match(blk_text) or CAPTION_PREFIX_RE.match(blk_text):
        return False

    # Always include text completely inside or overlapping a visual cluster if short or smaller font.
    if visual_union.intersects(blk_rect):
        if len(blk_text) <= 95 or blk_rect.width < 0.48 * pw or blk_size <= body_font_size - 0.4:
            return True

    # Include axis ticks / axis titles directly outside visual cluster bounds.
    if _rect_distance(blk_rect, visual_union) <= 18.0:
        if len(blk_text) <= 45 and (blk_size <= body_font_size + 0.5 or blk_rect.width < 0.30 * pw):
            return True
    return False


def _extract_caption_figure_bbox(
    page: pymupdf.Page,
    cap_idx: int,
    captions: list[CaptionBlock],
    text_blocks: list[tuple[pymupdf.Rect, str, float, bool]],
    body_font_size: float,
    rasters: list[pymupdf.Rect],
    drawings: list[pymupdf.Rect],
) -> tuple[Optional[pymupdf.Rect], str]:
    """
    Locate the cleanest tight diagram bounding box anchored to captions[cap_idx].

    Handles three textbook conventions cleanly:
    1. Semi-scanned / full-block raster pages (Goodfellow 2017): raster image overlaps
       caption and underlying prose -> clip bottom strictly at caption.y0 - 4pt.
    2. Standard bottom-captioned figures (Sutton, Murphy, Raschka, Tiep Vu, Prince top):
       cluster drawings/rasters in the band above the caption down to caption.y0 - 3pt,
       bounded above by previous prose paragraph or running header.
    3. Side-captioned two-column figures (Prince UDL Figure 3.7): visual sits horizontally
       beside the caption in the vertical band between surrounding elements.
    """
    pw = float(page.rect.width)
    ph = float(page.rect.height)
    cap = captions[cap_idx]
    cap_r = cap.bbox

    # Check Case 1 first: large raster above/overlapping the caption top.
    for r in rasters:
        # In Goodfellow, the figure scan overlaps caption.y0 or ends just above caption.y0.
        if r.width > 0.45 * pw and r.y0 < cap_r.y0 - 30.0 and cap_r.y0 - 24.0 <= r.y1 <= cap_r.y0 + 180.0:
            # Find header floor above raster without cutting top axes.
            top_floor = max(18.0, r.y0)
            for tb_r, tb_txt, _, _ in text_blocks:
                if tb_r.y1 < r.y0 + 6.0 and tb_r.y1 < 0.14 * ph:
                    top_floor = max(top_floor, tb_r.y1 + 2.0)
            bottom_ceil = min(r.y1 + 2.0, cap_r.y0 - 3.0)
            if bottom_ceil - top_floor >= 45.0:
                crop = pymupdf.Rect(max(12.0, r.x0 - 4.0), top_floor, min(pw - 12.0, r.x1 + 4.0), bottom_ceil)
                return crop, "scan_crop"

    # Compute dynamic page top/bottom margins from actual running header/footer blocks.
    header_floor = 16.0
    footer_ceil = ph - 16.0
    for tb_r, _, _, _ in text_blocks:
        if tb_r.y1 < 0.09 * ph:
            header_floor = max(header_floor, tb_r.y1 + 2.0)
        elif tb_r.y0 > 0.90 * ph:
            footer_ceil = min(footer_ceil, tb_r.y0 - 2.0)

    # Identify prose boundaries (headers/footers, body paragraphs, previous captions, section titles).
    prev_cap_y1 = captions[cap_idx - 1].bbox.y1 + 4.0 if cap_idx > 0 else header_floor
    next_cap_y0 = captions[cap_idx + 1].bbox.y0 - 4.0 if cap_idx + 1 < len(captions) else footer_ceil

    # Filter prose anchors (exclude short diagram labels and equations inside diagrams).
    prose_blocks: list[pymupdf.Rect] = []
    for tb_r, tb_txt, tb_sz, _ in text_blocks:
        if tb_r == cap_r:
            continue
        if tb_r.y1 <= 0.11 * ph or tb_r.y0 >= 0.89 * ph:
            prose_blocks.append(tb_r)
            continue
        # Substantial body paragraph or heading: wide block with normal/large font and many chars.
        if len(tb_txt) > 95 and tb_r.width > 0.45 * pw and tb_sz >= body_font_size - 0.6:
            prose_blocks.append(tb_r)
        elif re.match(r"^\d+\.\d+(?:\.\d+)?\s+[A-Z]", tb_txt) and tb_r.width > 0.25 * pw:
            prose_blocks.append(tb_r)

    # Candidate 2A: Diagram ABOVE caption.
    above_floor = prev_cap_y1
    for pr in prose_blocks:
        if pr.y1 <= cap_r.y0 - 15.0:
            # Verify whether substantial visuals exist below pr.y1; if none, skip.
            above_floor = max(above_floor, pr.y1 + 3.0) if pr.y1 < cap_r.y0 - 45.0 else above_floor

    # Refine above_floor carefully: only raise above a prose block that sits strictly above
    # the visual elements attached to this caption, using contiguous visual gap analysis.
    above_rasters = [r for r in rasters if prev_cap_y1 - 8.0 <= r.y0 and r.y1 <= cap_r.y0 + 10.0]
    above_drawings = [d for d in drawings if prev_cap_y1 - 8.0 <= d.y0 and d.y1 <= cap_r.y0 + 10.0]

    # Exclude visuals sitting strictly above a prose paragraph above the caption.
    candidate_above = above_rasters + above_drawings
    # Filter out drawings that lie mostly above a prose block between prev_cap and cap.
    blocking_prose_above = sorted(
        [pr for pr in prose_blocks if prev_cap_y1 <= pr.y0 and pr.y1 <= cap_r.y0 - 8.0],
        key=lambda r: r.y1,
    )
    zone_top = prev_cap_y1
    if blocking_prose_above:
        # Check from nearest prose block up: keep only visuals below the lowest blocking prose
        # UNLESS there are no visuals between that prose block and the caption.
        lowest_pr = blocking_prose_above[-1]
        vis_below_lowest = [v for v in candidate_above if v.y0 >= lowest_pr.y1 - 4.0]
        if any(v.height >= 20.0 or v.width >= 20.0 for v in vis_below_lowest) or len(vis_below_lowest) >= 4:
            zone_top = max(zone_top, lowest_pr.y1 + 3.0)

    zone_above_rect = pymupdf.Rect(16.0, zone_top, pw - 16.0, cap_r.y0 + 6.0)
    vis_in_above = [v for v in candidate_above if _rect_intersection_area(v, zone_above_rect) > 0.5 * v.get_area()]

    # Candidate 2B: Side-by-side diagram BESIDE caption (e.g., Prince UDL Figure 3.7).
    side_vis: list[pymupdf.Rect] = []
    side_top = max(prev_cap_y1, cap_r.y0 - 70.0)
    side_bot = min(next_cap_y0, cap_r.y1 + 110.0)
    for pr in prose_blocks:
        if pr.y1 <= cap_r.y0:
            side_top = max(side_top, pr.y1 + 3.0)
        if pr.y0 >= cap_r.y1:
            side_bot = min(side_bot, pr.y0 - 3.0)
    side_band = pymupdf.Rect(16.0, side_top, pw - 16.0, side_bot)
    # Must be horizontally separated from caption rect (not directly overlapping caption text).
    for v in rasters + drawings:
        if not v.intersects(side_band):
            continue
        if v.x1 <= cap_r.x0 + 8.0 or v.x0 >= cap_r.x1 - 8.0:
            side_vis.append(v)

    chosen_vis: list[pymupdf.Rect] = []
    if len(vis_in_above) >= 4 or any(r in vis_in_above for r in above_rasters):
        chosen_vis = vis_in_above
        cap_bottom_limit = cap_r.y0 - 3.0
    elif len(side_vis) >= 6 or any(r in side_vis for r in rasters):
        chosen_vis = side_vis
        cap_bottom_limit = side_bot
    else:
        # Candidate 2C: Figure placed BELOW caption (some papers/guides use top captions).
        below_floor = next_cap_y0
        for pr in prose_blocks:
            if pr.y0 >= cap_r.y1 + 25.0:
                below_floor = min(below_floor, pr.y0 - 3.0)
        zone_below_rect = pymupdf.Rect(16.0, cap_r.y1 - 4.0, pw - 16.0, below_floor)
        vis_below = [v for v in rasters + drawings if _rect_intersection_area(v, zone_below_rect) > 0.5 * v.get_area()]
        if len(vis_below) >= 4 or any(r in vis_below for r in rasters):
            chosen_vis = vis_below
            cap_bottom_limit = below_floor

    if not chosen_vis:
        return None, "none"

    union = pymupdf.Rect(chosen_vis[0])
    for v in chosen_vis[1:]:
        union.include_rect(v)

    # Reject minuscule false triggers.
    if union.width < 55.0 or union.height < 40.0:
        return None, "none"

    # Expand to include small embedded diagram labels (axis numbers, legends, node tags)
    # without accidentally swallowing caption text or prose blocks.
    final_rect = pymupdf.Rect(union)
    for tb_r, tb_txt, tb_sz, _ in text_blocks:
        if tb_r == cap_r or tb_r in prose_blocks:
            continue
        if any(tb_r == c.bbox for c in captions):
            continue
        if tb_r.y0 >= cap_bottom_limit and chosen_vis is vis_in_above:
            continue
        if _is_diagram_label_block(tb_r, tb_txt, tb_sz, body_font_size, union, pw):
            # Do not expand past caption top for above-caption figures.
            if chosen_vis is vis_in_above and tb_r.y1 > cap_r.y0 - 1.5:
                continue
            final_rect.include_rect(tb_r)

    # Determine source mode (raster / vector / hybrid).
    has_r = any(r.intersects(final_rect) for r in rasters)
    has_v = len([d for d in drawings if d.intersects(final_rect)]) >= 3
    mode = "hybrid" if (has_r and has_v) else ("raster" if has_r else "vector")

    # Apply clean padding while staying safely clear of caption and page edge.
    pad = 6.0
    padded = pymupdf.Rect(
        max(8.0, final_rect.x0 - pad),
        max(8.0, final_rect.y0 - pad),
        min(pw - 8.0, final_rect.x1 + pad),
        min(cap_bottom_limit, final_rect.y1 + pad) if chosen_vis is vis_in_above else min(ph - 8.0, final_rect.y1 + pad),
    )
    if padded.width < 55.0 or padded.height < 40.0:
        return None, "none"
    return padded, mode


def _detect_uncaptioned_visuals(
    page: pymupdf.Page,
    text_blocks: list[tuple[pymupdf.Rect, str, float, bool]],
    body_font_size: float,
    rasters: list[pymupdf.Rect],
    drawings: list[pymupdf.Rect],
) -> list[tuple[pymupdf.Rect, str]]:
    """
    Detect standalone architecture diagrams or figures on pages without 'Figure X' captions
    (common in illustrated guidebooks, slide decks, and engineering whitepapers).
    """
    pw = float(page.rect.width)
    ph = float(page.rect.height)
    candidates: list[tuple[pymupdf.Rect, str]] = []

    # Skip pages dominated by tables.
    if any(TABLE_OR_ALGO_PREFIX_RE.match(txt) and r.y0 < 0.35 * ph for r, txt, _, _ in text_blocks):
        if len(rasters) == 0:
            return []

    # Cluster non-background drawings into contiguous diagram regions.
    merged_drawings = _merge_close_rects(drawings, x_tol=24.0, y_tol=20.0)
    rich_vector_clusters: list[pymupdf.Rect] = []
    for cl in merged_drawings:
        count = sum(1 for d in drawings if cl.intersects(d))
        # Require substantial vector complexity and nontrivial geometry (skip simple boxes/tables).
        if count >= 10 and cl.width >= 150.0 and cl.height >= 90.0:
            # Reject likely full-page callout text boxes that mostly wrap body prose.
            overlapping_prose = sum(
                len(txt) for r, txt, sz, _ in text_blocks if cl.intersects(r) and sz >= body_font_size - 0.5 and r.width > 0.5 * pw
            )
            if overlapping_prose < 650:
                rich_vector_clusters.append(cl)

    # Combine raster figures with nearby vector overlays.
    all_regions: list[tuple[pymupdf.Rect, str]] = []
    used_clusters = [False] * len(rich_vector_clusters)

    for r in rasters:
        if r.width < 120.0 or r.height < 80.0:
            continue
        box = pymupdf.Rect(r)
        is_hybrid = False
        for i, cl in enumerate(rich_vector_clusters):
            if box.intersects(cl) or _rect_distance(box, cl) <= 18.0:
                box.include_rect(cl)
                used_clusters[i] = True
                is_hybrid = True
        all_regions.append((box, "hybrid" if is_hybrid else "raster"))

    for i, cl in enumerate(rich_vector_clusters):
        if not used_clusters[i]:
            all_regions.append((cl, "vector"))

    # Pad slightly and absorb enclosed short diagram labels.
    for box, mode in all_regions:
        out = pymupdf.Rect(box)
        for tb_r, tb_txt, tb_sz, _ in text_blocks:
            if _is_diagram_label_block(tb_r, tb_txt, tb_sz, body_font_size, box, pw):
                out.include_rect(tb_r)
        padded = pymupdf.Rect(
            max(8.0, out.x0 - 6.0),
            max(8.0, out.y0 - 6.0),
            min(pw - 8.0, out.x1 + 6.0),
            min(ph - 8.0, out.y1 + 6.0),
        )
        candidates.append((padded, mode))

    # Deduplicate overlapping regions on the same page.
    deduped: list[tuple[pymupdf.Rect, str]] = []
    for rect, mode in sorted(candidates, key=lambda x: x[0].get_area(), reverse=True):
        if any(_rect_intersection_area(rect, prev) > 0.45 * min(rect.get_area(), prev.get_area()) for prev, _ in deduped):
            continue
        deduped.append((rect, mode))

    deduped.sort(key=lambda x: (x[0].y0, x[0].x0))
    return deduped


def _trim_pixmap_whitespace(
    pix: pymupdf.Pixmap,
    margin_px: int = 16,
    white_threshold: int = 247,
) -> Optional[pymupdf.Pixmap]:
    """
    Tightly crop near-white outer margins from a rendered RGB Pixmap while keeping a clean margin.

    Returns None if the rendered crop contains virtually no foreground content.
    """
    if pix.n < 3 or pix.width <= 0 or pix.height <= 0:
        return pix
    arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    rgb = arr[:, :, :3]
    mask = np.any(rgb < white_threshold, axis=2)
    ys, xs = np.where(mask)
    if ys.size < 25:
        return None

    x0 = max(0, int(xs.min()) - margin_px)
    y0 = max(0, int(ys.min()) - margin_px)
    x1 = min(pix.width, int(xs.max()) + 1 + margin_px)
    y1 = min(pix.height, int(ys.max()) + 1 + margin_px)

    w = x1 - x0
    h = y1 - y0
    if w < 64 or h < 48:
        return None
    if w == pix.width and h == pix.height:
        return pix

    cropped = np.ascontiguousarray(rgb[y0:y1, x0:x1, :])
    return pymupdf.Pixmap(pymupdf.csRGB, w, h, cropped.tobytes(), False)


def extract_page_figures(
    page: pymupdf.Page,
    book_slug: str,
    page_num: int,
    out_dir: Path,
    dpi: int = 300,
    dry_run: bool = False,
) -> list[ExtractedFigure]:
    """Detect and optionally render all qualifying figures on a single 1-based PDF page."""
    rasters, drawings = _extract_clean_visuals(page)
    if not rasters and len(drawings) < 4:
        return []

    text_blocks, body_font_size = _analyze_text_blocks(page)
    captions = _detect_captions(page, text_blocks, body_font_size, rasters, drawings)

    targets: list[tuple[pymupdf.Rect, str, str, str, str]] = []  # (rect, kind, label, caption, mode)

    if captions:
        for idx in range(len(captions)):
            bbox, mode = _extract_caption_figure_bbox(
                page, idx, captions, text_blocks, body_font_size, rasters, drawings
            )
            if bbox is not None:
                targets.append((bbox, "captioned", captions[idx].label, captions[idx].full_text, mode))
    else:
        for rect, mode in _detect_uncaptioned_visuals(page, text_blocks, body_font_size, rasters, drawings):
            targets.append((rect, "uncaptioned_visual", f"p.{page_num}", "", mode))

    results: list[ExtractedFigure] = []
    scale = float(dpi) / 72.0
    mat = pymupdf.Matrix(scale, scale)

    if not dry_run:
        out_dir.mkdir(parents=True, exist_ok=True)

    saved_idx = 0
    for rect, kind, label, caption, mode in targets:
        if dry_run:
            saved_idx += 1
            fname = f"{book_slug}_p{page_num}_fig{saved_idx}.png"
            out_path = out_dir / fname
            pw_px = int(round(rect.width * scale))
            ph_px = int(round(rect.height * scale))
        else:
            raw_pix = page.get_pixmap(matrix=mat, clip=rect, alpha=False)
            pix = _trim_pixmap_whitespace(raw_pix, margin_px=16)
            if pix is None:
                continue
            saved_idx += 1
            fname = f"{book_slug}_p{page_num}_fig{saved_idx}.png"
            out_path = out_dir / fname
            pix.save(str(out_path))
            pw_px = int(pix.width)
            ph_px = int(pix.height)

        results.append(
            ExtractedFigure(
                book_slug=book_slug,
                page=page_num,
                fig_idx=saved_idx,
                kind=kind,
                label=label,
                caption=caption,
                bbox=[round(float(x), 2) for x in (rect.x0, rect.y0, rect.x1, rect.y1)],
                pixel_width=pw_px,
                pixel_height=ph_px,
                dpi=dpi,
                output_path=str(out_path),
                source_mode=mode,
            )
        )

    return results


def _collect_ranges_from_taxonomy(
    taxonomy_path: Path,
    leaf_id: Optional[str],
    domain_id: Optional[str],
) -> list[tuple[str, int, int, str]]:
    """Collect (book_slug, page_start, page_end, leaf_id) tasks from taxonomy_tree.json."""
    with taxonomy_path.open("r", encoding="utf-8") as f:
        tree = json.load(f)

    jobs: list[tuple[str, int, int, str]] = []
    for dom in tree.get("domains", []):
        if domain_id and dom.get("id") != domain_id:
            continue
        for pt in dom.get("parent_topics", []):
            for st in pt.get("sub_topics", []):
                for lf in st.get("leaves", []):
                    lid = str(lf.get("id"))
                    if leaf_id and lid != leaf_id:
                        continue
                    for src in lf.get("sources", []):
                        jobs.append((src["book_slug"], int(src["page_start"]), int(src["page_end"]), lid))
    return jobs


def _build_arg_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    p = argparse.ArgumentParser(description="Extract 300 DPI textbook figures using PyMuPDF.")
    p.add_argument("--index", type=Path, default=Path("data/book_index.json"), help="Path to book_index.json")
    p.add_argument("--taxonomy", type=Path, default=Path("data/taxonomy_tree.json"), help="Path to taxonomy_tree.json")
    p.add_argument("--out-dir", type=Path, default=Path("assets/figures"), help="Output figure directory")
    p.add_argument("--dpi", type=int, default=300, help="Rendering resolution in DPI (default: 300)")
    p.add_argument("--book", type=str, default=None, help="Book slug to extract from")
    p.add_argument("--pages", type=str, default=None, help="1-based page list/ranges, e.g. '44-46,50'")
    p.add_argument("--leaf", type=str, default=None, help="Leaf ID from taxonomy_tree.json")
    p.add_argument("--domain", type=str, default=None, help="Domain ID from taxonomy_tree.json")
    p.add_argument("--dry-run", action="store_true", help="Detect bounding boxes without writing PNGs")
    p.add_argument("--manifest-out", type=Path, default=None, help="Optional path to save JSON extraction manifest")
    return p


def main(argv: Optional[list[str]] = None) -> int:
    """Entry point for CLI execution."""
    args = _build_arg_parser().parse_args(argv)
    books = _load_book_index(args.index)

    extracted: list[ExtractedFigure] = []
    seen_book_pages: set[tuple[str, int]] = set()
    open_docs: dict[str, pymupdf.Document] = {}

    try:
        page_tasks: list[tuple[str, int]] = []
        if args.book:
            if args.book not in books:
                raise KeyError(f"Unknown book_slug '{args.book}'.")
            pc = int(books[args.book]["page_count"])
            target_pages = _parse_page_spec(args.pages, pc) if args.pages else list(range(1, pc + 1))
            for pno in target_pages:
                page_tasks.append((args.book, pno))
        elif args.leaf or args.domain:
            jobs = _collect_ranges_from_taxonomy(args.taxonomy, args.leaf, args.domain)
            if not jobs:
                raise ValueError(f"No taxonomy mappings matched leaf={args.leaf!r}, domain={args.domain!r}.")
            for slug, p_start, p_end, _ in jobs:
                if slug not in books:
                    raise KeyError(f"Unknown book_slug in taxonomy: '{slug}'")
                pc = int(books[slug]["page_count"])
                for pno in range(max(1, p_start), min(pc, p_end) + 1):
                    if (slug, pno) not in seen_book_pages:
                        seen_book_pages.add((slug, pno))
                        page_tasks.append((slug, pno))
        else:
            raise ValueError("Specify either --book (+ optional --pages), --leaf, or --domain.")

        for slug, pno in page_tasks:
            if slug not in open_docs:
                open_docs[slug] = pymupdf.open(books[slug]["file_path"])
            doc = open_docs[slug]
            figs = extract_page_figures(
                doc[pno - 1],
                book_slug=slug,
                page_num=pno,
                out_dir=args.out_dir,
                dpi=args.dpi,
                dry_run=args.dry_run,
            )
            extracted.extend(figs)
    finally:
        for doc in open_docs.values():
            doc.close()

    payload = {
        "dpi": args.dpi,
        "dry_run": args.dry_run,
        "count": len(extracted),
        "figures": [asdict(f) for f in extracted],
    }

    if args.manifest_out:
        args.manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with args.manifest_out.open("w", encoding="utf-8") as mf:
            json.dump(payload, mf, indent=2, ensure_ascii=False)
        print(f"Wrote {len(extracted)} figures to manifest: {args.manifest_out}")
    else:
        print(json.dumps(payload, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
