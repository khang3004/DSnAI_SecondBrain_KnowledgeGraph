#!/usr/bin/env python3
"""Comprehensive geometric PDF audit for all 47 Mindmap + Cheatsheet companions.

Detects 6 classes of visual defects in visual (rotated) coordinates:
  1. box_overflows     : Text spilling outside outer tcolorboxes (csboxenv, csnoteenv, cscard).
  2. box_overlaps      : Overlapping tcolorboxes or tcolorboxes overlapping csroot.
  3. text_overlaps     : Overlapping text spans (colliding labels/headers).
  4. margin_violations : Content extending too close to or outside page margins.
  5. mini_box_spills   : Text inside mini-TikZ boxes/circles overflowing its mini-box border.
  6. diagram_crossings : PGFPlots labels clipping axis borders or arrows/curves crossing through text.
"""

from __future__ import annotations

import json
import pathlib
import sys
from dataclasses import dataclass, asdict
import pymupdf


@dataclass
class PageAudit:
    page_num: int
    kind: str
    title: str
    box_overflows: list[str]
    box_overlaps: list[str]
    text_overlaps: list[str]
    margin_violations: list[str]
    mini_box_spills: list[str]
    diagram_crossings: list[str]

    @property
    def total_issues(self) -> int:
        return (
            len(self.box_overflows)
            + len(self.box_overlaps)
            + len(self.text_overlaps)
            + len(self.margin_violations)
            + len(self.mini_box_spills)
            + len(self.diagram_crossings)
        )


def rect_area(r: pymupdf.Rect) -> float:
    return max(0.0, r.width) * max(0.0, r.height)


def intersect_ratio(a: pymupdf.Rect, b: pymupdf.Rect) -> float:
    inter = a & b
    if inter.is_empty:
        return 0.0
    m = min(rect_area(a), rect_area(b))
    return rect_area(inter) / m if m > 0 else 0.0


def to_visual(r: pymupdf.Rect, page: pymupdf.Page) -> pymupdf.Rect:
    if page.rotation:
        return r * page.rotation_matrix
    return r


def pt_visual(pt: pymupdf.Point, page: pymupdf.Page) -> pymupdf.Point:
    if page.rotation:
        return pt * page.rotation_matrix
    return pt


def extract_boxes(page: pymupdf.Page) -> list[pymupdf.Rect]:
    """Extract outer tcolorbox frames from filled PDF vector drawings in visual coords."""
    pw, ph = page.rect.width, page.rect.height
    cx, cy = pw / 2.0, ph / 2.0
    raw: list[pymupdf.Rect] = []
    for d in page.get_drawings():
        if d.get("fill") is None:
            continue
        r = to_visual(pymupdf.Rect(d["rect"]), page)
        if r.width >= 140 and r.height >= 35 and r.y0 > 45:
            rcx, rcy = (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0
            if abs(rcx - cx) < 35 and abs(rcy - cy) < 35 and abs(r.width - r.height) < 15:
                continue
            raw.append(r)
    raw.sort(key=rect_area, reverse=True)
    boxes: list[pymupdf.Rect] = []
    for r in raw:
        if any(b.contains(r) or intersect_ratio(r, b) > 0.85 for b in boxes):
            continue
        boxes.append(r)
    return boxes


def extract_root_circle(page: pymupdf.Page) -> pymupdf.Rect | None:
    """Find the central csroot circle on a MINDMAP page in visual coords."""
    pw, ph = page.rect.width, page.rect.height
    cx, cy = pw / 2.0, ph / 2.0
    best: pymupdf.Rect | None = None
    for d in page.get_drawings():
        r = to_visual(pymupdf.Rect(d["rect"]), page)
        if 120 <= r.width <= 260 and 120 <= r.height <= 260 and abs(r.width - r.height) < 15:
            rcx, rcy = (r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0
            if abs(rcx - cx) < 60 and abs(rcy - cy) < 60:
                if best is None or rect_area(r) > rect_area(best):
                    best = r
    return best


def audit_page(page: pymupdf.Page, page_num: int) -> PageAudit:
    pw, ph = page.rect.width, page.rect.height
    boxes = extract_boxes(page)
    root_circle = extract_root_circle(page)

    text_dict = page.get_text("dict")
    spans: list[tuple[pymupdf.Rect, str]] = []
    lines: list[tuple[pymupdf.Rect, str]] = []
    kind = "UNKNOWN"
    title = ""

    for b in text_dict["blocks"]:
        if b["type"] != 0:
            continue
        for l in b["lines"]:
            lrect = to_visual(pymupdf.Rect(l["bbox"]), page)
            ltxt = "".join(s["text"] for s in l["spans"]).strip()
            if ltxt and lrect.y1 > 56 and lrect.y0 < 553:
                lines.append((lrect, ltxt))
            for s in l["spans"]:
                txt = s["text"].strip()
                if not txt:
                    continue
                r = to_visual(pymupdf.Rect(s["bbox"]), page)
                if txt in ("MINDMAP", "CHEATSHEET") and r.y0 < 60:
                    kind = txt
                elif r.y0 < 40 and r.x0 > 90 and not title:
                    title = txt
                spans.append((r, txt))

    box_overflows: list[str] = []
    margin_violations: list[str] = []
    box_overlaps: list[str] = []
    text_overlaps: list[str] = []
    mini_box_spills: list[str] = []
    diagram_crossings: list[str] = []

    # 1. Check outer box overlaps and margin violations
    for i, b1 in enumerate(boxes):
        if b1.x0 < 8 or b1.x1 > pw - 8 or b1.y0 < 48 or b1.y1 > ph - 8:
            margin_violations.append(
                f"Box ({b1.x0:.1f},{b1.y0:.1f},{b1.x1:.1f},{b1.y1:.1f}) near/outside margin"
            )
        for j in range(i + 1, len(boxes)):
            b2 = boxes[j]
            inter = b1 & b2
            if not inter.is_empty and inter.width > 2 and inter.height > 2:
                box_overlaps.append(
                    f"Box overlap {inter.width:.1f}x{inter.height:.1f}pt at ({inter.x0:.1f},{inter.y0:.1f})"
                )
        if root_circle is not None:
            rcx = (root_circle.x0 + root_circle.x1) / 2.0
            rcy = (root_circle.y0 + root_circle.y1) / 2.0
            rad = root_circle.width / 2.0
            nx = max(b1.x0, min(rcx, b1.x1))
            ny = max(b1.y0, min(rcy, b1.y1))
            dist = ((nx - rcx) ** 2 + (ny - rcy) ** 2) ** 0.5
            if dist < rad - 2:
                box_overlaps.append(
                    f"Box ({b1.x0:.1f},{b1.y0:.1f}) overlaps root circle by {rad - dist:.1f}pt"
                )

    # 2. Check text spans against outer boxes
    for r, txt in spans:
        if r.y1 < 56:
            continue
        if kind == "MINDMAP" and 553 < r.y0 < 572:
            continue
        if root_circle is not None and root_circle.contains(r):
            continue
        containing = [
            b for b in boxes
            if r.x0 >= b.x0 - 2 and r.x1 <= b.x1 + 2 and r.y0 >= b.y0 - 2 and r.y1 <= b.y1 + 2
        ]
        if not containing:
            partially = [b for b in boxes if not (r & b).is_empty]
            if partially:
                b = partially[0]
                over_x = max(0.0, r.x1 - b.x1, b.x0 - r.x0)
                over_y = max(0.0, r.y1 - b.y1, b.y0 - r.y0)
                if over_x > 2.5 or over_y > 2.5:
                    box_overflows.append(
                        f"'{txt[:32]}' spills box by dx={over_x:.1f}, dy={over_y:.1f}pt"
                    )
            else:
                box_overflows.append(
                    f"'{txt[:32]}' outside all boxes at ({r.x0:.1f},{r.y0:.1f})"
                )

    # 3. Check overlapping text spans
    for i in range(len(spans)):
        r1, t1 = spans[i]
        if len(t1) <= 1 or r1.y1 < 56:
            continue
        for j in range(i + 1, len(spans)):
            r2, t2 = spans[j]
            if len(t2) <= 1 or r2.y1 < 56:
                continue
            inter = r1 & r2
            if inter.is_empty:
                continue
            ratio = intersect_ratio(r1, r2)
            if ratio > 0.35 and inter.width > 4 and inter.height > 4:
                text_overlaps.append(
                    f"'{t1[:22]}' overlaps '{t2[:22]}' ({ratio*100:.0f}%)"
                )

    # 4. Check mini-TikZ box/circle text spills
    drawings = page.get_drawings()
    for d in drawings:
        r = to_visual(pymupdf.Rect(d["rect"]), page)
        if 10 <= r.width <= 135 and 6 <= r.height <= 85 and r.y0 > 56:
            items = d.get("items", [])
            if d.get("fill") is not None or d.get("closePath") or any(it[0] == "re" for it in items):
                for lr, lt in lines:
                    lcx, lcy = (lr.x0 + lr.x1) / 2.0, (lr.y0 + lr.y1) / 2.0
                    if r.x0 <= lcx <= r.x1 and r.y0 <= lcy <= r.y1:
                        ox = max(0.0, lr.x1 - r.x1, r.x0 - lr.x0)
                        oy = max(0.0, lr.y1 - r.y1, r.y0 - lr.y0)
                        if ox > 1.8 or oy > 1.8:
                            msg = f"'{lt[:28]}' overflows mini-box ({r.width:.0f}x{r.height:.0f}) by dx={ox:.1f}, dy={oy:.1f}pt"
                            if msg not in mini_box_spills:
                                mini_box_spills.append(msg)

    # 5. Check PGFPlots axis border clipping & curve/arrow crossings through labels
    short_labels: list[tuple[pymupdf.Rect, pymupdf.Rect, str]] = []
    for lr, lt in lines:
        if len(lt) >= 3 and lr.width < 120:
            inner = pymupdf.Rect(lr.x0 + 2.5, lr.y0 + 2.2, lr.x1 - 2.5, lr.y1 - 2.2)
            if inner.width > 4 and inner.height > 2.5:
                short_labels.append((inner, lr, lt))

    for d in drawings:
        r = to_visual(pymupdf.Rect(d["rect"]), page)
        if kind == "CHEATSHEET" and 150 <= r.width <= 250 and 75 <= r.height <= 155 and d.get("fill") is None and (d.get("width") or 0) >= 0.3:
            for _, lr, lt in short_labels:
                if lr.x1 > r.x0 + 10 and lr.x0 < r.x1 - 10:
                    if lr.y0 < r.y0 - 0.5 and lr.y1 > r.y0 + 1.0:
                        msg = f"'{lt[:28]}' straddles top axis border"
                        if msg not in diagram_crossings:
                            diagram_crossings.append(msg)
                    if lr.y0 < r.y1 - 1.0 and lr.y1 > r.y1 + 0.5:
                        msg = f"'{lt[:28]}' straddles bottom axis border"
                        if msg not in diagram_crossings:
                            diagram_crossings.append(msg)

        w = d.get("width") or 0
        if w > 2.5:
            continue
        for it in d.get("items", []):
            if it[0] == "l":
                p1 = pt_visual(it[1], page)
                p2 = pt_visual(it[2], page)
                dx, dy = abs(p2.x - p1.x), abs(p2.y - p1.y)
                if dx > 1.5 and dy > 1.5:
                    seg_rect = pymupdf.Rect(
                        min(p1.x, p2.x), min(p1.y, p2.y), max(p1.x, p2.x), max(p1.y, p2.y)
                    )
                    for inner, lr, lt in short_labels:
                        if not (seg_rect & inner).is_empty:
                            mx, my = (p1.x + p2.x) / 2.0, (p1.y + p2.y) / 2.0
                            if (
                                inner.contains(p1)
                                or inner.contains(p2)
                                or inner.contains(pymupdf.Point(mx, my))
                            ):
                                msg = f"'{lt[:28]}' crossed by curve/arrow"
                                if msg not in diagram_crossings:
                                    diagram_crossings.append(msg)

    return PageAudit(
        page_num=page_num,
        kind=kind,
        title=title,
        box_overflows=box_overflows[:10],
        box_overlaps=box_overlaps[:10],
        text_overlaps=text_overlaps[:10],
        margin_violations=margin_violations[:10],
        mini_box_spills=mini_box_spills[:10],
        diagram_crossings=diagram_crossings[:10],
    )


def main() -> None:
    pdf_path = sys.argv[2] if len(sys.argv) >= 3 and sys.argv[1] == "--single" else "atlas.pdf"
    doc = pymupdf.open(pdf_path)
    audits = [audit_page(doc[i], i + 1) for i in range(len(doc))]
    bad = [a for a in audits if a.total_issues > 0]
    print(f"Audited {len(audits)} pages in {pdf_path}: {len(bad)} pages have geometric issues.")
    for a in bad:
        print(
            f"  P{a.page_num:02d} [{a.kind}] {a.title[:40]} | "
            f"spill={len(a.box_overflows)} box_ov={len(a.box_overlaps)} "
            f"txt_ov={len(a.text_overlaps)} mini={len(a.mini_box_spills)} "
            f"cross={len(a.diagram_crossings)}"
        )
        for msg in a.box_overflows[:3]:
            print(f"     - SPILL: {msg}")
        for msg in a.text_overlaps[:3]:
            print(f"     - TXT_OV: {msg}")
        for msg in a.mini_box_spills[:4]:
            print(f"     - MINI_SPILL: {msg}")
        for msg in a.diagram_crossings[:4]:
            print(f"     - CROSS/CLIP: {msg}")

    if pdf_path == "atlas.pdf":
        out_path = pathlib.Path("data/audit_cheatsheets.json")
        out_path.write_text(json.dumps([asdict(a) for a in audits], indent=2))
        print(f"Wrote detailed report to {out_path}")
    elif bad:
        sys.exit(1)


if __name__ == "__main__":
    main()
