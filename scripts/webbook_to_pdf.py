"""Convert an online Quarto/HTML book (visionbook.mit.edu) into a bookmarked PDF.

Uses headless Google Chrome to render each chapter with full MathJax and figures,
then merges all chapter PDFs in TOC order with PyMuPDF bookmarks.
"""

from __future__ import annotations

import concurrent.futures
import html
import re
import shutil
import subprocess
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import pymupdf

CHROME_BIN = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
BASE_URL = "https://visionbook.mit.edu/"
OUT_PDF = (
    Path(__file__).resolve().parent.parent
    / "books_monographs"
    / "Machine Learning & Deep Learning Books"
    / "Foundations_of_Computer_Vision_MIT.pdf"
)


def fetch_ordered_chapters(base_url: str = BASE_URL) -> list[tuple[int, str, str]]:
    """Parse sidebar TOC links from the book index page in exact order.

    Returns list of (level, title, url).
    """
    raw = urllib.request.urlopen(base_url, timeout=20).read().decode("utf-8", "ignore")
    # Extract sidebar items with class sidebar-item-text
    pattern = re.compile(
        r'<a[^>]*href="([^"]+\.html)"[^>]*class="[^"]*sidebar-item-text[^"]*"[^>]*>(.*?)</a>',
        re.DOTALL,
    )
    chapters: list[tuple[int, str, str]] = []
    seen_urls: set[str] = set()
    for href, inner in pattern.findall(raw):
        url = urllib.parse.urljoin(base_url, href)
        if url in seen_urls or not url.startswith(base_url):
            continue
        seen_urls.add(url)
        title = html.unescape(re.sub(r"<[^>]+>", " ", inner))
        title = re.sub(r"\s+", " ", title).strip()
        if not title or "copyright" in href.lower():
            continue
        level = 1 if href.lstrip("./").startswith("part_") or href.lstrip("./") == "index.html" else 2
        chapters.append((level, title, url))
    return chapters


def render_chapter_pdf(args: tuple[int, str, Path]) -> tuple[int, Path]:
    """Render a single HTML chapter URL to PDF via headless Chrome with retry."""
    idx, url, out_dir = args
    pdf_path = out_dir / f"ch_{idx:03d}.pdf"
    if pdf_path.exists() and pdf_path.stat().st_size > 1000:
        return idx, pdf_path
    cmd = [
        CHROME_BIN,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--no-pdf-header-footer",
        "--virtual-time-budget=6000",
        f"--print-to-pdf={pdf_path}",
        url,
    ]
    for attempt in range(2):
        try:
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=120)
            return idx, pdf_path
        except subprocess.TimeoutExpired:
            if attempt == 1:
                raise
    return idx, pdf_path


def build_book_pdf() -> None:
    """Crawl visionbook.mit.edu, render all chapters, and merge into a bookmarked PDF."""
    chapters = fetch_ordered_chapters()
    print(f"Discovered {len(chapters)} chapters from {BASE_URL}")
    tmp_dir = Path(tempfile.gettempdir()) / "visionbook_pdf_cache"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    try:
        tasks = [(i, url, tmp_dir) for i, (_, _, url) in enumerate(chapters)]
        rendered: dict[int, Path] = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            for idx, pdf_path in pool.map(render_chapter_pdf, tasks):
                rendered[idx] = pdf_path
                if (idx + 1) % 15 == 0 or idx + 1 == len(tasks):
                    print(f"  Rendered {idx + 1}/{len(tasks)} chapters...")

        merged = pymupdf.open()
        toc: list[list[object]] = []
        for i, (level, title, _url) in enumerate(chapters):
            start_page = merged.page_count + 1
            ch_doc = pymupdf.open(rendered[i])
            merged.insert_pdf(ch_doc)
            ch_doc.close()
            toc.append([level, title, start_page])

        merged.set_toc(toc)
        OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
        merged.save(str(OUT_PDF), garbage=4, deflate=True)
        page_count = merged.page_count
        merged.close()
        size_mb = OUT_PDF.stat().st_size / 1e6
        print(f"Saved {OUT_PDF} ({page_count} pages, {size_mb:.2f} MB)")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    build_book_pdf()
