---
name: book-ingestion-and-indexing
description: >-
  Ingests new PDF or online HTML/Quarto textbooks into books_monographs/,
  converts web books to bookmarked PDFs via headless Chrome + PyMuPDF, runs
  deduplication and inverted indexing (pdf_indexer.py), and updates
  data/taxonomy_tree.json. Use when adding new source books or web textbooks.
---

# Book Ingestion & Library Indexing

## When to Use
- Adding a new PDF textbook to `books_monographs/`.
- Converting an online HTML/Quarto textbook (e.g., `visionbook.mit.edu`) into a local bookmarked PDF.
- Updating `data/book_index.json` and `data/taxonomy_tree.json` to map new chapters/pages.

## Step-by-Step Procedure

### 1. Acquire / Convert Source PDF
- **Direct PDF:** Download into the appropriate category folder under `books_monographs/` (`Maths & Statistics/`, `Machine Learning & Deep Learning Books/`, `Reinforement_Learning/`, or `LLMs&AgenticAI_Books/`).
- **Online Quarto / HTML Book:**
  1. Inspect the book's sidebar `<nav>` HTML to confirm the `<a href=... class="sidebar-item-text">` order.
  2. Adapt and run `scripts/webbook_to_pdf.py` using `/Applications/Google Chrome.app/Contents/MacOS/Google Chrome --headless=new --no-pdf-header-footer --virtual-time-budget=6000` with `max_workers=4`, `timeout=120`, and a persistent chapter cache dir so heavy MathJax/image chapters never lose progress.
  3. Merge all rendered chapter PDFs with `pymupdf` and write hierarchical PDF bookmarks (`doc.set_toc(toc)`).

### 2. Re-Index the Library
Run the indexer (never modify or delete existing files in `books_monographs/`):
```bash
.venv/bin/python scripts/pdf_indexer.py
```
Verify:
- Output `data/book_index.json` reports `0` errors and records the new `book_slug` and `page_count`.
- Query `data/book_index.json` (`inverted_index` + TOC) to identify exact chapters and page ranges before synthesizing monographs.

### 3. Update Hierarchical Taxonomy
Add new Parent Topics, Sub-Topics, or Leaf nodes to `data/taxonomy_tree.json` with:
- `id`, `title`, `tex_path`, `description`, `key_visuals`, and exact `sources` (`book_slug`, `chapters`, `pages`).
- Create the corresponding directory tree under `monographs/`.
