---
name: obsidian-knowledge-graph-builder
description: >-
  Extends the typed Knowledge Graph ontology (scripts/concept_graph_spec.py),
  rebuilds obsidian_vault/ from monographs, cheatsheets, and book_index.json,
  and validates zero broken links, zero asymmetry errors, and zero Obsidian
  Markdown/MathJax render defects via audit_obsidian_vault.py.
---

# Obsidian Knowledge Graph Builder & Render QA

## Prerequisites
Read `obsidian_vault/AGENTS.md` before modifying any ontology or vault builder code.

## Step-by-Step Procedure

### 1. Extend the Typed Ontology (`scripts/concept_graph_spec.py`)
Add `ConceptSpec` entries for new concepts, theorems, algorithms, objects, architectures, or metrics:
- Set `leaf_id` to an exact `id` from `data/taxonomy_tree.json`.
- Never use `/`, `:`, `\`, or `#` in `name` (Obsidian filename/wikilink safety).
- Wire **directed** edges (`prerequisites`, `generalizes`) and **undirected** edges (`related`, `contrasts_with`, `dual_to`) to connect with existing Domain 01–07 nodes.
- Expand all custom LaTeX macros in `formula` (`\boldsymbol`, `\arg\max`, `\lVert...\rVert`, `\parallel`, etc.).

### 2. Verify Converter Rules (`scripts/build_obsidian_vault.py`)
`build_obsidian_vault.py` enforces Obsidian Live Preview / Reading View compatibility:
- Balanced-brace macro expansion (`read_balanced`) with `(?![A-Za-z])` word boundaries.
- `\[...\]` / `\(...\)` $\to$ `$$...$$` / `$...$` (excluding `\\[4pt]` line breaks).
- `algorithm2e` $\to$ `[!example] Algorithm` callouts (3-pass nested loop/if + `\Return`).
- `tabular`/`tabularx` $\to$ Markdown tables with ASCII `| --- |` separators (protected from `---` $\to$ `—` dash normalization), mandatory blank lines (`>` in callouts) before/after every table, `|`/`\|` inside math converted to `\mid`/`\parallel`, `|alias` stripped, table rows excluded from concept auto-linking, and `{\small ...}`/`\resizebox` unwrapped.
- Inline `$...$` trimmed (no leading/trailing spaces) and `\texttt` expanded to `\mathtt` inside math rather than backticks.
- LaTeX accents converted to Unicode (`è`, `é`, `ô`, `ö`, `ü`, `ä`, `Ł`).
- Mermaid diagrams rendered via sanitized `flowchart LR`.

### 3. Rebuild & Audit
```bash
.venv/bin/python scripts/build_obsidian_vault.py
.venv/bin/python scripts/audit_obsidian_vault.py
```
Must exit `0` with:
- `Connected components: 1`
- `Orphans: 0`
- `Broken links: 0`
- `Asymmetry errors: 0`
- `Render errors: 0`
