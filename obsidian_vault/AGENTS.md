# Obsidian Knowledge Graph & Ontology Guidelines (`obsidian_vault/`)

`obsidian_vault/` is a **generated, strictly validated Obsidian Knowledge Graph** built by `scripts/build_obsidian_vault.py` from `scripts/concept_graph_spec.py`, `data/taxonomy_tree.json`, `data/book_index.json`, and `monographs/**/*.tex`.

**NEVER hand-edit generated files inside `obsidian_vault/`** — all changes must be made in `scripts/concept_graph_spec.py`, `scripts/build_obsidian_vault.py`, or the source `.tex` monographs, then rebuilt and audited. (`build_obsidian_vault.py` preserves this `AGENTS.md` file when regenerating subdirectories.)

## 1. Vault & Graph Topology
- `00_MOC/` — `00_Home_MOC.md`, `00_Concept_Graph_Index.md`, Domain MOCs (`D01..`), Parent-Topic MOCs (`PT ..`), Sub-Topic MOCs (`ST ..`).
- `01_Monographs/` — Full monograph notes converted from `.tex` with inline concept auto-linking and citation wikilinks.
- `02_Concepts/` — Atomic knowledge nodes (`type: concept | theorem | algorithm | object | architecture | metric`).
- `03_Sources/` — Textbook source notes linked from every monograph and concept that cites them.
- `04_Cheatsheets/` — Companion visual notes with `mermaid flowchart LR` mindmaps and key equations.
- `.obsidian/` — Pre-configured `graph.json` (5 color groups, `showArrow: true`), `app.json`, and `core-plugins.json`.

## 2. Typed Graph Semantics (Directed + Undirected)
When adding nodes to `scripts/concept_graph_spec.py`:
- **Safe Filenames:** Never use `/`, `:`, `\`, or `#` in `ConceptSpec.name` (use `-` instead of `/`, e.g. `User-Item`).
- **Directed Relations** (auto-inverted by `build_obsidian_vault.py`):
  - `prerequisites` ($\leftarrow$) $\iff$ `enables` ($\rightarrow$)
  - `generalizes` ($\supset$) $\iff$ `special_case_of` ($\subset$)
  - `part_of` ($\to$ Monograph / Cheatsheet / Domain)
- **Undirected / Symmetric Relations** (auto-symmetrized):
  - `related` ($\leftrightarrow$), `contrasts_with` ($\leftrightarrow$), `dual_to` ($\rightleftarrows$).
- Every node must connect to at least one leaf monograph and ≥ 1 other concept so the vault remains **1 single connected component with 0 orphans**.

## 3. Obsidian-Specific Markdown & MathJax Render Rules
Standard Markdown validity does NOT guarantee clean rendering in Obsidian Live Preview / Reading View. Always enforce:
1. **MathJax Macro & Delimiter Compatibility:**
   - Never emit `\bm`, `\vecb`, `\mat`, `\abs`, `\norm`, `\KL`, `\JS`, `\Wass`, `\argmax`, `\argmin`, `\coloneqq`, or `\eqqcolon`. Expand to `\boldsymbol`, `\lVert...\rVert`, `\lvert...\rvert`, `\parallel`, `\arg\max`, `\arg\min`, `:=`, `=:`.
   - Use standard `$$ ... $$` and single-line `$ ... $` (never leave `\[...\]` or `\(...\)`, and never match `\\[4pt]` line breaks as `\[`).
   - Never leave leading/trailing spaces inside inline math (`$ x$` or `$x $` breaks Obsidian MathJax) and never emit Markdown backticks (from `\texttt`) inside `$...$` (expand `\texttt` $\to$ `\mathtt` inside math).
   - Do not use `\Vval` or `\Qval` without word-boundary spacing (avoid `\gammaV`).
2. **Markdown Tables (Especially Inside Callouts):**
   - **Never run `---` $\to$ `—` dash replacement on table separator rows:** Obsidian strictly requires ASCII `| --- | --- |` (never Unicode `| — |`).
   - Obsidian **requires a blank line** (or bare `>` inside a `> [!callout]`) immediately before every table header `| ... |` AND immediately after the last table row.
   - Never leave unescaped `|` or `\|` inside `$...$` in table cells (convert `|` $\to$ `\mid` and `\|` $\to$ `\parallel`), strip `|alias` from `[[Note|alias]]` inside tables, and skip concept auto-linking on table rows so injected aliases never split columns.
   - Unwrap TeX layout groups (`{\small ...}`, `{\sloppy ...}`, `\resizebox{...}{...}{...}`) so no stray `{` or `}` lines appear around tables.
3. **Algorithms & Accents:**
   - Convert `algorithm2e` blocks via `convert_algorithm_block` (3-pass nested macro expansion + `\Return` handling).
   - Convert LaTeX accents (`\`e`, `\'e`, `\^o`, `\"o`, `\"u`, `\"a`, `\L{}`) to Unicode (`è`, `é`, `ô`, `ö`, `ü`, `ä`, `Ł`).
4. **Mermaid:** Use `flowchart LR` with sanitized node labels (never `::` prefix in `mindmap`).

## 4. Mandatory Rebuild & Audit Gate
```bash
.venv/bin/python scripts/build_obsidian_vault.py
.venv/bin/python scripts/audit_obsidian_vault.py
```
Must pass with `exit=0`, `Broken links: 0`, `Asymmetry errors: 0`, `Render errors: 0`, `Orphans: 0`, `Connected components: 1`.
