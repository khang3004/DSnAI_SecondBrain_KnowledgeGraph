---
type: algorithm
domain: "D07"
aliases: 
  - "HNSW"
  - "IVF-PQ"
part_of: 
  - "[[Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]"
  - "[[Cheatsheet - Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: 
  - "[[Sparse vs Dense Retrieval & RRF]]"
enables: 
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
generalizes: []
special_case_of: []
related: 
  - "[[Sparse vs Dense Retrieval & RRF]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Agents Book]]"
  - "[[Source - Engineering -]]"
  - "[[Source - LLM Engineer’s Handbook]]"
  - "[[Source - Speech and Language Processing]]"
tags:
  - algorithm
  - domain-07
---

# HNSW & IVF-PQ Indexing

> [!abstract] Definition (Algorithm)
> Approximate nearest-neighbor vector search trading RAM vs. recall: hierarchical small-world graphs (HNSW) vs. product-quantized Voronoi cells.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Sparse vs Dense Retrieval & RRF]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Two-Tower Retrieval & Multi-Stage Ranking]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Sparse vs Dense Retrieval & RRF]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]
- **Visual Cheatsheet:** [[Cheatsheet - Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]
- **Referenced Across Monographs:** [[Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]
- **Primary Textbooks:** [[Source - Agents Book]], [[Source - Engineering -]], [[Source - LLM Engineer’s Handbook]], [[Source - Speech and Language Processing]]
