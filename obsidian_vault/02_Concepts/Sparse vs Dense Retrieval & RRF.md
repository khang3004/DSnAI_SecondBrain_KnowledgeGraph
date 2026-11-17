---
type: algorithm
domain: "D07"
aliases: 
  - "RAG"
  - "BM25"
  - "ColBERT"
  - "Reciprocal Rank Fusion"
part_of: 
  - "[[Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]"
  - "[[Cheatsheet - Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: []
enables: 
  - "[[HNSW & IVF-PQ Indexing]]"
generalizes: []
special_case_of: []
related: 
  - "[[CLIP & Vision-Language Alignment]]"
  - "[[Context Memory & Compaction]]"
  - "[[HNSW & IVF-PQ Indexing]]"
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
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

# Sparse vs Dense Retrieval & RRF

> [!abstract] Definition (Algorithm)
> Combines lexical BM25 exact matching with Bi-Encoder semantic embeddings and ColBERT late interaction via Reciprocal Rank Fusion.

$$
\mathrm{RRF}(d;q) = \sum_{r\in\mathcal{R}} \frac{1}{\kappa + \mathrm{rank}_r(d)}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[HNSW & IVF-PQ Indexing]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[CLIP & Vision-Language Alignment]], [[Context Memory & Compaction]], [[HNSW & IVF-PQ Indexing]], [[Two-Tower Retrieval & Multi-Stage Ranking]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]
- **Visual Cheatsheet:** [[Cheatsheet - Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]
- **Referenced Across Monographs:** [[Retrieval-Augmented Generation, Vector Indexing, Context Engineering & Agent Memory]]
- **Primary Textbooks:** [[Source - Agents Book]], [[Source - Engineering -]], [[Source - LLM Engineer’s Handbook]], [[Source - Speech and Language Processing]]
