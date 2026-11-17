---
type: architecture
domain: "D07"
aliases: 
  - "Two-Tower Model"
  - "Multi-Stage Funnel"
part_of: 
  - "[[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]"
  - "[[Cheatsheet - Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: 
  - "[[Factorization Machines & DeepFM]]"
  - "[[HNSW & IVF-PQ Indexing]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Factorization Machines & DeepFM]]"
  - "[[NDCG, MAP & Calibration]]"
  - "[[Sparse vs Dense Retrieval & RRF]]"
contrasts_with: 
  - "[[Collaborative Filtering (User-Item)]]"
dual_to: []
sources: 
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - architecture
  - domain-07
---

# Two-Tower Retrieval & Multi-Stage Ranking

> [!abstract] Definition (Architecture)
> Decoupled user/item encoders with log-Q corrected in-batch softmax enabling O(log N) ANN candidate retrieval before DeepFM scoring.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Factorization Machines & DeepFM]], [[HNSW & IVF-PQ Indexing]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Factorization Machines & DeepFM]], [[NDCG, MAP & Calibration]], [[Sparse vs Dense Retrieval & RRF]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Collaborative Filtering (User-Item)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Visual Cheatsheet:** [[Cheatsheet - Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Referenced Across Monographs:** [[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Primary Textbooks:** [[Source - Designing Machine Learning Systems]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]]
