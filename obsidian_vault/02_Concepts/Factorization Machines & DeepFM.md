---
type: architecture
domain: "D07"
aliases: 
  - "Factorization Machines"
  - "DeepFM"
  - "NeuMF"
part_of: 
  - "[[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]"
  - "[[Cheatsheet - Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: 
  - "[[MLP & Universal Approximation]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
enables: 
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
generalizes: []
special_case_of: []
related: 
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - architecture
  - domain-07
---

# Factorization Machines & DeepFM

> [!abstract] Definition (Architecture)
> Computes all degree-2 sparse feature interactions in linear O(kd) time via shared embeddings, augmented with a parallel deep MLP in DeepFM.

$$
\sum_{i<j}\langle \boldsymbol{v}_i,\boldsymbol{v}_j\rangle x_i x_j = \frac{1}{2}\sum_{f=1}^k\Bigl[\Bigl(\sum_i v_{i,f}x_i\Bigr)^2 - \sum_i v_{i,f}^2 x_i^2\Bigr]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[MLP & Universal Approximation]], [[Matrix Factorization (ALS & SGD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Two-Tower Retrieval & Multi-Stage Ranking]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Two-Tower Retrieval & Multi-Stage Ranking]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Visual Cheatsheet:** [[Cheatsheet - Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Referenced Across Monographs:** [[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Primary Textbooks:** [[Source - Designing Machine Learning Systems]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]]
