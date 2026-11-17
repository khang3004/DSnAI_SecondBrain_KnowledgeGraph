---
type: metric
domain: "D07"
aliases: 
  - "NDCG@K"
  - "Position Bias & IPS"
part_of: 
  - "[[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]"
  - "[[Cheatsheet - Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: []
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[Two-Tower Retrieval & Multi-Stage Ranking]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - metric
  - domain-07
---

# NDCG, MAP & Calibration

> [!abstract] Definition (Metric)
> Position-discounted ranking quality (DCG/IDCG) and IPS propensity debiasing for evaluation under presentation bias.

$$
\mathrm{DCG}@K = \sum_{r=1}^K \frac{2^{\mathrm{rel}_r}-1}{\log_2(r+1)}, \qquad \mathrm{NDCG}@K = \frac{\mathrm{DCG}@K}{\mathrm{IDCG}@K}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Bayesian Personalized Ranking (BPR)]], [[Two-Tower Retrieval & Multi-Stage Ranking]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Visual Cheatsheet:** [[Cheatsheet - Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Referenced Across Monographs:** [[Factorization Machines, DeepFM, NeuMF, Two-Tower Retrieval & Multi-Stage Ranking]]
- **Primary Textbooks:** [[Source - Designing Machine Learning Systems]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]]
