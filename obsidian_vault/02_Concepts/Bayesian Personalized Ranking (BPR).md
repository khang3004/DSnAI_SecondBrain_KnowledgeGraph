---
type: algorithm
domain: "D07"
aliases: 
  - "BPR"
part_of: 
  - "[[Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]"
  - "[[Cheatsheet - Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: 
  - "[[Bradley-Terry Reward Model]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Matrix Factorization (ALS & SGD)]]"
  - "[[NDCG, MAP & Calibration]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-07
---

# Bayesian Personalized Ranking (BPR)

> [!abstract] Definition (Algorithm)
> Pairwise implicit-feedback ranking loss maximizing log-sigmoid preference margin between observed positives and sampled negatives.

$$
\mathcal{L}_{\mathrm{BPR}} = -\sum_{(u,i,j)} \log\sigma(\hat{y}_{ui} - \hat{y}_{uj}) + \lambda\|\Theta\|_2^2
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Bradley-Terry Reward Model]], [[Matrix Factorization (ALS & SGD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Matrix Factorization (ALS & SGD)]], [[NDCG, MAP & Calibration]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]
- **Visual Cheatsheet:** [[Cheatsheet - Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]
- **Referenced Across Monographs:** [[Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]
- **Primary Textbooks:** [[Source - Dive into Deep Learning]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]]
