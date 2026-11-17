---
type: algorithm
domain: "D07"
aliases: 
  - "Biased SVD"
  - "Weighted ALS"
part_of: 
  - "[[Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]"
  - "[[Cheatsheet - Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]"
  - "[[D07 - Agentic AI Architectures, Protocols & Production ML Systems]]"
prerequisites: 
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Singular Value Decomposition (SVD)]]"
enables: 
  - "[[AutoRec]]"
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[Factorization Machines & DeepFM]]"
generalizes: []
special_case_of: []
related: 
  - "[[AutoRec]]"
  - "[[Bayesian Personalized Ranking (BPR)]]"
  - "[[Collaborative Filtering (User-Item)]]"
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

# Matrix Factorization (ALS & SGD)

> [!abstract] Definition (Algorithm)
> Low-rank biased completion Y approx mu + b_u + b_i + P Q^T solved via SGD or closed-form Alternating Least Squares (including implicit WALS).

$$
\hat{y}_{ui} = \mu + b_u + b_i + \boldsymbol{p}_u^\top \boldsymbol{q}_i
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Low-Rank Approximation (Eckart-Young)]], [[Singular Value Decomposition (SVD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[AutoRec]], [[Bayesian Personalized Ranking (BPR)]], [[Factorization Machines & DeepFM]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[AutoRec]], [[Bayesian Personalized Ranking (BPR)]], [[Collaborative Filtering (User-Item)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]
- **Visual Cheatsheet:** [[Cheatsheet - Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]
- **Referenced Across Monographs:** [[Collaborative Filtering, Matrix Factorization (ALS - SGD), AutoRec & Bayesian Personalized Ranking (BPR)]]
- **Primary Textbooks:** [[Source - Dive into Deep Learning]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]]
