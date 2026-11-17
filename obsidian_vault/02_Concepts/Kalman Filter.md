---
type: algorithm
domain: "D03"
aliases: []
part_of: 
  - "[[Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]"
  - "[[Cheatsheet - Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Multivariate Gaussian & Schur Complement]]"
enables: 
  - "[[State Space Models (S4 & Mamba)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[State Space Models (S4 & Mamba)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations of Large Language Models]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Report - Mamba When Transformer is Not All You Need - AIVN]]"
tags:
  - algorithm
  - domain-03
---

# Kalman Filter

> [!abstract] Definition (Algorithm)
> Exact closed-form Bayesian filtering and smoothing for linear-Gaussian state-space dynamics via predict-update recursion.

$$
\boldsymbol{K}_t = \boldsymbol{\Sigma}_{t|t-1}\boldsymbol{H}_t^\top(\boldsymbol{H}_t\boldsymbol{\Sigma}_{t|t-1}\boldsymbol{H}_t^\top + \boldsymbol{R}_t)^{-1}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Multivariate Gaussian & Schur Complement]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[State Space Models (S4 & Mamba)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Multivariate Gaussian & Schur Complement]], [[State Space Models (S4 & Mamba)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]
- **Visual Cheatsheet:** [[Cheatsheet - Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]
- **Referenced Across Monographs:** [[Classical State-Space Models, Structured SSMs & Selective Scan (Mamba)]]
- **Primary Textbooks:** [[Source - Foundations of Large Language Models]], [[Source - Probabilistic Machine Learning]], [[Source - Report - Mamba When Transformer is Not All You Need - AIVN]]
