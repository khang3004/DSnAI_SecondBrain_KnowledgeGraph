---
type: algorithm
domain: "D01"
aliases: 
  - "MLE"
  - "Fisher Information Matrix"
part_of: 
  - "[[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
  - "[[Cheatsheet - Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[KL Divergence]]"
  - "[[Law of Large Numbers & CLT]]"
enables: 
  - "[[Logistic Regression]]"
  - "[[Natural Gradient & K-FAC]]"
generalizes: []
special_case_of: []
related: 
  - "[[KL Divergence]]"
  - "[[Law of Large Numbers & CLT]]"
  - "[[Logistic Regression]]"
  - "[[Natural Gradient & K-FAC]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-01
---

# Maximum Likelihood & Fisher Info

> [!abstract] Definition (Algorithm)
> Minimizes forward KL to empirical data; asymptotically efficient achieving the Cramer-Rao lower bound governed by Fisher information.

$$
\boldsymbol{F}(\boldsymbol{\theta}) = \mathbb{E}[\nabla\log p\,\nabla\log p^\top], \quad \mathrm{Cov}(\hat{\boldsymbol{\theta}}_{\mathrm{MLE}}) \succeq (N\boldsymbol{F})^{-1}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[KL Divergence]], [[Law of Large Numbers & CLT]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Logistic Regression]], [[Natural Gradient & K-FAC]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[KL Divergence]], [[Law of Large Numbers & CLT]], [[Logistic Regression]], [[Natural Gradient & K-FAC]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Visual Cheatsheet:** [[Cheatsheet - Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Referenced Across Monographs:** [[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Primary Textbooks:** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]]
