---
type: metric
domain: "D01"
aliases: []
part_of: 
  - "[[Probability Spaces, Multivariate Gaussians & Exponential Families]]"
  - "[[Cheatsheet - Probability Spaces, Multivariate Gaussians & Exponential Families]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Inner Product & Norms]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Inner Product & Norms]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - metric
  - domain-01
---

# Mahalanobis Distance

> [!abstract] Definition (Metric)
> Precision-weighted quadratic distance invariant under invertible affine transforms; equals Euclidean distance after whitening.

$$
\Delta^2(\boldsymbol{x}) = (\boldsymbol{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\boldsymbol{x}-\boldsymbol{\mu})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Inner Product & Norms]], [[Multivariate Gaussian & Schur Complement]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Inner Product & Norms]], [[Multivariate Gaussian & Schur Complement]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Visual Cheatsheet:** [[Cheatsheet - Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Referenced Across Monographs:** [[Probability Spaces, Multivariate Gaussians & Exponential Families]]
- **Primary Textbooks:** [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
