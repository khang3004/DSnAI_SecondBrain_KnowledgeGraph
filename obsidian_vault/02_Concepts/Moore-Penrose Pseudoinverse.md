---
type: object
domain: "D01"
aliases: 
  - "Pseudoinverse"
part_of: 
  - "[[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Singular Value Decomposition (SVD)]]"
enables: 
  - "[[Double Descent]]"
generalizes: []
special_case_of: []
related: 
  - "[[Double Descent]]"
  - "[[Linear & Ridge Regression]]"
  - "[[Singular Value Decomposition (SVD)]]"
contrasts_with: 
  - "[[QR Factorization]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - E D I T I O N]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - object
  - domain-01
---

# Moore-Penrose Pseudoinverse

> [!abstract] Definition (Object)
> Minimum-norm least-squares inverse defined for arbitrary rectangular or rank-deficient matrices via SVD.

$$
\boldsymbol{A}^+ = \boldsymbol{V}\boldsymbol{\Sigma}^+\boldsymbol{U}^\top, \quad \hat{\boldsymbol{x}}_{\mathrm{MNLS}} = \boldsymbol{A}^+\boldsymbol{b}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Singular Value Decomposition (SVD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Double Descent]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Double Descent]], [[Linear & Ridge Regression]], [[Singular Value Decomposition (SVD)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[QR Factorization]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Visual Cheatsheet:** [[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Referenced Across Monographs:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
