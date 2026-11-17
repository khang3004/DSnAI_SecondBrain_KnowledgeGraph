---
type: theorem
domain: "D01"
aliases: 
  - "Spectral Theorem"
  - "Eigendecomposition"
part_of: 
  - "[[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Inner Product & Norms]]"
  - "[[Vector Space & Subspaces]]"
enables: 
  - "[[Condition Number]]"
  - "[[Mercer Kernels & RKHS]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
  - "[[Principal Component Analysis (PCA)]]"
  - "[[Singular Value Decomposition (SVD)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Condition Number]]"
  - "[[Principal Component Analysis (PCA)]]"
contrasts_with: 
  - "[[Singular Value Decomposition (SVD)]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - E D I T I O N]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - theorem
  - domain-01
---

# Eigendecomposition & Spectral Theorem

> [!abstract] Definition (Theorem)
> Every real symmetric matrix admits an orthonormal eigenbasis with real eigenvalues; SPD matrices have strictly positive spectrum.

$$
\boldsymbol{S} = \boldsymbol{Q}\boldsymbol{\Lambda}\boldsymbol{Q}^\top, \quad \lambda_{\min}\|\boldsymbol{x}\|_2^2 \le \boldsymbol{x}^\top\boldsymbol{S}\boldsymbol{x} \le \lambda_{\max}\|\boldsymbol{x}\|_2^2
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Inner Product & Norms]], [[Vector Space & Subspaces]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Condition Number]], [[Mercer Kernels & RKHS]], [[Multivariate Gaussian & Schur Complement]], [[Principal Component Analysis (PCA)]], [[Singular Value Decomposition (SVD)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Condition Number]], [[Principal Component Analysis (PCA)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Singular Value Decomposition (SVD)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Visual Cheatsheet:** [[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Referenced Across Monographs:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
