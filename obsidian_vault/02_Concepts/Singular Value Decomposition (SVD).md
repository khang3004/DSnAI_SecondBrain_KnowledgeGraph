---
type: theorem
domain: "D01"
aliases: 
  - "SVD"
part_of: 
  - "[[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Orthogonal Projection]]"
enables: 
  - "[[Camera Calibration & Homography]]"
  - "[[Condition Number]]"
  - "[[Linear & Ridge Regression]]"
  - "[[LoRA & QLoRA]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Principal Component Analysis (PCA)]]"
generalizes: []
special_case_of: []
related: 
  - "[[LoRA & QLoRA]]"
  - "[[Low-Rank Approximation (Eckart-Young)]]"
  - "[[Moore-Penrose Pseudoinverse]]"
  - "[[Principal Component Analysis (PCA)]]"
contrasts_with: 
  - "[[Eigendecomposition & Spectral Theorem]]"
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

# Singular Value Decomposition (SVD)

> [!abstract] Definition (Theorem)
> Universal orthogonal factorization of any rectangular matrix into left/right singular bases and nonnegative singular values.

$$
\boldsymbol{A} = \boldsymbol{U}\boldsymbol{\Sigma}\boldsymbol{V}^\top = \sum_{i=1}^r \sigma_i \boldsymbol{u}_i \boldsymbol{v}_i^\top
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Eigendecomposition & Spectral Theorem]], [[Orthogonal Projection]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Camera Calibration & Homography]], [[Condition Number]], [[Linear & Ridge Regression]], [[LoRA & QLoRA]], [[Low-Rank Approximation (Eckart-Young)]], [[Matrix Factorization (ALS & SGD)]], [[Moore-Penrose Pseudoinverse]], [[Principal Component Analysis (PCA)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[LoRA & QLoRA]], [[Low-Rank Approximation (Eckart-Young)]], [[Moore-Penrose Pseudoinverse]], [[Principal Component Analysis (PCA)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Eigendecomposition & Spectral Theorem]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Visual Cheatsheet:** [[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Referenced Across Monographs:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
