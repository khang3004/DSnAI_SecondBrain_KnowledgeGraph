---
type: theorem
domain: "D01"
aliases: 
  - "Eckart-Young-Mirsky Theorem"
part_of: 
  - "[[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Singular Value Decomposition (SVD)]]"
enables: 
  - "[[LoRA & QLoRA]]"
  - "[[Matrix Factorization (ALS & SGD)]]"
generalizes: []
special_case_of: []
related: 
  - "[[LoRA & QLoRA]]"
  - "[[Principal Component Analysis (PCA)]]"
  - "[[Singular Value Decomposition (SVD)]]"
contrasts_with: []
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

# Low-Rank Approximation (Eckart-Young)

> [!abstract] Definition (Theorem)
> Truncated SVD gives the exact best rank-k approximation in both spectral and Frobenius norms.

$$
\min_{\mathrm{rank}(\boldsymbol{B})\le k}\|\boldsymbol{A}-\boldsymbol{B}\|_{2,F} = \|\boldsymbol{A}-\boldsymbol{A}_k\|_{2,F}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Singular Value Decomposition (SVD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[LoRA & QLoRA]], [[Matrix Factorization (ALS & SGD)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[LoRA & QLoRA]], [[Principal Component Analysis (PCA)]], [[Singular Value Decomposition (SVD)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Visual Cheatsheet:** [[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Referenced Across Monographs:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
