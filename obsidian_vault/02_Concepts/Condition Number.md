---
type: metric
domain: "D01"
aliases: []
part_of: 
  - "[[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Singular Value Decomposition (SVD)]]"
enables: 
  - "[[Gradient Descent]]"
  - "[[Momentum & Nesterov NAG]]"
generalizes: []
special_case_of: []
related: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Gradient Descent]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[QR Factorization]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - E D I T I O N]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - metric
  - domain-01
---

# Condition Number

> [!abstract] Definition (Metric)
> Ratio of largest to smallest singular value (or eigenvalue); governs numerical error amplification and GD convergence speed.

$$
\kappa_2(\boldsymbol{A}) = \frac{\sigma_{\max}(\boldsymbol{A})}{\sigma_{\min}(\boldsymbol{A})}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Eigendecomposition & Spectral Theorem]], [[Singular Value Decomposition (SVD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Gradient Descent]], [[Momentum & Nesterov NAG]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Eigendecomposition & Spectral Theorem]], [[Gradient Descent]], [[Momentum & Nesterov NAG]], [[QR Factorization]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Visual Cheatsheet:** [[Cheatsheet - Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Referenced Across Monographs:** [[Spectral Theory, Singular Value Decomposition & Low-Rank Approximations]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
