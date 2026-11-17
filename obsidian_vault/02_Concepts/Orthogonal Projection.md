---
type: object
domain: "D01"
aliases: 
  - "Orthogonal Projector"
  - "Least Squares Projection"
part_of: 
  - "[[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Inner Product & Norms]]"
  - "[[Vector Space & Subspaces]]"
enables: 
  - "[[Linear & Ridge Regression]]"
  - "[[QR Factorization]]"
  - "[[Singular Value Decomposition (SVD)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Inner Product & Norms]]"
  - "[[Linear & Ridge Regression]]"
  - "[[QR Factorization]]"
  - "[[Vector Space & Subspaces]]"
contrasts_with: []
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

# Orthogonal Projection

> [!abstract] Definition (Object)
> Idempotent symmetric linear operator mapping vectors to their closest Euclidean approximation in a subspace.

$$
\boldsymbol{P}_W = \boldsymbol{Q}\boldsymbol{Q}^\top = \boldsymbol{A}(\boldsymbol{A}^\top\boldsymbol{A})^{-1}\boldsymbol{A}^\top, \quad \boldsymbol{P}_W^2 = \boldsymbol{P}_W = \boldsymbol{P}_W^\top
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Inner Product & Norms]], [[Vector Space & Subspaces]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Linear & Ridge Regression]], [[QR Factorization]], [[Singular Value Decomposition (SVD)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Inner Product & Norms]], [[Linear & Ridge Regression]], [[QR Factorization]], [[Vector Space & Subspaces]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Visual Cheatsheet:** [[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Referenced Across Monographs:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
