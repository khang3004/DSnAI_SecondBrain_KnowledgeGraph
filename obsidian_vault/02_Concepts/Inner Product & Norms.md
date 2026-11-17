---
type: object
domain: "D01"
aliases: 
  - "SPD Inner Product"
  - "Lp Norm"
part_of: 
  - "[[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Vector Space & Subspaces]]"
enables: 
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Mahalanobis Distance]]"
  - "[[Mercer Kernels & RKHS]]"
  - "[[Orthogonal Projection]]"
generalizes: []
special_case_of: []
related: 
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Mahalanobis Distance]]"
  - "[[Orthogonal Projection]]"
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

# Inner Product & Norms

> [!abstract] Definition (Object)
> Bilinear SPD form inducing geometry (angles, orthogonality) and topological norm equivalence on finite-dimensional spaces.

$$
\langle \boldsymbol{x}, \boldsymbol{y} \rangle_{\boldsymbol{M}} = \boldsymbol{x}^\top \boldsymbol{M} \boldsymbol{y}, \quad \|\boldsymbol{x}\|_\infty \le \|\boldsymbol{x}\|_2 \le \|\boldsymbol{x}\|_1 \le \sqrt{n}\|\boldsymbol{x}\|_2
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Vector Space & Subspaces]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Cauchy-Schwarz Inequality]], [[Eigendecomposition & Spectral Theorem]], [[Mahalanobis Distance]], [[Mercer Kernels & RKHS]], [[Orthogonal Projection]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Cauchy-Schwarz Inequality]], [[Mahalanobis Distance]], [[Orthogonal Projection]], [[Vector Space & Subspaces]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Visual Cheatsheet:** [[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Referenced Across Monographs:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
