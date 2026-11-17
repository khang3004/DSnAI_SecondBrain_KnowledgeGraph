---
type: object
domain: "D01"
aliases: 
  - "Vector Space"
  - "Four Fundamental Subspaces"
part_of: 
  - "[[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: []
enables: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Inner Product & Norms]]"
  - "[[Matrix Calculus & Jacobians]]"
  - "[[Orthogonal Projection]]"
generalizes: []
special_case_of: []
related: 
  - "[[Inner Product & Norms]]"
  - "[[Orthogonal Projection]]"
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

# Vector Space & Subspaces

> [!abstract] Definition (Object)
> Linear space closed under addition and scalar multiplication; every matrix defines four fundamental subspaces.

$$
\mathbb{R}^n = \mathrm{Row}(\boldsymbol{A}) \oplus \mathrm{null}(\boldsymbol{A}), \quad \mathbb{R}^m = \mathrm{col}(\boldsymbol{A}) \oplus \mathrm{null}(\boldsymbol{A}^\top)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Eigendecomposition & Spectral Theorem]], [[Inner Product & Norms]], [[Matrix Calculus & Jacobians]], [[Orthogonal Projection]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Inner Product & Norms]], [[Orthogonal Projection]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Visual Cheatsheet:** [[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Referenced Across Monographs:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
