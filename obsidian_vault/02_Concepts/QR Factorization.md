---
type: algorithm
domain: "D01"
aliases: 
  - "Householder QR"
  - "Modified Gram-Schmidt"
part_of: 
  - "[[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Orthogonal Projection]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Condition Number]]"
  - "[[Orthogonal Projection]]"
contrasts_with: 
  - "[[Moore-Penrose Pseudoinverse]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - E D I T I O N]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-01
---

# QR Factorization

> [!abstract] Definition (Algorithm)
> Decomposes a matrix into an orthonormal basis Q and upper-triangular R via Householder reflections or MGS without squaring condition number.

$$
\boldsymbol{A} = \hat{\boldsymbol{Q}}\hat{\boldsymbol{R}}, \quad \kappa_2(\boldsymbol{A}) \text{ vs } \kappa_2(\boldsymbol{A}^\top\boldsymbol{A}) = \kappa_2(\boldsymbol{A})^2
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Orthogonal Projection]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Condition Number]], [[Orthogonal Projection]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Moore-Penrose Pseudoinverse]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Visual Cheatsheet:** [[Cheatsheet - Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Referenced Across Monographs:** [[Vector Spaces, Norms, Orthogonality & Least-Squares Projections]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - E D I T I O N]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
