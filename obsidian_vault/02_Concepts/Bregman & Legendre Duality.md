---
type: object
domain: "D01"
aliases: []
part_of: 
  - "[[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]"
  - "[[Cheatsheet - Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Convexity & Subgradients]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Sinkhorn Algorithm]]"
contrasts_with: []
dual_to: 
  - "[[Exponential Families]]"
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - object
  - domain-01
---

# Bregman & Legendre Duality

> [!abstract] Definition (Object)
> Fenchel conjugate f*(y) and Bregman divergence D_psi(x,y) unify mirror descent, exponential families, and Sinkhorn projections.

$$
f^*(\boldsymbol{y}) = \sup_{\boldsymbol{x}}\bigl(\langle\boldsymbol{y},\boldsymbol{x}\rangle - f(\boldsymbol{x})\bigr), \quad D_\psi(\boldsymbol{x},\boldsymbol{y}) = \psi(\boldsymbol{x})-\psi(\boldsymbol{y})-\langle\nabla\psi(\boldsymbol{y}),\boldsymbol{x}-\boldsymbol{y}\rangle
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Convexity & Subgradients]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Sinkhorn Algorithm]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | [[Exponential Families]] |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Visual Cheatsheet:** [[Cheatsheet - Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Referenced Across Monographs:** [[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
