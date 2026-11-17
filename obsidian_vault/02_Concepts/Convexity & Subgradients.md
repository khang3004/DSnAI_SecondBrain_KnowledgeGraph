---
type: object
domain: "D01"
aliases: []
part_of: 
  - "[[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]"
  - "[[Cheatsheet - Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: []
enables: 
  - "[[Bregman & Legendre Duality]]"
  - "[[Gradient Descent]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Proximal & Subgradient Methods]]"
generalizes: []
special_case_of: []
related: 
  - "[[Gradient Descent]]"
  - "[[Lagrange Duality & KKT]]"
  - "[[Proximal & Subgradient Methods]]"
contrasts_with: []
dual_to: []
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

# Convexity & Subgradients

> [!abstract] Definition (Object)
> Jensen's inequality, first/second-order conditions, L-smoothness, and strong convexity defining global optimality guarantees.

$$
f(\boldsymbol{y}) \ge f(\boldsymbol{x}) + \langle \boldsymbol{g}, \boldsymbol{y}-\boldsymbol{x}\rangle \quad \forall \boldsymbol{g}\in\partial f(\boldsymbol{x})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Bregman & Legendre Duality]], [[Gradient Descent]], [[Lagrange Duality & KKT]], [[Proximal & Subgradient Methods]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Gradient Descent]], [[Lagrange Duality & KKT]], [[Proximal & Subgradient Methods]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Visual Cheatsheet:** [[Cheatsheet - Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Referenced Across Monographs:** [[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
