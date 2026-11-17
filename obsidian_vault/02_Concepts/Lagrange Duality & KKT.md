---
type: theorem
domain: "D01"
aliases: 
  - "KKT Conditions"
  - "Lagrange Multipliers"
part_of: 
  - "[[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]"
  - "[[Cheatsheet - Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Convexity & Subgradients]]"
enables: 
  - "[[Kantorovich Duality]]"
  - "[[Support Vector Machines (SVM)]]"
  - "[[TRPO]]"
generalizes: []
special_case_of: []
related: 
  - "[[Convexity & Subgradients]]"
  - "[[Kantorovich Duality]]"
  - "[[Support Vector Machines (SVM)]]"
  - "[[TRPO]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - theorem
  - domain-01
---

# Lagrange Duality & KKT

> [!abstract] Definition (Theorem)
> Weak duality always holds (d* <= p*); Slater's condition guarantees strong duality and necessary/sufficient KKT optimality conditions.

$$
\mathcal{L}(\boldsymbol{x},\boldsymbol{\lambda},\boldsymbol{\nu}) = f_0(\boldsymbol{x}) + \sum_i \lambda_i f_i(\boldsymbol{x}) + \sum_j \nu_j h_j(\boldsymbol{x})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Convexity & Subgradients]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Kantorovich Duality]], [[Support Vector Machines (SVM)]], [[TRPO]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Convexity & Subgradients]], [[Kantorovich Duality]], [[Support Vector Machines (SVM)]], [[TRPO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Visual Cheatsheet:** [[Cheatsheet - Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Referenced Across Monographs:** [[Convex Sets, Fenchel - Lagrange Duality & KKT Conditions]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
