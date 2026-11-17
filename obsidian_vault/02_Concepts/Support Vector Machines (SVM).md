---
type: algorithm
domain: "D02"
aliases: 
  - "SVM"
part_of: 
  - "[[Support Vector Machines, Margin Duality & Mercer Kernel Methods]]"
  - "[[Cheatsheet - Support Vector Machines, Margin Duality & Mercer Kernel Methods]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Lagrange Duality & KKT]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Lagrange Duality & KKT]]"
  - "[[Mercer Kernels & RKHS]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-02
---

# Support Vector Machines (SVM)

> [!abstract] Definition (Algorithm)
> Maximum-margin hyperplane classifier solved via dual quadratic program over sparse support vectors.

$$
\min_{\boldsymbol{w},b}\frac{1}{2}\|\boldsymbol{w}\|_2^2 + C\sum_i \xi_i \quad \text{s.t. } y_i(\boldsymbol{w}^\top\boldsymbol{x}_i+b)\ge 1-\xi_i
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Lagrange Duality & KKT]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Lagrange Duality & KKT]], [[Mercer Kernels & RKHS]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Support Vector Machines, Margin Duality & Mercer Kernel Methods]]
- **Visual Cheatsheet:** [[Cheatsheet - Support Vector Machines, Margin Duality & Mercer Kernel Methods]]
- **Referenced Across Monographs:** [[Support Vector Machines, Margin Duality & Mercer Kernel Methods]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
