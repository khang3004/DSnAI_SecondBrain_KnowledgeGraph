---
type: object
domain: "D01"
aliases: 
  - "Jacobian"
  - "Hessian"
part_of: 
  - "[[Matrix Calculus, Computational Graphs & Automatic Differentiation]]"
  - "[[Cheatsheet - Matrix Calculus, Computational Graphs & Automatic Differentiation]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Vector Space & Subspaces]]"
enables: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[Normalizing Flows]]"
generalizes: []
special_case_of: []
related: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[Normalizing Flows]]"
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

# Matrix Calculus & Jacobians

> [!abstract] Definition (Object)
> Differential and trace-duality rules for vector/matrix-valued functions, Hessians, and Kronecker-vec identities.

$$
\mathrm{d}f = \mathrm{Tr}(\boldsymbol{G}^\top \mathrm{d}\boldsymbol{X}), \quad \boldsymbol{J}_{\boldsymbol{f}}(\boldsymbol{x})_{ij} = \frac{\partial f_i}{\partial x_j}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Vector Space & Subspaces]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Automatic Differentiation (AD)]], [[Normalizing Flows]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Automatic Differentiation (AD)]], [[Normalizing Flows]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Matrix Calculus, Computational Graphs & Automatic Differentiation]]
- **Visual Cheatsheet:** [[Cheatsheet - Matrix Calculus, Computational Graphs & Automatic Differentiation]]
- **Referenced Across Monographs:** [[Matrix Calculus, Computational Graphs & Automatic Differentiation]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
