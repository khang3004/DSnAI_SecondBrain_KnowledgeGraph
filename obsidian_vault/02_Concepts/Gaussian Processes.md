---
type: object
domain: "D02"
aliases: 
  - "GP"
part_of: 
  - "[[Probabilistic Graphical Models, Message Passing & Gaussian Processes]]"
  - "[[Cheatsheet - Probabilistic Graphical Models, Message Passing & Gaussian Processes]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Mercer Kernels & RKHS]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Mercer Kernels & RKHS]]"
  - "[[Multivariate Gaussian & Schur Complement]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - object
  - domain-02
---

# Gaussian Processes

> [!abstract] Definition (Object)
> Nonparametric Bayesian prior over functions whose finite-dimensional marginals are multivariate Gaussians governed by a kernel.

$$
f_* \mid \boldsymbol{X},\boldsymbol{y},\boldsymbol{X}_* \sim \mathcal{N}\bigl(\boldsymbol{K}_*^\top\boldsymbol{K}_y^{-1}\boldsymbol{y},\; \boldsymbol{K}_{**}-\boldsymbol{K}_*^\top\boldsymbol{K}_y^{-1}\boldsymbol{K}_*\bigr)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Mercer Kernels & RKHS]], [[Multivariate Gaussian & Schur Complement]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Mercer Kernels & RKHS]], [[Multivariate Gaussian & Schur Complement]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Probabilistic Graphical Models, Message Passing & Gaussian Processes]]
- **Visual Cheatsheet:** [[Cheatsheet - Probabilistic Graphical Models, Message Passing & Gaussian Processes]]
- **Referenced Across Monographs:** [[Probabilistic Graphical Models, Message Passing & Gaussian Processes]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]]
