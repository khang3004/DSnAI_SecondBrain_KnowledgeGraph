---
type: architecture
domain: "D03"
aliases: 
  - "KAN"
part_of: 
  - "[[Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]"
  - "[[Cheatsheet - Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: []
enables: []
generalizes: []
special_case_of: []
related: []
contrasts_with: 
  - "[[MLP & Universal Approximation]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Math behind KAN]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-03
---

# Kolmogorov-Arnold Networks (KAN)

> [!abstract] Definition (Architecture)
> Replaces fixed node activations with learnable univariate B-spline functions on graph edges.

$$
\boldsymbol{\Phi}(\boldsymbol{x})_q = \sum_{p=1}^{n_{\mathrm{in}}} \phi_{q,p}(x_p)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | — |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[MLP & Universal Approximation]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]
- **Visual Cheatsheet:** [[Cheatsheet - Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]
- **Referenced Across Monographs:** [[Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Math behind KAN]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
