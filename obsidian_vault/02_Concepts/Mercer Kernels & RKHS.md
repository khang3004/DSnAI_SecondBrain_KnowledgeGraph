---
type: theorem
domain: "D02"
aliases: 
  - "Kernel Trick"
  - "RKHS"
  - "Representer Theorem"
part_of: 
  - "[[Support Vector Machines, Margin Duality & Mercer Kernel Methods]]"
  - "[[Cheatsheet - Support Vector Machines, Margin Duality & Mercer Kernel Methods]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Eigendecomposition & Spectral Theorem]]"
  - "[[Inner Product & Norms]]"
enables: 
  - "[[Gaussian Processes]]"
generalizes: []
special_case_of: []
related: 
  - "[[Gaussian Processes]]"
  - "[[Support Vector Machines (SVM)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - theorem
  - domain-02
---

# Mercer Kernels & RKHS

> [!abstract] Definition (Theorem)
> Positive-definite kernels induce implicit inner products in Reproducing Kernel Hilbert Spaces; Representer Theorem guarantees finite-sum solutions.

$$
k(\boldsymbol{x},\boldsymbol{x}') = \langle \phi(\boldsymbol{x}), \phi(\boldsymbol{x}') \rangle_{\mathcal{H}}, \quad f^*(\cdot) = \sum_{i=1}^N \alpha_i k(\boldsymbol{x}_i, \cdot)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Eigendecomposition & Spectral Theorem]], [[Inner Product & Norms]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Gaussian Processes]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Gaussian Processes]], [[Support Vector Machines (SVM)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Support Vector Machines, Margin Duality & Mercer Kernel Methods]]
- **Visual Cheatsheet:** [[Cheatsheet - Support Vector Machines, Margin Duality & Mercer Kernel Methods]]
- **Referenced Across Monographs:** [[Support Vector Machines, Margin Duality & Mercer Kernel Methods]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
