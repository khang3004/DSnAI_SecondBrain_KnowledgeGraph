---
type: theorem
domain: "D03"
aliases: 
  - "Multilayer Perceptron"
  - "Universal Approximation Theorem"
part_of: 
  - "[[Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]"
  - "[[Cheatsheet - Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Automatic Differentiation (AD)]]"
enables: 
  - "[[Factorization Machines & DeepFM]]"
  - "[[He & Xavier Initialization]]"
generalizes: []
special_case_of: []
related: 
  - "[[Automatic Differentiation (AD)]]"
  - "[[Radiance Fields (NeRF & 3DGS)]]"
contrasts_with: 
  - "[[Kolmogorov-Arnold Networks (KAN)]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Math behind KAN]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - theorem
  - domain-03
---

# MLP & Universal Approximation

> [!abstract] Definition (Theorem)
> Feedforward ReLU networks partition input space into exponentially many linear regions O((D/d)^d K) with depth K.

$$
\boldsymbol{h}^{(l+1)} = \sigma\bigl(\boldsymbol{W}^{(l)}\boldsymbol{h}^{(l)} + \boldsymbol{b}^{(l)}\bigr)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Automatic Differentiation (AD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Factorization Machines & DeepFM]], [[He & Xavier Initialization]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Automatic Differentiation (AD)]], [[Radiance Fields (NeRF & 3DGS)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Kolmogorov-Arnold Networks (KAN)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]
- **Visual Cheatsheet:** [[Cheatsheet - Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]
- **Referenced Across Monographs:** [[Feedforward Networks, Universal Approximation & Kolmogorov-Arnold Networks (KAN)]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Math behind KAN]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
