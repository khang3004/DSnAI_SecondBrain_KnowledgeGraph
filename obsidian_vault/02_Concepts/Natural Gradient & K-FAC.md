---
type: algorithm
domain: "D01"
aliases: 
  - "Natural Gradient"
  - "L-BFGS"
part_of: 
  - "[[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]"
  - "[[Cheatsheet - Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Gradient Descent]]"
  - "[[Maximum Likelihood & Fisher Info]]"
enables: 
  - "[[TRPO]]"
generalizes: []
special_case_of: []
related: 
  - "[[Maximum Likelihood & Fisher Info]]"
  - "[[TRPO]]"
contrasts_with: 
  - "[[AdamW]]"
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Deep Learning]]"
  - "[[Source - Handbook of Convergence Theorems]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - algorithm
  - domain-01
---

# Natural Gradient & K-FAC

> [!abstract] Definition (Algorithm)
> Steepest descent in KL-metric via inverse Fisher Information Matrix (or Kronecker-factored block approximation), invariant to parameterization.

$$
\Delta\boldsymbol{\theta} \propto -\boldsymbol{F}(\boldsymbol{\theta})^{-1}\nabla\mathcal{L}(\boldsymbol{\theta})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Gradient Descent]], [[Maximum Likelihood & Fisher Info]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[TRPO]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Maximum Likelihood & Fisher Info]], [[TRPO]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[AdamW]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Visual Cheatsheet:** [[Cheatsheet - Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Referenced Across Monographs:** [[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Handbook of Convergence Theorems]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
