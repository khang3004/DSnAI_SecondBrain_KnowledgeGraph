---
type: algorithm
domain: "D01"
aliases: 
  - "Heavy-Ball Momentum"
  - "Nesterov Accelerated Gradient"
part_of: 
  - "[[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]"
  - "[[Cheatsheet - Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Condition Number]]"
  - "[[Gradient Descent]]"
enables: 
  - "[[AdamW]]"
generalizes: []
special_case_of: []
related: 
  - "[[AdamW]]"
  - "[[Condition Number]]"
  - "[[Gradient Descent]]"
contrasts_with: []
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

# Momentum & Nesterov NAG

> [!abstract] Definition (Algorithm)
> Velocity accumulation accelerates convergence across ill-conditioned valleys from O(kappa) to O(sqrt(kappa)) and O(1/T^2) convex.

$$
\boldsymbol{v}_{t+1} = \beta\boldsymbol{v}_t + \nabla f(\boldsymbol{\theta}_t - \eta\beta\boldsymbol{v}_t), \quad \boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta\boldsymbol{v}_{t+1}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Condition Number]], [[Gradient Descent]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[AdamW]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[AdamW]], [[Condition Number]], [[Gradient Descent]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Visual Cheatsheet:** [[Cheatsheet - Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Referenced Across Monographs:** [[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Handbook of Convergence Theorems]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
