---
type: algorithm
domain: "D01"
aliases: []
part_of: 
  - "[[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]"
  - "[[Cheatsheet - Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Momentum & Nesterov NAG]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Momentum & Nesterov NAG]]"
  - "[[Weight Decay & Dropout]]"
contrasts_with: 
  - "[[Natural Gradient & K-FAC]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
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

# AdamW

> [!abstract] Definition (Algorithm)
> Adaptive per-coordinate RMS scaling with bias-corrected moments and decoupled weight decay; standard optimizer for Transformers.

$$
\boldsymbol{\theta}_{t+1} = (1-\eta\lambda)\boldsymbol{\theta}_t - \eta \frac{\hat{\boldsymbol{m}}_t}{\sqrt{\hat{\boldsymbol{v}}_t} + \varepsilon}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Momentum & Nesterov NAG]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Momentum & Nesterov NAG]], [[Weight Decay & Dropout]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Natural Gradient & K-FAC]], [[Stochastic Gradient Descent (SGD)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Visual Cheatsheet:** [[Cheatsheet - Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Referenced Across Monographs:** [[Momentum, Nesterov, Adam - AdamW, LR Schedules & Second-Order - Natural Gradient]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Handbook of Convergence Theorems]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
