---
type: algorithm
domain: "D01"
aliases: 
  - "SGD"
part_of: 
  - "[[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]"
  - "[[Cheatsheet - Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Gradient Descent]]"
enables: 
  - "[[Proximal & Subgradient Methods]]"
  - "[[SVRG & Variance Reduction]]"
generalizes: []
special_case_of: []
related: 
  - "[[Gradient Descent]]"
  - "[[Proximal & Subgradient Methods]]"
  - "[[SVRG & Variance Reduction]]"
contrasts_with: 
  - "[[AdamW]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Handbook of Convergence Theorems]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - algorithm
  - domain-01
---

# Stochastic Gradient Descent (SGD)

> [!abstract] Definition (Algorithm)
> Unbiased noisy gradient updates converge to an O(eta sigma^2) noise ball under constant step size or O(1/t) under Robbins-Monro decay.

$$
\boldsymbol{\theta}_{t+1} = \boldsymbol{\theta}_t - \eta_t \boldsymbol{g}_t(\boldsymbol{\theta}_t), \quad \mathbb{E}[\boldsymbol{g}_t] = \nabla f(\boldsymbol{\theta}_t)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Gradient Descent]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Proximal & Subgradient Methods]], [[SVRG & Variance Reduction]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Gradient Descent]], [[Proximal & Subgradient Methods]], [[SVRG & Variance Reduction]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[AdamW]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Visual Cheatsheet:** [[Cheatsheet - Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Referenced Across Monographs:** [[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Handbook of Convergence Theorems]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
