---
type: algorithm
domain: "D01"
aliases: []
part_of: 
  - "[[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]"
  - "[[Cheatsheet - Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Stochastic Gradient Descent (SGD)]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Stochastic Gradient Descent (SGD)]]"
contrasts_with: []
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

# SVRG & Variance Reduction

> [!abstract] Definition (Algorithm)
> Control-variate anchor snapshots eliminate asymptotic gradient variance, recovering linear convergence on finite sums.

$$
\tilde{\boldsymbol{g}}_t = \nabla f_i(\boldsymbol{\theta}_t) - \nabla f_i(\tilde{\boldsymbol{\theta}}) + \nabla f(\tilde{\boldsymbol{\theta}})
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Stochastic Gradient Descent (SGD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Stochastic Gradient Descent (SGD)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Visual Cheatsheet:** [[Cheatsheet - Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Referenced Across Monographs:** [[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Handbook of Convergence Theorems]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
