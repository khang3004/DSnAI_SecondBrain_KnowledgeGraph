---
type: algorithm
domain: "D01"
aliases: []
part_of: 
  - "[[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]"
  - "[[Cheatsheet - Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Convexity & Subgradients]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Convexity & Subgradients]]"
  - "[[Linear & Ridge Regression]]"
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

# Proximal & Subgradient Methods

> [!abstract] Definition (Algorithm)
> Splits smooth + nonsmooth composite objectives via proximal operator (e.g., soft-thresholding for L1 sparsity) instead of slow subgradients.

$$
\mathrm{prox}_{\eta g}(\boldsymbol{v}) = \arg\min_{\boldsymbol{x}}\left\{g(\boldsymbol{x}) + \frac{1}{2\eta}\|\boldsymbol{x}-\boldsymbol{v}\|_2^2\right\}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Convexity & Subgradients]], [[Stochastic Gradient Descent (SGD)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Convexity & Subgradients]], [[Linear & Ridge Regression]], [[Stochastic Gradient Descent (SGD)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Visual Cheatsheet:** [[Cheatsheet - Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Referenced Across Monographs:** [[Stochastic & Minibatch SGD, Expected Smoothness, Subgradients & Proximal Methods]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Handbook of Convergence Theorems]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
