---
type: algorithm
domain: "D01"
aliases: []
part_of: 
  - "[[Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]"
  - "[[Cheatsheet - Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Condition Number]]"
  - "[[Convexity & Subgradients]]"
enables: 
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[Natural Gradient & K-FAC]]"
  - "[[Polyak-Lojasiewicz (PL) Condition]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Condition Number]]"
  - "[[Convexity & Subgradients]]"
  - "[[Momentum & Nesterov NAG]]"
  - "[[Polyak-Lojasiewicz (PL) Condition]]"
  - "[[Stochastic Gradient Descent (SGD)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Fall 2022]]"
  - "[[Source - Foundations]]"
  - "[[Source - Handbook of Convergence Theorems]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-01
---

# Gradient Descent

> [!abstract] Definition (Algorithm)
> Steepest Euclidean descent with step eta <= 1/L; achieves O(1/T) on convex smooth and linear O(kappa log 1/eps) under SC or PL.

$$
\boldsymbol{x}_{t+1} = \boldsymbol{x}_t - \eta_t \nabla f(\boldsymbol{x}_t)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Condition Number]], [[Convexity & Subgradients]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Gradient Boosted Decision Trees (GBDT)]], [[Momentum & Nesterov NAG]], [[Natural Gradient & K-FAC]], [[Polyak-Lojasiewicz (PL) Condition]], [[Stochastic Gradient Descent (SGD)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Condition Number]], [[Convexity & Subgradients]], [[Momentum & Nesterov NAG]], [[Polyak-Lojasiewicz (PL) Condition]], [[Stochastic Gradient Descent (SGD)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]
- **Visual Cheatsheet:** [[Cheatsheet - Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]
- **Referenced Across Monographs:** [[Gradient Descent Dynamics - Strong Convexity, Smoothness, PL Condition & Saddle Escape]]
- **Primary Textbooks:** [[Source - Fall 2022]], [[Source - Foundations]], [[Source - Handbook of Convergence Theorems]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
