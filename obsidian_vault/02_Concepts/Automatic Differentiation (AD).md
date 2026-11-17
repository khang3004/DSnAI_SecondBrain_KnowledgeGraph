---
type: algorithm
domain: "D01"
aliases: 
  - "Backpropagation"
  - "Reverse-Mode AD"
  - "Forward-Mode AD"
part_of: 
  - "[[Matrix Calculus, Computational Graphs & Automatic Differentiation]]"
  - "[[Cheatsheet - Matrix Calculus, Computational Graphs & Automatic Differentiation]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Matrix Calculus & Jacobians]]"
enables: 
  - "[[MLP & Universal Approximation]]"
  - "[[RNN, LSTM & GRU]]"
  - "[[Reparameterization Trick]]"
  - "[[Score Matching (DSM & SSM)]]"
generalizes: []
special_case_of: []
related: 
  - "[[MLP & Universal Approximation]]"
  - "[[Matrix Calculus & Jacobians]]"
  - "[[Score Matching (DSM & SSM)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Mathematics for Machine Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-01
---

# Automatic Differentiation (AD)

> [!abstract] Definition (Algorithm)
> Exact machine-precision derivatives via computational-graph chain rule in Forward (JVP) or Reverse (VJP / backprop) mode.

$$
\text{Forward (JVP): } \dot{\boldsymbol{v}} = \boldsymbol{J}\dot{\boldsymbol{x}} \quad\text{vs.}\quad \text{Reverse (VJP): } \bar{\boldsymbol{x}} = \boldsymbol{J}^\top \bar{\boldsymbol{y}}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Matrix Calculus & Jacobians]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[MLP & Universal Approximation]], [[RNN, LSTM & GRU]], [[Reparameterization Trick]], [[Score Matching (DSM & SSM)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[MLP & Universal Approximation]], [[Matrix Calculus & Jacobians]], [[Score Matching (DSM & SSM)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Matrix Calculus, Computational Graphs & Automatic Differentiation]]
- **Visual Cheatsheet:** [[Cheatsheet - Matrix Calculus, Computational Graphs & Automatic Differentiation]]
- **Referenced Across Monographs:** [[Matrix Calculus, Computational Graphs & Automatic Differentiation]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Machine Learning co ban]], [[Source - Mathematics for Machine Learning]], [[Source - Probabilistic Machine Learning]]
