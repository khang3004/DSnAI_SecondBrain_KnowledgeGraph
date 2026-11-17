---
type: theorem
domain: "D02"
aliases: []
part_of: 
  - "[[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]"
  - "[[Cheatsheet - Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: []
enables: 
  - "[[Double Descent]]"
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Random Forests & Bagging]]"
generalizes: []
special_case_of: []
related: 
  - "[[Double Descent]]"
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Linear & Ridge Regression]]"
  - "[[Random Forests & Bagging]]"
  - "[[Weight Decay & Dropout]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - theorem
  - domain-02
---

# Bias-Variance Trade-off

> [!abstract] Definition (Theorem)
> Expected squared error decomposes into irreducible noise, squared approximation bias, and estimation variance.

$$
\mathbb{E}[(y-\hat{f}(\boldsymbol{x}))^2] = \sigma_\varepsilon^2 + \mathrm{Bias}[\hat{f}(\boldsymbol{x})]^2 + \mathrm{Var}[\hat{f}(\boldsymbol{x})]
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Double Descent]], [[Gradient Boosted Decision Trees (GBDT)]], [[Random Forests & Bagging]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Double Descent]], [[Gradient Boosted Decision Trees (GBDT)]], [[Linear & Ridge Regression]], [[Random Forests & Bagging]], [[Weight Decay & Dropout]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]
- **Visual Cheatsheet:** [[Cheatsheet - Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]
- **Referenced Across Monographs:** [[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Understanding Deep Learning]]
