---
type: algorithm
domain: "D02"
aliases: []
part_of: 
  - "[[Decision Trees, Bagging, Random Forests & Gradient Boosting]]"
  - "[[Cheatsheet - Decision Trees, Bagging, Random Forests & Gradient Boosting]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Bias-Variance Trade-off]]"
  - "[[CART Decision Trees]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Bias-Variance Trade-off]]"
  - "[[CART Decision Trees]]"
contrasts_with: 
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-02
---

# Random Forests & Bagging

> [!abstract] Definition (Algorithm)
> Parallel ensemble of decorrelated bootstrap trees reducing prediction variance without increasing bias.

$$
\mathrm{Var}\!\left(\frac{1}{B}\sum_{b=1}^B T_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Bias-Variance Trade-off]], [[CART Decision Trees]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Bias-Variance Trade-off]], [[CART Decision Trees]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Gradient Boosted Decision Trees (GBDT)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Visual Cheatsheet:** [[Cheatsheet - Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Referenced Across Monographs:** [[Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Designing Machine Learning Systems]], [[Source - Probabilistic Machine Learning]]
