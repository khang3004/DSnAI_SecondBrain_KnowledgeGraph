---
type: algorithm
domain: "D02"
aliases: 
  - "GBDT"
  - "XGBoost"
part_of: 
  - "[[Decision Trees, Bagging, Random Forests & Gradient Boosting]]"
  - "[[Cheatsheet - Decision Trees, Bagging, Random Forests & Gradient Boosting]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Bias-Variance Trade-off]]"
  - "[[CART Decision Trees]]"
  - "[[Gradient Descent]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Bias-Variance Trade-off]]"
  - "[[CART Decision Trees]]"
contrasts_with: 
  - "[[Random Forests & Bagging]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-02
---

# Gradient Boosted Decision Trees (GBDT)

> [!abstract] Definition (Algorithm)
> Sequential functional gradient descent in tree space (XGBoost/LightGBM) using second-order Newton leaf weights to slash bias.

$$
F_m(\boldsymbol{x}) = F_{m-1}(\boldsymbol{x}) + \nu f_m(\boldsymbol{x}), \quad w_j^* = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Bias-Variance Trade-off]], [[CART Decision Trees]], [[Gradient Descent]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Bias-Variance Trade-off]], [[CART Decision Trees]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Random Forests & Bagging]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Visual Cheatsheet:** [[Cheatsheet - Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Referenced Across Monographs:** [[Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Designing Machine Learning Systems]], [[Source - Probabilistic Machine Learning]]
