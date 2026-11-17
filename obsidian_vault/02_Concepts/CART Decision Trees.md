---
type: algorithm
domain: "D02"
aliases: []
part_of: 
  - "[[Decision Trees, Bagging, Random Forests & Gradient Boosting]]"
  - "[[Cheatsheet - Decision Trees, Bagging, Random Forests & Gradient Boosting]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: []
enables: 
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Random Forests & Bagging]]"
generalizes: []
special_case_of: []
related: 
  - "[[Gradient Boosted Decision Trees (GBDT)]]"
  - "[[Random Forests & Bagging]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Designing Machine Learning Systems]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - algorithm
  - domain-02
---

# CART Decision Trees

> [!abstract] Definition (Algorithm)
> Greedy recursive axis-aligned partitioning minimizing Gini impurity, entropy, or MSE within leaf regions.

$$
F(\boldsymbol{x}) = \sum_{j=1}^J w_j \mathbb{I}(\boldsymbol{x}\in R_j)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Gradient Boosted Decision Trees (GBDT)]], [[Random Forests & Bagging]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Gradient Boosted Decision Trees (GBDT)]], [[Random Forests & Bagging]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Visual Cheatsheet:** [[Cheatsheet - Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Referenced Across Monographs:** [[Decision Trees, Bagging, Random Forests & Gradient Boosting]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Designing Machine Learning Systems]], [[Source - Probabilistic Machine Learning]]
