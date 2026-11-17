---
type: theorem
domain: "D01"
aliases: 
  - "Hoeffding Inequality"
  - "Sub-Gaussian Tail Bound"
part_of: 
  - "[[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
  - "[[Cheatsheet - Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[Law of Large Numbers & CLT]]"
enables: 
  - "[[Multi-Armed Bandits & UCB]]"
  - "[[Rademacher Complexity & VC Dimension]]"
generalizes: []
special_case_of: []
related: 
  - "[[Law of Large Numbers & CLT]]"
  - "[[Multi-Armed Bandits & UCB]]"
  - "[[Rademacher Complexity & VC Dimension]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Probabilistic Machine Learning]]"
tags:
  - theorem
  - domain-01
---

# Concentration Inequalities

> [!abstract] Definition (Theorem)
> Non-asymptotic tail bounds (Markov, Chebyshev, Hoeffding, Bernstein) giving exponential decay for bounded/sub-Gaussian variables.

$$
\mathbb{P}(|\bar{X}_N - \mu| \ge \varepsilon) \le 2\exp\!\left(-\frac{N\varepsilon^2}{2\sigma^2}\right)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Law of Large Numbers & CLT]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Multi-Armed Bandits & UCB]], [[Rademacher Complexity & VC Dimension]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Law of Large Numbers & CLT]], [[Multi-Armed Bandits & UCB]], [[Rademacher Complexity & VC Dimension]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Visual Cheatsheet:** [[Cheatsheet - Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Referenced Across Monographs:** [[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Primary Textbooks:** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]]
