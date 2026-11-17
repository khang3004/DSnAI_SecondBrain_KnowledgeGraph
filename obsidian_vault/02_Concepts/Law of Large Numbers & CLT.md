---
type: theorem
domain: "D01"
aliases: 
  - "LLN"
  - "Central Limit Theorem"
part_of: 
  - "[[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
  - "[[Cheatsheet - Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: []
enables: 
  - "[[Concentration Inequalities]]"
  - "[[Maximum Likelihood & Fisher Info]]"
generalizes: []
special_case_of: []
related: 
  - "[[Concentration Inequalities]]"
  - "[[Maximum Likelihood & Fisher Info]]"
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

# Law of Large Numbers & CLT

> [!abstract] Definition (Theorem)
> Sample averages converge almost surely to population means (LLN) with sqrt(N)-scaled Gaussian fluctuations (CLT).

$$
\sqrt{N}(\bar{X}_N - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Concentration Inequalities]], [[Maximum Likelihood & Fisher Info]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Concentration Inequalities]], [[Maximum Likelihood & Fisher Info]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Visual Cheatsheet:** [[Cheatsheet - Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Referenced Across Monographs:** [[Concentration Inequalities, Limit Theorems & Statistical Estimation (MLE - MAP)]]
- **Primary Textbooks:** [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Probabilistic Machine Learning]]
