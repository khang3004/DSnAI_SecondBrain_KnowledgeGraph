---
type: metric
domain: "D02"
aliases: 
  - "VC Dimension"
  - "PAC Learning"
part_of: 
  - "[[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]"
  - "[[Cheatsheet - Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]"
  - "[[D02 - Classical, Kernel & Probabilistic Machine Learning]]"
prerequisites: 
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Concentration Inequalities]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Cauchy-Schwarz Inequality]]"
  - "[[Concentration Inequalities]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]]"
  - "[[Source - Machine Learning co ban]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - metric
  - domain-02
---

# Rademacher Complexity & VC Dimension

> [!abstract] Definition (Metric)
> Distribution-dependent and combinatorial capacity measures yielding uniform convergence generalization bounds.

$$
R(\mathcal{F}) - \hat{R}_N(\mathcal{F}) \le 2\mathfrak{R}_N(\mathcal{F}) + \mathcal{O}\!\left(\sqrt{\frac{\log(1/\delta)}{N}}\right)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Cauchy-Schwarz Inequality]], [[Concentration Inequalities]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Cauchy-Schwarz Inequality]], [[Concentration Inequalities]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]
- **Visual Cheatsheet:** [[Cheatsheet - Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]
- **Referenced Across Monographs:** [[Bias-Variance Decomposition, Overfitting, Generalization & Double Descent]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Introduction-PROBABILITY-for-DATA-SCIENCE]], [[Source - Machine Learning co ban]], [[Source - Understanding Deep Learning]]
