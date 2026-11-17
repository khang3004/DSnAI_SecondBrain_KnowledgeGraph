---
type: metric
domain: "D01"
aliases: []
part_of: 
  - "[[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: []
enables: []
generalizes: 
  - "[[KL Divergence]]"
special_case_of: []
related: []
contrasts_with: 
  - "[[Wasserstein Distance]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - metric
  - domain-01
---

# f-Divergence

> [!abstract] Definition (Metric)
> Convex-generator family of probability divergences satisfying the Data Processing Inequality and Pinsker-type bounds.

$$
D_f(p \| q) = \int q(x)\, f\!\left(\frac{p(x)}{q(x)}\right)\mathrm{d}x
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | [[KL Divergence]] |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | — |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Wasserstein Distance]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Visual Cheatsheet:** [[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Referenced Across Monographs:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
