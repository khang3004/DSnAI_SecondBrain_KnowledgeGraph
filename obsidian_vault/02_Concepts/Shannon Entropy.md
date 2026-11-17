---
type: metric
domain: "D01"
aliases: 
  - "Entropy"
part_of: 
  - "[[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: []
enables: 
  - "[[KL Divergence]]"
  - "[[Mutual Information & DPI]]"
generalizes: []
special_case_of: []
related: 
  - "[[KL Divergence]]"
  - "[[Mutual Information & DPI]]"
contrasts_with: []
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

# Shannon Entropy

> [!abstract] Definition (Metric)
> Expected surprisal measuring uncertainty of a random variable and the fundamental lower bound on lossless source coding.

$$
H(X) = -\sum_{x} p(x)\log p(x)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[KL Divergence]], [[Mutual Information & DPI]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[KL Divergence]], [[Mutual Information & DPI]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Visual Cheatsheet:** [[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Referenced Across Monographs:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
