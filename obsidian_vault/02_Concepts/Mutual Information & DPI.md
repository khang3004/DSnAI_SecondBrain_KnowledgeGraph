---
type: theorem
domain: "D01"
aliases: 
  - "Mutual Information"
  - "Data Processing Inequality"
part_of: 
  - "[[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]"
  - "[[D01 - Mathematical Foundations for Machine Intelligence]]"
prerequisites: 
  - "[[KL Divergence]]"
  - "[[Shannon Entropy]]"
enables: 
  - "[[Information Bottleneck]]"
generalizes: []
special_case_of: []
related: 
  - "[[Information Bottleneck]]"
  - "[[Shannon Entropy]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - theorem
  - domain-01
---

# Mutual Information & DPI

> [!abstract] Definition (Theorem)
> Measures shared information I(X;Y) = D_KL(p(x,y) || p(x)p(y)); cannot increase along Markov chains (Data Processing Inequality).

$$
X \to Y \to Z \implies I(X; Y) \ge I(X; Z)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[KL Divergence]], [[Shannon Entropy]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Information Bottleneck]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Information Bottleneck]], [[Shannon Entropy]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Visual Cheatsheet:** [[Cheatsheet - Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Referenced Across Monographs:** [[Shannon Entropy, KL Divergence, f-Divergences & Information Bottleneck]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
