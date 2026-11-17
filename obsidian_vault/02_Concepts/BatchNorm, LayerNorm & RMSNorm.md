---
type: architecture
domain: "D03"
aliases: 
  - "BatchNorm"
  - "LayerNorm"
  - "RMSNorm"
part_of: 
  - "[[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
  - "[[Cheatsheet - Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: []
enables: 
  - "[[Transformer Architecture]]"
generalizes: []
special_case_of: []
related: 
  - "[[Residual Connections]]"
  - "[[Transformer Architecture]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Build a Large Language Model (From Scratch) (for sam speaks)]]"
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-03
---

# BatchNorm, LayerNorm & RMSNorm

> [!abstract] Definition (Architecture)
> Stabilizes activation scales across layers: BatchNorm over (N,H,W), LayerNorm over feature dim D, RMSNorm without mean-centering.

$$
\mathrm{RMSNorm}(\boldsymbol{x}) = \frac{\boldsymbol{x}}{\sqrt{\frac{1}{D}\sum_i x_i^2 + \varepsilon}} \odot \boldsymbol{\gamma}
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Transformer Architecture]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Residual Connections]], [[Transformer Architecture]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Visual Cheatsheet:** [[Cheatsheet - Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Referenced Across Monographs:** [[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Understanding Deep Learning]]
