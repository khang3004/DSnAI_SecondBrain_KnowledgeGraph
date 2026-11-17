---
type: architecture
domain: "D03"
aliases: 
  - "Skip Connections"
  - "ResNet"
part_of: 
  - "[[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
  - "[[Cheatsheet - Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: []
enables: 
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
  - "[[Transformer Architecture]]"
generalizes: []
special_case_of: []
related: 
  - "[[BatchNorm, LayerNorm & RMSNorm]]"
  - "[[CNN & Equivariance]]"
  - "[[He & Xavier Initialization]]"
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

# Residual Connections

> [!abstract] Definition (Architecture)
> Identity skip paths x + F(x) preserve gradient norms across hundreds of layers and yield implicit ensemble behavior.

$$
\boldsymbol{x}_{l+1} = \boldsymbol{x}_l + \mathcal{F}_l(\boldsymbol{x}_l; \boldsymbol{\theta}_l)
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]], [[Transformer Architecture]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[BatchNorm, LayerNorm & RMSNorm]], [[CNN & Equivariance]], [[He & Xavier Initialization]], [[Transformer Architecture]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Visual Cheatsheet:** [[Cheatsheet - Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Referenced Across Monographs:** [[Initialization, Batch - LayerNorm, Residual Connections, Weight Decay & Dropout]]
- **Primary Textbooks:** [[Source - Build a Large Language Model (From Scratch) (for sam speaks)]], [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Understanding Deep Learning]]
