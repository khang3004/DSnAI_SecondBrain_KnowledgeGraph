---
type: architecture
domain: "D03"
aliases: 
  - "ViT"
part_of: 
  - "[[Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]"
  - "[[Cheatsheet - Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Self-Attention & Multi-Head Attention]]"
enables: 
  - "[[CLIP & Vision-Language Alignment]]"
generalizes: []
special_case_of: []
related: 
  - "[[Transformer Architecture]]"
contrasts_with: 
  - "[[CNN & Equivariance]]"
dual_to: []
sources: 
  - "[[Source - Deep Learning]]"
  - "[[Source - Foundations]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-03
---

# Vision Transformer (ViT)

> [!abstract] Definition (Architecture)
> Tokenizes images into patches and applies global self-attention with lower inductive bias and higher data-scaling capacity than CNNs.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Self-Attention & Multi-Head Attention]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[CLIP & Vision-Language Alignment]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Transformer Architecture]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[CNN & Equivariance]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]
- **Visual Cheatsheet:** [[Cheatsheet - Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]
- **Referenced Across Monographs:** [[Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
