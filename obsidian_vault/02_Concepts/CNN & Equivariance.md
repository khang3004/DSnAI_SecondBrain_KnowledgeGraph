---
type: architecture
domain: "D03"
aliases: 
  - "Convolutional Neural Network"
  - "Translation Equivariance"
part_of: 
  - "[[Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]"
  - "[[Cheatsheet - Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: []
enables: 
  - "[[Image Filtering & 2D Fourier Transform]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Residual Connections]]"
contrasts_with: 
  - "[[Vision Transformer (ViT)]]"
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

# CNN & Equivariance

> [!abstract] Definition (Architecture)
> Weight-shared local convolutions enforce translation equivariance f(T_g x) = T_g f(x) and hierarchical receptive fields.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Image Filtering & 2D Fourier Transform]], [[Object Detection (R-CNN, SSD & DETR)]], [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Residual Connections]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | [[Vision Transformer (ViT)]] |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]
- **Visual Cheatsheet:** [[Cheatsheet - Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]
- **Referenced Across Monographs:** [[Convolutional Neural Networks, Translation Equivariance & Vision Backbones]]
- **Primary Textbooks:** [[Source - Deep Learning]], [[Source - Foundations]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
