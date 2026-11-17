---
type: architecture
domain: "D03"
aliases: 
  - "FCN"
  - "U-Net"
  - "SAM"
  - "Transposed Convolution"
part_of: 
  - "[[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
  - "[[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[CNN & Equivariance]]"
  - "[[Residual Connections]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[DDPM]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-03
---

# Semantic & Instance Segmentation (FCN, U-Net & SAM)

> [!abstract] Definition (Architecture)
> Dense pixel-level classification using transposed convolutions or skip-connected encoder-decoders (FCN, U-Net) and promptable SAM.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[CNN & Equivariance]], [[Residual Connections]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[DDPM]], [[Object Detection (R-CNN, SSD & DETR)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Visual Cheatsheet:** [[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Referenced Across Monographs:** [[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
