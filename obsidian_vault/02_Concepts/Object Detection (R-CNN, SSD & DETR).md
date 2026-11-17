---
type: architecture
domain: "D03"
aliases: 
  - "Faster R-CNN"
  - "SSD"
  - "DETR"
part_of: 
  - "[[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
  - "[[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[CNN & Equivariance]]"
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[IoU, Anchor Boxes & NMS]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Gaussian & Laplacian Pyramids]]"
  - "[[IoU, Anchor Boxes & NMS]]"
  - "[[Semantic & Instance Segmentation (FCN, U-Net & SAM)]]"
  - "[[Transformer Architecture]]"
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

# Object Detection (R-CNN, SSD & DETR)

> [!abstract] Definition (Architecture)
> Two-stage R-CNN (RPN + RoIAlign) vs. one-stage SSD/YOLO anchor grids vs. anchor-free DETR transformer bipartite Hungarian matching.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[CNN & Equivariance]], [[Gaussian & Laplacian Pyramids]], [[IoU, Anchor Boxes & NMS]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Gaussian & Laplacian Pyramids]], [[IoU, Anchor Boxes & NMS]], [[Semantic & Instance Segmentation (FCN, U-Net & SAM)]], [[Transformer Architecture]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Visual Cheatsheet:** [[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Referenced Across Monographs:** [[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
