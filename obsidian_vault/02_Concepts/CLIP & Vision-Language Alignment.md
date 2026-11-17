---
type: architecture
domain: "D03"
aliases: 
  - "CLIP"
  - "Vision-Language Models"
part_of: 
  - "[[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
  - "[[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Transformer Architecture]]"
  - "[[Vision Transformer (ViT)]]"
enables: []
generalizes: []
special_case_of: []
related: 
  - "[[Classifier-Free Guidance (CFG)]]"
  - "[[Sparse vs Dense Retrieval & RRF]]"
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

# CLIP & Vision-Language Alignment

> [!abstract] Definition (Architecture)
> Dual image-text encoders trained with symmetric InfoNCE contrastive loss enabling zero-shot classification and open-vocabulary vision.

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Transformer Architecture]], [[Vision Transformer (ViT)]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | — |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Classifier-Free Guidance (CFG)]], [[Sparse vs Dense Retrieval & RRF]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Visual Cheatsheet:** [[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Referenced Across Monographs:** [[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
