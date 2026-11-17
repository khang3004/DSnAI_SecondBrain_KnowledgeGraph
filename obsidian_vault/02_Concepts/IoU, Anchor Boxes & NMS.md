---
type: metric
domain: "D03"
aliases: 
  - "IoU"
  - "Non-Maximum Suppression"
  - "Focal Loss"
part_of: 
  - "[[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
  - "[[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: []
enables: 
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Dive into Deep Learning]]"
  - "[[Source - Probabilistic Machine Learning]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - metric
  - domain-03
---

# IoU, Anchor Boxes & NMS

> [!abstract] Definition (Metric)
> Jaccard bounding-box overlap, scale/aspect anchor priors, Focal Loss for class imbalance, and greedy Non-Maximum Suppression.

$$
\mathrm{IoU}(A,B) = \frac{|A\cap B|}{|A\cup B|}, \qquad \mathcal{L}_{\mathrm{FL}}(p_t) = -\alpha_t(1-p_t)^\gamma\log p_t
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | — |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Object Detection (R-CNN, SSD & DETR)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Object Detection (R-CNN, SSD & DETR)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Visual Cheatsheet:** [[Cheatsheet - Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Referenced Across Monographs:** [[Object Detection (R-CNN, SSD, DETR), Semantic Segmentation (FCN, U-Net, SAM) & Vision-Language Models]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Dive into Deep Learning]], [[Source - Probabilistic Machine Learning]], [[Source - Understanding Deep Learning]]
