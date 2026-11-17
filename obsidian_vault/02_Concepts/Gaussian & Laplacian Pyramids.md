---
type: architecture
domain: "D03"
aliases: 
  - "Laplacian Pyramid"
  - "Feature Pyramid Network (FPN)"
part_of: 
  - "[[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]"
  - "[[D03 - Neural Architectures, Representations & Regularization]]"
prerequisites: 
  - "[[Image Filtering & 2D Fourier Transform]]"
enables: 
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Optical Flow (Lucas-Kanade & Horn-Schunck)]]"
generalizes: []
special_case_of: []
related: 
  - "[[Image Filtering & 2D Fourier Transform]]"
  - "[[Object Detection (R-CNN, SSD & DETR)]]"
  - "[[Optical Flow (Lucas-Kanade & Horn-Schunck)]]"
contrasts_with: []
dual_to: []
sources: 
  - "[[Source - Antonio Torralba, Phillip Isola, and William Freeman]]"
  - "[[Source - Understanding Deep Learning]]"
tags:
  - architecture
  - domain-03
---

# Gaussian & Laplacian Pyramids

> [!abstract] Definition (Architecture)
> Multiresolution bandpass decomposition storing downsampled Gaussian levels and Laplacian residuals with 4/3 memory overhead.

$$
L_\ell = G_\ell - \uparrow_2(G_{\ell+1}), \quad I = \sum_\ell \uparrow_2^\ell L_\ell
$$

## Semantic Graph Relations

| Direction | Relation | Connected Nodes |
| :--- | :--- | :--- |
| Directed ($\leftarrow$) | **Prerequisites** (`requires`) | [[Image Filtering & 2D Fourier Transform]] |
| Directed ($\rightarrow$) | **Enables / Unlocks** (`enables`) | [[Object Detection (R-CNN, SSD & DETR)]], [[Optical Flow (Lucas-Kanade & Horn-Schunck)]] |
| Directed ($\supset$) | **Generalizes** | — |
| Directed ($\subset$) | **Special Case Of** | — |
| Undirected ($\leftrightarrow$) | **Related Concepts** | [[Image Filtering & 2D Fourier Transform]], [[Object Detection (R-CNN, SSD & DETR)]], [[Optical Flow (Lucas-Kanade & Horn-Schunck)]] |
| Undirected ($\leftrightarrow$) | **Contrasts With** | — |
| Undirected ($\rightleftarrows$) | **Dual To** | — |

## Monograph & Visual Anchors

- **Primary Monograph:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Visual Cheatsheet:** [[Cheatsheet - Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Referenced Across Monographs:** [[Multiscale Pyramids, Optical Flow, Epipolar Geometry, Stereo & Radiance Fields]]
- **Primary Textbooks:** [[Source - Antonio Torralba, Phillip Isola, and William Freeman]], [[Source - Understanding Deep Learning]]
